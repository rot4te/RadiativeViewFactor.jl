# src/ViewFactorKernel.jl
# ---------------------------------------------------------------------------
# View-factor kernel and element-pair integrator.
#
# 3D (surface meshes, mesh_dim=2):
#   Fᵢⱼ = (1/Aᵢ) ∬_Aᵢ ∬_Aⱼ  [cos θᵢ cos θⱼ / (π r²)]  H_ij  dAⱼ dAᵢ
#
# 2D (curve meshes, mesh_dim=1, per unit depth):
#   Fᵢⱼ = (1/Lᵢ) ∫_Lᵢ  ∫_Lⱼ  [cos θᵢ cos θⱼ / (2 r)]   H_ij  dLⱼ dLᵢ
#
# The dimension is inferred from the element family of elem_i:
#   :line3, :line2  → 2D kernel and 1-D Gauss–Legendre quadrature
#   :quad,  :quad4  → 3D kernel and 2-D Gauss–Legendre quadrature
#   :tri,   :tri3   → 3D kernel and 2-D Dunavant quadrature
# ---------------------------------------------------------------------------

module ViewFactorKernel

using StaticArrays
using LinearAlgebra

import ..Quadrature:  gauss_legendre_1d, gauss_legendre_2d
import ..Geometry:    quad8_physical_point, quad8_normal_and_area_element,
                      quad4_physical_point, quad4_normal_and_area_element,
                      tri3_physical_point,  tri3_normal_and_area_element,
                      line3_physical_point, line3_normal_and_length_element,
                      line2_physical_point, line2_normal_and_length_element
import ..BVH:         BVHTree
import ..RayCast:     is_visible
import ..MeshIO:      SurfaceElement

export element_pair_view_factor

# ---------------------------------------------------------------------------
# Dunavant quadrature rules for the reference triangle
# ---------------------------------------------------------------------------

struct TriQuadRule
    points  :: Matrix{Float64}   # 2 × N  (ξ, η)
    weights :: Vector{Float64}   # N  (sum = 0.5 = area of reference triangle)
end

function tri_quad_rule(nquad::Int)::TriQuadRule
    if nquad <= 1
        return TriQuadRule(reshape([1/3, 1/3], 2, 1), [0.5])
    elseif nquad == 2
        pts = [1/6 2/3 1/6; 1/6 1/6 2/3]
        return TriQuadRule(pts, fill(0.5/3, 3))
    elseif nquad == 3
        a1=0.101286507323456; b1=1-2a1
        a2=0.470142064105115; b2=1-2a2
        pts = [a1 b1 a1 a2 b2 a2 1/3;
               a1 a1 b1 a2 a2 b2 1/3]
        w1=0.125939180544827/2; w2=0.132394440720100/2; w3=0.225/2
        return TriQuadRule(pts, [w1,w1,w1, w2,w2,w2, w3])
    else   # 13-point Dunavant degree 7
        a1=0.0651301029022; b1=1-2a1
        a2=0.3128654960049; b2=1-2a2
        a3=0.0486903154254; b3=0.6384441885698; c3=1-a3-b3
        pts = [a1  b1  a1  a2  b2  a2  a3  b3  c3  a3  b3  c3  1/3;
               a1  a1  b1  a2  a2  b2  b3  a3  a3  c3  c3  b3  1/3]
        w1=0.0533472356088/2; w2=0.0771137146903/2
        w3=0.0764649319397/2; w4=0.1498275574648/2  # corrected: 0.1756152576332/2 was wrong
        return TriQuadRule(pts, [w1,w1,w1, w2,w2,w2, w3,w3,w3,w3,w3,w3, w4])
    end
end

# ---------------------------------------------------------------------------
# Tri6 shape functions
# ---------------------------------------------------------------------------

# tri3_shape is defined here (not only in Geometry) so tests can access it as
# RadiativeViewFactor.ViewFactorKernel.tri3_shape
@inline function tri3_shape(ξ::Float64, η::Float64)
    N    = SVector(1.0-ξ-η, ξ, η)
    dNdξ = SVector(-1.0, 1.0, 0.0)
    dNdη = SVector(-1.0, 0.0, 1.0)
    return N, dNdξ, dNdη
end

@inline function tri6_shape(ξ::Float64, η::Float64)
    L1 = 1.0-ξ-η; L2 = ξ; L3 = η
    N    = SVector(L1*(2L1-1), L2*(2L2-1), L3*(2L3-1), 4L1*L2, 4L2*L3, 4L1*L3)
    dNdξ = SVector((4L1-1)*(-1.0), 4L2-1, 0.0, 4*(L2*(-1.0)+L1), 4L3, 4L3*(-1.0))
    dNdη = SVector((4L1-1)*(-1.0), 0.0, 4L3-1, 4*L2*(-1.0), 4L2, 4*(L3*(-1.0)+L1))
    return N, dNdξ, dNdη
end

@inline function tri6_physical_point(coords::Matrix{Float64},
                                      nodes ::Vector{Int},
                                      ξ::Float64, η::Float64)::SVector{3,Float64}
    N, _, _ = tri6_shape(ξ, η)
    x = @SVector zeros(3)
    for a in 1:6
        xa = SVector{3,Float64}(coords[1,nodes[a]], coords[2,nodes[a]], coords[3,nodes[a]])
        x  = x + N[a]*xa
    end
    return x
end

@inline function tri6_normal_and_area_element(coords::Matrix{Float64},
                                               nodes ::Vector{Int},
                                               ξ::Float64, η::Float64)
    _, dNdξ, dNdη = tri6_shape(ξ, η)
    dxdξ = @SVector zeros(3); dxdη = @SVector zeros(3)
    for a in 1:6
        xa   = SVector{3,Float64}(coords[1,nodes[a]], coords[2,nodes[a]], coords[3,nodes[a]])
        dxdξ = dxdξ + dNdξ[a]*xa
        dxdη = dxdη + dNdη[a]*xa
    end
    c  = cross(dxdξ, dxdη)
    dA = norm(c)
    return c/dA, dA
end

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

"""3D kernel: cos θᵢ cos θⱼ / (π r²)"""
@inline function vf_kernel(xi::SVector{3,Float64}, ni::SVector{3,Float64},
                            xj::SVector{3,Float64}, nj::SVector{3,Float64})::Float64
    r_vec = xj - xi
    r²    = dot(r_vec, r_vec)
    r²    < 1e-30 && return 0.0
    r     = sqrt(r²)
    r̂     = r_vec / r
    cos_i = dot(ni,  r̂)
    cos_j = dot(nj, -r̂)
    (cos_i <= 0.0 || cos_j <= 0.0) && return 0.0
    return cos_i * cos_j / (π * r²)
end

"""2D kernel (per unit depth): cos θᵢ cos θⱼ / (2 r)"""
@inline function vf_kernel_2d(xi::SVector{3,Float64}, ni::SVector{3,Float64},
                               xj::SVector{3,Float64}, nj::SVector{3,Float64})::Float64
    r_vec = xj - xi
    r²    = dot(r_vec, r_vec)
    r²    < 1e-30 && return 0.0
    r     = sqrt(r²)
    r̂     = r_vec / r
    cos_i = dot(ni,  r̂)
    cos_j = dot(nj, -r̂)
    (cos_i <= 0.0 || cos_j <= 0.0) && return 0.0
    return cos_i * cos_j / (2.0 * r)
end

# ---------------------------------------------------------------------------
# Element-pair integrator
# ---------------------------------------------------------------------------

"""
    element_pair_view_factor(coords, elem_i, elem_j, nquad, bvh) -> (raw, Li)

Compute the raw double integral and the measure of elem_i.

For surface elements (:quad, :tri):
  raw = ∬_Aᵢ ∬_Aⱼ K₃D dAⱼ dAᵢ,   Li = area of elem_i

For curve elements (:line3):
  raw = ∫_Lᵢ  ∫_Lⱼ  K₂D dLⱼ dLᵢ,  Li = arc length of elem_i

The dimension is inferred from `elem_i.family`.
"""
function element_pair_view_factor(coords::Matrix{Float64},
                                   elem_i::SurfaceElement,
                                   elem_j::SurfaceElement,
                                   nquad ::Int,
                                   bvh   ::Union{BVHTree, Nothing},
                                   mesh_dim::Int = 2)::Tuple{Float64,Float64}

    do_vis = bvh !== nothing
    is_2d  = mesh_dim == 1 ||
              elem_i.family === :line2 || elem_i.family === :line3

    pts_i, wts_i, nds_i = _quad_points(coords, elem_i, nquad)
    pts_j, wts_j, nds_j = _quad_points(coords, elem_j, nquad)

    Fij = 0.0
    Li  = 0.0

    for p in eachindex(wts_i)
        wi  = wts_i[p]
        xi  = pts_i[p]
        ni  = nds_i[p][1]
        dLi = nds_i[p][2]

        Li += wi * dLi

        inner = 0.0
        for q in eachindex(wts_j)
            wj  = wts_j[q]
            xj  = pts_j[q]
            nj  = nds_j[q][1]
            dLj = nds_j[q][2]

            K = is_2d ? vf_kernel_2d(xi, ni, xj, nj) :
                        vf_kernel(xi, ni, xj, nj)
            K == 0.0 && continue
            do_vis && !is_visible(bvh, xi, xj; mesh_dim=mesh_dim) && continue

            inner += wj * K * dLj
        end
        Fij += wi * inner * dLi
    end

    return Fij, Li
end

# ---------------------------------------------------------------------------
# Quadrature point pre-evaluation
# ---------------------------------------------------------------------------

function _quad_points(coords::Matrix{Float64}, elem::SurfaceElement, nquad::Int)
    if elem.family === :quad
        rule = gauss_legendre_2d(nquad)
        nq   = size(rule.points, 2)
        xs   = Vector{SVector{3,Float64}}(undef, nq)
        nds  = Vector{Tuple{SVector{3,Float64},Float64}}(undef, nq)
        for k in 1:nq
            ξ, η   = rule.points[1,k], rule.points[2,k]
            xs[k]  = quad8_physical_point(coords, elem.nodes, ξ, η)
            n, dA  = quad8_normal_and_area_element(coords, elem.nodes, ξ, η)
            nds[k] = (n, dA)
        end
        return xs, rule.weights, nds
    elseif elem.family === :quad4
        rule = gauss_legendre_2d(nquad)
        nq   = size(rule.points, 2)
        xs   = Vector{SVector{3,Float64}}(undef, nq)
        nds  = Vector{Tuple{SVector{3,Float64},Float64}}(undef, nq)
        for k in 1:nq
            ξ, η   = rule.points[1,k], rule.points[2,k]
            xs[k]  = quad4_physical_point(coords, elem.nodes, ξ, η)
            n, dA  = quad4_normal_and_area_element(coords, elem.nodes, ξ, η)
            nds[k] = (n, dA)
        end
        return xs, rule.weights, nds
    elseif elem.family === :tri
        rule = tri_quad_rule(nquad)
        nq   = size(rule.points, 2)
        xs   = Vector{SVector{3,Float64}}(undef, nq)
        nds  = Vector{Tuple{SVector{3,Float64},Float64}}(undef, nq)
        for k in 1:nq
            ξ, η   = rule.points[1,k], rule.points[2,k]
            xs[k]  = tri6_physical_point(coords, elem.nodes, ξ, η)
            n, dA  = tri6_normal_and_area_element(coords, elem.nodes, ξ, η)
            nds[k] = (n, dA)
        end
        return xs, rule.weights, nds
    elseif elem.family === :tri3
        rule = tri_quad_rule(nquad)
        nq   = size(rule.points, 2)
        xs   = Vector{SVector{3,Float64}}(undef, nq)
        nds  = Vector{Tuple{SVector{3,Float64},Float64}}(undef, nq)
        for k in 1:nq
            ξ, η   = rule.points[1,k], rule.points[2,k]
            xs[k]  = tri3_physical_point(coords, elem.nodes, ξ, η)
            n, dA  = tri3_normal_and_area_element(coords, elem.nodes, ξ, η)
            nds[k] = (n, dA)
        end
        return xs, rule.weights, nds
    elseif elem.family === :line3
        pts1d, wts1d = gauss_legendre_1d(nquad)
        nq   = length(pts1d)
        xs   = Vector{SVector{3,Float64}}(undef, nq)
        nds  = Vector{Tuple{SVector{3,Float64},Float64}}(undef, nq)
        for k in 1:nq
            ξ      = pts1d[k]
            xs[k]  = line3_physical_point(coords, elem.nodes, ξ)
            n, dL  = line3_normal_and_length_element(coords, elem.nodes, ξ)
            nds[k] = (n, dL)
        end
        return xs, wts1d, nds
    else  # :line2 — 1-D Gauss–Legendre on [-1,1]
        pts1d, wts1d = gauss_legendre_1d(nquad)
        nq   = length(pts1d)
        xs   = Vector{SVector{3,Float64}}(undef, nq)
        nds  = Vector{Tuple{SVector{3,Float64},Float64}}(undef, nq)
        for k in 1:nq
            ξ      = pts1d[k]
            xs[k]  = line2_physical_point(coords, elem.nodes, ξ)
            n, dL  = line2_normal_and_length_element(coords, elem.nodes, ξ)
            nds[k] = (n, dL)
        end
        return xs, wts1d, nds
    end
end

end # module ViewFactorKernel
