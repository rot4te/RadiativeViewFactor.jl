# src/MCKernel.jl
# ---------------------------------------------------------------------------
# Monte Carlo view factor integrator (CPU path).
#
# Estimator
# ---------
# For each element pair (i,j), draw N stratified samples (xᵢ, xⱼ) where
# xᵢ is drawn uniformly on Aᵢ and xⱼ uniformly on Aⱼ.  The unbiased MC
# estimator for the raw double integral is:
#
#   ∬_Aᵢ ∬_Aⱼ K dAⱼ dAᵢ  ≈  (Aᵢ · Aⱼ / N) · Σₖ K(xᵢₖ, nᵢₖ, xⱼₖ, nⱼₖ) · H_ij
#
# F_ij = raw / Aᵢ  =  (Aⱼ / N) · Σₖ K · H_ij
#
# Stratified sampling
# -------------------
# The N samples are divided into √N × √N strata on each element's reference
# domain.  Within each stratum one point is drawn uniformly at random.  This
# gives variance reduction proportional to 1/N² rather than 1/N for smooth
# integrands (stratified MC converges at O(1/N) vs O(1/√N) for plain MC).
# If √N is not an integer, the strata count is floor(√N) × floor(√N) and
# the remaining samples are drawn from the full reference domain.
#
# Random point sampling on reference elements
# --------------------------------------------
# Quad8 / Tri6: uniform on the reference square [-1,1]² or reference triangle.
# Line3: uniform on [-1,1].
# The isoparametric map is applied to get the physical point and normal.
# ---------------------------------------------------------------------------

module MCKernel

using LinearAlgebra
using StaticArrays
using Random

import ..Geometry:    quad8_physical_point, quad8_normal_and_area_element,
                      line3_physical_point, line3_normal_and_length_element
import ..BVH:         BVHTree
import ..RayCast:     is_visible
import ..MeshIO:      SurfaceElement

export element_pair_view_factor_mc

# ---------------------------------------------------------------------------
# Random point generation on reference elements
# ---------------------------------------------------------------------------

"""
Draw `n` stratified random points on the reference square [-1,1]².
Returns a vector of (ξ,η) tuples and a vector of Jacobian weights.
For a uniform distribution on [-1,1]² each point has weight 4/n.
"""
function _stratified_quad_points(n::Int, rng::AbstractRNG)
    s      = floor(Int, sqrt(n))   # strata per side
    pts    = Vector{Tuple{Float64,Float64}}(undef, n)
    # s² stratified points
    k = 0
    for si in 0:s-1, sj in 0:s-1
        k += 1
        u = (si + rand(rng)) / s   # uniform in [0,1]
        v = (sj + rand(rng)) / s
        pts[k] = (2u - 1, 2v - 1) # map to [-1,1]²
    end
    # remaining n - s² points drawn uniformly
    for k2 in k+1:n
        pts[k2] = (2*rand(rng) - 1, 2*rand(rng) - 1)
    end
    wt = 4.0 / n   # area of [-1,1]² = 4, divided equally
    return pts, wt
end

"""
Draw `n` stratified random points on the reference triangle {ξ≥0, η≥0, ξ+η≤1}.
Uses the Larcher–Pillichshammer or simple random-number mapping for triangles.
Area of reference triangle = 0.5, so weight = 0.5/n per point.
"""
function _stratified_tri_points(n::Int, rng::AbstractRNG)
    s   = floor(Int, sqrt(n))
    pts = Vector{Tuple{Float64,Float64}}(undef, n)
    k   = 0
    for si in 0:s-1, sj in 0:s-1
        k += 1
        # Uniform point in unit square, then fold into triangle
        u = (si + rand(rng)) / s
        v = (sj + rand(rng)) / s
        if u + v > 1.0
            u, v = 1.0 - u, 1.0 - v
        end
        pts[k] = (u, v)
    end
    for k2 in k+1:n
        u, v = rand(rng), rand(rng)
        if u + v > 1.0; u, v = 1.0-u, 1.0-v; end
        pts[k2] = (u, v)
    end
    wt = 0.5 / n
    return pts, wt
end

"""
Draw `n` stratified random points on the reference interval [-1,1].
Weight = 2/n per point.
"""
function _stratified_line_points(n::Int, rng::AbstractRNG)
    pts = Vector{Float64}(undef, n)
    s   = n   # 1-D: n strata of width 2/n each
    for si in 0:s-1
        pts[si+1] = -1.0 + (si + rand(rng)) * 2.0 / s
    end
    wt = 2.0 / n
    return pts, wt
end

# ---------------------------------------------------------------------------
# Physical-space sampling
# ---------------------------------------------------------------------------

"""
Sample `n` stratified random points on `elem`, returning:
- `xs`   : Vector of physical positions  (SVector{3,Float64})
- `ns`   : Vector of unit normals        (SVector{3,Float64})
- `dAs`  : Vector of area/length elements (Float64) — Jacobian at sample point
- `A`    : Total area/length of the element (from the same MC samples)
"""
function _sample_element(coords::Matrix{Float64},
                          elem  ::SurfaceElement,
                          n     ::Int,
                          rng   ::AbstractRNG)
    xs  = Vector{SVector{3,Float64}}(undef, n)
    ns  = Vector{SVector{3,Float64}}(undef, n)
    dAs = Vector{Float64}(undef, n)

    if elem.family === :quad
        ref_pts, wt = _stratified_quad_points(n, rng)
        for k in 1:n
            ξ, η    = ref_pts[k]
            xs[k]   = quad8_physical_point(coords, elem.nodes, ξ, η)
            nk, dAk = quad8_normal_and_area_element(coords, elem.nodes, ξ, η)
            ns[k]   = nk
            dAs[k]  = dAk
        end
        A = wt * sum(dAs)   # MC estimate of element area

    elseif elem.family === :tri
        ref_pts, wt = _stratified_tri_points(n, rng)
        for k in 1:n
            ξ, η = ref_pts[k]
            # Use inline Tri6 evaluation (Geometry module not imported for tri6)
            # — reuse the same logic as ViewFactorKernel
            xs[k], ns[k], dAs[k] = _tri6_point_normal_dA(coords, elem.nodes, ξ, η)
        end
        A = wt * sum(dAs)

    else  # :line3
        ref_pts, wt = _stratified_line_points(n, rng)
        for k in 1:n
            ξ      = ref_pts[k]
            xs[k]  = line3_physical_point(coords, elem.nodes, ξ)
            nk, dLk = line3_normal_and_length_element(coords, elem.nodes, ξ)
            ns[k]  = nk
            dAs[k] = dLk
        end
        A = wt * sum(dAs)
    end

    return xs, ns, dAs, A
end

# Inline Tri6 point/normal/dA (avoids circular import with ViewFactorKernel)
@inline function _tri6_point_normal_dA(coords, nodes, ξ::Float64, η::Float64)
    L1=1-ξ-η; L2=ξ; L3=η
    N    = SVector(L1*(2L1-1), L2*(2L2-1), L3*(2L3-1), 4L1*L2, 4L2*L3, 4L1*L3)
    dNdξ = SVector((4L1-1)*(-1.0), 4L2-1, 0.0, 4*(L2*(-1.0)+L1), 4L3, 4L3*(-1.0))
    dNdη = SVector((4L1-1)*(-1.0), 0.0, 4L3-1, 4*L2*(-1.0), 4L2, 4*(L3*(-1.0)+L1))
    x=@SVector zeros(3); dxdξ=@SVector zeros(3); dxdη=@SVector zeros(3)
    for a in 1:6
        xa=SVector{3,Float64}(coords[1,nodes[a]],coords[2,nodes[a]],coords[3,nodes[a]])
        x=x+N[a]*xa; dxdξ=dxdξ+dNdξ[a]*xa; dxdη=dxdη+dNdη[a]*xa
    end
    c=cross(dxdξ,dxdη); dA=norm(c)
    return x, c/dA, dA
end

# ---------------------------------------------------------------------------
# MC view factor kernel
# ---------------------------------------------------------------------------

"""
    element_pair_view_factor_mc(coords, elem_i, elem_j, n_samples, bvh,
                                 mesh_dim, rng) -> (raw, Ai)

Monte Carlo estimate of the raw double integral ∬K dAⱼ dAᵢ and element
area Aᵢ, using `n_samples` stratified random sample pairs.

The estimator is:

    raw ≈ (Aᵢ_mc · Aⱼ_mc / n_samples) · Σₖ K(xᵢₖ, nᵢₖ, xⱼₖ, nⱼₖ) · H_ij

where Aᵢ_mc and Aⱼ_mc are MC estimates of the element areas from the same
sample points, keeping the estimator consistent.
"""
function element_pair_view_factor_mc(coords   ::Matrix{Float64},
                                      elem_i   ::SurfaceElement,
                                      elem_j   ::SurfaceElement,
                                      n_samples::Int,
                                      bvh      ::Union{BVHTree,Nothing},
                                      mesh_dim ::Int,
                                      rng      ::AbstractRNG)::Tuple{Float64,Float64}

    do_vis = bvh !== nothing
    is_2d  = mesh_dim == 1

    xs_i, ns_i, dAs_i, Ai = _sample_element(coords, elem_i, n_samples, rng)
    xs_j, ns_j, dAs_j, Aj = _sample_element(coords, elem_j, n_samples, rng)

    K_sum = 0.0
    for k in 1:n_samples
        xi = xs_i[k]; ni = ns_i[k]; dAi = dAs_i[k]
        xj = xs_j[k]; nj = ns_j[k]; dAj = dAs_j[k]

        K = is_2d ? _kernel_2d(xi, ni, xj, nj) :
                    _kernel_3d(xi, ni, xj, nj)
        K == 0.0 && continue
        do_vis && !is_visible(bvh, xi, xj; mesh_dim=mesh_dim) && continue

        # Weight each sample by the Jacobian ratio (importance sampling
        # correction since we sampled uniformly in reference space)
        K_sum += K * dAi * dAj
    end

    # MC estimate: (Ai * Aj / n) * (1/(Ai*Aj)) * Σ K*dAi*dAj
    # = (1/n) * Σ K*dAi*dAj   ... but we need to normalise by the
    # reference domain areas already absorbed into dAi,dAj via the
    # stratified weights.  The stratified sampler already divides by n
    # in the weight wt = ref_area/n, so:
    #   _sample_element returns dAs without the 1/n factor (raw Jacobian)
    #   Ai = wt * Σ dAs_i  =  (ref_area/n) * Σ dAs_i
    # The MC estimator for ∬K dAi dAj is:
    #   (ref_area_i / n) * (ref_area_j / n) * Σ K * dAi * dAj
    # But since Ai = (ref_area_i/n)*Σ dAs_i and similarly for Aj,
    # we absorb the normalisation directly:
    raw = K_sum * _ref_area(elem_i) * _ref_area(elem_j) / n_samples

    return raw, Ai
end

@inline _ref_area(el::SurfaceElement) =
    el.family === :quad  ? 4.0 :   # [-1,1]²
    el.family === :tri   ? 0.5 :   # reference triangle
                           2.0     # [-1,1]

@inline function _kernel_3d(xi, ni, xj, nj)
    r_vec = xj - xi
    r²    = dot(r_vec, r_vec)
    r²    < 1e-30 && return 0.0
    r     = sqrt(r²); r̂ = r_vec/r
    ci    = dot(ni,  r̂); cj = dot(nj, -r̂)
    (ci <= 0.0 || cj <= 0.0) && return 0.0
    return ci * cj / (π * r²)
end

@inline function _kernel_2d(xi, ni, xj, nj)
    r_vec = xj - xi
    r²    = dot(r_vec, r_vec)
    r²    < 1e-30 && return 0.0
    r     = sqrt(r²); r̂ = r_vec/r
    ci    = dot(ni,  r̂); cj = dot(nj, -r̂)
    (ci <= 0.0 || cj <= 0.0) && return 0.0
    return ci * cj / (2.0 * r)
end

end # module MCKernel
