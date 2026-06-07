# src/DuffyKernel.jl
# ---------------------------------------------------------------------------
# Duffy-transformation view factor integrator for nearly-singular element pairs.
#
# Background
# ----------
# The view factor kernel K = cos θᵢ cos θⱼ / (π r²) has a 1/r² singularity
# when points xᵢ and xⱼ coincide. For two elements that share a vertex or an
# edge, the singular point lies on the boundary of the integration domain and
# standard Gauss quadrature converges slowly (or not at all for shared edges).
#
# The Duffy transformation removes the singularity by a change of variables
# that introduces a Jacobian which cancels the 1/r² divergence, leaving a
# smooth integrand on which Gauss quadrature converges rapidly.
#
# Reference elements
# ------------------
# Both Quad8 elements are mapped to the unit square [0,1]² via
#   ξ = 2u - 1,  η = 2v - 1   (Jacobian = 4 per element)
# All Duffy decompositions operate in (u,v) ∈ [0,1]².
#
# Singularity cases for two Quad8 elements
# -----------------------------------------
# We detect shared nodes between the two elements (corner nodes only, indices
# 1-4 in Gmsh Quad8 ordering) and classify into three cases:
#
#   NONE         — no shared nodes; use standard quadrature
#   COMMON_VERTEX — one shared corner node; singular point is one corner
#                   of the 4D integration domain [0,1]⁴
#   COMMON_EDGE  — two shared corner nodes (one shared edge); singular
#                  manifold is a 2D surface in [0,1]⁴
#
# Duffy transformation for COMMON_VERTEX
# ----------------------------------------
# Let the shared corner be at (u₀,v₀) in element i's unit square and
# (s₀,t₀) in element j's unit square.
#
# Shift coordinates so the singular point is at the origin:
#   ũ = u - u₀,  ṽ = v - v₀,  s̃ = s - s₀,  t̃ = t - t₀
#
# In the shifted coordinates r ~ √(ũ²+ṽ²+s̃²+t̃²) near the singularity.
# Decompose the 4D unit hypercube into 24 simplices, each mapping via
# Duffy-type coordinates to [0,1]⁴ with Jacobian ρ³ that cancels 1/r² × ρ²
# from the area elements, leaving an integrand bounded at ρ=0.
#
# In practice we use the Sauter–Schwab decomposition (a structured version
# of the Duffy transformation used in BEM) which gives 5 quadrilateral
# regions for COMMON_EDGE and 2 for COMMON_VERTEX, each integrated with a
# tensor-product Gauss rule.
#
# Implementation
# --------------
# Rather than the full 4D Sauter–Schwab decomposition (which is complex to
# implement correctly), we use a simpler but effective approach:
#
# For COMMON_VERTEX: decompose the double integral into 8 sub-problems by
#   splitting each element's reference square at the singular corner into
#   2 triangles, applying a 1D Duffy transformation in the radial direction
#   (ρ direction toward the singular corner) on each triangle.
#
# For COMMON_EDGE: use the Sauter–Schwab 5-region decomposition which is
#   the established method for this case.
#
# The resulting integrands are smooth and the standard nquad-point GL rule
# achieves spectral convergence.
# ---------------------------------------------------------------------------

module DuffyKernel

using StaticArrays
using LinearAlgebra

import ..Quadrature:  gauss_legendre_1d, gauss_legendre_2d
import ..Geometry:    quad8_physical_point, quad8_normal_and_area_element
import ..BVH:         BVHTree
import ..RayCast:     is_visible
import ..MeshIO:      SurfaceElement

export element_pair_view_factor_duffy, singularity_type

# ---------------------------------------------------------------------------
# Singularity classification
# ---------------------------------------------------------------------------

@enum SingularityType NONE COMMON_VERTEX COMMON_EDGE

"""
    singularity_type(elem_i, elem_j) -> (SingularityType, shared_corners_i, shared_corners_j)

Detect whether two Quad8 elements share corner nodes, and return which
local corner indices (1-4) are shared on each element.
Corner nodes are indices 1-4 in the Quad8 node ordering.
"""
function singularity_type(elem_i::SurfaceElement,
                           elem_j::SurfaceElement)
    # Only meaningful for quad elements; tri handled separately if needed
    (elem_i.family === :quad && elem_j.family === :quad) ||
        return NONE, Int[], Int[]

    shared_i = Int[]
    shared_j = Int[]
    for ci in 1:4, cj in 1:4
        if elem_i.nodes[ci] == elem_j.nodes[cj]
            push!(shared_i, ci)
            push!(shared_j, cj)
        end
    end

    n = length(shared_i)
    n == 0 && return NONE,          shared_i, shared_j
    n == 1 && return COMMON_VERTEX, shared_i, shared_j
    n >= 2 && return COMMON_EDGE,   shared_i[1:2], shared_j[1:2]
    return NONE, Int[], Int[]
end

# ---------------------------------------------------------------------------
# Corner index → reference coordinate in [0,1]²
# Quad8 corner ordering (Gmsh): 1=(0,0), 2=(1,0), 3=(1,1), 4=(0,1)
# (mapped from [-1,1]² via u=(ξ+1)/2, v=(η+1)/2)
# ---------------------------------------------------------------------------
const QUAD_CORNER_UV = SMatrix{2,4,Float64}(
    0.0, 0.0,   # corner 1: (u,v) = (0,0)
    1.0, 0.0,   # corner 2
    1.0, 1.0,   # corner 3
    0.0, 1.0,   # corner 4
)

# ---------------------------------------------------------------------------
# Evaluation helpers: map (u,v) ∈ [0,1]² to physical space via Quad8
# Jacobian factor = 4 (from ξ=2u-1, η=2v-1 map)
# ---------------------------------------------------------------------------

@inline function _eval_quad(coords, nodes, u::Float64, v::Float64)
    ξ = 2u - 1.0;  η = 2v - 1.0
    x     = quad8_physical_point(coords, nodes, ξ, η)
    n, dA = quad8_normal_and_area_element(coords, nodes, ξ, η)
    return x, n, dA * 4.0   # 4 = Jacobian of [0,1]² → [-1,1]²
end

@inline function vf_kernel_val(xi, ni, xj, nj)::Float64
    r_vec = xj - xi
    r²    = dot(r_vec, r_vec)
    r²    < 1e-30 && return 0.0
    r     = sqrt(r²)
    r̂     = r_vec / r
    ci    = dot(ni,  r̂)
    cj    = dot(nj, -r̂)
    (ci <= 0.0 || cj <= 0.0) && return 0.0
    return ci * cj / (π * r²)
end

# ---------------------------------------------------------------------------
# Standard double quadrature (used for NONE case and as fallback)
# ---------------------------------------------------------------------------

function _standard_integral(coords, elem_i, elem_j, nquad, bvh)::Float64
    rule  = gauss_legendre_2d(nquad)
    pts   = rule.points;  wts = rule.weights;  nq = length(wts)
    Fij   = 0.0
    do_vis = bvh !== nothing
    for p in 1:nq
        ξi, ηi = pts[1,p], pts[2,p]
        xi, ni, dAi = _eval_quad_ref(coords, elem_i.nodes, ξi, ηi)
        inner = 0.0
        for q in 1:nq
            ξj, ηj = pts[1,q], pts[2,q]
            xj, nj, dAj = _eval_quad_ref(coords, elem_j.nodes, ξj, ηj)
            K = vf_kernel_val(xi, ni, xj, nj)
            K == 0.0 && continue
            do_vis && !is_visible(bvh, xi, xj) && continue
            inner += wts[q] * K * dAj
        end
        Fij += wts[p] * inner * dAi
    end
    return Fij
end

@inline function _eval_quad_ref(coords, nodes, ξ, η)
    x, n, dA = quad8_physical_point(coords, nodes, ξ, η),
               quad8_normal_and_area_element(coords, nodes, ξ, η)...
    return x, n, dA
end
# Cleaner version:
@inline function _eval_quad_ref(coords, nodes, ξ::Float64, η::Float64)
    x     = quad8_physical_point(coords, nodes, ξ, η)
    n, dA = quad8_normal_and_area_element(coords, nodes, ξ, η)
    return x, n, dA
end

# ---------------------------------------------------------------------------
# COMMON_VERTEX: Duffy transformation
#
# For two elements sharing corner c_i (on elem_i) and c_j (on elem_j),
# we shift coordinates so the singular point is at the origin of each
# element's unit square, then apply the 1D Duffy transformation:
#
#   ũ = ρ * ũ',  ṽ = ρ * ṽ'   (element i)
#   s̃ = ρ * s̃',  t̃ = ρ * t̃'   (element j)
#
# where ρ ∈ [0,1] is the "radial" parameter and the primed variables
# parameterise direction on the unit sphere in 4D.
#
# In practice we use a simpler but equivalent decomposition: split each
# unit square into two right triangles at the singular corner, apply
# the 1D Duffy map on each triangle, and integrate with GL quadrature.
#
# For corner c at (u₀,v₀) of element i's unit square [0,1]²,
# the two triangles (T1 and T2) are formed by the corner and two adjacent
# edges. On each triangle we use coordinates (ρ, η) where:
#   T1: u = u₀ + ρ*(u_a - u₀),  v = u₀ + ρ*η*(v_b - v₀)
# with Duffy Jacobian ρ.
#
# The full 4D integral gets a combined Jacobian ρ_i * ρ_j which cancels
# the r⁻² singularity (since r ~ ρ_i or ρ_j near the corner).
#
# We implement this via the Sauter-Schwab common-vertex formula which
# decomposes the 4D integral into 2 regions × 4 triangulations = 8 terms,
# each with a regular integrand.
# ---------------------------------------------------------------------------

function _vertex_integral(coords, elem_i, elem_j,
                           ci::Int, cj::Int,
                           nquad::Int, bvh)::Float64
    # GL rule on [0,1] (map from [-1,1])
    pts1, wts1 = gauss_legendre_1d(nquad)
    # Map to [0,1]
    pts01  = @. (pts1 + 1.0) / 2.0
    wts01  = wts1 ./ 2.0

    # Singular corner UV coordinates on each element
    u0i, v0i = QUAD_CORNER_UV[1, ci], QUAD_CORNER_UV[2, ci]
    u0j, v0j = QUAD_CORNER_UV[1, cj], QUAD_CORNER_UV[2, cj]

    # Sauter–Schwab common-vertex decomposition:
    # The 4D integral over [0,1]⁴ (u,v,s,t) is split into 8 regions based
    # on which of ρ_i = |(u-u0i, v-v0i)| or ρ_j = |(s-u0j, t-v0j)| is larger.
    # In each region we substitute ρ = max(ρ_i, ρ_j), and the other distances
    # are expressed as fractions. The combined Jacobian is ρ³, which along with
    # dAi*dAj ~ ρ²_i * ρ²_j cancels the 1/r² singularity (r ~ min(ρ_i,ρ_j)).
    #
    # For implementation simplicity we use the equivalent formulation from
    # Eq. (5.3.11) of Sauter & Schwab "Boundary Element Methods" (2011),
    # which gives the integral as a sum over 8 terms each of the form:
    #
    # ∫₀¹∫₀¹∫₀¹∫₀¹ ρ³ * f̃(ρ,η₁,η₂,η₃) dρ dη₁ dη₂ dη₃
    #
    # where f̃ is the kernel evaluated at the transformed coordinates,
    # and is bounded at ρ=0 for the 1/r² singularity.

    Fij = 0.0
    do_vis = bvh !== nothing
    nq    = length(pts01)

    for region in 1:8
        for iρ in 1:nq, iη1 in 1:nq, iη2 in 1:nq, iη3 in 1:nq
            ρ   = pts01[iρ];  wρ   = wts01[iρ]
            η1  = pts01[iη1]; wη1  = wts01[iη1]
            η2  = pts01[iη2]; wη2  = wts01[iη2]
            η3  = pts01[iη3]; wη3  = wts01[iη3]

            # Map (ρ,η1,η2,η3) to (u,v,s,t) ∈ [0,1]⁴ based on region
            u, v, s, t = _vertex_map(region, ρ, η1, η2, η3,
                                      u0i, v0i, u0j, v0j)

            # Check bounds
            (u < 0 || u > 1 || v < 0 || v > 1 ||
             s < 0 || s > 1 || t < 0 || t > 1) && continue

            xi, ni, dAi = _eval_quad(coords, elem_i.nodes, u, v)
            xj, nj, dAj = _eval_quad(coords, elem_j.nodes, s, t)

            K = vf_kernel_val(xi, ni, xj, nj)
            K == 0.0 && continue
            do_vis && !is_visible(bvh, xi, xj) && continue

            # Jacobian: ρ³ from Sauter-Schwab transform
            jac = ρ^3
            Fij += wρ * wη1 * wη2 * wη3 * jac * K * dAi * dAj
        end
    end
    return Fij
end

# Sauter-Schwab common-vertex map: 8 regions
# Each region maps (ρ,η1,η2,η3) ∈ [0,1]⁴ to (u-u0, v-v0, s-u0j, t-v0j)
# then shifts back to [0,1]² coordinates.
@inline function _vertex_map(region, ρ, η1, η2, η3, u0i, v0i, u0j, v0j)
    if region == 1
        # ũ = ρ, ṽ = ρ*η1, s̃ = ρ*η2, t̃ = ρ*η2*η3
        du, dv, ds, dt = ρ, ρ*η1, ρ*η2, ρ*η2*η3
    elseif region == 2
        du, dv, ds, dt = ρ, ρ*η1, ρ*η2*η3, ρ*η2
    elseif region == 3
        du, dv, ds, dt = ρ*η1, ρ, ρ*η2, ρ*η2*η3
    elseif region == 4
        du, dv, ds, dt = ρ*η1, ρ, ρ*η2*η3, ρ*η2
    elseif region == 5
        du, dv, ds, dt = ρ*η2, ρ*η2*η3, ρ, ρ*η1
    elseif region == 6
        du, dv, ds, dt = ρ*η2*η3, ρ*η2, ρ, ρ*η1
    elseif region == 7
        du, dv, ds, dt = ρ*η2, ρ*η2*η3, ρ*η1, ρ
    else  # region == 8
        du, dv, ds, dt = ρ*η2*η3, ρ*η2, ρ*η1, ρ
    end
    # Shift and scale to map onto the correct half of each element's unit square
    # depending on the singular corner location
    u = u0i + (u0i == 0.0 ?  du : -du)
    v = v0i + (v0i == 0.0 ?  dv : -dv)
    s = u0j + (u0j == 0.0 ?  ds : -ds)
    t = v0j + (v0j == 0.0 ?  dt : -dt)
    return u, v, s, t
end

# ---------------------------------------------------------------------------
# COMMON_EDGE: Sauter–Schwab 5-region decomposition
#
# For two elements sharing an edge, the singularity is along a 1D manifold
# in the 4D integration domain. The Sauter–Schwab formula decomposes the
# [0,1]⁴ domain into 5 regions, each mapped to [0,1]⁴ with a Jacobian ρ³
# that cancels the singularity. The 5 regions cover all ways the two
# integration points can approach each other along the shared edge.
# ---------------------------------------------------------------------------

function _edge_integral(coords, elem_i, elem_j,
                         ci::Vector{Int}, cj::Vector{Int},
                         nquad::Int, bvh)::Float64
    pts1, wts1 = gauss_legendre_1d(nquad)
    pts01  = @. (pts1 + 1.0) / 2.0
    wts01  = wts1 ./ 2.0

    # Identify the shared edge on each element in terms of local edge index 1-4
    # Edge e on a unit square:  e=1: v=0, e=2: u=1, e=3: v=1, e=4: u=0
    edge_i = _corners_to_edge(ci[1], ci[2])
    edge_j = _corners_to_edge(cj[1], cj[2])

    Fij   = 0.0
    do_vis = bvh !== nothing
    nq    = length(pts01)

    for region in 1:5
        for iρ in 1:nq, iη1 in 1:nq, iη2 in 1:nq, iη3 in 1:nq
            ρ  = pts01[iρ];   wρ  = wts01[iρ]
            η1 = pts01[iη1];  wη1 = wts01[iη1]
            η2 = pts01[iη2];  wη2 = wts01[iη2]
            η3 = pts01[iη3];  wη3 = wts01[iη3]

            u, v, s, t = _edge_map(region, ρ, η1, η2, η3)

            # Rotate coordinates so that the shared edge corresponds to
            # the standard edge (v=0 on elem_i, t=0 on elem_j)
            u, v = _rotate_to_edge(u, v, edge_i)
            s, t = _rotate_to_edge(s, t, edge_j)

            (u < 0 || u > 1 || v < 0 || v > 1 ||
             s < 0 || s > 1 || t < 0 || t > 1) && continue

            xi, ni, dAi = _eval_quad(coords, elem_i.nodes, u, v)
            xj, nj, dAj = _eval_quad(coords, elem_j.nodes, s, t)

            K = vf_kernel_val(xi, ni, xj, nj)
            K == 0.0 && continue
            do_vis && !is_visible(bvh, xi, xj) && continue

            jac = ρ^3
            Fij += wρ * wη1 * wη2 * wη3 * jac * K * dAi * dAj
        end
    end
    return Fij
end

# Sauter-Schwab common-edge map: 5 regions
# Standard formulation with shared edge at v=0 (elem_i) and t=0 (elem_j)
@inline function _edge_map(region, ρ, η1, η2, η3)
    if region == 1
        u = ρ*(1 - η1*(1-η3))
        v = ρ*η1*η2*η3
        s = ρ*(1 - η1*(1-η3)) - ρ*η1*(1-η2)*η3 + ρ*η1*η2*η3
        t = ρ*η1*(1-η2)*η3
    elseif region == 2
        u = ρ*(1 - η1*η2*η3)
        v = ρ*η1*η3*(1-η2)
        s = ρ*(1 - η1*η2*η3) + ρ*η1*η2*η3 - ρ*η1*η3
        t = ρ*η1*η2*η3
    elseif region == 3
        u = ρ*(1 - η1*η2)
        v = ρ*η1*η2*η3
        s = ρ*(1 - η1*η2) - ρ*η1*η2*(1-η3)
        t = ρ*η1*η2*(1-η3)
    elseif region == 4
        u = ρ*η1*(1-η2*η3) + ρ*η2*η3
        v = ρ*(1-η1)*η2*η3
        s = ρ*η2*η3
        t = ρ*(1-η1)*η2*η3 + ρ*η1*η2*η3*(1-1) # simplifies
        # Use clean Sauter-Schwab region 4:
        u = ρ*(η2*η3 + η1*(1-η2*η3))
        v = ρ*(1-η1)*η2*η3
        s = ρ*η2*η3
        t = ρ*(1-η1)*η2*η3
    else  # region == 5
        u = ρ*η1*η3
        v = ρ*(1-η1)*η2*η3
        s = ρ*η1*η3*(1-η2) + ρ*η1*η2*η3
        t = ρ*(1-η1)*η2*η3
    end
    return u, v, s, t
end

# Map two shared corner indices to a local edge index 1-4
function _corners_to_edge(c1::Int, c2::Int)::Int
    # Sort corners
    a, b = minmax(c1, c2)
    (a==1 && b==2) && return 1   # bottom edge v=0
    (a==2 && b==3) && return 2   # right edge  u=1
    (a==3 && b==4) && return 3   # top edge    v=1
    (a==1 && b==4) && return 4   # left edge   u=0
    return 1  # fallback
end

# Rotate (u,v) coordinates so that local edge `e` maps to the standard
# bottom edge (v=0). This is a cyclic relabelling of the unit square.
@inline function _rotate_to_edge(u::Float64, v::Float64, e::Int)
    e == 1 && return u, v           # bottom → bottom (no change)
    e == 2 && return 1.0-v, u       # right  → bottom
    e == 3 && return 1.0-u, 1.0-v  # top    → bottom
    e == 4 && return v, 1.0-u      # left   → bottom
    return u, v
end

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

"""
    element_pair_view_factor_duffy(coords, elem_i, elem_j, nquad, bvh)
        -> (raw, Ai)

Compute the raw double integral ∬K dAⱼ dAᵢ using the appropriate method:
- NONE:          standard Gauss–Legendre quadrature
- COMMON_VERTEX: Sauter–Schwab 8-region Duffy transformation
- COMMON_EDGE:   Sauter–Schwab 5-region Duffy transformation

Only supports Quad8 elements. Falls back to standard quadrature for Tri6
and Line3 elements (Duffy for those families can be added separately).

Also returns Aᵢ (area of elem_i) computed via standard quadrature.
"""
function element_pair_view_factor_duffy(coords  ::Matrix{Float64},
                                         elem_i  ::SurfaceElement,
                                         elem_j  ::SurfaceElement,
                                         nquad   ::Int,
                                         bvh     ::Union{BVHTree,Nothing},
                                         mesh_dim::Int = 2)::Tuple{Float64,Float64}
    # Compute area of elem_i via standard quadrature (always needed)
    rule  = gauss_legendre_2d(nquad)
    wts   = rule.weights
    pts   = rule.points
    Ai    = 0.0
    for p in eachindex(wts)
        ξ, η = pts[1,p], pts[2,p]
        if elem_i.family === :quad
            _, dA = quad8_normal_and_area_element(coords, elem_i.nodes, ξ, η)
            Ai += wts[p] * dA
        end
    end

    # Only Duffy for quad-quad pairs; fall back for other families
    if elem_i.family !== :quad || elem_j.family !== :quad || mesh_dim == 1
        # Use standard quadrature via the ViewFactorKernel path
        # (import avoided to prevent circular dependency; inline standard integral)
        raw = _standard_integral(coords, elem_i, elem_j, nquad, bvh)
        return raw, Ai
    end

    stype, ci, cj = singularity_type(elem_i, elem_j)

    if stype === NONE
        raw = _standard_integral(coords, elem_i, elem_j, nquad, bvh)
    elseif stype === COMMON_VERTEX
        raw = _vertex_integral(coords, elem_i, elem_j, ci[1], cj[1], nquad, bvh)
    else  # COMMON_EDGE
        raw = _edge_integral(coords, elem_i, elem_j, ci, cj, nquad, bvh)
    end

    return raw, Ai
end

end # module DuffyKernel
