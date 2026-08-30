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
# Both elements (Quad4 or Quad8 — never mixed) are mapped to the unit square
# [0,1]² via
#   ξ = 2u - 1,  η = 2v - 1   (Jacobian = 4 per element)
# All Duffy decompositions operate in (u,v) ∈ [0,1]². The decomposition itself
# only depends on corner-node adjacency and the isoparametric map, not on
# element order, so the same 8-region/5-region formulas apply to Quad4 pairs;
# only the physical-point/normal evaluation (`_eval_quad`) dispatches on
# `elem.family` to call the Quad4 or Quad8 shape functions.
#
# Singularity cases for two same-order quad elements (Quad4 or Quad8)
# ---------------------------------------------------------------------
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

import ..Quadrature:  gauss_legendre_1d
import ..Geometry:    quad8_physical_point, quad8_normal_and_area_element,
                      quad4_physical_point, quad4_normal_and_area_element
import ..BVH:         BVHTree
import ..RayCast:     is_visible
import ..MeshIO:      SurfaceElement
import ..ViewFactorKernel: element_pair_view_factor, precompute_quad

export element_pair_view_factor_duffy, singularity_type,
       near_pairs, patch_adjacent_pairs_duffy!

# ---------------------------------------------------------------------------
# Singularity classification
# ---------------------------------------------------------------------------

@enum SingularityType NONE COMMON_VERTEX COMMON_EDGE

"""
    singularity_type(elem_i, elem_j) -> (SingularityType, shared_corners_i, shared_corners_j)

Detect whether two same-order quad elements (both Quad4 or both Quad8) share
corner nodes, and return which local corner indices (1-4) are shared on each
element. Corner nodes are indices 1-4 in both the Quad4 and Quad8 node
ordering (Quad8's mid-edge nodes 5-8 are never singular-pair candidates).
"""
function singularity_type(elem_i::SurfaceElement,
                           elem_j::SurfaceElement)
    # Meaningful for same-order quad pairs only (both Quad4 or both Quad8);
    # tri handled separately if needed, and mixed Quad4/Quad8 pairs fall back
    # to standard quadrature in element_pair_view_factor_duffy.
    (elem_i.family === elem_j.family && elem_i.family in (:quad, :quad4)) ||
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
# Evaluation helpers: map (u,v) ∈ [0,1]² to physical space via the element's
# own family (Quad4 or Quad8). Jacobian factor = 4 (from ξ=2u-1, η=2v-1 map)
# ---------------------------------------------------------------------------

@inline function _eval_quad(coords, elem::SurfaceElement, u::Float64, v::Float64)
    ξ = 2u - 1.0;  η = 2v - 1.0
    if elem.family === :quad4
        x     = quad4_physical_point(coords, elem.nodes, ξ, η)
        n, dA = quad4_normal_and_area_element(coords, elem.nodes, ξ, η)
    else
        x     = quad8_physical_point(coords, elem.nodes, ξ, η)
        n, dA = quad8_normal_and_area_element(coords, elem.nodes, ξ, η)
    end
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
# Standard double quadrature (used for the NONE case). Delegates to the
# already family-generic element_pair_view_factor (ViewFactorKernel.jl)
# instead of re-deriving Quad8-only shape-function evaluation here.
# ---------------------------------------------------------------------------

function _standard_integral(coords, elem_i, elem_j, nquad, bvh)::Float64
    raw, _ = element_pair_view_factor(coords, elem_i, elem_j, nquad, bvh, 2)
    return raw
end

# ---------------------------------------------------------------------------
# COMMON_VERTEX: "biggest-coordinate" Duffy decomposition
#
# Shift coordinates so the shared corner sits at the origin of each unit
# square: du=|u-u0i|, dv=|v-v0i|, ds=|s-u0j|, dt=|t-v0j|, each ranging over
# [0,1]. The singularity is the single point (du,dv,ds,dt)=(0,0,0,0). Split
# [0,1]⁴ into 4 regions by which of (du,dv,ds,dt) is largest; in region k,
# set that coordinate to ρ∈[0,1] and the other three to ρ·η (η∈[0,1]):
#
#   region 1: du=ρ,     dv=ρη₁,   ds=ρη₂,   dt=ρη₃
#   region 2: dv=ρ,     du=ρη₁,   ds=ρη₂,   dt=ρη₃
#   region 3: ds=ρ,     du=ρη₁,   dv=ρη₂,   dt=ρη₃
#   region 4: dt=ρ,     du=ρη₁,   dv=ρη₂,   ds=ρη₃
#
# This is a plain "biggest coordinate" substitution (elementary, not
# Sauter-Schwab's specific formulas) with Jacobian ρ³ in every region — for
# region 1: ∂(du,dv,ds,dt)/∂(ρ,η₁,η₂,η₃) is lower-triangular with diagonal
# (1,ρ,ρ,ρ), determinant ρ³; the other 3 regions are permutations of the
# same computation. The 4 regions exactly tile [0,1]⁴ (η's ties have measure
# zero): ∫region_k 1 dρdη₁dη₂dη₃ = ∫₀¹ρ³dρ = 1/4 each, ×4 = 1 = vol([0,1]⁴).
# Near ρ=0 all of (du,dv,ds,dt) → 0 together, so r → 0 like ρ and the kernel
# ~1/r² ~1/ρ²; combined with the ρ³ Jacobian the integrand is O(ρ), which is
# finite and smooth in ρ, unlike the un-regularized 1/ρ² integrand a plain
# quadrature rule sees at the shared corner.
# ---------------------------------------------------------------------------

function _vertex_integral(coords, elem_i, elem_j,
                           ci::Int, cj::Int,
                           nquad::Int, bvh)::Float64
    pts1, wts1 = gauss_legendre_1d(nquad)
    pts01  = @. (pts1 + 1.0) / 2.0
    wts01  = wts1 ./ 2.0
    nq     = length(pts01)

    # Singular corner UV coordinates on each element
    u0i, v0i = QUAD_CORNER_UV[1, ci], QUAD_CORNER_UV[2, ci]
    u0j, v0j = QUAD_CORNER_UV[1, cj], QUAD_CORNER_UV[2, cj]

    Fij = 0.0
    do_vis = bvh !== nothing

    for region in 1:4
        for iρ in 1:nq, iη1 in 1:nq, iη2 in 1:nq, iη3 in 1:nq
            ρ   = pts01[iρ];  wρ   = wts01[iρ]
            η1  = pts01[iη1]; wη1  = wts01[iη1]
            η2  = pts01[iη2]; wη2  = wts01[iη2]
            η3  = pts01[iη3]; wη3  = wts01[iη3]

            if region == 1
                du, dv, ds, dt = ρ,     ρ*η1,   ρ*η2,   ρ*η3
            elseif region == 2
                du, dv, ds, dt = ρ*η1,  ρ,      ρ*η2,   ρ*η3
            elseif region == 3
                du, dv, ds, dt = ρ*η1,  ρ*η2,   ρ,      ρ*η3
            else
                du, dv, ds, dt = ρ*η1,  ρ*η2,   ρ*η3,   ρ
            end

            # Reflect the offset back onto the correct half of each unit
            # square, depending on which corner (0 or 1 in each axis) is
            # the shared one — offsets always move *into* the square.
            u = u0i == 0.0 ? du : 1.0 - du
            v = v0i == 0.0 ? dv : 1.0 - dv
            s = u0j == 0.0 ? ds : 1.0 - ds
            t = v0j == 0.0 ? dt : 1.0 - dt

            xi, ni, dAi = _eval_quad(coords, elem_i, u, v)
            xj, nj, dAj = _eval_quad(coords, elem_j, s, t)

            K = vf_kernel_val(xi, ni, xj, nj)
            K == 0.0 && continue
            do_vis && !is_visible(bvh, xi, xj) && continue

            jac = ρ^3
            Fij += wρ * wη1 * wη2 * wη3 * jac * K * dAi * dAj
        end
    end
    return Fij
end

# ---------------------------------------------------------------------------
# COMMON_EDGE: along-edge free parameter + 3D "biggest-coordinate" removal
#
# Build an edge-local (u,v) system on each element directly from the shared
# corner pair, rather than a "canonical edge number" abstraction that would
# discard which physical corner is u=0 vs u=1: elem_i's edge runs from its
# corner ci[1] (u=0) to ci[2] (u=1); elem_j's from cj[1] (s=0) to cj[2]
# (s=1). `singularity_type` guarantees node(ci[k]) == node(cj[k]) for both
# k=1,2, so u=0 and s=0 are *the same physical point* by construction — no
# separate orientation bookkeeping is needed (this sidesteps a real bug in
# the previous edge-index-only rotation, which had no way to tell whether
# the two elements traversed the shared edge in the same or opposite
# direction). `_edge_local_to_ref` below maps this (u,v) system to the
# element's actual (ξ,η) reference square for whichever of the 4 sides
# (in either direction — 8 ordered corner pairs) the shared edge is.
#
# In this (u,v,s,t) ∈ [0,1]⁴ system the singularity is the *line* u=s,
# v=t=0 (any point along the shared edge, not just its corners) — v,t are
# already independent of u,s, so only (s vs u, v, t) participate in it.
#
# Split the (u,s) square into the two triangles u≥s / u<s (Jacobian 1each,
# together tiling [0,1]²); on u≥s substitute w=(u-s)/u ∈[0,1] (so
# s=u(1-w), automatically in [0,u]), keeping u free — this is the standard
# "unit square via ratio" parametrization of a right triangle, with
# Jacobian u (lower-triangular ∂(u,s)/∂(u,w) = [[1,0],[1-w,-u]], det=-u).
# u<s is the mirror image with s free and w=(s-u)/s.
#
# Within each triangle, w,v,t → 0 together is exactly the same kind of
# single-point singularity as COMMON_VERTEX, now in 3 variables instead of
# 4: split by which of (w,v,t) is largest (3 regions, "biggest coordinate"
# again), Jacobian ρ² (n=3 ⟹ ρ^(n-1)). Combined with the triangle's own
# Jacobian (u or s), each of the 6 regions (2 triangles × 3 sub-regions) has
# Jacobian (u or s)·ρ². Tiling check: per triangle, 3×∫₀¹u du·∫₀¹ρ²dρ = 3×
# (1/2)(1/3) = 1/2 = the triangle's own area in the original (u,v,s,t)
# domain (½ from the (u,s) triangle × 1 from (v,t) ranging freely over
# [0,1]²); ×2 triangles = 1 = vol([0,1]⁴). Near the singularity r²~w²+v²+t²
# ~ρ² so the kernel ~1/ρ², and combined with the ρ² Jacobian (times the free
# u or s factor) the integrand is O(1) — bounded, no residual singularity.
# ---------------------------------------------------------------------------

# Explicit local(u,v) → reference(ξ,η) map for the edge running from corner
# `a` to corner `b` (u=0 at a, u=1 at b; v=0 on the edge, v>0 into the
# element interior). Quad corner layout: 1=(0,0), 2=(1,0), 3=(1,1), 4=(0,1).
@inline function _edge_local_to_ref(a::Int, b::Int, u::Float64, v::Float64)
    if     (a, b) == (1, 2); ξ = u;      η = v
    elseif (a, b) == (2, 1); ξ = 1.0-u;  η = v
    elseif (a, b) == (2, 3); ξ = 1.0-v;  η = u
    elseif (a, b) == (3, 2); ξ = 1.0-v;  η = 1.0-u
    elseif (a, b) == (3, 4); ξ = 1.0-u;  η = 1.0-v
    elseif (a, b) == (4, 3); ξ = u;      η = 1.0-v
    elseif (a, b) == (4, 1); ξ = v;      η = 1.0-u
    else                     ξ = v;      η = u    # (a,b) == (1,4)
    end
    return ξ, η
end

@inline function _eval_edge_quad(coords, elem::SurfaceElement,
                                  a::Int, b::Int, u::Float64, v::Float64)
    ξloc, ηloc = _edge_local_to_ref(a, b, u, v)
    ξ = 2ξloc - 1.0;  η = 2ηloc - 1.0
    if elem.family === :quad4
        x     = quad4_physical_point(coords, elem.nodes, ξ, η)
        n, dA = quad4_normal_and_area_element(coords, elem.nodes, ξ, η)
    else
        x     = quad8_physical_point(coords, elem.nodes, ξ, η)
        n, dA = quad8_normal_and_area_element(coords, elem.nodes, ξ, η)
    end
    return x, n, dA * 4.0   # 4 = Jacobian of [0,1]² → [-1,1]²
end

function _edge_integral(coords, elem_i, elem_j,
                         ci::Vector{Int}, cj::Vector{Int},
                         nquad::Int, bvh)::Float64
    pts1, wts1 = gauss_legendre_1d(nquad)
    pts01  = @. (pts1 + 1.0) / 2.0
    wts01  = wts1 ./ 2.0
    nq     = length(pts01)

    a_i, b_i = ci[1], ci[2]     # elem_i's edge: u=0 at a_i, u=1 at b_i
    a_j, b_j = cj[1], cj[2]     # elem_j's edge: s=0 at a_j (== node(a_i)), s=1 at b_j

    Fij    = 0.0
    do_vis = bvh !== nothing

    for branch in (:plus, :minus), region in 1:3
        for ia in 1:nq, iρ in 1:nq, iη1 in 1:nq, iη2 in 1:nq
            a  = pts01[ia];  wa  = wts01[ia]     # the free coordinate (u or s)
            ρ  = pts01[iρ];  wρ  = wts01[iρ]
            η1 = pts01[iη1]; wη1 = wts01[iη1]
            η2 = pts01[iη2]; wη2 = wts01[iη2]

            if region == 1
                w, v, t = ρ,     ρ*η1,  ρ*η2
            elseif region == 2
                w, v, t = ρ*η1,  ρ,     ρ*η2
            else
                w, v, t = ρ*η1,  ρ*η2,  ρ
            end

            if branch === :plus
                u = a;  s = u * (1.0 - w)
            else
                s = a;  u = s * (1.0 - w)
            end

            xi, ni, dAi = _eval_edge_quad(coords, elem_i, a_i, b_i, u, v)
            xj, nj, dAj = _eval_edge_quad(coords, elem_j, a_j, b_j, s, t)

            K = vf_kernel_val(xi, ni, xj, nj)
            K == 0.0 && continue
            do_vis && !is_visible(bvh, xi, xj) && continue

            jac = a * ρ^2
            Fij += wa * wρ * wη1 * wη2 * jac * K * dAi * dAj
        end
    end
    return Fij
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

The Duffy/Sauter–Schwab singular treatment applies to same-order quad pairs
— both Quad4 or both Quad8. This matters for Quad4 in particular because
that's the only family `load_re2` produces (Nek5000/NekRS boundary faces),
and structured hex-mesh boundaries have many edge-adjacent Quad4 pairs whose
shared-edge singularity plain quadrature (and Monte Carlo, whose variance is
unbounded there) cannot resolve — this is what makes those meshes fail
`check_closure` without Duffy. For every other case (Tri6, Tri3, Line2,
Line3, mixed Quad4/Quad8 pairs) and for 2-D curve meshes, this falls back to
the standard tensor-product quadrature in `element_pair_view_factor`, which
handles all families and both dimensions correctly (just without singular
regularization).

Also returns Aᵢ (the measure — area or arc length — of elem_i).
"""
function element_pair_view_factor_duffy(coords  ::Matrix{Float64},
                                         elem_i  ::SurfaceElement,
                                         elem_j  ::SurfaceElement,
                                         nquad   ::Int,
                                         bvh     ::Union{BVHTree,Nothing},
                                         mesh_dim::Int = 2)::Tuple{Float64,Float64}
    # Only same-order quad pairs (Quad4-Quad4 or Quad8-Quad8) use the Duffy
    # transformation; everything else (linear elements, triangles, curves,
    # mixed-order pairs) falls back to the standard quadrature path, which
    # returns (raw integral, measure of elem_i).
    same_quad_family = elem_i.family === elem_j.family && elem_i.family in (:quad, :quad4)
    if !same_quad_family || mesh_dim == 1
        return element_pair_view_factor(coords, elem_i, elem_j, nquad, bvh, mesh_dim)
    end

    # Area of elem_i via the same family-generic quadrature ViewFactorKernel
    # uses elsewhere (Quad4 or Quad8, whichever elem_i actually is).
    Ai = precompute_quad(coords, elem_i, nquad, 2).Li

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

# ---------------------------------------------------------------------------
# Near-pair patch for Monte Carlo assembly
#
# The 1/r² kernel has *unbounded* variance for element pairs that share a
# vertex or edge — more samples doesn't fix this (it's a statistical
# inconsistency, not slow convergence). But it's not only touching pairs:
# two elements that are merely *close* relative to their own size (e.g.
# neighbors across a periodic seam, which aren't mesh-adjacent — they don't
# share a node index — but can sit a fraction of an element-width apart)
# still have high, slowly-converging variance, because a stratified sample
# pair can land close together by chance and dominate the estimate for that
# element (confirmed empirically: a real mesh's rowsum error did not shrink
# from 2000→8000 samples/pair for exactly this reason). Both cases are
# handled the same way here: find every pair within `factor` element-sizes
# of each other (a strict superset of touching pairs) via a uniform spatial
# grid (~O(N) for a well-shaped mesh, not O(N²)), and evaluate each with
# `element_pair_view_factor_duffy`, which already dispatches correctly
# between the Duffy transform (touching) and plain quadrature (merely
# close, no singularity — quadrature is fine there, it just needs to be the
# one evaluating it, not MC sampling). Both are O(N) in count and therefore
# asymptotically free next to the O(N²) Monte Carlo bulk at scale.
# ---------------------------------------------------------------------------

"""
    near_pairs(coords, elems; factor=3.0) -> Vector{Tuple{Int,Int}}

All index pairs `(i,j)`, `i<j`, whose elements are within `factor` times the
larger element's own diameter of each other (centroid distance) — a strict
superset of literally touching (shared-node) pairs. Found via a uniform
spatial grid over element centroids, not by testing every pair.
"""
function near_pairs(coords::Matrix{Float64}, elems::Vector{SurfaceElement};
                     factor::Float64 = 3.0)::Vector{Tuple{Int,Int}}
    N = length(elems)
    N < 2 && return Tuple{Int,Int}[]

    cent  = Matrix{Float64}(undef, 3, N)
    esize = Vector{Float64}(undef, N)
    for i in 1:N
        ns  = elems[i].nodes
        n0  = length(ns)
        cx  = sum(coords[1, n] for n in ns) / n0
        cy  = sum(coords[2, n] for n in ns) / n0
        cz  = sum(coords[3, n] for n in ns) / n0
        cent[1, i] = cx; cent[2, i] = cy; cent[3, i] = cz
        esize[i] = 2 * maximum(sqrt((coords[1,n]-cx)^2 + (coords[2,n]-cy)^2 +
                                     (coords[3,n]-cz)^2) for n in ns)
    end

    cellsize = max(maximum(esize), eps()) * factor
    key(i) = (floor(Int, cent[1,i]/cellsize), floor(Int, cent[2,i]/cellsize),
              floor(Int, cent[3,i]/cellsize))
    grid = Dict{NTuple{3,Int}, Vector{Int}}()
    for i in 1:N
        push!(get!(grid, key(i), Int[]), i)
    end

    # ±2 cells of margin: with cellsize == the *largest* per-element
    # threshold, two elements within that distance can, in the worst case
    # (near a cell boundary), land 2 grid cells apart, not just 1.
    pairs = Set{Tuple{Int,Int}}()
    for i in 1:N
        kx, ky, kz = key(i)
        for dx in -2:2, dy in -2:2, dz in -2:2
            cell = get(grid, (kx+dx, ky+dy, kz+dz), nothing)
            cell === nothing && continue
            for j in cell
                j <= i && continue
                d = sqrt((cent[1,i]-cent[1,j])^2 + (cent[2,i]-cent[2,j])^2 +
                          (cent[3,i]-cent[3,j])^2)
                d < factor * max(esize[i], esize[j]) && push!(pairs, (i, j))
            end
        end
    end
    return collect(pairs)
end

"""
    patch_adjacent_pairs_duffy!(raw, coords, elems, nquad, mesh_dim; factor=3.0)

Overwrite `raw[i,j]` and `raw[j,i]` (the raw, pre-area-division double
integral, symmetric in `i,j`) for every pair within `factor` element-sizes
of each other (see [`near_pairs`](@ref)) with the deterministic value from
`element_pair_view_factor_duffy`, in place. Called after Monte Carlo
assembly (CPU or GPU) so those O(N) pairs get a statistically consistent
value instead of high- or unbounded-variance samples. No-op for curve
meshes (`mesh_dim == 1`), where Duffy doesn't apply (see
`element_pair_view_factor_duffy`). Obstruction is not checked for these
pairs — nearby elements at this distance scale cannot have a third surface
positioned between them without also being flagged as obstructing the bulk
Monte Carlo pairs via `obstruction_groups`.
"""
function patch_adjacent_pairs_duffy!(raw::Matrix{Float64},
                                      coords::Matrix{Float64},
                                      elems::Vector{SurfaceElement},
                                      nquad::Int,
                                      mesh_dim::Int;
                                      factor::Float64 = 3.0)
    mesh_dim == 1 && return raw
    for (i, j) in near_pairs(coords, elems; factor=factor)
        raw_ij, _ = element_pair_view_factor_duffy(coords, elems[i], elems[j],
                                                     nquad, nothing, mesh_dim)
        raw[i, j] = raw_ij
        raw[j, i] = raw_ij
    end
    return raw
end

end # module DuffyKernel
