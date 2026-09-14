# src/ElementBounds.jl
# ---------------------------------------------------------------------------
# Conservative per-element bounds used to reject element pairs that provably
# cannot exchange radiation, before any integration is attempted.
#
# Motivation
# ----------
# Both the quadrature and Monte Carlo pair integrators already return zero for
# a *point* pair whose cosines are non-positive (`K == 0 && continue`), but they
# discover that one point pair at a time — after entering the pair and, with
# obstruction enabled, potentially after a BVH ray cast. On closed convex
# bodies (a tube bundle, a pebble bed) most element pairs face away from each
# other entirely, so that per-point test is paid `nquad⁴` (or `n_samples`)
# times per pair to conclude nothing.
#
# This module bounds each element by a sphere plus a normal cone, which lets a
# pair be rejected with a handful of flops. The test is *conservative*: it
# rejects only pairs for which the kernel is provably zero at every point pair,
# so the assembled matrix is unchanged (bitwise) — this is purely an
# evaluation-order optimization.
#
# The bound
# ---------
# Each element carries two axis-aligned boxes: one bounding its points
# [plo, phi] and one bounding its unit normals [nlo, nhi]. For p ∈ i, q ∈ j the
# separation v = q - p then lies in the Minkowski-difference box
#
#     v ∈ [ploⱼ - phiᵢ,  phiⱼ - ploᵢ]
#
# and an upper bound on the cosine numerator follows by maximising each
# component independently:
#
#     dot(n, q - p) ≤ Σₖ max(nloₖ·vloₖ, nloₖ·vhiₖ, nhiₖ·vloₖ, nhiₖ·vhiₖ)
#
# If that bound is ≤ 0 then cos θᵢ ≤ 0 at every point pair, the kernel vanishes
# identically, and the pair contributes nothing. The same test is applied from
# j's side with v negated. Maximising per component ignores the coupling
# between them, which can only overestimate the true maximum — the direction
# that keeps the test conservative.
#
# Why boxes and not a sphere plus a normal cone (both were tried):
#
#   * A circular normal cone of half-angle α contributes slack growing like
#     |d|·sin α, which swamps the test at long range. On an extruded mesh — a
#     tube bundle, this package's motivating case — the normals of a
#     cylindrical element have *no* axial component at all, but a circular cone
#     inflates that 1-D circumferential arc into a 2-D disc admitting axial
#     tilt. The componentwise box represents "n_z ≡ 0" exactly.
#   * A bounding sphere contributes Rᵢ + Rⱼ regardless of direction, which
#     badly over-bounds elements that are elongated rather than square — again
#     the extruded case, where elements are far taller than they are wide. The
#     position box contributes only each axis' own extent.
#
# See `changelog.md` for the measured rejection rates of all three variants on
# a reactor-pin fixture.
#
# Validity
# --------
# Correctness needs the sphere to contain the element and the cone to contain
# every surface normal. Both are estimated by sampling the isoparametric map on
# a fixed grid (independent of `nquad`, so bounds do not drift with quadrature
# order) and then padded outward. For 1st-order elements the sphere bound is
# exact up to the padding, since a Quad4/Tri3 lies in the convex hull of its
# nodes. 2nd-order elements can in principle bulge beyond the sampled points —
# the serendipity Quad8 basis is not a non-negative partition of unity, so the
# convex-hull property does not hold — which is what the padding covers. Both
# paddings widen the bounds, i.e. they can only make the test reject *fewer*
# pairs, never more. `compute_view_factors(...; facing_cull=false)` disables
# the test outright for meshes pathological enough to worry about.
# ---------------------------------------------------------------------------

module ElementBounds

using LinearAlgebra
using StaticArrays

import ..MeshIO: SurfaceElement
import ..Geometry: quad8_physical_point, quad8_normal_and_area_element,
                   quad4_physical_point, quad4_normal_and_area_element,
                   tri3_physical_point,  tri3_normal_and_area_element,
                   line3_physical_point, line3_normal_and_length_element,
                   line2_physical_point, line2_normal_and_length_element

export ElementBound, build_element_bound, build_element_bounds, pair_can_see

# Sampling density for the bound estimate. 5 points per parametric direction
# (25 per quad) is far denser than any curvature a usable 2nd-order element
# carries, and costs O(N), not O(N²).
const NSAMP = 5

# Outward padding. `PPAD` expands the position box by this fraction of its own
# diagonal; `NPAD` expands the normal box in every component. Both loosen the
# bounds, so both can only reduce the number of pairs rejected, never cause a
# wrong rejection.
const PPAD = 0.02
const NPAD = 0.02

"""
    ElementBound

Conservative axis-aligned bounds for one element: a box containing its points
and a box containing its unit normals.

Fields
------
- `plo` : componentwise lower bound on the element's points (padded outward)
- `phi` : componentwise upper bound on the same
- `nlo` : componentwise lower bound on the element's unit normals (padded)
- `nhi` : componentwise upper bound on the same
"""
struct ElementBound
    plo :: SVector{3,Float64}
    phi :: SVector{3,Float64}
    nlo :: SVector{3,Float64}
    nhi :: SVector{3,Float64}
end

# Reference-domain sample grid for each family, as (ξ, η) pairs.
function _ref_samples(family::Symbol)
    ts = range(-1.0, 1.0; length=NSAMP)
    if family === :quad || family === :quad4
        return [(ξ, η) for ξ in ts, η in ts] |> vec
    elseif family === :tri || family === :tri3
        # Barycentric grid on {ξ≥0, η≥0, ξ+η≤1}
        us = range(0.0, 1.0; length=NSAMP)
        return [(u, v) for u in us, v in us if u + v <= 1.0 + 1e-12] |> vec
    else  # :line2, :line3 — η unused
        return [(ξ, 0.0) for ξ in ts]
    end
end

# Physical point and unit normal at a reference-domain location.
@inline function _point_and_normal(coords, elem::SurfaceElement, ξ, η)
    f = elem.family
    if f === :quad
        x = quad8_physical_point(coords, elem.nodes, ξ, η)
        n, _ = quad8_normal_and_area_element(coords, elem.nodes, ξ, η)
        return x, n
    elseif f === :quad4
        x = quad4_physical_point(coords, elem.nodes, ξ, η)
        n, _ = quad4_normal_and_area_element(coords, elem.nodes, ξ, η)
        return x, n
    elseif f === :tri3
        x = tri3_physical_point(coords, elem.nodes, ξ, η)
        n, _ = tri3_normal_and_area_element(coords, elem.nodes, ξ, η)
        return x, n
    elseif f === :tri
        return _tri6_point_and_normal(coords, elem.nodes, ξ, η)
    elseif f === :line3
        x = line3_physical_point(coords, elem.nodes, ξ)
        n, _ = line3_normal_and_length_element(coords, elem.nodes, ξ)
        return x, n
    else  # :line2
        x = line2_physical_point(coords, elem.nodes, ξ)
        n, _ = line2_normal_and_length_element(coords, elem.nodes, ξ)
        return x, n
    end
end

# Tri6 point/normal, inlined here for the same reason MCKernel inlines it:
# the Tri6 evaluators live in ViewFactorKernel, and importing them would make
# this module depend on it.
@inline function _tri6_point_and_normal(coords, nodes, ξ::Float64, η::Float64)
    L1 = 1-ξ-η; L2 = ξ; L3 = η
    N    = SVector(L1*(2L1-1), L2*(2L2-1), L3*(2L3-1), 4L1*L2, 4L2*L3, 4L1*L3)
    dNdξ = SVector((4L1-1)*(-1.0), 4L2-1, 0.0, 4*(L2*(-1.0)+L1), 4L3, 4L3*(-1.0))
    dNdη = SVector((4L1-1)*(-1.0), 0.0, 4L3-1, 4*L2*(-1.0), 4L2, 4*(L3*(-1.0)+L1))
    x = @SVector zeros(3); dxdξ = @SVector zeros(3); dxdη = @SVector zeros(3)
    for a in 1:6
        xa = SVector{3,Float64}(coords[1,nodes[a]], coords[2,nodes[a]], coords[3,nodes[a]])
        x = x + N[a]*xa; dxdξ = dxdξ + dNdξ[a]*xa; dxdη = dxdη + dNdη[a]*xa
    end
    c = cross(dxdξ, dxdη)
    return x, c / norm(c)
end

"""
    build_element_bound(coords, elem) -> ElementBound

Estimate a padded bounding sphere and normal cone for one element by sampling
its isoparametric map. See the module docstring for what the bounds guarantee.
"""
function build_element_bound(coords::Matrix{Float64},
                              elem  ::SurfaceElement)::ElementBound
    refs = _ref_samples(elem.family)
    xs = SVector{3,Float64}[]
    ns = SVector{3,Float64}[]
    sizehint!(xs, length(refs) + length(elem.nodes))
    sizehint!(ns, length(refs))
    for (ξ, η) in refs
        x, n = _point_and_normal(coords, elem, ξ, η)
        push!(xs, x); push!(ns, n)
    end
    # The nodes themselves are on the element (or are its control points) and
    # cost nothing to include in the sphere estimate.
    for nd in elem.nodes
        push!(xs, SVector{3,Float64}(coords[1,nd], coords[2,nd], coords[3,nd]))
    end

    plo0 = SVector{3,Float64}(minimum(x[1] for x in xs),
                              minimum(x[2] for x in xs),
                              minimum(x[3] for x in xs))
    phi0 = SVector{3,Float64}(maximum(x[1] for x in xs),
                              maximum(x[2] for x in xs),
                              maximum(x[3] for x in xs))
    # Pad by a fraction of the box diagonal, with an absolute floor so that a
    # box that is degenerate along one axis (a planar, axis-aligned element)
    # still gets a nonzero margin there.
    diag = norm(phi0 - plo0)
    pad  = max(diag * PPAD, 1e-12)
    plo  = plo0 .- pad
    phi  = phi0 .+ pad

    nlo = SVector{3,Float64}(
        minimum(n[1] for n in ns) - NPAD,
        minimum(n[2] for n in ns) - NPAD,
        minimum(n[3] for n in ns) - NPAD)
    nhi = SVector{3,Float64}(
        maximum(n[1] for n in ns) + NPAD,
        maximum(n[2] for n in ns) + NPAD,
        maximum(n[3] for n in ns) + NPAD)
    return ElementBound(plo, phi, nlo, nhi)
end

"""
    build_element_bounds(coords, elems) -> Vector{ElementBound}

Build bounds for every element, in parallel. O(N).
"""
function build_element_bounds(coords::Matrix{Float64},
                               elems ::Vector{SurfaceElement})::Vector{ElementBound}
    bounds = Vector{ElementBound}(undef, length(elems))
    Threads.@threads for i in eachindex(elems)
        bounds[i] = build_element_bound(coords, elems[i])
    end
    return bounds
end

# Largest value of a·b for a ∈ [alo, ahi], b ∈ [blo, bhi]: the extremes of a
# bilinear form on a rectangle are attained at its corners.
@inline _max_prod(alo, ahi, blo, bhi) =
    max(max(alo*blo, alo*bhi), max(ahi*blo, ahi*bhi))

# Upper bound on dot(n, v) over n ∈ b's normal box and v ∈ [vlo, vhi].
@inline function _max_dot(b::ElementBound, vlo::SVector{3,Float64},
                           vhi::SVector{3,Float64})::Float64
    @inbounds (_max_prod(b.nlo[1], b.nhi[1], vlo[1], vhi[1]) +
               _max_prod(b.nlo[2], b.nhi[2], vlo[2], vhi[2]) +
               _max_prod(b.nlo[3], b.nhi[3], vlo[3], vhi[3]))
end

"""
    pair_can_see(bi, bj) -> Bool

`false` only when the view factor kernel is provably zero at *every* point pair
of the two elements, so the pair may be skipped without changing the result.
`true` means "cannot rule it out" — the caller must integrate normally.
"""
@inline function pair_can_see(bi::ElementBound, bj::ElementBound)::Bool
    # v = q - p for p ∈ i, q ∈ j lies in this Minkowski-difference box.
    vlo = bj.plo - bi.phi
    vhi = bj.phi - bi.plo
    # i must be able to face j, and j to face i (same separations, negated).
    _max_dot(bi, vlo, vhi) > 0.0 || return false
    _max_dot(bj, -vhi, -vlo) > 0.0 || return false
    return true
end

end # module ElementBounds
