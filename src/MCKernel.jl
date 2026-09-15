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
                      quad4_physical_point, quad4_normal_and_area_element,
                      tri3_physical_point,  tri3_normal_and_area_element,
                      line3_physical_point, line3_normal_and_length_element,
                      line2_physical_point, line2_normal_and_length_element
import ..BVH:         BVHTree
import ..RayCast:     is_visible
import ..MeshIO:      SurfaceElement

export element_pair_view_factor_mc, ElementSamples, sample_element_mc

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
            ns[k]   = nk; dAs[k] = dAk
        end
        A = wt * sum(dAs)
    elseif elem.family === :quad4
        ref_pts, wt = _stratified_quad_points(n, rng)
        for k in 1:n
            ξ, η    = ref_pts[k]
            xs[k]   = quad4_physical_point(coords, elem.nodes, ξ, η)
            nk, dAk = quad4_normal_and_area_element(coords, elem.nodes, ξ, η)
            ns[k]   = nk; dAs[k] = dAk
        end
        A = wt * sum(dAs)
    elseif elem.family === :tri
        ref_pts, wt = _stratified_tri_points(n, rng)
        for k in 1:n
            ξ, η = ref_pts[k]
            xs[k], ns[k], dAs[k] = _tri6_point_normal_dA(coords, elem.nodes, ξ, η)
        end
        A = wt * sum(dAs)
    elseif elem.family === :tri3
        ref_pts, wt = _stratified_tri_points(n, rng)
        for k in 1:n
            ξ, η    = ref_pts[k]
            xs[k]   = tri3_physical_point(coords, elem.nodes, ξ, η)
            nk, dAk = tri3_normal_and_area_element(coords, elem.nodes, ξ, η)
            ns[k]   = nk; dAs[k] = dAk
        end
        A = wt * sum(dAs)
    elseif elem.family === :line3
        ref_pts, wt = _stratified_line_points(n, rng)
        for k in 1:n
            ξ        = ref_pts[k]
            xs[k]    = line3_physical_point(coords, elem.nodes, ξ)
            nk, dLk  = line3_normal_and_length_element(coords, elem.nodes, ξ)
            ns[k]    = nk; dAs[k] = dLk
        end
        A = wt * sum(dAs)
    else  # :line2
        ref_pts, wt = _stratified_line_points(n, rng)
        for k in 1:n
            ξ        = ref_pts[k]
            xs[k]    = line2_physical_point(coords, elem.nodes, ξ)
            nk, dLk  = line2_normal_and_length_element(coords, elem.nodes, ξ)
            ns[k]    = nk; dAs[k] = dLk
        end
        A = wt * sum(dAs)
    end

    return xs, ns, dAs, A
end

"""
    ElementSamples

Pre-drawn Monte Carlo samples for one element: physical positions `xs`, unit
normals `ns`, raw Jacobians `dAs`, MC area estimate `A`, the reference-domain
area `ref_area`, and sample count `n`.

Drawing these once per element (O(N)) and reusing them across all element pairs
avoids the O(N²) re-sampling a naive pairwise loop incurs. Each per-pair
estimate stays unbiased: `xs` is uniform on the element, and the pair kernel
pairs each sample of one element with a uniformly random sample of the other
(see [`element_pair_view_factor_mc`](@ref)).
"""
struct ElementSamples
    xs       :: Vector{SVector{3,Float64}}
    ns       :: Vector{SVector{3,Float64}}
    dAs      :: Vector{Float64}
    A        :: Float64
    ref_area :: Float64
    n        :: Int
end

"""
    sample_element_mc(coords, elem, n, rng) -> ElementSamples

Draw and cache `n` stratified samples on `elem` for reuse across pairs.

The samples are stored in a uniformly random order (one `randperm` draw),
not the stratum-scan order `_sample_element` produces them in. This is what
lets the pair integrator below pair same-index samples of two elements
directly — see [`element_pair_view_factor_mc`](@ref) for why that is both
faster and lower-variance than the per-sample random index it replaces.
"""
function sample_element_mc(coords::Matrix{Float64}, elem::SurfaceElement,
                            n::Int, rng::AbstractRNG)::ElementSamples
    xs, ns, dAs, A = _sample_element(coords, elem, n, rng)
    perm = randperm(rng, n)
    return ElementSamples(xs[perm], ns[perm], dAs[perm], A, _ref_area(elem), n)
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

    si = sample_element_mc(coords, elem_i, n_samples, rng)
    sj = sample_element_mc(coords, elem_j, n_samples, rng)
    return element_pair_view_factor_mc(si, sj, bvh, mesh_dim, rng)
end

"""
    element_pair_view_factor_mc(si::ElementSamples, sj::ElementSamples, bvh,
                                 mesh_dim, rng) -> (raw, Ai)

Fast path used by the assembly loop: estimate the raw double integral from
pre-drawn samples ([`sample_element_mc`](@ref)). `si` and `sj` must hold the
same number of samples. `rng` is accepted but unused — all randomness for
this pair was already spent when `si` and `sj` were drawn.

Both sample sets are stratified in the same deterministic stratum order and
then stored in a uniformly random (`randperm`) order, so the k-th stored
sample of an element is uniform over the element, independent of `k`. Pairing
same-index samples (`xᵢ[k]` with `xⱼ[k]`) therefore pairs each `xᵢ[k]` with a
uniformly random `xⱼ`, exactly like the per-sample random index this
replaced — but reads both arrays sequentially, which is faster and
(measured) *lower*-variance than the per-sample index.

!!! note "Why this differs from the single-offset approach that was rejected"
    An earlier version drew `xⱼ`'s partner as `rand(rng, 1:n)` per sample —
    correct, but the dominant cost of the loop, since indexing `sj` out of
    order defeats prefetching (~2.2x slower than sequential access,
    measured). A single random cyclic offset per pair is equally unbiased
    and just as fast as this sequential scheme, but was rejected: it
    preserves the *adjacency* between neighbouring stratum indices (a shift
    keeps neighbours as neighbours), so if the integrand varies smoothly
    across the element the shift correlates nearby terms and the per-pair
    standard deviation rose ~30x in that experiment (7.5e-6 → 2.4e-4 over
    40 seeds on a reactor-pin Quad8 pair).

    A `randperm` is a different kind of single random draw: it is one of
    `n!` bijections rather than one of `n` shifts, and — unlike a shift —
    it does not preserve adjacency between stratum indices, so it does not
    reproduce that correlation. Measured on the same kind of pair, the
    per-pair standard deviation with this scheme is *lower* than the
    original per-sample-random-index version (about 5x lower on a Quad8
    pair, 1.0e-4 vs 5.0e-4 relative), and full-assembly comparisons against
    the Howell catalog (`benchmarks/howell/`) across Quad8, Tri6 and Line3
    element families, obstructed and unobstructed, show equal or lower
    noise at the same `n_samples`. See `changelog.md` for the measurements.
"""
function element_pair_view_factor_mc(si      ::ElementSamples,
                                      sj      ::ElementSamples,
                                      bvh     ::Union{BVHTree,Nothing},
                                      mesh_dim::Int,
                                      rng     ::AbstractRNG = Random.default_rng()
                                      )::Tuple{Float64,Float64}
    do_vis = bvh !== nothing
    is_2d  = mesh_dim == 1
    n      = si.n
    xs_i, ns_i, dAs_i = si.xs, si.ns, si.dAs
    xs_j, ns_j, dAs_j = sj.xs, sj.ns, sj.dAs

    K_sum = 0.0
    @inbounds for k in 1:n
        # Same-index pairing: both arrays are already in a uniformly random
        # (randperm'd) order, so this pairs xᵢ[k] with a uniformly random
        # xⱼ — see the docstring above.
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

    # MC estimator for ∬K dAi dAj.  The stratified sampler returns raw
    # Jacobians dAs (no 1/n factor), with Ai = (ref_area_i/n)*Σ dAs_i, so
    # the reference-domain areas and the 1/n normalisation are folded in here:
    raw = K_sum * si.ref_area * sj.ref_area / n
    return raw, si.A
end

@inline _ref_area(el::SurfaceElement) =
    (el.family === :quad || el.family === :quad4) ? 4.0 :   # [-1,1]²
    (el.family === :tri  || el.family === :tri3)  ? 0.5 :   # reference triangle
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
