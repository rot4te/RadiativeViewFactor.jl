# src/RayTraceKernel.jl
# ---------------------------------------------------------------------------
# Ray-shooting Monte Carlo view-factor integrator (CPU path).
#
# This is a different Monte Carlo method from `MCKernel.jl`'s pair-area
# sampling, not a faster version of it. Pair-area sampling estimates the
# double integral for one element PAIR at a time (O(N²) pairs, each needing
# its own visibility test against `obstruction_groups`). Ray-shooting instead
# treats the *whole* radiating mesh as one scene: for each element i, draw a
# point on it and a direction from the cosine-weighted hemisphere around its
# normal, and find the first surface that ray hits via ONE nearest-hit BVH
# query over every element (not just flagged `obstruction_groups`). This is
# O(N · n_rays · log N) instead of O(N² · n_samples), and obstruction is a
# side effect of the same query rather than a separate O(N²) check.
#
# The math
# --------
# The standard double-area view-factor integral can be rewritten as a single
# area integral over the (occlusion-limited) solid angle A_j subtends from
# each point x ∈ A_i:
#
#   F_{i→j} = (1/A_i) ∫_{A_i} [ ∫_{Ω_j visible from x} cos θ_i / π  dΩ ] dA_i
#
# (this holds because dA_j cos θ_j / r² = dΩ by definition of solid angle —
# a standard identity, e.g. Cohen & Wallace, *Radiosity and Realistic Image
# Synthesis*, or Siegel & Howell, *Thermal Radiation Heat Transfer*). If a
# ray direction ω is drawn from the cosine-weighted hemisphere pdf(ω) =
# cos θ_i / π, then
#
#   E[ 1(ray's first hit is j) ] = ∫_{Ω_j visible} (cos θ_i/π) dΩ = F_{x→j}
#
# — the indicator's expectation is *exactly* the point-to-area form factor,
# occlusion included (an occluded portion of j simply isn't reachable by any
# ray, so it can't be "the first hit"). Averaging over points x on i (drawn
# uniform-by-area) gives F_{i→j} directly: no cos θ_j, no 1/r², no separate
# visibility test — a ray hitting the wrong thing first *is* the visibility
# test. See `changelog.md` for the derivation checked against known cases
# and a real front/back-face bug found while prototyping this (next section).
#
# Front vs. back hits
# --------------------
# `nearest_hit_bvh` (BVH.jl) finds the closest geometric intersection with
# no winding/normal culling — correct for "is the ray blocked", since an
# opaque surface blocks a ray incident on either face (matching
# `intersect_ray_bvh`'s existing convention for `obstruction_groups`). But
# it is *not* sufficient on its own to decide "did the ray land on element
# j": a ray whose closest hit is the *back* of j (dot(hit normal, -direction)
# <= 0) has reached j's non-radiating side and stops there (the surface is
# still opaque, so nothing behind j is reachable either) — but this must not
# be attributed to F_{i→j}. This is the same check the existing kernels make
# via `cos θ_j <= 0 → K = 0` (`MCKernel.jl`'s `_kernel_3d`), just evaluated
# post-hoc on whichever element the ray actually hit rather than on a
# pre-selected target point.
#
# What counts as "the scene"
# ---------------------------
# The scene BVH is built from every element in `mesh.surface_elems` (the
# radiating set, already possibly restricted by `radiating_groups` before
# this module ever sees it), each element tagged with its own index, plus
# any additional geometry from `obstruction_groups` that is *not* already
# part of that radiating set (tagged 0: opaque, but not a valid target).
# Because every radiating element is already in the scene, radiating
# elements obstruct each other automatically — unlike the pair-area kernels,
# which only apply an obstruction check for groups explicitly listed in
# `obstruction_groups` (see `Assembly.jl`'s docstring, note 3: "a group never
# obstructs a pair involving itself"). This is an intentional behavioural
# difference, not a bug: it is more physically complete (self-shadowing
# between different elements of the same or different radiating bodies is
# real and no longer needs to be told apart via group bookkeeping), but it
# means ray-tracing results will not bit-match the other kernels' results on
# a mesh where the user relied on omitting true blockers from
# `obstruction_groups`.
#
# No singular pairs, no Duffy patch
# -----------------------------------
# The pair-area kernel's 1/r² kernel has *unbounded* variance for touching
# or nearly-touching element pairs, which is why `monte_carlo=true` always
# patches those with the deterministic Duffy transform. Ray-shooting has no
# such singularity: it never evaluates 1/r² at all, so every element pair —
# adjacent or not — has ordinary bounded (binomial) variance. `use_duffy`
# and the near-pair patch are therefore irrelevant here and not applied.
#
# Reciprocity
# -----------
# Because every element is *both* a ray source and a potential target, each
# unordered pair {i,j} gets two independent estimates: F_{i→j} from i's own
# rays and F_{j→i} from j's. `Assembly.jl` averages the corresponding raw
# double-integral values (using the true, quadrature-computed element areas)
# into one shared value before dividing back out to F — exactly the
# convention the other kernels already use (`raw[i,j] == raw[j,i]` by
# construction) — which makes the result exactly reciprocal (not just in
# expectation) and, being the average of two independent unbiased estimates,
# lower-variance than either alone.
# ---------------------------------------------------------------------------

module RayTraceKernel

using LinearAlgebra
using StaticArrays
using Random

import ..BVH:      BVHTree, build_bvh, nearest_hit_bvh
import ..MeshIO:   SurfaceElement, _is_quad
import ..MCKernel: _sample_element
import ..Geometry: quad8_physical_point

export SceneBVH, build_scene_bvh, raytrace_element

# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

"""
    SceneBVH

A BVH over every triangle in the ray-tracing scene, alongside `elem_of`,
which maps each triangle (by the same index `nearest_hit_bvh` returns) to
the radiating element index (`1:N`, into `mesh.surface_elems`) it belongs
to, or `0` for non-radiating blocker-only geometry (extra
`obstruction_groups` bodies not already part of the radiating set).
"""
struct SceneBVH
    bvh     :: BVHTree
    elem_of :: Vector{Int}
end

"""Number of cells per parametric direction used to facet a curved (Quad8)
element for the ray-tracing scene. Rays *leave* an element from its exact
isoparametric surface (`_sample_element`), so the scene they *land* on has to
follow the same surface or the two disagree: with a single corner quad, a
sphere faceted 6x6 per cubed-sphere block loses 1.5 % of its area, and the
missing solid angle lands on whatever is behind it. Chord error falls as the
square of the cell size, so 3 x 3 cuts that to about 0.2 % for 18 triangles per
element, which is cheap next to the nearest-hit queries themselves."""
const QUAD8_SCENE_SUBDIV = 3

"""Triangulate every element of `elems`, tagged with its 1-based index into
`elems`. Straight-sided elements give 2 triangles per quad (the same
v1,v2,v3 / v1,v3,v4 split `MeshIO._build_group_obs_soups` uses); curved Quad8
elements are subdivided on a `QUAD8_SCENE_SUBDIV`² parametric lattice through
their isoparametric map, so the scene follows the real surface."""
function _triangulate_tagged(coords::Matrix{Float64}, elems::Vector{SurfaceElement})
    k    = QUAD8_SCENE_SUBDIV
    ntri = 0
    for e in elems
        ntri += e.family === :quad ? 2k^2 : (_is_quad(e.family) ? 2 : 1)
    end
    soup    = Array{Float64,3}(undef, 3, 3, ntri)
    elem_of = Vector{Int}(undef, ntri)
    t = 0
    @inbounds for (ei, e) in enumerate(elems)
        c = e.nodes
        if e.family === :quad
            # curved: walk a (k+1)² lattice of the [-1,1]² parametric square
            for a in 1:k, b in 1:k
                ξ0, ξ1 = -1 + 2(a-1)/k, -1 + 2a/k
                η0, η1 = -1 + 2(b-1)/k, -1 + 2b/k
                p00 = quad8_physical_point(coords, c, ξ0, η0)
                p10 = quad8_physical_point(coords, c, ξ1, η0)
                p11 = quad8_physical_point(coords, c, ξ1, η1)
                p01 = quad8_physical_point(coords, c, ξ0, η1)
                t += 1
                soup[:,1,t] = p00; soup[:,2,t] = p10; soup[:,3,t] = p11; elem_of[t] = ei
                t += 1
                soup[:,1,t] = p00; soup[:,2,t] = p11; soup[:,3,t] = p01; elem_of[t] = ei
            end
            continue
        end
        v1 = SVector{3,Float64}(coords[1,c[1]], coords[2,c[1]], coords[3,c[1]])
        v2 = SVector{3,Float64}(coords[1,c[2]], coords[2,c[2]], coords[3,c[2]])
        v3 = SVector{3,Float64}(coords[1,c[3]], coords[2,c[3]], coords[3,c[3]])
        t += 1
        soup[:,1,t] = v1; soup[:,2,t] = v2; soup[:,3,t] = v3; elem_of[t] = ei
        if _is_quad(e.family)
            v4 = SVector{3,Float64}(coords[1,c[4]], coords[2,c[4]], coords[3,c[4]])
            t += 1
            soup[:,1,t] = v1; soup[:,2,t] = v3; soup[:,3,t] = v4; elem_of[t] = ei
        end
    end
    return soup, elem_of
end

"""
    build_scene_bvh(coords, elems, group_tri_soup, obstruction_groups) -> SceneBVH

Build the ray-tracing scene: every element in `elems` (tagged `1:length(elems)`),
plus any `obstruction_groups` triangle soup not already covered by `elems`
(tagged `0`, i.e. opaque but never a valid ray-shooting target — geometry the
user listed purely as a blocker, not as part of the radiating set). Looking
up group membership from `elems` themselves (not from `mesh.group_elems`)
means this is correct even after `restrict_to_radiating` has pruned `elems`
down to a subset of the original groups.
"""
function build_scene_bvh(coords::Matrix{Float64}, elems::Vector{SurfaceElement},
                          group_tri_soup::Dict{Int,Array{Float64,3}},
                          obstruction_groups::Vector{Int})::SceneBVH
    soup, elem_of = _triangulate_tagged(coords, elems)

    radiating_groups = Set(e.group for e in elems)
    extra_groups = [g for g in obstruction_groups
                    if g ∉ radiating_groups && haskey(group_tri_soup, g)]
    if !isempty(extra_groups)
        extra_soups = [group_tri_soup[g] for g in extra_groups]
        n_extra = sum(size(s, 3) for s in extra_soups)
        merged  = Array{Float64,3}(undef, 3, 3, size(soup, 3) + n_extra)
        merged[:, :, 1:size(soup,3)] = soup
        t = size(soup, 3)
        for s in extra_soups
            nt = size(s, 3)
            merged[:, :, t+1:t+nt] = s
            t += nt
        end
        soup    = merged
        elem_of = vcat(elem_of, zeros(Int, n_extra))
    end

    return SceneBVH(build_bvh(soup), elem_of)
end

# ---------------------------------------------------------------------------
# Cosine-weighted hemisphere direction sampling
# ---------------------------------------------------------------------------

"""Draw a direction from the cosine-weighted hemisphere around unit normal
`n` (pdf(ω) = cos θ/π). Uses the branchless orthonormal-basis construction
of Duff, Burgess, Christensen, Hery, Kensler, Liani & Villemin, "Building an
Orthonormal Basis, Revisited" (JCGT 2017) — robust for `n` arbitrarily close
to ±ẑ, unlike a naive cross-product basis."""
@inline function cosine_dir(n::SVector{3,Float64}, rng::AbstractRNG)::SVector{3,Float64}
    s = n[3] >= 0.0 ? 1.0 : -1.0
    a = -1.0 / (s + n[3])
    b = n[1] * n[2] * a
    t  = SVector(1.0 + s*n[1]*n[1]*a, s*b, -s*n[1])
    bt = SVector(b, s + n[2]*n[2]*a, -n[2])

    u1 = rand(rng); u2 = rand(rng)
    r = sqrt(u1); phi = 2pi*u2
    lx, ly, lz = r*cos(phi), r*sin(phi), sqrt(max(0.0, 1.0 - u1))
    return lx*t + ly*bt + lz*n
end

@inline function _tri_normal(soup::Array{Float64,3}, tidx::Int)::SVector{3,Float64}
    v0 = SVector{3,Float64}(soup[1,1,tidx], soup[2,1,tidx], soup[3,1,tidx])
    v1 = SVector{3,Float64}(soup[1,2,tidx], soup[2,2,tidx], soup[3,2,tidx])
    v2 = SVector{3,Float64}(soup[1,3,tidx], soup[2,3,tidx], soup[3,3,tidx])
    c = cross(v1 - v0, v2 - v0)
    return c / norm(c)
end

# ---------------------------------------------------------------------------
# Per-element ray shooting
# ---------------------------------------------------------------------------

"""
    raytrace_element(coords, elem, elem_idx, scene, n_rays, N, rng)
        -> (hits::Vector{Int}, n_escaped::Int, n_absorbed::Int)

Shoot `n_rays` rays from `elem` (index `elem_idx` in the scene) and tally,
for each, which radiating element (`1:N`) it landed on. `hits[j]` is the
number of rays whose first hit was element j's front face; `n_escaped`
counts rays that hit nothing (open enclosure); `n_absorbed` counts rays
whose first hit was geometrically blocked but not a valid target — either a
radiating element's back face, or any hit on non-radiating blocker-only
geometry (`scene.elem_of == 0`, front or back: it carries no radiating
identity, so no orientation check applies) — see the module docstring.

`hits[elem_idx]` stays 0: rays are not tested against their own origin
element (`self_vf` is not supported by this kernel — see `Assembly.jl`).
"""
function raytrace_element(coords ::Matrix{Float64},
                           elem   ::SurfaceElement,
                           elem_idx::Int,
                           scene  ::SceneBVH,
                           n_rays ::Int,
                           N      ::Int,
                           rng    ::AbstractRNG)::Tuple{Vector{Int},Int,Int}
    xs, ns, _, _ = _sample_element(coords, elem, n_rays, rng)
    hits = zeros(Int, N)
    n_escaped  = 0
    n_absorbed = 0
    elem_of = scene.elem_of
    soup    = scene.bvh.tri_soup
    @inbounds for k in 1:n_rays
        x = xs[k]; n = ns[k]
        d = cosine_dir(n, rng)
        tidx, _ = nearest_hit_bvh(scene.bvh, x, d; skip = t -> elem_of[t] == elem_idx)
        if tidx == 0
            n_escaped += 1
            continue
        end
        j = elem_of[tidx]
        if j == 0 || dot(_tri_normal(soup, tidx), -d) <= 0.0
            n_absorbed += 1
        else
            hits[j] += 1
        end
    end
    return hits, n_escaped, n_absorbed
end

end # module RayTraceKernel
