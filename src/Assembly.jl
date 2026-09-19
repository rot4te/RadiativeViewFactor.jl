# src/Assembly.jl
module Assembly

using LinearAlgebra
using SparseArrays
using KernelAbstractions
using Random

import ..MeshIO:           MeshData, SurfaceElement, restrict_to_radiating
import ..Quadrature:       gauss_legendre_2d
import ..BVH:              BVHTree, build_bvh
import ..ViewFactorKernel: element_pair_view_factor, precompute_quad, ElementQuad
import ..MCKernel:         element_pair_view_factor_mc, sample_element_mc,
                           ElementSamples
import ..DuffyKernel:      element_pair_view_factor_duffy, singularity_type,
                           patch_adjacent_pairs_duffy!, near_pairs
import ..RayTraceKernel:   SceneBVH, build_scene_bvh, raytrace_element
import ..ElementBounds:    ElementBound, build_element_bounds, pair_can_see
import ..Results:          ViewFactorResult, _aggregate, aggregate_by_group,
                           check_reciprocity, check_closure

export compute_view_factors,
       aggregate_by_group,
       check_reciprocity,
       check_closure,
       ViewFactorResult,
       register_gpu_hook!,
       build_bvh_lookup

"""
    compute_view_factors(mesh; nquad=4, obstruction_groups=Int[],
                         radiating_groups=Int[], backend=CPU(),
                         self_vf=false, monte_carlo=false, n_samples=5000,
                         rng=Random.default_rng(), use_duffy=false,
                         factor=3.0, facing_cull=true,
                         raytrace=false, n_rays=10000, verbose=true)
                         -> ViewFactorResult

Assemble the full view factor matrix at element and physical-group level.

# Arguments
- `mesh`                : `MeshData` returned by `load_mesh`
- `nquad`               : Gauss points per direction; `nquad²` points on each
                          quadrilateral element (`nquad` on each curve element),
                          so the pair integral costs `nquad⁴` point-pairs per
                          quadrilateral pair (`nquad²` per curve pair) — 256
                          point-pairs at the default `nquad=4`. Triangles use
                          a Dunavant rule instead, with 1, 3, 7 or 13 points for
                          `nquad` = 1, 2, 3 or ≥ 4. Compare `n_samples` below,
                          which counts point-pairs directly. When
                          `monte_carlo=true`, only used for the adjacent-pair
                          Duffy patch (see `monte_carlo` below), not the bulk
                          sampling. With `raytrace=true`, only used to compute
                          element areas.
- `obstruction_groups`  : physical group tags that may occlude rays. Source and
                          destination groups are excluded automatically per pair.
                          Also applied to the Monte Carlo near-pair Duffy patch.
                          Obstruction geometry is triangulated from element
                          corner nodes only, so a curved 2nd-order blocker is
                          treated as flat.
- `radiating_groups`    : restrict the *radiating* surface to these physical
                          group tags; every other group still obstructs (if
                          listed in `obstruction_groups`) but is not itself
                          assembled. Empty (default) means every group radiates,
                          the previous behaviour. Because assembly is dense over
                          whatever it is handed, this turns an `O(N_total²)`
                          problem into `O(N_radiating²)`: asking for one pair of
                          surfaces in a mesh full of shadowing bodies no longer
                          computes the view factors *between* those shadowing
                          bodies. On a reactor fuel-assembly slice, `FA_wall →
                          Pin_11` with 19 shadowing pins loaded drops from 18400
                          radiating elements to 3040 — 37x fewer pairs.
                          The returned `F_group` covers only the radiating
                          groups. **The enclosure is deliberately open, so row
                          sums no longer approach 1 and `check_closure` is not
                          meaningful**; reciprocity is unaffected. See
                          [`restrict_to_radiating`](@ref).
- `backend`             : `CPU()` (default), `CUDABackend()`, or `MetalBackend()`
- `self_vf`             : include self view factors (concave elements). CPU only
                          — silently ignored on GPU backends — and an error with
                          `raytrace=true`.
- `monte_carlo`         : use stratified Monte Carlo area-sampling instead of
                          Gauss–Legendre quadrature for the O(N²) bulk of
                          element pairs — the fastest option for large meshes,
                          especially on GPU. The 1/r² kernel has *unbounded*
                          variance for pairs sharing a vertex or edge, so no
                          amount of sampling fixes those; this function
                          therefore always patches those O(N) pairs
                          afterward with the deterministic Duffy transform
                          (CPU-only, using `nquad` above), on top of
                          whichever backend the bulk ran on. `use_duffy` has
                          no separate effect here — the patch is
                          unconditional whenever `monte_carlo=true`.
- `n_samples`           : MC sample pairs **per element pair** (default 5000).
                          Ignored when `monte_carlo=false`. Variance decreases
                          as O(1/N) for the (non-adjacent) pairs Monte Carlo
                          handles. This counts point-pairs directly, so it is
                          `nquad⁴` — not `nquad²` — that it should be compared
                          against: `n_samples=5000` does ~20x the kernel
                          evaluations of the default `nquad=4`, and with
                          `obstruction_groups` set, ~20x the BVH ray casts,
                          since both paths ray-cast once per kernel-positive
                          point-pair. On CPU that made a 4320-element
                          obstructed case take 1368 s at `n_samples=5000`
                          versus 60 s at `nquad=4`, for answers agreeing to 5
                          decimals. The default of 5000 is the value used
                          throughout `benchmarks/howell/` (115 catalog points,
                          96% within 1% of the published value, median error
                          0.008%) — at that sample count the MC estimator's
                          own noise is typically 10-1000x smaller than the
                          mesh's discretization error against the analytic
                          answer, so raising it further mostly buys accuracy
                          the mesh cannot use. Lower it for a quick look at a
                          large mesh; raise it only if `n_samples` sweeps on
                          your own geometry show the noise still dominates.
- `rng`                 : RNG for the CPU Monte Carlo paths (`monte_carlo` and
                          `raytrace`). Pass a seeded RNG (e.g.
                          `MersenneTwister(42)`) for reproducible results.
                          Ignored on GPU, where a fresh random seed is drawn on
                          each call.
- `use_duffy`           : apply the Duffy singularity transformation (see
                          `DuffyKernel.jl`) for same-order quad pairs (Quad4
                          or Quad8) sharing a vertex or edge, in the plain
                          quadrature path. CPU only: on a GPU backend it is
                          ignored with a warning. Has no effect when
                          `monte_carlo=true` (that path always Duffy-patches
                          adjacent pairs regardless of this flag) or
                          `raytrace=true`. Not applicable to curve meshes.
- `factor`              : near-pair patch radius, in element diameters (see
                          `near_pairs` in `DuffyKernel.jl`) — only used when
                          `monte_carlo=true`, to decide which O(N) pairs get
                          the deterministic Duffy patch instead of the raw
                          MC estimate. Raising it does not, by itself, make
                          those pairs more accurate — `nquad` (above) is
                          what governs the patch's own resolution, and is
                          the more effective lever for closure error on
                          meshes with large or elongated elements.
- `raytrace`            : use ray-shooting Monte Carlo (`RayTraceKernel.jl`)
                          instead of either quadrature or the pair-area
                          `monte_carlo` above. For each element, shoots
                          `n_rays` cosine-weighted rays and tallies which
                          element each first hits, via one BVH over the
                          *whole* radiating mesh — O(N·n_rays·log N) instead
                          of the O(N²) pair loop the other two paths always
                          pay. Obstruction is then a side effect of the same
                          query, not a separate check: radiating elements
                          obstruct each other automatically (a real
                          behavioural difference from the other two paths,
                          which only apply obstruction for groups explicitly
                          listed in `obstruction_groups` — see that argument
                          above);
                          `obstruction_groups` here only adds *extra*,
                          non-radiating blocker geometry to the scene. No
                          adjacent-pair singularity exists for this
                          estimator (it never evaluates 1/r²), so
                          `use_duffy`/the Duffy patch/`factor` do not apply.
                          `facing_cull` does not apply either (there is no
                          O(N²) pair loop to cull). Runs on CPU or GPU
                          (`backend=CUDABackend()`/`MetalBackend()`, same as
                          `monte_carlo`) with its own GPU kernel
                          (`GPURayTraceKernels.jl`); one thread per *element*
                          on GPU, not per pair. 3-D meshes only
                          (`mesh.mesh_dim == 2`); incompatible with
                          `self_vf` (a ray is never tested against its own
                          origin element) and with `monte_carlo=true`.
                          Reciprocity is enforced by construction (each
                          unordered pair's two independent ray-based
                          estimates — from each side — are averaged), so
                          results are exactly symmetric despite being
                          stochastic, unlike a raw hit-count would be. This
                          same averaging is why row sums on a closed
                          enclosure are only approximately 1 (ordinary MC
                          noise, shrinking with `n_rays`) rather than exact:
                          each element's own rays alone would close exactly,
                          but every off-diagonal entry blends in the
                          *other* element's independent estimate too. See
                          the module docstring in `RayTraceKernel.jl` for the
                          derivation and `changelog.md` for validation
                          against analytic cases and against quadrature
                          (including a case where quadrature itself was the
                          less accurate one, under-resolving a sharp
                          obstruction shadow boundary).
- `n_rays`              : rays **per element** (default 10000) when
                          `raytrace=true`; ignored otherwise. Unlike
                          `n_samples` (which counts point-pairs per element
                          *pair*), one ray-shooting pass from an element
                          estimates its view factor to *every* other element
                          at once, so this is not directly comparable to
                          `n_samples` — tune per-case rather than assuming
                          the same order of magnitude applies.
- `facing_cull`         : reject element pairs that provably cannot see each
                          other before integrating them (default `true`). Each
                          element gets a conservative bounding box for its points
                          and another for its normals (O(N)); a pair whose kernel
                          is then provably zero at every point pair is skipped
                          in O(1) instead of
                          being discovered zero one quadrature point — or one
                          MC sample, or one BVH ray cast — at a time. Results
                          are unchanged; on closed convex bodies (tube bundles,
                          pebble beds) most pairs face away and this is a large
                          saving. Set `false` to disable, e.g. for a mesh whose
                          2nd-order elements are curved sharply enough that the
                          sampled bounds might not contain them (see
                          `ElementBounds`).
- `verbose`             : print progress and row-sum diagnostics

# Returns
A [`ViewFactorResult`](@ref). Its `F_group` is ordered by the sorted group tags in
`result.group_tags`; with `radiating_groups` set it covers only those groups.
Curve meshes (`mesh.mesh_dim == 1`) give 2D view factors; they run on the CPU
only and cannot use `raytrace`.

# Examples
```julia
result = compute_view_factors(mesh; nquad=6)
result = compute_view_factors(mesh; nquad=6, use_duffy=true)
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000)               # CPU
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               backend=CUDABackend())                                 # GPU — fastest for large meshes
result = compute_view_factors(mesh; raytrace=true, n_rays=10000)                     # CPU, fastest for large/obstructed 3D meshes

# Just one pair of surfaces, with everything else still shadowing them:
all_tags = collect(keys(mesh.group_tags))
result = compute_view_factors(mesh; nquad=4,
                               radiating_groups   = [wall_tag, pin_tag],
                               obstruction_groups = all_tags)
```
"""
function compute_view_factors(mesh               ::MeshData;
                               nquad             ::Int          = 4,
                               obstruction_groups::Vector{Int}  = Int[],
                               radiating_groups  ::Vector{Int}  = Int[],
                               backend                          = CPU(),
                               self_vf           ::Bool         = false,
                               monte_carlo       ::Bool         = false,
                               n_samples         ::Int          = 5000,
                               rng               ::AbstractRNG  = Random.default_rng(),
                               use_duffy         ::Bool         = false,
                               factor            ::Float64      = 3.0,
                               facing_cull       ::Bool         = true,
                               raytrace          ::Bool         = false,
                               n_rays            ::Int          = 10000,
                               verbose           ::Bool         = true)::ViewFactorResult

    backend isa Type && (backend = backend())

    if raytrace
        monte_carlo &&
            error("raytrace and monte_carlo are two different Monte Carlo " *
                  "estimators; pick one (raytrace=true, monte_carlo=false).")
        self_vf &&
            error("raytrace does not support self_vf: a ray is never tested " *
                  "against its own origin element. Use the default (quadrature) " *
                  "or monte_carlo=true path for self-view factors.")
    end

    # Restrict the radiating surface before anything else: every path below
    # assembles a dense matrix over `mesh.surface_elems`, so this is what makes
    # the cost O(N_radiating²) rather than O(N_total²). Obstruction geometry is
    # carried over in full, so occlusion is unaffected.
    if !isempty(radiating_groups)
        n_before  = length(mesh.surface_elems)
        n_groups  = length(mesh.group_tags)
        mesh      = restrict_to_radiating(mesh, radiating_groups)
        n_after   = length(mesh.surface_elems)
        verbose && println("  Radiating subset: $n_after of $n_before elements, ",
                           "$(length(mesh.group_tags)) of $n_groups groups ",
                           "— pair count reduced ",
                           "$(round((n_before/n_after)^2, digits=1))x. ",
                           "The enclosure is now open, so row sums will not close to 1.")
    end

    use_duffy && !monte_carlo && !(backend isa CPU) &&
        @warn "use_duffy is CPU-only; ignored for GPU backends."

    if mesh.mesh_dim == 1
        raytrace &&
            error("raytrace does not support curve meshes (mesh_dim=1) yet. " *
                  "Use monte_carlo=true or the default quadrature path.")
        (backend isa CPU) ||
            error("GPU backend does not support curve meshes (mesh_dim=1). " *
                  "Use the CPU backend for 2D per-unit-depth view factors.")
    end

    if !(backend isa CPU)
        ArrayT = _gpu_array_type(backend)
        FloatT = _gpu_float_type(backend)
        return _gpu_compute_hook(mesh, nquad, backend, FloatT, ArrayT,
                                  obstruction_groups, verbose,
                                  monte_carlo, n_samples, factor, facing_cull,
                                  raytrace, n_rays)
    end

    raytrace &&
        return _compute_cpu_raytrace(mesh, nquad, obstruction_groups, n_rays,
                                     rng, verbose)

    return _compute_cpu(mesh, nquad, obstruction_groups, self_vf, verbose,
                         mesh.mesh_dim, monte_carlo, n_samples, rng, use_duffy,
                         factor, facing_cull)
end

"""
    build_bvh_lookup(mesh, obstruction_groups) -> (group_i, group_j) -> Union{BVHTree,Nothing}

Return a memoized closure mapping a pair of *radiating*-element group tags to
the merged obstruction `BVHTree` built from every `obstruction_groups` entry
other than `group_i`/`group_j`, or `nothing` when no obstruction geometry
applies. Shared by the CPU and GPU assembly paths (including the near-pair
Duffy patch) so obstruction is checked consistently everywhere a pair of
elements is evaluated.
"""
function build_bvh_lookup(mesh::MeshData, obstruction_groups::Vector{Int})
    check_obs = !isempty(obstruction_groups)
    bvh_cache = Dict{Vector{Int}, Union{BVHTree,Nothing}}()
    # The assembly loop is threaded and a Julia Dict is not thread-safe: two
    # threads inserting distinct keys can rehash concurrently and corrupt it.
    # Distinct keys are numerous (one per pair of radiating groups), so this is
    # reached often; guard it the same way Quadrature memoises its rules.
    cache_lock = ReentrantLock()

    return function get_bvh(group_i::Int, group_j::Int)::Union{BVHTree,Nothing}
        check_obs || return nothing
        active = sort(filter(g -> g != group_i && g != group_j, obstruction_groups))
        isempty(active) && return nothing
        lock(cache_lock) do
        get!(bvh_cache, active) do
            soups = [mesh.group_tri_soup[g]
                     for g in active if haskey(mesh.group_tri_soup, g)]
            isempty(soups) && return nothing
            total  = sum(size(s, 3) for s in soups)
            # Segment soups are (3,2,N); triangle soups are (3,3,N).
            # build_bvh only requires size(soup,3) and the vertex data,
            # so both layouts work transparently.
            dim2   = size(first(soups), 2)
            merged = Array{Float64,3}(undef, 3, dim2, total)
            t = 0
            for s in soups
                nt = size(s, 3)
                merged[:, :, t+1:t+nt] .= s
                t += nt
            end
            build_bvh(merged)
        end
        end
    end
end

# ---------------------------------------------------------------------------
# CPU path
# ---------------------------------------------------------------------------

function _compute_cpu(mesh              ::MeshData,
                       nquad            ::Int,
                       obstruction_groups::Vector{Int},
                       self_vf          ::Bool,
                       verbose          ::Bool,
                       mesh_dim         ::Int         = 2,
                       monte_carlo      ::Bool        = false,
                       n_samples        ::Int         = 5000,
                       rng              ::AbstractRNG = Random.default_rng(),
                       use_duffy        ::Bool        = false,
                       factor           ::Float64     = 3.0,
                       facing_cull      ::Bool        = true)::ViewFactorResult

    elems  = mesh.surface_elems
    coords = mesh.coords
    N      = length(elems)

    check_obs = !isempty(obstruction_groups)
    get_bvh   = build_bvh_lookup(mesh, obstruction_groups)

    # Conservative sphere + normal-cone bounds, used to reject pairs whose
    # kernel is provably zero everywhere before any integration. O(N).
    bounds = facing_cull ? build_element_bounds(coords, elems) : ElementBound[]
    n_culled = Threads.Atomic{Int}(0)

    if verbose
        if monte_carlo
            println("CPU compute_view_factors: $N elements, n_samples=$n_samples (Monte Carlo)")
        elseif use_duffy
            println("CPU compute_view_factors: $N elements, nquad=$nquad (Duffy for singular pairs)")
        else
            println("CPU compute_view_factors: $N elements, nquad=$nquad")
        end
    end
    # After `radiating_groups` restriction `mesh.group_tags` only holds the
    # radiating groups, so a non-radiating blocker's name is not available here.
    check_obs && verbose &&
        println("  Obstruction groups: ",
                [get(mesh.group_tags, g, "tag $g") for g in obstruction_groups])

    raw_integral = zeros(Float64, N, N)
    A_elem       = zeros(Float64, N)

    if monte_carlo
        # Pre-generate one independent RNG per row to avoid both thread contention
        # and the threadid() > nthreads() issue in Julia 1.9+ task-based threading.
        # Using threadid() as an index is unreliable; per-row RNGs are safe regardless
        # of how many threads or tasks Julia uses internally.
        row_rngs = [Random.seed!(copy(rng), rand(rng, UInt64)) for _ in 1:N]
        # Draw one independent sample set per element once (O(N)) and reuse it
        # across all pairs, instead of re-sampling both elements for every pair.
        samples = Vector{ElementSamples}(undef, N)
        Threads.@threads for i in 1:N
            samples[i] = sample_element_mc(coords, elems[i], n_samples, row_rngs[i])
            A_elem[i]  = samples[i].A
        end
        # Every pair this near will be overwritten by the Duffy patch below
        # regardless of what the MC bulk loop computes for it, so skip it
        # there entirely — O(N) pairs, saving `n_samples` kernel evaluations
        # (and, with obstruction, that many BVH ray casts) each. Computed
        # once and reused for both the skip and the patch itself. Curve
        # meshes (mesh_dim == 1) get an empty list: the Duffy patch is a
        # no-op there (see `patch_adjacent_pairs_duffy!`), so skipping would
        # leave those entries at their initialised zero permanently.
        near = mesh_dim == 1 ? Tuple{Int,Int}[] : near_pairs(coords, elems; factor=factor)
        near_set = Set(near)
        Threads.@threads for i in 1:N
            gi      = elems[i].group
            si      = samples[i]
            j_start = self_vf ? i : i + 1
            ncull_i = 0
            for j in j_start:N
                # Provably zero kernel over the whole pair — leave both
                # entries at their initialised 0.0 and skip the sampling.
                if facing_cull && j != i && !pair_can_see(bounds[i], bounds[j])
                    ncull_i += 1
                    continue
                end
                j != i && (i, j) in near_set && continue
                gj    = elems[j].group
                bvh   = get_bvh(gi, gj)
                # Diagonal self-pair needs an independent second sample set,
                # otherwise xᵢ == xⱼ at every k gives r=0 and a spurious zero.
                sj = j == i ?
                     sample_element_mc(coords, elems[i], n_samples, row_rngs[i]) :
                     samples[j]
                integ, _ = element_pair_view_factor_mc(si, sj, bvh, mesh_dim,
                                                        row_rngs[i])
                raw_integral[i, j] = integ
                raw_integral[j, i] = integ
            end
            Threads.atomic_add!(n_culled, ncull_i)
            verbose && i % max(1, N÷10) == 0 && println("  … row $i / $N done")
        end
        # The 1/r² kernel has unbounded variance for vertex/edge-adjacent
        # pairs — no amount of sampling fixes this. Patch those O(N) pairs
        # with the deterministic Duffy transform (see DuffyKernel.jl).
        # `near` was already computed above, so it isn't rebuilt here.
        verbose && print("  Patching adjacent-pair singularities (Duffy)… ")
        patch_adjacent_pairs_duffy!(raw_integral, coords, elems, nquad, mesh_dim,
                                     get_bvh; factor=factor, pairs=near)
        verbose && println("done.")
    else
        # Pre-evaluate each element's quadrature points once (O(N)) instead of
        # re-deriving them for every pair inside the O(N²) loop below.
        quads = Vector{ElementQuad}(undef, N)
        Threads.@threads for i in 1:N
            quads[i]  = precompute_quad(coords, elems[i], nquad, mesh_dim)
            A_elem[i] = quads[i].Li
        end
        Threads.@threads for i in 1:N
            gi      = elems[i].group
            qi      = quads[i]
            j_start = self_vf ? i : i + 1
            ncull_i = 0
            for j in j_start:N
                # Provably zero kernel over the whole pair — leave both
                # entries at their initialised 0.0 and skip the integration.
                if facing_cull && j != i && !pair_can_see(bounds[i], bounds[j])
                    ncull_i += 1
                    continue
                end
                gj  = elems[j].group
                bvh = get_bvh(gi, gj)
                integ, _ = if use_duffy
                    element_pair_view_factor_duffy(coords, elems[i], elems[j],
                                                    nquad, bvh, mesh_dim)
                else
                    element_pair_view_factor(qi, quads[j], bvh, mesh_dim)
                end
                raw_integral[i, j] = integ
                raw_integral[j, i] = integ
            end
            Threads.atomic_add!(n_culled, ncull_i)
            verbose && i % max(1, N÷10) == 0 && println("  … row $i / $N done")
        end
    end

    if verbose && facing_cull
        total_pairs = self_vf ? N*(N+1)÷2 : N*(N-1)÷2
        pct = 100 * n_culled[] / max(total_pairs, 1)
        println("  Facing cull: skipped $(n_culled[]) of $total_pairs pairs ",
                "($(round(pct, digits=1))%) as provably non-facing.")
    end

    F_elem = raw_integral ./ reshape(A_elem, N, 1)

    group_tags, group_names, F_group, A_group = _aggregate(mesh, F_elem, A_elem)

    if verbose
        println("Done.")
        println("  Row-sum check (element level) — max |Σⱼ Fᵢⱼ - 1| : ",
                maximum(abs.(vec(sum(F_elem, dims=2)) .- 1.0)))
        println("  Row-sum check (group level)   — max |Σⱼ Fᵢⱼ - 1| : ",
                maximum(abs.(vec(sum(F_group, dims=2)) .- 1.0)))
    end

    return ViewFactorResult(F_elem, A_elem, F_group, A_group,
                             group_tags, group_names)
end

# ---------------------------------------------------------------------------
# CPU path — ray-shooting Monte Carlo (see RayTraceKernel.jl for the method)
# ---------------------------------------------------------------------------

function _compute_cpu_raytrace(mesh              ::MeshData,
                                nquad             ::Int,
                                obstruction_groups::Vector{Int},
                                n_rays            ::Int,
                                rng               ::AbstractRNG,
                                verbose           ::Bool)::ViewFactorResult

    elems  = mesh.surface_elems
    coords = mesh.coords
    N      = length(elems)

    verbose && println("CPU compute_view_factors: $N elements, n_rays=$n_rays ",
                        "(ray-shooting Monte Carlo)")
    if verbose && !isempty(obstruction_groups)
        radiating_groups = Set(e.group for e in elems)
        extra = [g for g in obstruction_groups if g ∉ radiating_groups]
        isempty(extra) ||
            println("  Extra (non-radiating) blocker groups: ",
                    [get(mesh.group_tags, g, "tag $g") for g in extra])
    end

    # One BVH over the whole radiating mesh (self-obstructing by
    # construction — see the module docstring in RayTraceKernel.jl) plus any
    # extra `obstruction_groups` geometry not already part of it.
    scene = build_scene_bvh(coords, elems, mesh.group_tri_soup, obstruction_groups)

    # True (quadrature) element areas, used both as the final `A_elem` and to
    # convert each row's hit-count fractions into "raw" double-integral
    # values for the reciprocity averaging below. Using quadrature here (there
    # is no ray-based area estimate) keeps areas exact, consistent with the
    # other CPU paths' `A_elem`.
    A_elem = zeros(Float64, N)
    Threads.@threads for i in 1:N
        A_elem[i] = precompute_quad(coords, elems[i], nquad, mesh.mesh_dim).Li
    end

    # F_raw[i,j] = fraction of i's own rays whose first hit was j's front
    # face: an independent, not-yet-symmetrized estimate of F_{i->j}. One
    # full row at a time, each with its own RNG stream (avoids both thread
    # contention and the threadid()>nthreads() hazard — same pattern as the
    # pair-area Monte Carlo path above).
    F_raw     = zeros(Float64, N, N)
    n_escaped = zeros(Float64, N)
    row_rngs  = [Random.seed!(copy(rng), rand(rng, UInt64)) for _ in 1:N]

    Threads.@threads for i in 1:N
        hits, nesc, _ = raytrace_element(coords, elems[i], i, scene, n_rays, N, row_rngs[i])
        F_raw[i, :] .= hits ./ n_rays
        n_escaped[i] = nesc / n_rays
        verbose && i % max(1, N÷10) == 0 && println("  … row $i / $N done")
    end

    # Reciprocity by construction: average the two independent raw
    # double-integral estimates each unordered pair has (raw = F * A_source),
    # then store F_{i->j} = raw/A_i and F_{j->i} = raw/A_j directly back into
    # F_raw's own storage. This used to go through two more N×N arrays
    # (`raw_integral`, then the broadcasted division into a new `F_elem`) on
    # top of F_raw itself — a 3x peak-memory multiplier that matters once N
    # is in the tens of thousands (a 48,993-element mesh needs ~19.2 GB per
    # N×N Float64 matrix, so the old code's ~58 GB transient peak could
    # approach or exceed a workstation's RAM well before compute cost did;
    # see changelog.md). Each F_raw[i,j]/F_raw[j,i] pair is read exactly
    # once before being overwritten, so this in-place pass is safe.
    @inbounds for i in 1:N, j in i+1:N
        r = 0.5 * (F_raw[i,j] * A_elem[i] + F_raw[j,i] * A_elem[j])
        F_raw[i,j] = r / A_elem[i]
        F_raw[j,i] = r / A_elem[j]
    end
    @inbounds for i in 1:N
        F_raw[i,i] = 0.0   # raytrace never estimates a self view factor
    end
    F_elem = F_raw

    group_tags, group_names, F_group, A_group = _aggregate(mesh, F_elem, A_elem)

    if verbose
        println("Done.")
        println("  Mean escaped-ray fraction (open enclosure; excludes rays absorbed by a ",
                 "back face or blocker) : ", sum(n_escaped) / N)
        println("  Row-sum check (element level) — max |Σⱼ Fᵢⱼ - 1| : ",
                maximum(abs.(vec(sum(F_elem, dims=2)) .- 1.0)))
        println("  Row-sum check (group level)   — max |Σⱼ Fᵢⱼ - 1| : ",
                maximum(abs.(vec(sum(F_group, dims=2)) .- 1.0)))
    end

    return ViewFactorResult(F_elem, A_elem, F_group, A_group,
                             group_tags, group_names)
end

# ---------------------------------------------------------------------------
# GPU backend registry
# Default methods — overridden by ext/ modules when backends are loaded.
# _gpu_compute_hook is set by GPUAssembly.register_gpu_hook!() which is
# called from the main module after all submodules are included.
# ---------------------------------------------------------------------------

_gpu_array_type(backend) =
    error("No GPU array type registered for $(typeof(backend)). " *
          "Load CUDA.jl (for CUDABackend) or Metal.jl (for MetalBackend).")
_gpu_float_type(backend) =
    error("No GPU float type registered for $(typeof(backend)).")

# Mutable ref so GPUAssembly can register itself without a circular import
const _GPU_HOOK_REF = Ref{Any}(nothing)

function _gpu_compute_hook(mesh, nquad, backend, FloatT, ArrayT,
                            obstruction_groups, verbose, monte_carlo, n_samples,
                            factor, facing_cull, raytrace, n_rays)
    _GPU_HOOK_REF[] === nothing &&
        error("GPU compute hook not registered. Ensure GPUAssembly is loaded.")
    return _GPU_HOOK_REF[](mesh, nquad, backend, FloatT, ArrayT;
                             obstruction_groups=obstruction_groups,
                             verbose=verbose,
                             monte_carlo=monte_carlo,
                             n_samples=n_samples,
                             factor=factor,
                             facing_cull=facing_cull,
                             raytrace=raytrace,
                             n_rays=n_rays)
end

"""
    register_gpu_hook!(f)

Called by `GPUAssembly` at load time to register `compute_view_factors_gpu`
as the GPU dispatch target.  This avoids a circular module import.
"""
register_gpu_hook!(f) = (_GPU_HOOK_REF[] = f)

end # module Assembly