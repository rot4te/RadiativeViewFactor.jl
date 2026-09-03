# src/Assembly.jl
module Assembly

using LinearAlgebra
using SparseArrays
using KernelAbstractions
using Random

import ..MeshIO:           MeshData, SurfaceElement
import ..Quadrature:       gauss_legendre_2d
import ..BVH:              BVHTree, build_bvh
import ..ViewFactorKernel: element_pair_view_factor, precompute_quad, ElementQuad
import ..MCKernel:         element_pair_view_factor_mc, sample_element_mc,
                           ElementSamples
import ..DuffyKernel:      element_pair_view_factor_duffy, singularity_type,
                           patch_adjacent_pairs_duffy!
import ..Results:          ViewFactorResult, _aggregate, aggregate_by_group,
                           check_reciprocity, check_closure

export compute_view_factors,
       aggregate_by_group,
       check_reciprocity,
       check_closure,
       ViewFactorResult,
       register_gpu_hook!

"""
    compute_view_factors(mesh; nquad=4, obstruction_groups=Int[],
                         backend=CPU(), self_vf=false, verbose=true)
                         -> ViewFactorResult

Assemble the full view factor matrix at element and physical-group level.

# Arguments
- `mesh`                : `MeshData` returned by `load_mesh`
- `nquad`               : Gauss points per direction; `nquad²` points per surface
                          element pair, `nquad` per curve element pair. When
                          `monte_carlo=true`, only used for the adjacent-pair
                          Duffy patch (see `monte_carlo` below), not the bulk
                          sampling.
- `obstruction_groups`  : physical group tags that may occlude rays. Source and
                          destination groups are excluded automatically per pair.
                          Not applied to the Monte Carlo adjacent-pair patch —
                          two elements sharing a vertex/edge cannot have a
                          third surface positioned between them.
- `backend`             : `CPU()` (default), `CUDABackend()`, or `MetalBackend()`
- `self_vf`             : include self view factors (concave elements). CPU only.
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
- `n_samples`           : MC sample pairs **per element pair**. Ignored when
                          `monte_carlo=false`. Variance decreases as O(1/N)
                          for the (non-adjacent) pairs Monte Carlo handles.
- `rng`                 : RNG for the CPU MC path. Pass a seeded RNG (e.g.
                          `MersenneTwister(42)`) for reproducible results.
                          Ignored on GPU.
- `use_duffy`           : apply the Duffy singularity transformation (see
                          `DuffyKernel.jl`) for same-order quad pairs (Quad4
                          or Quad8) sharing a vertex or edge, in the plain
                          quadrature path. CPU only. Ignored when
                          `monte_carlo=true` (that path always Duffy-patches
                          adjacent pairs regardless of this flag).
- `factor`              : near-pair patch radius, in element diameters (see
                          `near_pairs` in `DuffyKernel.jl`) — only used when
                          `monte_carlo=true`, to decide which O(N) pairs get
                          the deterministic Duffy patch instead of the raw
                          MC estimate. Raising it does not, by itself, make
                          those pairs more accurate — `nquad` (above) is
                          what governs the patch's own resolution, and is
                          the more effective lever for closure error on
                          meshes with large or elongated elements.
- `verbose`             : print progress and row-sum diagnostics

# Returns
A `ViewFactorResult`.

# Examples
```julia
result = compute_view_factors(mesh; nquad=6)
result = compute_view_factors(mesh; nquad=6, use_duffy=true)
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000)               # CPU
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               backend=CUDABackend())                                 # GPU — fastest for large meshes
```
"""
function compute_view_factors(mesh               ::MeshData;
                               nquad             ::Int          = 4,
                               obstruction_groups::Vector{Int}  = Int[],
                               backend                          = CPU(),
                               self_vf           ::Bool         = false,
                               monte_carlo       ::Bool         = false,
                               n_samples         ::Int          = 10000,
                               rng               ::AbstractRNG  = Random.default_rng(),
                               use_duffy         ::Bool         = false,
                               factor            ::Float64      = 3.0,
                               verbose           ::Bool         = true)::ViewFactorResult

    backend isa Type && (backend = backend())

    use_duffy && !monte_carlo && !(backend isa CPU) &&
        @warn "use_duffy is CPU-only; ignored for GPU backends."

    if !(backend isa CPU)
        if mesh.mesh_dim == 1
            error("GPU backend does not support curve meshes (mesh_dim=1). " *
                  "Use the CPU backend for 2D per-unit-depth view factors.")
        end
        ArrayT = _gpu_array_type(backend)
        FloatT = _gpu_float_type(backend)
        return _gpu_compute_hook(mesh, nquad, backend, FloatT, ArrayT,
                                  obstruction_groups, verbose,
                                  monte_carlo, n_samples, factor)
    end

    return _compute_cpu(mesh, nquad, obstruction_groups, self_vf, verbose,
                         mesh.mesh_dim, monte_carlo, n_samples, rng, use_duffy,
                         factor)
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
                       n_samples        ::Int         = 10000,
                       rng              ::AbstractRNG = Random.default_rng(),
                       use_duffy        ::Bool        = false,
                       factor           ::Float64     = 3.0)::ViewFactorResult

    elems  = mesh.surface_elems
    coords = mesh.coords
    N      = length(elems)

    check_obs = !isempty(obstruction_groups)

    bvh_cache = Dict{Vector{Int}, Union{BVHTree,Nothing}}()

    function get_bvh(group_i::Int, group_j::Int)::Union{BVHTree,Nothing}
        check_obs || return nothing
        active = sort(filter(g -> g != group_i && g != group_j, obstruction_groups))
        isempty(active) && return nothing
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

    if verbose
        if monte_carlo
            println("CPU compute_view_factors: $N elements, n_samples=$n_samples (Monte Carlo)")
        elseif use_duffy
            println("CPU compute_view_factors: $N elements, nquad=$nquad (Duffy for singular pairs)")
        else
            println("CPU compute_view_factors: $N elements, nquad=$nquad")
        end
    end
    check_obs && verbose &&
        println("  Obstruction groups: ",
                [mesh.group_tags[g] for g in obstruction_groups])

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
        Threads.@threads for i in 1:N
            gi      = elems[i].group
            si      = samples[i]
            j_start = self_vf ? i : i + 1
            for j in j_start:N
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
            verbose && i % max(1, N÷10) == 0 && println("  … row $i / $N done")
        end
        # The 1/r² kernel has unbounded variance for vertex/edge-adjacent
        # pairs — no amount of sampling fixes this. Patch those O(N) pairs
        # with the deterministic Duffy transform (see DuffyKernel.jl).
        verbose && print("  Patching adjacent-pair singularities (Duffy)… ")
        patch_adjacent_pairs_duffy!(raw_integral, coords, elems, nquad, mesh_dim;
                                     factor=factor)
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
            for j in j_start:N
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
            verbose && i % max(1, N÷10) == 0 && println("  … row $i / $N done")
        end
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
                            factor)
    _GPU_HOOK_REF[] === nothing &&
        error("GPU compute hook not registered. Ensure GPUAssembly is loaded.")
    return _GPU_HOOK_REF[](mesh, nquad, backend, FloatT, ArrayT;
                             obstruction_groups=obstruction_groups,
                             verbose=verbose,
                             monte_carlo=monte_carlo,
                             n_samples=n_samples,
                             factor=factor)
end

"""
    register_gpu_hook!(f)

Called by `GPUAssembly` at load time to register `compute_view_factors_gpu`
as the GPU dispatch target.  This avoids a circular module import.
"""
register_gpu_hook!(f) = (_GPU_HOOK_REF[] = f)

end # module Assembly