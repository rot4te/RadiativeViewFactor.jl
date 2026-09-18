# src/GPUAssembly.jl
# ---------------------------------------------------------------------------
# GPU dispatch path for compute_view_factors.
# Called from Assembly.jl when a non-CPU backend is passed.
# ---------------------------------------------------------------------------

module GPUAssembly

using LinearAlgebra
using KernelAbstractions

import ..MeshIO:       MeshData
import ..GPUBVH:       build_flat_bvh_from_mesh, build_flat_scene_bvh
import ..GPUKernels:   build_gpu_arrays, launch_vf_kernel!
import ..GPUMCKernels: launch_mc_kernel!
import ..GPURayTraceKernels: launch_raytrace_kernel!, launch_area_kernel!
import ..Results:      ViewFactorResult, _aggregate
import ..Assembly:     register_gpu_hook!, build_bvh_lookup
import ..DuffyKernel:  patch_adjacent_pairs_duffy!

export compute_view_factors_gpu

"""
    compute_view_factors_gpu(mesh, nquad, backend, FloatT, ArrayT;
                             obstruction_groups, verbose) -> ViewFactorResult

GPU implementation of compute_view_factors.

`backend`            — a KernelAbstractions backend, e.g. `CUDABackend()` or `MetalBackend()`.
`FloatT`             — element type: `Float64` for CUDA, `Float32` for Metal.
`ArrayT`             — device array constructor, provided by the backend extension.
`obstruction_groups` — physical group tags whose geometry occludes rays. For
                        `raytrace=true`, radiating elements already obstruct
                        each other automatically (see `RayTraceKernel.jl`);
                        this only adds *extra* non-radiating blocker geometry
                        to the scene there.
`factor`             — near-pair Duffy-patch radius, in element diameters
                        (see `near_pairs` in `DuffyKernel.jl`); only used
                        when `monte_carlo=true`.
`raytrace`/`n_rays`  — ray-shooting Monte Carlo instead of either the
                        deterministic or pair-area-sampling kernels above;
                        see `compute_view_factors`'s own docstring in
                        `Assembly.jl` for the full description (identical
                        semantics and caveats on GPU as on CPU).
"""
function compute_view_factors_gpu(mesh               ::MeshData,
                                   nquad             ::Int,
                                   backend,
                                   FloatT            ::Type,
                                   ArrayT            ;
                                   obstruction_groups::Vector{Int} = Int[],
                                   verbose           ::Bool        = true,
                                   monte_carlo       ::Bool        = false,
                                   n_samples         ::Int         = 10000,
                                   factor            ::Float64     = 3.0,
                                   facing_cull       ::Bool        = true,
                                   raytrace          ::Bool        = false,
                                   n_rays            ::Int         = 10000)::ViewFactorResult
    N = length(mesh.surface_elems)
    if verbose
        if raytrace
            println("GPU compute_view_factors: $N elements, n_rays=$n_rays (ray-shooting Monte Carlo), ",
                    "FloatT=$FloatT, backend=$(typeof(backend))")
        elseif monte_carlo
            println("GPU compute_view_factors: $N elements, n_samples=$n_samples (Monte Carlo), ",
                    "FloatT=$FloatT, backend=$(typeof(backend))")
        else
            println("GPU compute_view_factors: $N elements, nquad=$nquad, ",
                    "FloatT=$FloatT, backend=$(typeof(backend))")
        end
    end

    # Flatten mesh data and transfer to device
    verbose && print("  Transferring mesh to device… ")
    ga = build_gpu_arrays(mesh, nquad, ArrayT, FloatT)
    verbose && println("done.")

    if raytrace
        return _compute_gpu_raytrace(mesh, ga, backend, FloatT, ArrayT,
                                     obstruction_groups, n_rays, verbose)
    end

    # Build flat BVH on CPU and upload to device (if obstruction groups given)
    flat_bvh = nothing
    if !isempty(obstruction_groups)
        verbose && print("  Building obstruction BVH… ")
        flat_bvh = build_flat_bvh_from_mesh(mesh, obstruction_groups, FloatT, ArrayT)
        if flat_bvh === nothing
            @warn "obstruction_groups specified but no triangle geometry found for those groups; " *
                  "proceeding without obstruction checking."
        end
        verbose && println("done.")
    end

    # Launch kernels
    verbose && println("  Running GPU kernel…")
    if monte_carlo
        seed = rand(UInt64)
        raw_dev, area_dev = launch_mc_kernel!(ga, backend;
                                               n_samples=n_samples,
                                               seed=seed,
                                               flat_bvh=flat_bvh,
                                               facing_cull=facing_cull,
                                               verbose=verbose)
    else
        raw_dev, area_dev = launch_vf_kernel!(ga, backend; flat_bvh=flat_bvh,
                                               facing_cull=facing_cull)
    end
    verbose && println("  …kernel done.")

    # Copy results back to CPU
    raw_cpu  = Array(raw_dev)
    area_cpu = Array(area_dev)

    # Every element must have a positive area estimate.  A zero means the
    # kernel never ran for that element: on Metal, macOS's GPU watchdog kills
    # long-running command buffers ("Failed to submit command buffer:
    # Impacting Interactivity"), and that error is only logged asynchronously
    # — execution continues with partially-written buffers, which would turn
    # into a silently NaN-filled view factor matrix below (0/0 in the row
    # normalisation).  Fail loudly instead.
    nzero = count(iszero, area_cpu)
    if nzero > 0
        error("GPU kernel returned a zero area for $nzero of $N elements — " *
              "the kernel did not run to completion (on Metal, check the log " *
              "for an asynchronous \"Impacting Interactivity\" command-buffer " *
              "error from the macOS GPU watchdog), or the mesh contains " *
              "degenerate elements.")
    end

    # Promote to Float64 for all post-processing (aggregation, reciprocity checks)
    raw_f64  = Float64.(raw_cpu)
    area_f64 = Float64.(area_cpu)

    # The 1/r² kernel has unbounded variance for vertex/edge-adjacent
    # pairs — sampling more doesn't fix this, on GPU any more than on CPU.
    # Patch those O(N) pairs on the CPU with the deterministic Duffy
    # transform, on top of whatever ran on the GPU for the O(N²) bulk.
    if monte_carlo
        verbose && print("  Patching adjacent-pair singularities (Duffy, CPU)… ")
        get_bvh = build_bvh_lookup(mesh, obstruction_groups)
        patch_adjacent_pairs_duffy!(raw_f64, mesh.coords, mesh.surface_elems,
                                     nquad, mesh.mesh_dim, get_bvh; factor=factor)
        verbose && println("done.")
    end

    # Divide each row i by A[i] to get F_elem
    F_elem = raw_f64 ./ reshape(area_f64, N, 1)

    group_tags, group_names, F_group, A_group =
        _aggregate(mesh, F_elem, area_f64)

    if verbose
        println("  Row-sum check (element level) — max |Σⱼ Fᵢⱼ - 1| : ",
                maximum(abs.(vec(sum(F_elem, dims=2)) .- 1.0)))
        println("  Row-sum check (group level)   — max |Σⱼ Fᵢⱼ - 1| : ",
                maximum(abs.(vec(sum(F_group, dims=2)) .- 1.0)))
    end

    return ViewFactorResult(F_elem, area_f64, F_group, A_group,
                             group_tags, group_names)
end

"""
    _compute_gpu_raytrace(mesh, ga, backend, FloatT, ArrayT,
                          obstruction_groups, n_rays, verbose) -> ViewFactorResult

GPU ray-shooting path, split out of `compute_view_factors_gpu` because its
structure (one BVH per whole scene, one thread per element, no Duffy patch,
CPU-side reciprocity averaging) doesn't share much with the pair-kernel path
above. Mirrors `Assembly._compute_cpu_raytrace` exactly past the point the
GPU kernel returns — see that function's comments for why the
reciprocity-averaging step needs the *true* element areas, and why row sums
on a closed enclosure end up only approximately (not exactly) 1.
"""
function _compute_gpu_raytrace(mesh, ga, backend, FloatT, ArrayT,
                                obstruction_groups::Vector{Int},
                                n_rays::Int, verbose::Bool)::ViewFactorResult
    N = ga.N

    verbose && print("  Building ray-tracing scene BVH… ")
    flat_scene = build_flat_scene_bvh(mesh, obstruction_groups, FloatT, ArrayT)
    verbose && println("done.")

    verbose && print("  Computing element areas (quadrature)… ")
    area_dev = launch_area_kernel!(ga, backend)
    verbose && println("done.")

    verbose && println("  Running GPU ray-tracing kernel…")
    hits_dev, escaped_dev = launch_raytrace_kernel!(ga, flat_scene, backend;
                                                     n_rays=n_rays, verbose=verbose)
    verbose && println("  …kernel done.")

    hits_cpu    = Array(hits_dev)      # N×N hit counts (F_raw[i,j]*n_rays)
    area_cpu    = Float64.(Array(area_dev))
    escaped_cpu = Float64.(Array(escaped_dev))

    nzero = count(iszero, area_cpu)
    nzero > 0 &&
        error("GPU ray-trace kernel returned a zero area for $nzero of $N elements — " *
              "the kernel did not run to completion (on Metal, check the log for an " *
              "asynchronous \"Impacting Interactivity\" command-buffer error from the " *
              "macOS GPU watchdog), or the mesh contains degenerate elements.")

    F_raw = Float64.(hits_cpu) ./ n_rays

    # Reciprocity by construction: average the two independent raw
    # double-integral estimates each unordered pair has (raw = F * A_source),
    # writing F_{i->j}/F_{j->i} back into F_raw's own storage in place
    # (rather than through two more N×N arrays) — identical to, and for the
    # same memory-scaling reason as, Assembly._compute_cpu_raytrace.
    @inbounds for i in 1:N, j in i+1:N
        r = 0.5 * (F_raw[i,j] * area_cpu[i] + F_raw[j,i] * area_cpu[j])
        F_raw[i,j] = r / area_cpu[i]
        F_raw[j,i] = r / area_cpu[j]
    end
    @inbounds for i in 1:N
        F_raw[i,i] = 0.0   # raytrace never estimates a self view factor
    end
    F_elem = F_raw

    group_tags, group_names, F_group, A_group = _aggregate(mesh, F_elem, area_cpu)

    if verbose
        println("Done.")
        println("  Mean escaped-ray fraction (open enclosure; excludes rays absorbed by a ",
                 "back face or blocker) : ", sum(escaped_cpu) / N)
        println("  Row-sum check (element level) — max |Σⱼ Fᵢⱼ - 1| : ",
                maximum(abs.(vec(sum(F_elem, dims=2)) .- 1.0)))
        println("  Row-sum check (group level)   — max |Σⱼ Fᵢⱼ - 1| : ",
                maximum(abs.(vec(sum(F_group, dims=2)) .- 1.0)))
    end

    return ViewFactorResult(F_elem, area_cpu, F_group, A_group,
                             group_tags, group_names)
end

# Register this module's compute function as the GPU dispatch target in
# Assembly.  This runs when GPUAssembly is first loaded (after Assembly),
# completing the dependency loop without a circular import at module-load time.
register_gpu_hook!(compute_view_factors_gpu)

end # module GPUAssembly
