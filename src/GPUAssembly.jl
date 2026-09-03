# src/GPUAssembly.jl
# ---------------------------------------------------------------------------
# GPU dispatch path for compute_view_factors.
# Called from Assembly.jl when a non-CPU backend is passed.
# ---------------------------------------------------------------------------

module GPUAssembly

using LinearAlgebra
using KernelAbstractions

import ..MeshIO:       MeshData
import ..GPUBVH:       build_flat_bvh_from_mesh
import ..GPUKernels:   build_gpu_arrays, launch_vf_kernel!
import ..GPUMCKernels: launch_mc_kernel!
import ..Results:      ViewFactorResult, _aggregate
import ..Assembly:     register_gpu_hook!
import ..DuffyKernel:  patch_adjacent_pairs_duffy!

export compute_view_factors_gpu

"""
    compute_view_factors_gpu(mesh, nquad, backend, FloatT, ArrayT;
                             obstruction_groups, verbose) -> ViewFactorResult

GPU implementation of compute_view_factors.

`backend`            — a KernelAbstractions backend, e.g. `CUDABackend()` or `MetalBackend()`.
`FloatT`             — element type: `Float64` for CUDA, `Float32` for Metal.
`ArrayT`             — device array constructor, provided by the backend extension.
`obstruction_groups` — physical group tags whose geometry occludes rays.
`factor`             — near-pair Duffy-patch radius, in element diameters
                        (see `near_pairs` in `DuffyKernel.jl`); only used
                        when `monte_carlo=true`.
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
                                   factor            ::Float64     = 3.0)::ViewFactorResult
    N = length(mesh.surface_elems)
    if verbose
        if monte_carlo
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
                                               verbose=verbose)
    else
        raw_dev, area_dev = launch_vf_kernel!(ga, backend; flat_bvh=flat_bvh)
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
        patch_adjacent_pairs_duffy!(raw_f64, mesh.coords, mesh.surface_elems,
                                     nquad, mesh.mesh_dim; factor=factor)
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


# Register this module's compute function as the GPU dispatch target in
# Assembly.  This runs when GPUAssembly is first loaded (after Assembly),
# completing the dependency loop without a circular import at module-load time.
register_gpu_hook!(compute_view_factors_gpu)

end # module GPUAssembly
