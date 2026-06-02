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

export compute_view_factors_gpu

"""
    compute_view_factors_gpu(mesh, nquad, backend, FloatT, ArrayT;
                             obstruction_groups, verbose) -> ViewFactorResult

GPU implementation of compute_view_factors.

`backend`            — a KernelAbstractions backend, e.g. `CUDABackend()` or `MetalBackend()`.
`FloatT`             — element type: `Float64` for CUDA, `Float32` for Metal.
`ArrayT`             — device array constructor, provided by the backend extension.
`obstruction_groups` — physical group tags whose geometry occludes rays.
"""
function compute_view_factors_gpu(mesh               ::MeshData,
                                   nquad             ::Int,
                                   backend,
                                   FloatT            ::Type,
                                   ArrayT            ;
                                   obstruction_groups::Vector{Int} = Int[],
                                   verbose           ::Bool        = true,
                                   monte_carlo       ::Bool        = false,
                                   n_samples         ::Int         = 10000)::ViewFactorResult
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
    verbose && print("  Running GPU kernel… ")
    if monte_carlo
        seed = rand(UInt64)
        raw_dev, area_dev = launch_mc_kernel!(ga, backend;
                                               n_samples=n_samples,
                                               seed=seed,
                                               flat_bvh=flat_bvh)
    else
        raw_dev, area_dev = launch_vf_kernel!(ga, backend; flat_bvh=flat_bvh)
    end
    verbose && println("done.")

    # Copy results back to CPU
    raw_cpu  = Array(raw_dev)
    area_cpu = Array(area_dev)

    # Promote to Float64 for all post-processing (aggregation, reciprocity checks)
    raw_f64  = Float64.(raw_cpu)
    area_f64 = Float64.(area_cpu)

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
