# src/Assembly.jl
module Assembly

using LinearAlgebra
using SparseArrays
using KernelAbstractions

import ..MeshIO:           MeshData, SurfaceElement
import ..Quadrature:       gauss_legendre_2d
import ..BVH:              BVHTree, build_bvh
import ..ViewFactorKernel: element_pair_view_factor
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

Compute all element-pair view factors.

Arguments
---------
- `mesh`                : `MeshData` from `load_mesh`
- `nquad`               : Gauss points per direction (nquad² per element pair
                          for surface elements; nquad points for curve elements)
- `obstruction_groups`  : physical group tags that may occlude rays. The source
                          and destination groups are automatically excluded per
                          pair on both CPU and GPU backends.
- `backend`             : a KernelAbstractions backend.
                          `CPU()` (default)  — multi-threaded via Threads.@threads
                          `CUDABackend()`    — NVIDIA GPU (requires CUDA.jl)
                          `MetalBackend()`   — Apple GPU (requires Metal.jl)
- `self_vf`             : include self view factors (curved elements). CPU only.
- `verbose`             : print progress and row-sum diagnostics
"""
function compute_view_factors(mesh               ::MeshData;
                               nquad             ::Int          = 4,
                               obstruction_groups::Vector{Int}  = Int[],
                               backend                          = CPU(),
                               self_vf           ::Bool         = false,
                               verbose           ::Bool         = true)::ViewFactorResult

    backend isa Type && (backend = backend())

    if !(backend isa CPU)
        if mesh.mesh_dim == 1
            error("GPU backend does not support curve meshes (mesh_dim=1). " *
                  "Use the CPU backend for 2D per-unit-depth view factors.")
        end
        ArrayT = _gpu_array_type(backend)
        FloatT = _gpu_float_type(backend)
        return _gpu_compute_hook(mesh, nquad, backend, FloatT, ArrayT,
                                  obstruction_groups, verbose)
    end

    return _compute_cpu(mesh, nquad, obstruction_groups, self_vf, verbose,
                         mesh.mesh_dim)
end

# ---------------------------------------------------------------------------
# CPU path
# ---------------------------------------------------------------------------

function _compute_cpu(mesh              ::MeshData,
                       nquad            ::Int,
                       obstruction_groups::Vector{Int},
                       self_vf          ::Bool,
                       verbose          ::Bool,
                       mesh_dim         ::Int = 2)::ViewFactorResult

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

    verbose && println("CPU compute_view_factors: $N elements, nquad=$nquad")
    check_obs && verbose &&
        println("  Obstruction groups: ",
                [mesh.group_tags[g] for g in obstruction_groups])

    raw_integral = zeros(Float64, N, N)
    A_elem       = zeros(Float64, N)

    for i in 1:N
        _, Ai     = element_pair_view_factor(coords, elems[i], elems[i],
                                              nquad, nothing, mesh_dim)
        A_elem[i] = Ai
    end

    Threads.@threads for i in 1:N
        gi      = elems[i].group
        j_start = self_vf ? i : i + 1
        for j in j_start:N
            gj  = elems[j].group
            bvh = get_bvh(gi, gj)
            integ, _ = element_pair_view_factor(coords, elems[i], elems[j],
                                                 nquad, bvh, mesh_dim)
            raw_integral[i, j] = integ
            raw_integral[j, i] = integ
        end
        verbose && i % max(1, N÷10) == 0 && println("  … row $i / $N done")
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
                            obstruction_groups, verbose)
    _GPU_HOOK_REF[] === nothing &&
        error("GPU compute hook not registered. Ensure GPUAssembly is loaded.")
    return _GPU_HOOK_REF[](mesh, nquad, backend, FloatT, ArrayT;
                             obstruction_groups=obstruction_groups,
                             verbose=verbose)
end

"""
    register_gpu_hook!(f)

Called by `GPUAssembly` at load time to register `compute_view_factors_gpu`
as the GPU dispatch target.  This avoids a circular module import.
"""
register_gpu_hook!(f) = (_GPU_HOOK_REF[] = f)

end # module Assembly