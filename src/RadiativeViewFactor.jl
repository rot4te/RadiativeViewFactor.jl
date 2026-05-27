# src/RadiativeViewFactor.jl
module RadiativeViewFactor

using LinearAlgebra
using StaticArrays
using SparseArrays
using KernelAbstractions

include("MeshIO.jl")
include("Quadrature.jl")
include("Geometry.jl")
include("BVH.jl")
include("RayCast.jl")
include("ViewFactorKernel.jl")
include("Results.jl")       # ViewFactorResult, _aggregate — no upstream deps
include("GPUBVH.jl")
include("GPUKernels.jl")
include("Assembly.jl")      # imports Results; defines register_gpu_hook!
include("GPUAssembly.jl")   # imports Results + Assembly.register_gpu_hook!;
                             # calls register_gpu_hook!(compute_view_factors_gpu)

using .MeshIO:    load_mesh, MeshData
using .Results:   ViewFactorResult, aggregate_by_group,
                  check_reciprocity, check_closure
using .Assembly:  compute_view_factors

export load_mesh,
       compute_view_factors,
       aggregate_by_group,
       check_reciprocity,
       check_closure,
       plot_mesh_normals,
       MeshData,
       ViewFactorResult

# Declare the function with no methods here; the extension adds the real method.
# A fallback on Any gives a helpful error when called without Makie loaded,
# and does not conflict with the extension's MeshData-typed method.
function plot_mesh_normals end

plot_mesh_normals(x; kwargs...) =
    error("plot_mesh_normals requires a Makie backend. " *
          "Load one first, e.g. `using GLMakie` or `using CairoMakie`.")

end # module
