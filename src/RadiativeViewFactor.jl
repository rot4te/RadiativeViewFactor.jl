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
       MeshData,
       ViewFactorResult

end # module
