# test/runtests.jl
using Test

include(joinpath(@__DIR__, "..", "src", "RadiativeViewFactor.jl"))
using .RadiativeViewFactor
using .RadiativeViewFactor.Quadrature
using .RadiativeViewFactor.Geometry
using .RadiativeViewFactor.BVH
using .RadiativeViewFactor.RayCast
using .RadiativeViewFactor.ViewFactorKernel
using .RadiativeViewFactor.MeshIO: SurfaceElement
using StaticArrays
using LinearAlgebra
using Random


# ---------------------------------------------------------------------------

include("quad_test.jl")

# ---------------------------------------------------------------------------

include("ray_test.jl")

# ---------------------------------------------------------------------------

include("vf_test.jl")

# ---------------------------------------------------------------------------

include("mesh_test.jl")

# ---------------------------------------------------------------------------

include("GPU_test.jl")

# ---------------------------------------------------------------------------

include("vtk_test.jl")

# ---------------------------------------------------------------------------

include("re2_test.jl")

# ---------------------------------------------------------------------------

include("duffy_correctness_test.jl")

# ---------------------------------------------------------------------------

include("nek_export_test.jl")

# ---------------------------------------------------------------------------

include("type_stable_test.jl")