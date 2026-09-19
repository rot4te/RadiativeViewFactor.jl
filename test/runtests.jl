# test/runtests.jl
using Test

using RadiativeViewFactor
using RadiativeViewFactor.Quadrature
using RadiativeViewFactor.Geometry
using RadiativeViewFactor.BVH
using RadiativeViewFactor.RayCast
using RadiativeViewFactor.ViewFactorKernel
using RadiativeViewFactor.MeshIO: SurfaceElement
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

include("plots_test.jl")

# ---------------------------------------------------------------------------

include("re2_test.jl")

# ---------------------------------------------------------------------------

include("duffy_correctness_test.jl")

# ---------------------------------------------------------------------------

include("nek_export_test.jl")

# ---------------------------------------------------------------------------

include("obstruction_test.jl")

# ---------------------------------------------------------------------------

include("type_stable_test.jl")

# ---------------------------------------------------------------------------

include("raytrace_test.jl")

# ---------------------------------------------------------------------------
# Coverage-gap tests: shared fixtures first, then the files that use them
# (assembly_paths_test.jl needs re2_test.jl's `_write_re2`, hence the order).

include("fixtures.jl")

# ---------------------------------------------------------------------------

include("gpu_families_test.jl")

# ---------------------------------------------------------------------------

include("curve_mesh_test.jl")

# ---------------------------------------------------------------------------

include("vtk_extra_test.jl")

# ---------------------------------------------------------------------------

include("assembly_paths_test.jl")

# ---------------------------------------------------------------------------

include("gpu_duffy_test.jl")
