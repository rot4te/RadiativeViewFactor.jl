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
include("DuffyKernel.jl")   # Sauter-Schwab Duffy transformation for singular pairs
include("MCKernel.jl")      # CPU Monte Carlo integrator
include("Results.jl")       # ViewFactorResult, _aggregate — no upstream deps
include("GPUBVH.jl")
include("GPUKernels.jl")
include("GPUMCKernels.jl")  # GPU Monte Carlo kernel
include("Assembly.jl")      # imports Results; defines register_gpu_hook!
include("GPUAssembly.jl")   # imports Results + Assembly.register_gpu_hook!;
                             # calls register_gpu_hook!(compute_view_factors_gpu)

using .MeshIO:    load_mesh, MeshData
using .MeshIO:    SurfaceElement
using .Geometry:  quad8_physical_point, quad8_normal_and_area_element,
                  quad4_shape, quad4_physical_point, quad4_normal_and_area_element,
                  line2_shape, line2_physical_point, line2_normal_and_length_element,
                  line3_physical_point, line3_normal_and_length_element
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
       SurfaceElement,
       ViewFactorResult,
       quad8_physical_point,
       quad8_normal_and_area_element,
       quad4_shape,
       quad4_physical_point,
       quad4_normal_and_area_element,
       line2_shape,
       line2_physical_point,
       line2_normal_and_length_element,
       line3_physical_point,
       line3_normal_and_length_element

"""
    plot_mesh_normals(mesh; normal_scale=nothing, group_colors=nothing,
                      show_nodes=false, show_indices=false, backend_3d=auto)
        -> Plots.Plot

Visualise mesh elements with normal arrows coloured by physical group.

Requires Plots.jl to be loaded first:
```julia
using Plots
```

# Arguments
- `mesh`          : [`MeshData`](@ref) from [`load_mesh`](@ref)
- `normal_scale`  : arrow length in mesh units. Auto-estimated from the
                    bounding box diagonal if omitted.
- `group_colors`  : `Dict{Int,Any}` mapping physical group tag → any colour
                    accepted by Plots.jl (e.g. `:red`, `"#FF0000"`).
                    overriding the automatic palette for specified groups.
- `show_nodes`    : scatter-plot all element nodes.
- `show_indices`  : annotate each element with its index number.

# Returns
The `Plots.Plot` object; save it with `savefig(fig, "file.png")`.

# Example
```julia
using Plots, RadiativeViewFactor
mesh = load_mesh("geometry.msh"; surface_dim=1)
fig  = plot_mesh_normals(mesh; normal_scale=0.05, show_nodes=true)
```
"""
function plot_mesh_normals end

plot_mesh_normals(x; kwargs...) =
    error("plot_mesh_normals requires Plots.jl. Load it first with `using Plots`.")

end # module
