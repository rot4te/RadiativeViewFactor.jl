# Mesh Visualisation

The `plot_mesh_normals` function renders the mesh with element outlines and
normal arrows coloured by physical group. It is provided as an optional
extension — loading `Plots` triggers it automatically.

## Setup

```julia
using Plots
using RadiativeViewFactor
```

If Plots is not loaded, calling `plot_mesh_normals` raises a clear error
message rather than a cryptic `MethodError`.

## Basic usage

```julia
mesh = load_mesh("geometry.msh"; surface_dim=1)
fig  = plot_mesh_normals(mesh)
```

The normal scale is estimated automatically from the mesh bounding box. For
2D curve meshes the figure uses a 2D `Axis`; for 3D surface meshes it uses
`Axis3`.

## Options

```julia
fig = plot_mesh_normals(mesh;
    normal_scale  = 0.05,              # arrow length in mesh units
    show_nodes    = true,              # scatter-plot element nodes
    show_indices  = true,              # label each element with its index
    group_colors  = Dict(1=>:red,      # override colours per physical group tag
                         2=>:blue),
    backend_3d    = true)              # force Axis3 even for curve meshes
```

## Saving to file

```julia
fig = plot_mesh_normals(mesh)
savefig(fig, "normals.pdf")
savefig(fig, "normals.png")
```

## Interpreting the output

Each element is drawn as:
- A line (Line2, Line3) or edge outline (Tri3, Quad4, Tri6, Quad8) in the group colour
- An arrow at the element centre pointing in the outward normal direction

Arrows pointing toward the opposing surface indicate correct orientation.
Arrows pointing away indicate a winding error — use `reverse_normals=true`
in [`load_mesh`](@ref) to flip all normals before computing view factors.
