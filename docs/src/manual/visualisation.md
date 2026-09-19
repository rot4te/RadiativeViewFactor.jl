# Mesh Visualisation

The `plot_mesh_normals` function renders the mesh with element outlines and
normal arrows coloured by physical group, so you can check normal orientation
before computing. It is provided as an optional extension built on
[Plots.jl](https://docs.juliaplots.org) — loading `Plots` triggers it
automatically.

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

The normal scale is estimated automatically from the mesh bounding box and the
element count. The result is a 2D `Plots.Plot` with equal aspect ratio and one
legend entry per group (name and tag).

Curve meshes (`surface_dim=1`) are the primary use: Line3 elements are drawn as
smooth quadratic curves, Line2 as straight segments, and each element has a
normal arrow at its midpoint.

Surface meshes (`surface_dim=2`) are drawn as an **xy-projection** of the
element edges and normals — a quick sanity check, not a 3-D view. For full 3-D
inspection of a surface mesh use Gmsh's built-in normal visualisation (View →
Mesh → Normals).

## Options

```julia
fig = plot_mesh_normals(mesh;
    normal_scale  = 0.05,              # arrow length in mesh units
    show_nodes    = true,              # scatter-plot element nodes
    show_indices  = true,              # label each element with its index
    group_colors  = Dict(1=>:red,      # override colours per physical group tag
                         2=>:blue))
```

`normal_scale` must be a `Float64` (write `1.0`, not `1`).

## Saving to file

```julia
fig = plot_mesh_normals(mesh)
savefig(fig, "normals.pdf")
savefig(fig, "normals.png")
```

## Interpreting the output

Each element is drawn as:

- A line (Line2, Line3) or edge outline (Tri3, Quad4, Tri6, Quad8) in the group
  colour
- An arrow at the element centre (midpoint, for curve elements) pointing along
  the element normal

Arrows pointing toward the opposing surface indicate correct orientation.
Arrows pointing away indicate a winding error — use `reverse_normals=true`
in [`load_mesh`](@ref) to flip all normals before computing view factors, or
[`reverse_group_normals`](@ref) to flip selected groups.
