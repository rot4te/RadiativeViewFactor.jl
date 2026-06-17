# Getting Started

## Loading a mesh

`load_mesh` reads any format the Gmsh SDK can open — `.msh` (v2.2 and v4),
`.stl`, `.step`/`.stp`, Nastran, `.med`, legacy `.vtk`, and more — and
auto-detects XML VTK (`.vtu`) files, which are read through ReadVTK.jl. Meshes
may be structured or unstructured, with 1st- or 2nd-order elements (or a mix).

Radiating geometry is partitioned by named groups (**Physical Surface** in 3D,
**Physical Curve** in 2D). Formats without named groups (e.g. STL) fall back to
a single `"default"` group.

```julia
using RadiativeViewFactor

# 3D surface mesh (default)
mesh = load_mesh("geometry.msh")

# 2D planar curve mesh (view factors per unit depth)
mesh = load_mesh("planar.msh"; surface_dim=1)

# Other formats are detected from the extension
mesh = load_mesh("part.stl")
mesh = load_mesh("assembly.step")

# XML VTK (.vtu) requires ReadVTK in scope
using ReadVTK
mesh = load_mesh("grid.vtu")
```

## Computing view factors

```julia
result = compute_view_factors(mesh; nquad=4)
```

This returns a [`ViewFactorResult`](@ref) containing view factors at both the
element level (`F_elem`) and the physical-group level (`F_group`).

## Reading the result

The rows and columns of `F_group` correspond to physical groups in the order
given by `result.group_tags` and `result.group_names`:

```julia
# Print all group names
println(result.group_names)

# Look up a specific pair by name
i = findfirst(==("hotplate"),  result.group_names)
j = findfirst(==("coldplate"), result.group_names)
println("F(hotplate → coldplate) = ", result.F_group[i, j])
```

`F_group[i, j]` is the view factor **from** group `i` **to** group `j`.

## Validation checks

```julia
check_reciprocity(result)   # verifies Aᵢ Fᵢⱼ ≈ Aⱼ Fⱼᵢ
check_closure(result)       # verifies row sums ≤ 1
```

Row sums less than 1 are expected for open geometries where radiation escapes
through open boundaries.
