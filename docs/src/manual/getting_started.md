# Getting Started

## Loading a mesh

`load_mesh` reads any format the Gmsh SDK can open — `.msh` (v2.2 and v4),
`.stl`, `.step`/`.stp`, Nastran, `.med`, legacy `.vtk`, and more. It
auto-detects XML VTK (`.vtu`) files, which are read through ReadVTK.jl, and
Nek5000/NekRS `.re2` files, which are read by a built-in parser. Meshes may be
structured or unstructured, with 1st- or 2nd-order elements (or a mix).

Radiating geometry is partitioned by named groups (**Physical Surface** in 3D,
**Physical Curve** in 2D). Formats without named groups (e.g. STL) fall back to
a single `"default"` group.

```julia
using RadiativeViewFactor

# 3D surface mesh (default)
mesh = load_mesh("geometry.msh")

# 2D planar curve mesh
mesh = load_mesh("planar.msh"; surface_dim=1)

# Other formats are detected from the extension
mesh = load_mesh("part.stl")
mesh = load_mesh("assembly.step")

# XML VTK (.vtu) requires ReadVTK in scope
using ReadVTK
mesh = load_mesh("grid.vtu")

# Nek5000/NekRS mesh: boundary faces of the hex volume mesh
mesh = load_mesh("case.re2")
```

See [Mesh Requirements](@ref) for supported element types, normal orientation,
and format-specific behaviour.

## Computing view factors

```julia
result = compute_view_factors(mesh; nquad=4)
```

This returns a [`ViewFactorResult`](@ref) containing view factors at both the
element level (`F_elem`) and the physical-group level (`F_group`), along with
the element and group areas (`A_elem`, `A_group`; lengths for curve meshes).
The integration method is chosen by keyword — see [Integration Methods](@ref):

```julia
# Duffy near singularities
compute_view_factors(mesh; nquad=6, use_duffy=true)
# pair-area Monte Carlo
compute_view_factors(mesh; monte_carlo=true, n_samples=5000)
# ray-shooting Monte Carlo
compute_view_factors(mesh; raytrace=true, n_rays=10000)
# with a blocking surface
compute_view_factors(mesh; nquad=4, obstruction_groups=[3])
```

## Reading the result

The rows and columns of `F_group` correspond to physical groups in the order
given by `result.group_tags` (sorted) and `result.group_names`:

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
check_closure(result)       # verifies no row sum exceeds 1
```

Both return a `Bool` and print what they measured (the maximum relative
reciprocity error, and the row-sum range). `check_closure` only fails when a row
sum is *above* 1 (by more than `tol`); row sums *below* 1 are expected for open
geometries where radiation escapes through open boundaries.

For a **closed** enclosure, [`enforce_closure`](@ref) returns a copy of the
result whose `F_elem` satisfies both reciprocity and `Σⱼ Fᵢⱼ = 1` to
machine precision. This matters when the view factors feed a radiosity solve,
because a small row-sum error is multiplied by the (large) radiosity and can
swamp a small net flux:

```julia
result = enforce_closure(result, mesh)
```

Do not use it on an open enclosure — including any result computed with
`radiating_groups` — since it would force rows to sum to 1 and invent
radiation that is not there.

## Computing one pair of surfaces in a crowded mesh

Assembly is dense over whatever it is handed, so a mesh full of shadowing bodies
costs `O(N_total²)` even when only one pair of surfaces is wanted.
`radiating_groups` restricts the radiating set while every group still
obstructs:

```julia
all_tags = collect(keys(mesh.group_tags))
result = compute_view_factors(mesh; nquad=4,
                              radiating_groups   = [wall_tag, pin_tag],
                              obstruction_groups = all_tags)
```

The returned `F_group` covers only the radiating groups, and the enclosure is
deliberately open, so use `check_reciprocity` (not `check_closure`) to validate
it. See [Obstruction Detection](@ref).

## Nek5000/NekRS export

A mesh loaded from a `.re2` file remembers each boundary face's Nek
`(element, face)` identity, so its view factors can be written straight back out
for the Nek-VF surface-to-surface radiation module:

```julia
mesh   = load_mesh("case.re2")
result = compute_view_factors(mesh; nquad=6, use_duffy=true)
result = enforce_closure(result, mesh)   # the enclosure is closed
write_nekrs_view_factors("case.vf", result, mesh)
```

See [`write_nekrs_view_factors`](@ref).
