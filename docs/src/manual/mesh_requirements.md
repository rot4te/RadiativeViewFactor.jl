# Mesh Requirements

Meshes may be **structured or unstructured** and **1st- or 2nd-order**, and the
two orders may be mixed within a single mesh. The solver identifies elements by
node connectivity alone — there is no structured-grid (`i,j`) assumption in
assembly, quadrature, or the obstruction BVH — so an unstructured Delaunay
triangulation, a structured/transfinite mesh, and a mixed-element mesh are all
handled the same way.

## File formats

`load_mesh` accepts any format the Gmsh SDK can open, inferred from the
extension: `.msh` (v2.2 and v4), `.stl`, `.step`/`.stp`, Nastran `.bdf`/`.nas`,
`.med`, legacy `.vtk`, and others. XML VTK unstructured grids (`.vtu`, and
XML-form `.vtk`) are detected automatically and read through ReadVTK.jl, which
must be loaded (`using ReadVTK`).

Radiating geometry is partitioned by **named groups**. In Gmsh these are
Physical Surface (3D) or Physical Curve (2D) groups. Formats that cannot carry
named groups (e.g. STL) fall back to a single synthetic `"default"` group; for
VTK, a per-cell integer region array can be used instead (see
[`load_vtu`](@ref)).

## Surface meshes (`surface_dim=2`)

Supported element types:

| Order | Gmsh type | Name | Nodes |
|---|---|---|---|
| 1st | 2  | Tri3  | 3 (linear triangle) |
| 1st | 3  | Quad4 | 4 (bilinear quadrilateral) |
| 2nd | 9  | Tri6  | 6 (quadratic triangle) |
| 2nd | 16 | Quad8 | 8 (serendipity quadrilateral) — preferred for curved geometry |
| 2nd | 10 | Quad9 | 9 (Lagrange quadrilateral) — centre node silently dropped |

Requirements:
- Radiating surfaces in named groups; obstruction surfaces in separate groups.
- `Mesh.ElementOrder` selects the order (1 = default → Tri3/Quad4; 2 → Tri6/Quad8).
- Element normals follow node winding, which Gmsh keeps consistent within a
  surface. **Opposing surfaces must be wound to face each other.** No automatic
  orientation is applied for surface meshes; use `reverse_normals=true` to flip
  all normals at load time if a mesh comes in back-to-front.

## Curve meshes (`surface_dim=1`)

Supported element types:

| Order | Gmsh type | Name | Nodes |
|---|---|---|---|
| 1st | 1 | Line2 | 2 (linear line) |
| 2nd | 8 | Line3 | 3 (quadratic line) |

Requirements:
- Radiating curves in named groups.
- `Mesh.ElementOrder` selects the order (1 → Line2; 2 → Line3).
- Normal orientation is corrected automatically at load time (see below).

## Normal orientation for curve meshes

For `surface_dim=1`, element normals are computed by rotating the tangent vector
90° counter-clockwise in the xy-plane. The correct sign depends on how the curve
is wound.

RadiativeViewFactor.jl corrects this automatically at load time by locating the
adjacent surface for each curve **from mesh connectivity** (shared nodes, then
the nearest surface centroid), not from CAD topology. Because it reads
connectivity rather than requiring a structured/transfinite mesh, it works for
unstructured curve meshes too, and with `.msh` v2.2 files that carry no CAD
topology. Any element whose normal points away from that surface interior is
flipped.

If the auto-correction is wrong for a mesh, flip all normals at load time:

```julia
mesh = load_mesh("planar.msh"; surface_dim=1, reverse_normals=true)
```

`reverse_normals` is applied **after** the auto-correction. Use
[`plot_mesh_normals`](@ref) to inspect normal directions before computing.

## Example Gmsh scripts

```gmsh
// 2D curve mesh. Order 1 (default) gives Line2; set 2 for Line3.
Mesh.ElementOrder = 1;

Physical Curve("emitter")     = {1};
Physical Curve("receiver")    = {2};
Physical Curve("obstruction") = {3};
```

```gmsh
// 3D surface mesh. Order 1 (default) gives Tri3/Quad4; set 2 for Tri6/Quad8.
// Recombination is optional — unstructured triangles are fully supported.
Mesh.ElementOrder    = 2;
Mesh.RecombineAll    = 1;   // produces quads instead of triangles
Mesh.Algorithm       = 8;   // Frontal-Delaunay for quads

Physical Surface("hotplate")  = {1};
Physical Surface("coldplate") = {2};
Physical Surface("fin")       = {3};
```
