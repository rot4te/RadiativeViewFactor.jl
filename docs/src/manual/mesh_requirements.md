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
must be loaded (`using ReadVTK`). Legacy (non-XML) `.vtk` files are not read by
ReadVTK and go through Gmsh instead.

Nek5000/NekRS `.re2` binary meshes are also supported, through a dedicated
in-tree parser (Gmsh cannot open them); see [`.re2` meshes](@ref) below.

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
| 2nd | 10 | Quad9 | 9 (Lagrange quadrilateral) — centre node dropped (reported by `verbose=true`) |

Requirements:

- Radiating surfaces in named groups; obstruction surfaces in separate groups.
- `Mesh.ElementOrder` selects the order (1 = default → Tri3/Quad4; 2 → Tri6 and
  Quad9). Gmsh writes Quad9 for second-order quads unless
  `Mesh.SecondOrderIncomplete = 1` is set, which gives true Quad8; either loads
  as Quad8. Third and higher orders are not supported and raise an error that
  lists the element types found.
- Element normals follow node winding, which Gmsh keeps consistent within a
  surface. **Opposing surfaces must be wound to face each other.** No automatic
  orientation is applied for Gmsh or VTK surface meshes; use
  `reverse_normals=true` to flip all normals at load time if a mesh comes in
  back-to-front, or [`reverse_group_normals`](@ref) to flip selected groups
  after loading (see [Flipping normals](@ref) below). `.re2` meshes are the
  exception: their normals are oriented automatically.

## Curve meshes (`surface_dim=1`)

Supported element types:

| Order | Gmsh type | Name | Nodes |
|---|---|---|---|
| 1st | 1 | Line2 | 2 (linear line) |
| 2nd | 8 | Line3 | 3 (quadratic line) |

Requirements:

- Radiating curves in named groups.
- `Mesh.ElementOrder` selects the order (1 → Line2; 2 → Line3).
- Normal orientation is corrected automatically at load time for meshes read
  through Gmsh, provided the adjacent 2-D surface mesh is in the file (see
  below).
- Curve meshes run on the CPU only.

## Normal orientation for curve meshes

For `surface_dim=1`, element normals are computed by rotating the tangent vector
90° counter-clockwise in the xy-plane. The correct sign depends on how the curve
is wound.

For meshes read through Gmsh, RadiativeViewFactor.jl corrects this
automatically at load time by locating the adjacent surface for each curve
**from mesh connectivity** (shared nodes, then the nearest surface centroid),
not from CAD topology. Because it reads connectivity rather than requiring a
structured/transfinite mesh, it works for unstructured curve meshes too, and
with `.msh` v2.2 files that carry no CAD topology. Any element whose normal
points away from that surface interior is flipped.

This needs the **surface elements themselves** in the file, and Gmsh writes only
elements that belong to a physical group. Either put the surface in a
`Physical Surface`, or save everything with `Mesh.SaveAll = 1`; a surface with
no physical group and no `SaveAll` is not in the `.msh` at all. When no adjacent
surface can be found for a curve (no surface in the file, or one that shares no
nodes with it, as with a free-standing baffle), `load_mesh` warns and leaves
that curve's normals as Gmsh wrote them, so check them with
[`plot_mesh_normals`](@ref) or pass `reverse_normals=true`. If a curve borders
several surfaces, the one whose centroid is nearest the curve wins.

Curve meshes read through [`load_vtu`](@ref) get **no** automatic correction (a
warning says so): normals follow the cell's node order.

If the auto-correction is wrong for a mesh, flip all normals at load time:

```julia
mesh = load_mesh("planar.msh"; surface_dim=1, reverse_normals=true)
```

`reverse_normals` is applied **after** the auto-correction. Use
[`plot_mesh_normals`](@ref) to inspect normal directions before computing.

## Flipping normals

`reverse_normals=true` flips every element. When only some groups need
flipping — typically a CSG-cut body sitting inside another, where one surface
is already correct and the other is not — reverse just those groups after
loading:

```julia
mesh = load_mesh("concentric.msh")
mesh = reverse_group_normals(mesh, 2)          # one tag, or a collection of tags
```

`reverse_group_normals` returns a new `MeshData` and permutes the mid-side nodes
of 2nd-order elements along with the corners, so the geometry stays intact.

A wrongly-oriented body is not always obvious from the result: the pair-area
kernels (quadrature, `monte_carlo`) can still give roughly the right *aggregated*
view factor while `check_reciprocity` and the row sums quietly do not, and
`raytrace=true` fails outright. See the [`reverse_group_normals`](@ref)
docstring for the reasoning.

## `.re2` meshes

A Nek5000/NekRS `.re2` file stores a 3D hex *volume* mesh. [`load_re2`](@ref)
(also reached through `load_mesh`) takes its **boundary faces** as the radiating
surface:

- **Groups** are the Nek boundary-condition labels (`W`, `P`, or a generic label
  such as `MSH`), numbered in order of first appearance. Only genuinely internal
  faces (`E`, or blank) are skipped; periodic (`P`) faces are kept as their own
  group, so the enclosure closes (rows sum to 1), matching Nek5000/NekRS's own
  view-factor convention. If the file has no usable labels, the topological
  boundary is taken as a single `"default"` group.
- **Normals** are oriented into the fluid cavity (toward the owning hex's
  centroid). `reverse_normals=true` gives the opposite convention.
- **Curved faces**: a Nek face is curved and `.re2` records the curvature as a
  mid-side point per curved edge (the `'m'` form). With `curved=true` (the
  default) every boundary face becomes a Quad8 built from those points, with
  straight mid-points for edges that have no record, so a mesh is all-Quad8 or
  all-Quad4. Flattening curved faces to the quad through their corners
  understates their area, and — because the resulting `F` is reciprocal with
  respect to those flat areas while the solver integrates over the curved ones —
  breaks the enclosure energy balance. `curved=false`, or a file with no `'m'`
  records, gives Quad4 faces. The analytic curved-side forms (`'C'`, `'s'`) are
  ignored.
- **Nek identity**: each element records its Nek global element number and
  local face index (`SurfaceElement.eg`, `.iface`), which
  [`write_nekrs_view_factors`](@ref) needs, so only `.re2`-loaded meshes can be
  exported that way.
- **Splitting groups**: `gmsh2nek` writes the same generic label for every
  ordinary surface, so several named surfaces can share one group. Each face
  also records its original Gmsh Physical Surface tag (`SurfaceElement.phys_tag`);
  call [`split_groups_by_tag`](@ref) once after loading to regroup at
  `(code, phys_tag)` granularity, e.g. to pass one particular surface as an
  `obstruction_groups` entry.
- Word size (4- or 8-byte reals) and byte order are auto-detected. Only 3D
  files (`surface_dim=2`) are handled.

## XML VTK (`.vtu`) meshes

VTK has no physical groups, so [`load_vtu`](@ref) uses a per-cell integer array
if one is available: the one named by `group_field`, otherwise the first of
`CellEntityIds`, `gmsh:physical`, `RegionId`, `MaterialIds`, `region`, `group`
found. Groups are then named `group_<value>`; with no such array everything goes
in one `"default"` group. Cell types map to element families as:

| VTK cell | Id | Element |
|---|---|---|
| Line | 3 | Line2 |
| QuadraticEdge | 21 | Line3 |
| Triangle | 5 | Tri3 |
| QuadraticTriangle | 22 | Tri6 |
| Quad | 9 | Quad4 |
| QuadraticQuad | 23 | Quad8 |

Cells of other types are skipped. Quadratic VTK cells already use the
corners-then-mid-edge node order this package expects.

## Obstruction geometry

Whatever the element order, obstruction geometry is built from **corner nodes
only**: each quadrilateral becomes 2 triangles, each triangle 1, and each curve
element 1 segment. A curved 2nd-order blocker is therefore treated as flat for
visibility (its radiating integration still uses the full curved map). See
[Obstruction Detection](@ref).

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
// Mesh.SecondOrderIncomplete = 1;  // optional: Quad8 instead of Quad9 (both load as Quad8)

Physical Surface("hotplate")  = {1};
Physical Surface("coldplate") = {2};
Physical Surface("fin")       = {3};
```
