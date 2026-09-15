# benchmarks/howell/geom.jl
# Gmsh geometry builders for the Howell catalog benchmark cases.
# Each builder writes a .msh file whose physical groups are named "S1", "S2",
# so the driver can pull F[1,2] from the aggregated view factor matrix.
#
# Normal orientation conventions:
#   3D surfaces — the node winding of each plane surface is set explicitly, so
#     the right-hand-rule normal points where the case needs it.
#   2D curves   — RadiativeViewFactor orients line normals toward the interior
#     of the adjacent meshed surface, so every curve group is built as the
#     boundary of a plane surface covering the radiating cavity.

import Gmsh: gmsh

const OCC = gmsh.model.occ

function _gmsh_start(name; order=2)
  gmsh.initialize()
  gmsh.option.setNumber("General.Terminal", 0)
  gmsh.option.setNumber("General.Verbosity", 0)
  gmsh.model.add(name)
  gmsh.option.setNumber("Mesh.ElementOrder", order)
  gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
  # Gmsh options survive finalize/initialize within one process, and a stale
  # RecombineAll from an earlier case crashes the mesher on curved bodies.
  gmsh.option.setNumber("Mesh.RecombineAll", 0)
  gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
  gmsh.option.setNumber("Mesh.MeshSizeMax", 1e22)
  gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
end

function _gmsh_finish(path, dim)
  gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
  gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
  gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
  gmsh.model.mesh.generate(dim)
  gmsh.write(path)
  gmsh.finalize()
  return path
end

_group!(dim, tags, n) = gmsh.model.addPhysicalGroup(dim, tags, -1, "S$n")

# Plane surface through an explicit ordered point list; the winding fixes the
# right-hand-rule normal. Returns (surface_tag, line_tags).
function _poly_surface(pts)
  ptags = [OCC.addPoint(p...) for p in pts]
  ltags = [OCC.addLine(ptags[i], ptags[mod1(i + 1, length(ptags))])
           for i in eachindex(ptags)]
  s = OCC.addPlaneSurface([OCC.addCurveLoop(ltags)])
  return s, ltags
end

function _transfinite_quad!(s, ltags, n)
  for l in ltags
    gmsh.model.mesh.setTransfiniteCurve(l, n + 1)
  end
  gmsh.model.mesh.setTransfiniteSurface(s)
  gmsh.model.mesh.setRecombine(2, s)
end

# ---------------------------------------------------------------------------
# 3D surface geometries
# ---------------------------------------------------------------------------

"""
    rect_pair_parallel(path; a, b, c, n=10)

Two identical `a × b` rectangles in parallel planes separated by `c`, wound to
face each other. Catalog case C-11.
"""
function rect_pair_parallel(path; a, b, c, n=10)
  _gmsh_start("C11")
  s1, l1 = _poly_surface([(0, 0, 0), (a, 0, 0), (a, b, 0), (0, b, 0)])       # +z
  s2, l2 = _poly_surface([(0, 0, c), (0, b, c), (a, b, c), (a, 0, c)])       # -z
  OCC.synchronize()
  _transfinite_quad!(s1, l1, n); _transfinite_quad!(s2, l2, n)
  _group!(2, [s1], 1); _group!(2, [s2], 2)
  _gmsh_finish(path, 2)
end

"""
    rect_pair_perp(path; l, w, h, n=10)

Two rectangles of common length `l` sharing an edge at 90°: A1 is `l × w` in
the z=0 plane (normal +z), A2 is `l × h` in the y=0 plane (normal +y).
Catalog case C-14.
"""
function rect_pair_perp(path; l, w, h, n=10)
  _gmsh_start("C14")
  s1, l1 = _poly_surface([(0, 0, 0), (l, 0, 0), (l, w, 0), (0, w, 0)])       # +z
  s2, l2 = _poly_surface([(0, 0, 0), (0, 0, h), (l, 0, h), (l, 0, 0)])       # +y
  OCC.synchronize()
  OCC.removeAllDuplicates(); OCC.synchronize()
  for s in (s1, s2)
    for cv in gmsh.model.getBoundary([(2, s)], false, false, false)
      gmsh.model.mesh.setTransfiniteCurve(abs(cv[2]), n + 1)
    end
    gmsh.model.mesh.setTransfiniteSurface(s)
    gmsh.model.mesh.setRecombine(2, s)
  end
  _group!(2, [s1], 1); _group!(2, [s2], 2)
  _gmsh_finish(path, 2)
end

"""
    coax_disks(path; r1, r2, L, nrad=6)

Coaxial parallel disks of radii `r1` (z=0, normal +z) and `r2` (z=L, normal −z).
Catalog cases C-40 (equal radii) and C-41 (unequal radii).
"""
function coax_disks(path; r1, r2, L, nrad=6)
  _gmsh_start("C41")
  d1 = OCC.addDisk(0, 0, 0, r1, r1)
  d2 = OCC.addDisk(0, 0, L, r2, r2)
  OCC.rotate([(2, d2)], 0, 0, L, 1, 0, 0, pi)   # flip d2 to face d1
  OCC.synchronize()
  h = min(r1, r2) / nrad
  gmsh.option.setNumber("Mesh.MeshSizeMax", h)
  gmsh.option.setNumber("Mesh.MeshSizeMin", h)
  gmsh.option.setNumber("Mesh.RecombineAll", 1)
  _group!(2, [d1], 1); _group!(2, [d2], 2)
  _gmsh_finish(path, 2)
end

"""
    cylinder_base_lateral(path; r, h, nsize=nothing)

Right circular cylinder: S1 is the base disk, S2 the inside lateral surface.
Both are faces of the same solid, so `reverse_normals=true` turns Gmsh's
outward normals into the inward-facing cavity normals this case needs.
Catalog case C-79.
"""
function cylinder_base_lateral(path; r, h, nsize=nothing)
  _gmsh_start("C79")
  OCC.addCylinder(0, 0, 0, 0, 0, h, r)
  OCC.synchronize()
  base = Int[]; lateral = Int[]
  for (dim, tag) in gmsh.model.getEntities(2)
    z = OCC.getCenterOfMass(dim, tag)[3]
    if abs(z) < 1e-6 * h
      push!(base, tag)
    elseif abs(z - h) > 1e-6 * h        # everything but the top cap
      push!(lateral, tag)
    end
  end
  msz = something(nsize, min(r, h) / 8)
  gmsh.option.setNumber("Mesh.MeshSizeMax", msz)
  gmsh.option.setNumber("Mesh.MeshSizeMin", msz)
  gmsh.option.setNumber("Mesh.RecombineAll", 1)   # quads: Duffy needs quad pairs
  _group!(2, base, 1); _group!(2, lateral, 2)
  _gmsh_finish(path, 2)
end

"""
    cone_interior_base(path; r, h, nsize=nothing)

Right circular cone: S1 is the interior lateral surface, S2 the base disk.
Use with `reverse_normals=true` so both face into the cone cavity.
Catalog case C-109.
"""
function cone_interior_base(path; r, h, nsize=nothing)
  _gmsh_start("C109")
  OCC.addCone(0, 0, 0, 0, 0, h, r, 0.0)
  OCC.synchronize()
  OCC.removeAllDuplicates(); OCC.synchronize()   # clean up the apex/seam
  base = Int[]; lateral = Int[]
  for (dim, tag) in gmsh.model.getEntities(2)
    z = OCC.getCenterOfMass(dim, tag)[3]
    abs(z) < 1e-6 * h ? push!(base, tag) : push!(lateral, tag)
  end
  msz = something(nsize, min(r, h) / 12)
  gmsh.option.setNumber("Mesh.MeshSizeMax", msz)
  gmsh.option.setNumber("Mesh.MeshSizeMin", msz)
  # Quads matter here: the Duffy correction for the base/lateral shared edge is
  # quad-only, and a triangle mesh of this geometry carries 4-9% error. Gmsh's
  # default blossom recombiner segfaults on the cone seam at some aspect
  # ratios, so use the simple algorithm.
  gmsh.option.setNumber("Mesh.RecombineAll", 1)
  gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
  _group!(2, lateral, 1); _group!(2, base, 2)
  _gmsh_finish(path, 2)
end

"""
    concentric_spheres(path; r1, r2, nsize=nothing)

Spherical shell between concentric spheres: S1 is the inner sphere, S2 the
outer. The meshed body is the gap, so `reverse_normals=true` points the inner
sphere outward and the outer sphere inward. Catalog case C-135.
"""
function concentric_spheres(path; r1, r2, nsize=nothing)
  _gmsh_start("C135")
  so = OCC.addSphere(0, 0, 0, r2)
  si = OCC.addSphere(0, 0, 0, r1)
  OCC.cut([(3, so)], [(3, si)])
  OCC.synchronize()
  inner = Int[]; outer = Int[]
  for (dim, tag) in gmsh.model.getEntities(2)
    bb = gmsh.model.getBoundingBox(dim, tag)
    (bb[4] - bb[1]) > (r1 + r2) ? push!(outer, tag) : push!(inner, tag)
  end
  msz = something(nsize, r1 / 2.5)
  gmsh.option.setNumber("Mesh.MeshSizeMax", msz)
  gmsh.option.setNumber("Mesh.MeshSizeMin", msz)
  _group!(2, inner, 1); _group!(2, outer, 2)
  _gmsh_finish(path, 2)
end

"""
    sphere_to_disk(path; rs, r, a, nsize=nothing)

Sphere of radius `rs` centred at the origin (S1, normals outward) and a coaxial
disk of radius `r` (S2) whose plane lies a distance `a` below the centre, wound
to face the sphere. Requires `a > rs`. Catalog case C-125.
"""
function sphere_to_disk(path; rs, r, a, nsize=nothing)
  a > rs || error("disk plane must not intersect the sphere (need a > rs)")
  _gmsh_start("C125")
  OCC.addSphere(0, 0, 0, rs)
  OCC.addDisk(0, 0, -a, r, r)          # addDisk's normal is +z, toward the sphere
  OCC.synchronize()
  sph = Int[]; disk = Int[]
  for (dim, tag) in gmsh.model.getEntities(2)
    z = OCC.getCenterOfMass(dim, tag)[3]
    abs(z + a) < 1e-6 * a ? push!(disk, tag) : push!(sph, tag)
  end
  msz = something(nsize, min(rs, r) / 2.5)
  gmsh.option.setNumber("Mesh.MeshSizeMax", msz)
  gmsh.option.setNumber("Mesh.MeshSizeMin", msz)
  _group!(2, sph, 1); _group!(2, disk, 2)
  _gmsh_finish(path, 2)
end

# ---------------------------------------------------------------------------
# 2D geometries (per unit depth) — curve meshes, loaded with surface_dim=1.
# The cavity surface is meshed only so line normals can be oriented inward.
# ---------------------------------------------------------------------------

"""
    parallel_plates_2d(path; w1, w2=w1, h, n=60)

Two infinitely long parallel plates of widths `w1` and `w2` separated by `h`,
sharing a common centre perpendicular. Catalog case C-1 (`w2 == w1`) and
C-2 (unequal widths).
"""
function parallel_plates_2d(path; w1, w2=w1, h, n=60)
  _gmsh_start("C1")
  s, l = _poly_surface([(-w1 / 2, 0, 0), (w1 / 2, 0, 0),
                        (w2 / 2, h, 0), (-w2 / 2, h, 0)])
  OCC.synchronize()
  _transfinite_quad!(s, l, n)
  _group!(1, [l[1]], 1)     # plate 1
  _group!(1, [l[3]], 2)     # plate 2
  _gmsh_finish(path, 2)
end

"""
    wedge_plates_2d(path; w1, w2, alpha, n=60)

Two infinitely long plates sharing a common edge at the origin with included
angle `alpha`: plate 1 of width `w1` along +x, plate 2 of width `w2` at angle
`alpha`. Catalog cases C-3 (90°, unequal widths) and C-4 (equal widths).
"""
function wedge_plates_2d(path; w1, w2, alpha, n=60)
  _gmsh_start("C4")
  p0 = OCC.addPoint(0, 0, 0)
  p1 = OCC.addPoint(w1, 0, 0)
  p2 = OCC.addPoint(w2 * cos(alpha), w2 * sin(alpha), 0)
  la = OCC.addLine(p0, p1)          # plate 1
  lb = OCC.addLine(p1, p2)          # closing edge of the cavity
  lc = OCC.addLine(p2, p0)          # plate 2
  s = OCC.addPlaneSurface([OCC.addCurveLoop([la, lb, lc])])
  OCC.synchronize()
  for l in (la, lb, lc)
    gmsh.model.mesh.setTransfiniteCurve(l, n + 1)
  end
  gmsh.model.mesh.setTransfiniteSurface(s)
  _group!(1, [la], 1); _group!(1, [lc], 2)
  _gmsh_finish(path, 2)
end

"""
    parallel_cylinders_2d(path; r1, r2, s, narc=64)

Cross-section of two infinitely long parallel cylinders of radii `r1`, `r2`
separated by a surface-to-surface gap `s` (the catalog's definition: axis
distance is `r1 + r2 + s`). The disks are meshed so that with
`reverse_normals=true` each circle's normals point outward.
Catalog cases C-68 (equal radii) and C-69 (unequal radii).
"""
function parallel_cylinders_2d(path; r1, r2, s, narc=64)
  _gmsh_start("C69")
  d1 = OCC.addDisk(0, 0, 0, r1, r1)
  d2 = OCC.addDisk(r1 + r2 + s, 0, 0, r2, r2)
  OCC.synchronize()
  c1 = [abs(c[2]) for c in gmsh.model.getBoundary([(2, d1)], false, false, false)]
  c2 = [abs(c[2]) for c in gmsh.model.getBoundary([(2, d2)], false, false, false)]
  for c in vcat(c1, c2)
    gmsh.model.mesh.setTransfiniteCurve(c, narc + 1)
  end
  gmsh.option.setNumber("Mesh.MeshSizeMax", min(r1, r2) / 4)
  gmsh.option.setNumber("Mesh.MeshSizeMin", min(r1, r2) / 4)
  _group!(1, c1, 1); _group!(1, c2, 2)
  _gmsh_finish(path, 2)
end

"""
    concentric_cylinders_2d(path; r1, r2, narc=80)

Cross-section of infinitely long concentric cylinders (inner `r1`, outer `r2`).
The annulus between them is meshed, so the inner circle's normals point outward
and the outer circle's inward. Catalog case C-63.
"""
function concentric_cylinders_2d(path; r1, r2, narc=80)
  _gmsh_start("C63")
  dout = OCC.addDisk(0, 0, 0, r2, r2)
  din = OCC.addDisk(0, 0, 0, r1, r1)
  ann, _ = OCC.cut([(2, dout)], [(2, din)])
  OCC.synchronize()
  cs = [abs(c[2]) for c in gmsh.model.getBoundary(ann, false, false, false)]
  inner = Int[]; outer = Int[]
  for c in cs
    bb = gmsh.model.getBoundingBox(1, c)
    (bb[4] - bb[1]) > (r1 + r2) ? push!(outer, c) : push!(inner, c)
  end
  for c in cs
    gmsh.model.mesh.setTransfiniteCurve(c, narc + 1)
  end
  gmsh.option.setNumber("Mesh.MeshSizeMax", (r2 - r1) / 4)
  gmsh.option.setNumber("Mesh.MeshSizeMin", (r2 - r1) / 4)
  _group!(1, inner, 1); _group!(1, outer, 2)
  _gmsh_finish(path, 2)
end

# ---------------------------------------------------------------------------
# Geometries for the tabulated cases
# ---------------------------------------------------------------------------

# Vertices of a regular n-gon of side `l`, counter-clockwise in the z=`z` plane.
function _ngon(n, l, z; reverse=false)
  Rc = l / (2 * sin(pi / n))
  ks = reverse ? (n-1:-1:0) : (0:n-1)
  [(Rc * cos(2pi * k / n), Rc * sin(2pi * k / n), z) for k in ks]
end

"""
    parallel_polygons(path; n, l, h, nsize=nothing)

Two identical regular n-gons of side `l` in parallel planes separated by `h`,
aligned and wound to face each other. Catalog case C-34.
"""
function parallel_polygons(path; n, l, h, nsize=nothing)
  _gmsh_start("C34")
  s1, _ = _poly_surface(_ngon(n, l, 0.0))                  # +z
  s2, _ = _poly_surface(_ngon(n, l, h; reverse=true))      # -z
  OCC.synchronize()
  msz = something(nsize, l / 12)
  gmsh.option.setNumber("Mesh.MeshSizeMax", msz)
  gmsh.option.setNumber("Mesh.MeshSizeMin", msz)
  gmsh.option.setNumber("Mesh.RecombineAll", 1)
  _group!(2, [s1], 1); _group!(2, [s2], 2)
  _gmsh_finish(path, 2)
end

"""
    two_spheres(path; r1, r2, s, nsize=nothing)

Two spheres of radii `r1` and `r2` separated by a surface-to-surface gap `s`,
so their centres are `r1 + r2 + s` apart. Normals are outward on both (Gmsh's
natural orientation for a solid), so no reversal is needed. Catalog case C-137.
"""
function two_spheres(path; r1, r2, s, nsize=nothing)
  _gmsh_start("C137")
  d = r1 + r2 + s
  OCC.addSphere(0, 0, 0, r1)
  OCC.addSphere(0, 0, d, r2)
  OCC.synchronize()
  g1 = Int[]; g2 = Int[]
  for (dim, tag) in gmsh.model.getEntities(2)
    z = OCC.getCenterOfMass(dim, tag)[3]
    abs(z) < abs(z - d) ? push!(g1, tag) : push!(g2, tag)
  end
  msz = something(nsize, min(r1, r2) / 3)
  gmsh.option.setNumber("Mesh.MeshSizeMax", msz)
  gmsh.option.setNumber("Mesh.MeshSizeMin", msz)
  _group!(2, g1, 1); _group!(2, g2, 2)
  _gmsh_finish(path, 2)
end

"""
    hex_prism(path; l, h, nsize=nothing)

Regular hexagonal prism of side `l` and height `h`. Groups are S1..S4 = side
walls at 0°, 60°, 120°, 180° from a reference wall (adjacent, next, opposite),
S5 = top hexagon, S6 = bottom hexagon. Use with `reverse_normals=true` so all
faces look into the cavity. Catalog case C-33.
"""
function hex_prism(path; l, h, nsize=nothing)
  _gmsh_start("C33")
  # Wind the base clockwise so its normal starts outward (-z) like the faces
  # the extrusion creates; the global reverse_normals then turns every face
  # inward together.
  s, _ = _poly_surface(_ngon(6, l, 0.0; reverse=true))
  OCC.synchronize()
  OCC.extrude([(2, s)], 0, 0, h)
  OCC.synchronize()
  OCC.removeAllDuplicates(); OCC.synchronize()
  walls = Tuple{Float64,Int}[]; top = Int[]; bot = Int[]
  for (dim, tag) in gmsh.model.getEntities(2)
    com = OCC.getCenterOfMass(dim, tag)
    if abs(com[3]) < 1e-6 * h
      push!(bot, tag)
    elseif abs(com[3] - h) < 1e-6 * h
      push!(top, tag)
    else
      push!(walls, (atan(com[2], com[1]), tag))       # angular position
    end
  end
  length(walls) == 6 || error("expected 6 side walls, got $(length(walls))")
  # order walls by angle relative to the first, so S1..S4 are 0°, 60°, 120°, 180°
  sort!(walls, by=first)
  ref = walls[1][1]
  key(a) = mod(a - ref + 2pi, 2pi)
  sel(target) = walls[argmin([abs(key(a) - target) for (a, _) in walls])][2]
  msz = something(nsize, min(l, h) / 8)
  gmsh.option.setNumber("Mesh.MeshSizeMax", msz)
  gmsh.option.setNumber("Mesh.MeshSizeMin", msz)
  gmsh.option.setNumber("Mesh.RecombineAll", 1)
  for (i, target) in enumerate((0.0, pi/3, 2pi/3, pi))
    _group!(2, [sel(target)], i)
  end
  _group!(2, top, 5); _group!(2, bot, 6)
  _gmsh_finish(path, 2)
end

"""
    cylinder_array_2d(path; P, square, nring=2, narc=48)

Cross-section of an infinite array of parallel cylinders of unit diameter at
pitch ratio `P = p/d`, either a square lattice (`square=true`) or an
equilateral triangular one. The centre cylinder is group 1, an edge/nearest
neighbour is group 2, and a diagonal/second-shell neighbour is group 3; every
other cylinder gets its own group so it can act as a blocker. Use with
`reverse_normals=true` and obstruction enabled.
Catalog cases C-72 (square) and C-73 (triangular).
"""
function cylinder_array_2d(path; P, square, nring=2, narc=48)
  _gmsh_start("C72")
  d = 1.0; r = d / 2; p = P * d
  centres = Tuple{Float64,Float64}[]
  for i in -nring:nring, j in -nring:nring
    square ? push!(centres, (i * p, j * p)) :
             push!(centres, ((i + j / 2) * p, j * p * sqrt(3) / 2))
  end
  # nearest and second-shell neighbour of the centre cylinder
  nb1 = square ? (p, 0.0) : (p, 0.0)
  nb2 = square ? (p, p)   : (p / 2, p * sqrt(3) / 2 + 0.0)   # placeholder, set below
  if !square
    nb2 = (0.0, p * sqrt(3))          # second shell of a hex array is at p*sqrt(3)
  end
  near(c, t) = hypot(c[1] - t[1], c[2] - t[2]) < 1e-6 * p
  order = Int[]
  push!(order, findfirst(c -> near(c, (0.0, 0.0)), centres))
  push!(order, findfirst(c -> near(c, nb1), centres))
  i3 = findfirst(c -> near(c, nb2), centres)
  i3 === nothing || push!(order, i3)
  rest = setdiff(1:length(centres), order)
  disks = [OCC.addDisk(x, y, 0, r, r) for (x, y) in centres]
  OCC.synchronize()
  curves = [[abs(c[2]) for c in gmsh.model.getBoundary([(2, dk)], false, false, false)]
            for dk in disks]
  for cs in curves, c in cs
    gmsh.model.mesh.setTransfiniteCurve(c, narc + 1)
  end
  gmsh.option.setNumber("Mesh.MeshSizeMax", r / 3)
  gmsh.option.setNumber("Mesh.MeshSizeMin", r / 3)
  for (n, idx) in enumerate(vcat(order, rest))
    _group!(1, curves[idx], n)
  end
  _gmsh_finish(path, 2)
end

"""
    plane_tube_rows_2d(path; R, ntube=21, gap=0.75, narc=40)

Cross-section of an infinite plane facing two rows of tubes of unit diameter in
an equilateral triangular array at pitch ratio `R = p/d`. The radiating cavity
(the region above the plane, with the tubes cut out of it) is the meshed
surface, so every normal is oriented into it: the plane radiates up and the
tubes radiate outward, with no reversal.

Group 1 is the central segment of the plane, one pitch wide — sampling the
middle avoids the end effects of a finite plane. Group 2 is the front row,
group 3 the second row, group 4 the rest of the plane.
Catalog case C-8.
"""
function plane_tube_rows_2d(path; R, ntube=21, gap=0.75, narc=40)
  _gmsh_start("C8")
  d = 1.0; r = d / 2; p = R * d
  y1 = gap * p + r                       # front-row centre height
  y2 = y1 + p * sqrt(3) / 2              # second row, offset half a pitch
  half = ntube ÷ 2
  W = (half + 0.5) * p
  ytop = y2 + 2p
  # cavity outline, bottom edge pre-split so the central segment stays separate
  pts = [(-W, 0.0, 0.0), (-p/2, 0.0, 0.0), (p/2, 0.0, 0.0), (W, 0.0, 0.0),
         (W, ytop, 0.0), (-W, ytop, 0.0)]
  ptags = [OCC.addPoint(q...) for q in pts]
  ltags = [OCC.addLine(ptags[i], ptags[mod1(i+1, length(ptags))]) for i in eachindex(ptags)]
  rect = OCC.addPlaneSurface([OCC.addCurveLoop(ltags)])
  fdisks = [OCC.addDisk(i * p, y1, 0, r, r) for i in -half:half]
  sdisks = [OCC.addDisk((i + 0.5) * p, y2, 0, r, r) for i in -half:half]
  cav, _ = OCC.cut([(2, rect)], [(2, dk) for dk in vcat(fdisks, sdisks)])
  OCC.synchronize()
  central = Int[]; outer = Int[]; frow = Int[]; srow = Int[]
  for c in [abs(cv[2]) for cv in gmsh.model.getBoundary(cav, false, false, false)]
    com = OCC.getCenterOfMass(1, c)
    bb  = gmsh.model.getBoundingBox(1, c)
    if abs(com[2]) < 1e-6 * p                     # lies on the plane y = 0
      (bb[4] <= p/2 + 1e-6 && bb[1] >= -p/2 - 1e-6) ? push!(central, c) : push!(outer, c)
    elseif abs(com[2] - y1) < 0.6r
      push!(frow, c)
    elseif abs(com[2] - y2) < 0.6r
      push!(srow, c)
    end
  end
  for c in vcat(frow, srow)
    gmsh.model.mesh.setTransfiniteCurve(c, max(4, narc ÷ 4) + 1)
  end
  gmsh.option.setNumber("Mesh.MeshSizeMax", r / 3)
  gmsh.option.setNumber("Mesh.MeshSizeMin", r / 3)
  # Every tube gets its own group. Obstruction never applies a group to a pair
  # involving itself, so tubes lumped into one group could not shadow each
  # other and the plane would see tubes that are actually hidden.
  bycentre(cs) = begin
    g = Dict{Int,Vector{Int}}()
    for c in cs
      k = round(Int, OCC.getCenterOfMass(1, c)[1] / (p / 2))
      push!(get!(g, k, Int[]), c)
    end
    [g[k] for k in sort(collect(keys(g)))]
  end
  _group!(1, central, 1)
  n = 1
  ftubes = bycentre(frow); stubes = bycentre(srow)
  for t in ftubes; _group!(1, t, n += 1); end
  for t in stubes; _group!(1, t, n += 1); end
  _group!(1, outer, n + 1)
  _gmsh_finish(path, 2)
  return (nfront=length(ftubes), nsecond=length(stubes))
end

"""
    rect_to_long_rect(path; a, b, phi, W=60.0, n=16)

A finite `b × a` rectangle (S1) sharing an edge of length `b` with a second
rectangle (S2) at included angle `phi`, the second extending a distance `W`
away from the common edge as a stand-in for infinity. Catalog case C-10.
"""
function rect_to_long_rect(path; a, b, phi, W=60.0, n=16)
  _gmsh_start("C10")
  # common edge along y from 0 to b; S2 in the z=0 plane out to x = W
  s2, l2 = _poly_surface([(0, 0, 0), (W, 0, 0), (W, b, 0), (0, b, 0)])           # +z
  s1, _  = _poly_surface([(0, 0, 0), (0, b, 0),
                          (a * cos(phi), b, a * sin(phi)),
                          (a * cos(phi), 0, a * sin(phi))])
  OCC.synchronize()
  OCC.removeAllDuplicates(); OCC.synchronize()
  for cv in gmsh.model.getBoundary([(2, s1)], false, false, false)
    gmsh.model.mesh.setTransfiniteCurve(abs(cv[2]), n + 1)
  end
  gmsh.model.mesh.setTransfiniteSurface(s1)
  gmsh.model.mesh.setRecombine(2, s1)
  # S2 stands in for an infinite strip, so it is graded: fine at the common
  # edge, coarsening with distance. A uniform mesh over a long W leaves
  # elements with a huge aspect ratio and the view factor stops converging.
  nx = max(n, ceil(Int, 6 * log(W / a + 1)) * 8)
  gmsh.model.mesh.setTransfiniteCurve(l2[1], nx + 1, "Progression", 1.06)
  gmsh.model.mesh.setTransfiniteCurve(l2[3], nx + 1, "Progression", 1 / 1.06)
  gmsh.model.mesh.setTransfiniteCurve(l2[2], n + 1)
  gmsh.model.mesh.setTransfiniteCurve(l2[4], n + 1)
  gmsh.model.mesh.setTransfiniteSurface(s2)
  gmsh.model.mesh.setRecombine(2, s2)
  _group!(2, [s1], 1); _group!(2, [s2], 2)
  _gmsh_finish(path, 2)
end

# C-154 (two hemispheres in contact) and C-35 (rectangle to a quarter of a
# parallel cylinder) are NOT implemented here. Both figures leave the exact
# solid angle genuinely ambiguous (which quarter of the cylinder faces the
# rectangle in C-35; whether the hemispheres' poles or rims face each other,
# and which side is concave, in C-154), and several distinct readings of each
# were tried against the published table without reproducing it — see
# RESULTS.md for what was tried and the numbers each gave.
