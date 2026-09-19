# test/fixtures.jl
#
# Small hand-built meshes and helpers shared by the tests that exercise
# element families, dimensions and code paths the older test files do not
# reach (tri3/tri6/quad8 on the GPU kernels, 2-D curve meshes, verbose
# output, obstruction on the GPU, ...). Included once from runtests.jl.

using RadiativeViewFactor.MeshIO: _build_group_obs_soups

"""Run `f()` and return `(result, captured_stdout)`."""
function capture_stdout(f)
  result = Ref{Any}(nothing)
  out = mktemp() do path, io
    redirect_stdout(io) do
      result[] = f()
    end
    close(io)
    read(path, String)
  end
  return result[], out
end

"""Assemble a `MeshData`, deriving `group_tri_soup` from the elements."""
function assemble_mesh(coords, elems, names; dim=2)
  group_elems = Dict{Int,Vector{Int}}(g => Int[] for g in keys(names))
  for (i, e) in enumerate(elems)
    push!(group_elems[e.group], i)
  end
  soups = _build_group_obs_soups(coords, elems, group_elems, dim)
  return MeshData(coords, elems, names, group_elems, soups, dim)
end

"""
Two unit squares facing each other one unit apart (analytic F = 0.19982),
as Quad4. Group 1 is the bottom plate (normal +z), group 2 the top (-z).
"""
function facing_plates_quad4()
  coords = [0.0 1 1 0  1 0 0 1;
            0.0 0 1 1  0 0 1 1;
            0.0 0 0 0  1 1 1 1]
  elems = [SurfaceElement([1, 2, 3, 4], 1, :quad4),
           SurfaceElement([5, 6, 7, 8], 2, :quad4)]
  return assemble_mesh(coords, elems, Dict(1 => "bottom", 2 => "top"))
end

"""Same plates with one extra unit-footprint blocker plate at z = 0.5 (group 3)."""
function facing_plates_with_blocker(; half_width=0.5)
  m = facing_plates_quad4()
  a = 0.5 - half_width; b = 0.5 + half_width
  extra = [a b b a;
           a a b b;
           0.5 0.5 0.5 0.5]
  coords = hcat(m.coords, extra)
  elems = vcat(m.surface_elems, [SurfaceElement([9, 10, 11, 12], 3, :quad4)])
  return assemble_mesh(coords, elems, Dict(1 => "bottom", 2 => "top", 3 => "blocker"))
end

"""
Same plates, but the blocker is an `n`x`n` grid of small Quad4 covering
`[0.5-half_width, 0.5+half_width]^2`, so its BVH has interior nodes (not just a
single leaf).
"""
function facing_plates_with_grid_blocker(; half_width=0.5, n=6)
  m = facing_plates_quad4()
  xs = range(0.5 - half_width, 0.5 + half_width; length=n + 1)
  cols = Vector{Float64}[]
  elems = SurfaceElement[SurfaceElement(e.nodes, e.group, e.family) for e in m.surface_elems]
  idx(i, j) = 8 + (j - 1) * (n + 1) + i
  for j in 1:n+1, i in 1:n+1
    push!(cols, [xs[i], xs[j], 0.5])
  end
  for j in 1:n, i in 1:n
    push!(elems, SurfaceElement([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)], 3, :quad4))
  end
  coords = hcat(m.coords, reduce(hcat, cols))
  return assemble_mesh(coords, elems, Dict(1 => "bottom", 2 => "top", 3 => "blocker"))
end

"""Split every Quad4 into two Tri3 with the same winding (same normal)."""
function to_tri3(mesh::MeshData)
  elems = SurfaceElement[]
  for e in mesh.surface_elems
    n = e.nodes
    if e.family === :quad4
      push!(elems, SurfaceElement([n[1], n[2], n[3]], e.group, :tri3))
      push!(elems, SurfaceElement([n[1], n[3], n[4]], e.group, :tri3))
    else
      push!(elems, e)
    end
  end
  return assemble_mesh(mesh.coords, elems, mesh.group_tags; dim=mesh.mesh_dim)
end

"""
Promote Line2/Tri3/Quad4 elements to Line3/Tri6/Quad8 by inserting straight
mid-side nodes (geometry unchanged, so results must match the linear mesh).
Gmsh node order: Quad8 5=(1,2) 6=(2,3) 7=(3,4) 8=(4,1); Tri6 4=(1,2) 5=(2,3)
6=(3,1); Line3 3=midpoint.
"""
function to_second_order(mesh::MeshData)
  cols = [mesh.coords[:, j] for j in 1:size(mesh.coords, 2)]
  mids = Dict{Tuple{Int,Int},Int}()
  mid(a, b) = get!(mids, minmax(a, b)) do
    push!(cols, (mesh.coords[:, a] + mesh.coords[:, b]) / 2)
    length(cols)
  end
  elems = SurfaceElement[]
  for e in mesh.surface_elems
    n = e.nodes
    if e.family === :quad4
      push!(elems, SurfaceElement([n; mid(n[1], n[2]); mid(n[2], n[3]);
                                   mid(n[3], n[4]); mid(n[4], n[1])], e.group, :quad))
    elseif e.family === :tri3
      push!(elems, SurfaceElement([n; mid(n[1], n[2]); mid(n[2], n[3]);
                                   mid(n[3], n[1])], e.group, :tri))
    elseif e.family === :line2
      push!(elems, SurfaceElement([n; mid(n[1], n[2])], e.group, :line3))
    else
      push!(elems, e)
    end
  end
  return assemble_mesh(reduce(hcat, cols), elems, mesh.group_tags; dim=mesh.mesh_dim)
end

"""Closed unit cube of six Quad4 faces, all normals pointing inward."""
function unit_cube_quad4()
  coords = [0.0 1 1 0 0 1 1 0;
            0.0 0 1 1 0 0 1 1;
            0.0 0 0 0 1 1 1 1]
  raw_faces = [(1, 2, 3, 4), (5, 6, 7, 8), (1, 2, 6, 5),
               (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]
  center = SVector(0.5, 0.5, 0.5)
  p(k) = SVector{3,Float64}(coords[1, k], coords[2, k], coords[3, k])
  function inward(f)
    a, b, c, d = f
    nvec = cross(p(b) - p(a), p(c) - p(a))
    centroid = (p(a) + p(b) + p(c) + p(d)) / 4
    dot(nvec, center - centroid) > 0 ? f : (a, d, c, b)
  end
  elems = [SurfaceElement(collect(inward(f)), 1, :quad4) for f in raw_faces]
  return assemble_mesh(coords, elems, Dict(1 => "cube"))
end

# ---------------------------------------------------------------------------
# 2-D curve-mesh fixtures (Gmsh-written .msh files)
# ---------------------------------------------------------------------------

import Gmsh: gmsh

struct NotCPU end   # any backend object that is not KernelAbstractions.CPU

const F_OPPOSITE = sqrt(2) - 1
const F_ADJACENT = (2 - sqrt(2)) / 2

# Unit square; sides are drawn so that `top` and `left` initially have their
# normals pointing OUT of the surface and `bottom`/`right` pointing in, so the
# loader's orientation pass has to flip some elements and leave others alone.
#
# Gmsh only writes elements that belong to a physical group, so unless the
# surface is in a physical group (`surface_group=true`) or `save_all=true`, the
# .msh contains no 2-D elements and the loader has nothing to orient against.
function write_square_msh(path; order=1, physical=true, surface_group=false,
                          save_all=false, generate_dim=2, h=0.25)
  gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", 0)
  gmsh.model.add("square")
  for (i, (x, y)) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)))
    gmsh.model.geo.addPoint(x, y, 0, h, i)
  end
  gmsh.model.geo.addLine(1, 2, 1)   # bottom, +x: normal (0, 1)   (inward)
  gmsh.model.geo.addLine(2, 3, 2)   # right,  +y: normal (-1, 0)  (inward)
  gmsh.model.geo.addLine(4, 3, 3)   # top,    +x: normal (0, 1)   (OUTWARD)
  gmsh.model.geo.addLine(1, 4, 4)   # left,   +y: normal (-1, 0)  (OUTWARD)
  gmsh.model.geo.addCurveLoop([1, 2, -3, -4], 1)
  gmsh.model.geo.addPlaneSurface([1], 1)
  gmsh.model.geo.synchronize()
  tags = Dict{String,Int}()
  if physical
    for (i, name) in enumerate(("bottom", "right", "top", "left"))
      tags[name] = gmsh.model.addPhysicalGroup(1, [i])
      gmsh.model.setPhysicalName(1, tags[name], name)
    end
  end
  if surface_group
    s = gmsh.model.addPhysicalGroup(2, [1])
    gmsh.model.setPhysicalName(2, s, "interior")
  end
  gmsh.option.setNumber("Mesh.ElementOrder", order)
  save_all && gmsh.option.setNumber("Mesh.SaveAll", 1)
  gmsh.model.mesh.generate(generate_dim)
  gmsh.write(path)
  gmsh.finalize()
  return tags
end

# Free-standing line segments, no surface at all: `bottom` (y=0, normal up) and
# `top` (y=1, normal down) one unit apart, plus an optional `blocker` at y=0.5.
function write_strips_msh(path; blocker=:none, h=0.25)
  gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", 0)
  gmsh.model.add("strips")
  pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
  blocker === :full    && append!(pts, [(-1.1, 0.5), (2.05, 0.5)])
  blocker === :partial && append!(pts, [(0.0, 0.5), (0.5, 0.5)])
  for (i, (x, y)) in enumerate(pts)
    gmsh.model.geo.addPoint(x, y, 0, h, i)
  end
  gmsh.model.geo.addLine(1, 2, 1)   # bottom, +x  -> normal +y
  gmsh.model.geo.addLine(3, 4, 2)   # top,    -x  -> normal -y
  blocker === :none || gmsh.model.geo.addLine(5, 6, 3)
  gmsh.model.geo.synchronize()
  tags = Dict{String,Int}()
  for (i, name) in enumerate(blocker === :none ? ("bottom", "top") :
                                                 ("bottom", "top", "blocker"))
    tags[name] = gmsh.model.addPhysicalGroup(1, [i])
    gmsh.model.setPhysicalName(1, tags[name], name)
  end
  gmsh.model.mesh.generate(1)
  gmsh.write(path)
  gmsh.finalize()
  return tags
end

# Normal of a curve element is the CCW rotation of its (node1 -> node2) tangent.
function curve_normal(coords, e)
  dx = coords[1, e.nodes[2]] - coords[1, e.nodes[1]]
  dy = coords[2, e.nodes[2]] - coords[2, e.nodes[1]]
  return (-dy, dx) ./ hypot(dx, dy)
end
function curve_midpoint(coords, e)
  e.family === :line3 ? coords[1:2, e.nodes[3]] :
                        (coords[1:2, e.nodes[1]] + coords[1:2, e.nodes[2]]) / 2
end
points_inward(mesh, e) =
  dot(curve_normal(mesh.coords, e), [0.5, 0.5] - curve_midpoint(mesh.coords, e)) > 0

# Two rooms side by side, [0,1]x[0,1] and [1,3]x[0,1], sharing the wall x = 1.
# Only that wall is a physical curve group, so its centroid is adjacent to both
# surfaces and the loader must pick the nearer one (the small room). `shared_down`
# draws the wall top-to-bottom; `small_first` gives the small room surface tag 1.
function write_two_rooms_msh(path; shared_down=false, small_first=true, h=0.5)
  gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", 0)
  gmsh.model.add("rooms")
  for (i, (x, y)) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1), (3, 0), (3, 1)))
    gmsh.model.geo.addPoint(x, y, 0, h, i)
  end
  gmsh.model.geo.addLine(1, 2, 1)
  shared_down ? gmsh.model.geo.addLine(3, 2, 2) : gmsh.model.geo.addLine(2, 3, 2)
  gmsh.model.geo.addLine(3, 4, 3)
  gmsh.model.geo.addLine(4, 1, 4)
  gmsh.model.geo.addLine(2, 5, 5)
  gmsh.model.geo.addLine(5, 6, 6)
  gmsh.model.geo.addLine(6, 3, 7)
  s = shared_down ? -1 : 1
  gmsh.model.geo.addCurveLoop([1, s * 2, 3, 4], 1)
  gmsh.model.geo.addCurveLoop([5, 6, 7, -s * 2], 2)
  small, large = small_first ? (1, 2) : (2, 1)
  gmsh.model.geo.addPlaneSurface([1], small)
  gmsh.model.geo.addPlaneSurface([2], large)
  gmsh.model.geo.synchronize()
  gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, [2]), "shared")
  gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [small]), "small_room")
  gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [large]), "large_room")
  gmsh.model.mesh.generate(2)
  gmsh.write(path)
  gmsh.finalize()
end
