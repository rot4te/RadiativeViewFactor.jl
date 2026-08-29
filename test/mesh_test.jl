# test/mesh_test.jl

@testset "load_mesh: 1st & 2nd order + group fallback" begin
  import Gmsh: gmsh

  # Build a single unit square plate, mesh at a given element order, write .msh
  function make_plate_msh(path::String, order::Int; physical::Bool=true)
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.model.add("plate")
    gmsh.model.geo.addPoint(0, 0, 0, 0.5, 1)
    gmsh.model.geo.addPoint(1, 0, 0, 0.5, 2)
    gmsh.model.geo.addPoint(1, 1, 0, 0.5, 3)
    gmsh.model.geo.addPoint(0, 1, 0, 0.5, 4)
    for (i, (a, b)) in enumerate([(1, 2), (2, 3), (3, 4), (4, 1)])
      gmsh.model.geo.addLine(a, b, i)
    end
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    if physical
      ptag = gmsh.model.addPhysicalGroup(2, [1])
      gmsh.model.setPhysicalName(2, ptag, "plate")
    end
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()
  end

  # Order 1 → linear elements (Tri3/Quad4)
  f1 = tempname() * ".msh"
  make_plate_msh(f1, 1)
  m1 = load_mesh(f1; verbose=false)
  @test !isempty(m1.surface_elems)
  @test all(e -> e.family in (:tri3, :quad4), m1.surface_elems)
  A1 = sum(element_pair_view_factor(m1.coords, e, e, 4, nothing)[2]
           for e in m1.surface_elems)
  @test isapprox(A1, 1.0; atol=1e-9)   # total plate area

  # Order 2 → quadratic elements (Tri6/Quad8)
  f2 = tempname() * ".msh"
  make_plate_msh(f2, 2)
  m2 = load_mesh(f2; verbose=false)
  @test all(e -> e.family in (:tri, :quad), m2.surface_elems)
  A2 = sum(element_pair_view_factor(m2.coords, e, e, 4, nothing)[2]
           for e in m2.surface_elems)
  @test isapprox(A2, 1.0; atol=1e-9)

  # Group fallback: a mesh with no physical groups gets a synthetic "default"
  f3 = tempname() * ".msh"
  make_plate_msh(f3, 1; physical=false)
  m3 = load_mesh(f3; verbose=false)
  @test !isempty(m3.surface_elems)
  @test collect(values(m3.group_tags)) == ["default"]

  foreach(rm, (f1, f2, f3))
end

# ---------------------------------------------------------------------------
@testset "Unstructured surface mesh end-to-end view factors" begin
  import Gmsh: gmsh

  # Two coaxial unit-square plates a distance 1 apart, each discretized with
  # Gmsh's DEFAULT (unstructured) triangulation — no transfinite/structured
  # meshing. Physical groups "bottom" and "top". Analytic F(bottom→top) for
  # two directly-opposed unit squares at separation 1 is ≈ 0.19982.
  function make_two_plates_msh(path::String, order::Int, h::Float64)
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.model.add("plates")
    # Bottom plate (z=0)
    gmsh.model.geo.addPoint(0, 0, 0, h, 1);
    gmsh.model.geo.addPoint(1, 0, 0, h, 2)
    gmsh.model.geo.addPoint(1, 1, 0, h, 3);
    gmsh.model.geo.addPoint(0, 1, 0, h, 4)
    for (i, (a, b)) in enumerate([(1, 2), (2, 3), (3, 4), (4, 1)])
      ;
      gmsh.model.geo.addLine(a, b, i);
    end
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1);
    gmsh.model.geo.addPlaneSurface([1], 1)
    # Top plate (z=1). Wind the loop clockwise (viewed from +z) so its
    # outward normal points DOWN (-z), i.e. toward the bottom plate.
    gmsh.model.geo.addPoint(0, 0, 1, h, 5);
    gmsh.model.geo.addPoint(1, 0, 1, h, 6)
    gmsh.model.geo.addPoint(1, 1, 1, h, 7);
    gmsh.model.geo.addPoint(0, 1, 1, h, 8)
    for (i, (a, b)) in enumerate([(5, 6), (6, 7), (7, 8), (8, 5)])
      ;
      gmsh.model.geo.addLine(a, b, i+4);
    end
    gmsh.model.geo.addCurveLoop([-8, -7, -6, -5], 2);
    gmsh.model.geo.addPlaneSurface([2], 2)
    gmsh.model.geo.synchronize()
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [1]), "bottom")
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [2]), "top")
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()
  end

  analytic = 0.19982

  # First-order unstructured triangle mesh (Tri3)
  f1 = tempname() * ".msh"
  make_two_plates_msh(f1, 1, 0.2)
  m1 = load_mesh(f1; verbose=false)
  @test all(e -> e.family === :tri3, m1.surface_elems)   # genuinely unstructured tris
  @test length(m1.group_tags) == 2
  r1 = compute_view_factors(m1; nquad=4, verbose=false)
  b = findfirst(==("bottom"), r1.group_names)
  t = findfirst(==("top"), r1.group_names)
  @test isapprox(r1.F_group[b, t], analytic; atol=5e-3)
  @test check_reciprocity(r1)

  # Second-order unstructured triangle mesh (Tri6) — cross-order agreement
  f2 = tempname() * ".msh"
  make_two_plates_msh(f2, 2, 0.2)
  m2 = load_mesh(f2; verbose=false)
  @test all(e -> e.family === :tri, m2.surface_elems)
  r2 = compute_view_factors(m2; nquad=4, verbose=false)
  b2 = findfirst(==("bottom"), r2.group_names)
  t2 = findfirst(==("top"), r2.group_names)
  @test isapprox(r2.F_group[b2, t2], analytic; atol=5e-3)

  foreach(rm, (f1, f2))
end
