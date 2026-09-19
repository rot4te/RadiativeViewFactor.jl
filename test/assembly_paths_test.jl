# test/assembly_paths_test.jl
#
# CPU assembly paths and loaders that no other test file reaches: verbose
# reports, `radiating_groups`, Monte Carlo between well-separated elements
# (every element family), Duffy on Quad8, ray-shooting on curved / blocker-only
# geometry, GPU-dispatch plumbing, mid-side-record filtering in the .re2 reader,
# and Gmsh loader error/verbose branches. Reference value throughout: opposing
# unit squares one unit apart, F = 0.19982.

import Gmsh: gmsh
import KernelAbstractions

# Backends that are not KernelAbstractions.CPU: one the dispatch layer knows how
# to allocate for (registered below), one it does not.
struct RegisteredBackend end
struct UnregisteredBackend end
RadiativeViewFactor.Assembly._gpu_array_type(::RegisteredBackend) = Array
RadiativeViewFactor.Assembly._gpu_float_type(::RegisteredBackend) = Float32

function write_plate_msh(path; order=1, recombine=false, h=0.5, second_order_incomplete=false)
  gmsh.initialize()
  gmsh.option.setNumber("General.Verbosity", 0)
  gmsh.option.setNumber("Mesh.RecombineAll", recombine ? 1 : 0)
  second_order_incomplete && gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
  gmsh.model.add("plate")
  for (i, (x, y)) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)))
    gmsh.model.geo.addPoint(x, y, 0, h, i)
  end
  for i in 1:4
    gmsh.model.geo.addLine(i, i % 4 + 1, i)
  end
  gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
  gmsh.model.geo.addPlaneSurface([1], 1)
  gmsh.model.geo.synchronize()
  gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [1]), "plate")
  gmsh.option.setNumber("Mesh.ElementOrder", order)
  gmsh.model.mesh.generate(2)
  gmsh.write(path)
  gmsh.finalize()
  return path
end

@testset "assembly paths and loaders (coverage gaps)" begin
  F_ANALYTIC = 0.19982
  plates = Dict(:quad4 => facing_plates_quad4(),
                :tri3  => to_tri3(facing_plates_quad4()),
                :quad8 => to_second_order(facing_plates_quad4()),
                :tri6  => to_second_order(to_tri3(facing_plates_quad4())))

  @testset "verbose CPU reports" begin
    mesh = facing_plates_with_blocker(half_width=0.25)
    for (kw, marker) in ((NamedTuple(),             "nquad=4"),
                         ((use_duffy=true,),        "Duffy for singular pairs"),
                         ((monte_carlo=true, n_samples=200), "(Monte Carlo)"))
      r, out = capture_stdout() do
        compute_view_factors(mesh; obstruction_groups=[3], verbose=true, kw...)
      end
      @test r isa RadiativeViewFactor.ViewFactorResult
      @test occursin("CPU compute_view_factors: 3 elements", out)
      @test occursin(marker, out)
      @test occursin("Obstruction groups: [\"blocker\"]", out)
      @test occursin("Facing cull:", out)
      @test occursin("Row-sum check (group level)", out)
    end
  end

  @testset "radiating_groups: same answer as the full mesh, cheaper" begin
    mesh = facing_plates_with_blocker(half_width=0.25)
    full = compute_view_factors(mesh; obstruction_groups=[3], verbose=false)
    sub = compute_view_factors(mesh; radiating_groups=[1, 2], obstruction_groups=[3],
                                verbose=false)
    @test size(sub.F_elem) == (2, 2)
    @test isapprox(sub.F_elem[1, 2], full.F_elem[1, 2]; atol=1e-12)
    _, out = capture_stdout() do
      compute_view_factors(mesh; radiating_groups=[1, 2], verbose=true)
    end
    @test occursin("Radiating subset: 2 of 3 elements", out)
    # Documented usage: radiating_groups + non-radiating blockers, default verbose=true.
    # restrict_to_radiating trims mesh.group_tags, so the blocker's name is gone by the
    # time _compute_cpu prints its "Obstruction groups:" line; that used to throw
    # KeyError(3). The line now falls back to the bare tag.
    r_v, out_v = capture_stdout() do
      compute_view_factors(mesh; radiating_groups=[1, 2], obstruction_groups=[3], verbose=true)
    end
    @test occursin("Obstruction groups: [\"tag 3\"]", out_v)
    @test isapprox(r_v.F_elem[1, 2], full.F_elem[1, 2]; atol=1e-12)

    @test restrict_to_radiating(mesh, Int[]) === mesh
    @test_throws ErrorException restrict_to_radiating(mesh, [1, 42])       # unknown tag
    hollow = MeshData(mesh.coords, mesh.surface_elems,
                      merge(mesh.group_tags, Dict(9 => "empty")),
                      merge(mesh.group_elems, Dict(9 => Int[])),
                      mesh.group_tri_soup, 2)
    @test_throws ErrorException restrict_to_radiating(hollow, [9])         # selects nothing
  end

  @testset "Monte Carlo between separated elements, $fam" for fam in (:quad4, :tri3, :quad8, :tri6)
    # 5000 samples is not a perfect square, so the stratified samplers' leftover-
    # point loops run too. factor=0.1 keeps the pair out of the Duffy patch.
    r = compute_view_factors(plates[fam]; monte_carlo=true, n_samples=5000, factor=0.1,
                              rng=Xoshiro(2), verbose=false)
    @test isapprox(r.F_group[1, 2], F_ANALYTIC; rtol=0.04)
    @test check_reciprocity(r)
  end

  @testset "Monte Carlo: self_vf and the near-pair patch on non-quad elements" begin
    r = compute_view_factors(plates[:quad4]; monte_carlo=true, n_samples=400, factor=0.1,
                              self_vf=true, rng=Xoshiro(4), verbose=false)
    @test r.F_elem[1, 1] == 0.0            # a flat element cannot see itself
    # default factor: every Tri3 pair is "near", and Duffy has no singular
    # formula for triangles, so the patch falls back to plain quadrature
    rt = compute_view_factors(plates[:tri3]; monte_carlo=true, n_samples=400, nquad=6,
                               rng=Xoshiro(5), verbose=false)
    rq = compute_view_factors(plates[:tri3]; nquad=6, verbose=false)
    @test isapprox(rt.F_elem, rq.F_elem; atol=1e-8)   # same rule up to ~1e-10
  end

  @testset "Duffy on Quad8 matches Quad4 (same straight-sided geometry)" begin
    cube4 = unit_cube_quad4()
    cube8 = to_second_order(cube4)
    r4 = compute_view_factors(cube4; nquad=10, use_duffy=true, verbose=false)
    r8 = compute_view_factors(cube8; nquad=10, use_duffy=true, verbose=false)
    @test isapprox(r8.F_elem, r4.F_elem; atol=1e-9)
    @test all(x -> isapprox(x, 1.0; atol=5e-4), vec(sum(r8.F_elem, dims=2)))
  end

  @testset "Duffy on a vertex-only pair: Quad8 matches Quad4" begin
    # Two squares meeting at the single point (1,1,0), at right angles. Unlike
    # any pair in a cube (all edge-adjacent or opposite), this takes the
    # COMMON_VERTEX branch, which evaluates elements through `_eval_quad`.
    coords = [0.0 1 1 0  1 1 1;
              0.0 0 1 1  2 2 1;
              0.0 0 0 0  0 1 1]
    elems = [SurfaceElement([1, 2, 3, 4], 1, :quad4),      # z = 0, normal +z
             SurfaceElement([3, 7, 6, 5], 2, :quad4)]      # x = 1, normal -x
    m4 = assemble_mesh(coords, elems, Dict(1 => "floor", 2 => "wall"))
    m8 = to_second_order(m4)
    D = RadiativeViewFactor.DuffyKernel
    @test D.singularity_type(m4.surface_elems[1], m4.surface_elems[2])[1] === D.COMMON_VERTEX
    r4 = compute_view_factors(m4; nquad=8, use_duffy=true, verbose=false)
    r8 = compute_view_factors(m8; nquad=8, use_duffy=true, verbose=false)
    @test r4.F_elem[1, 2] > 0
    @test isapprox(r8.F_elem, r4.F_elem; atol=1e-9)
    @test check_reciprocity(r8)
  end

  @testset "Geometry.element_area: every quadrature order" for nq in (1, 2, 3, 4, 6)
    m = plates[:quad8]
    @test isapprox(RadiativeViewFactor.Geometry.element_area(m.coords, m.surface_elems[1].nodes; nquad=nq),
                   1.0; atol=1e-12)
  end

  @testset "reverse_group_normals for every linear/quadratic family" begin
    G = RadiativeViewFactor.Geometry
    normal_of = Dict(
      :quad4 => (c, n) -> quad4_normal_and_area_element(c, n, 0.0, 0.0)[1],
      :tri3  => (c, n) -> G.tri3_normal_and_area_element(c, n, 1/3, 1/3)[1],
      :quad  => (c, n) -> quad8_normal_and_area_element(c, n, 0.0, 0.0)[1])
    for fam in (:quad4, :tri3, :quad)
      mesh = fam === :quad4 ? plates[:quad4] : fam === :tri3 ? plates[:tri3] : plates[:quad8]
      flipped = reverse_group_normals(mesh, [1, 2])
      for (e0, e1) in zip(mesh.surface_elems, flipped.surface_elems)
        @test isapprox(normal_of[fam](flipped.coords, e1.nodes),
                       -normal_of[fam](mesh.coords, e0.nodes); atol=1e-12)
      end
    end
  end

  @testset "ray-shooting (CPU): curved Quad8 scene, extra blocker-only groups" begin
    for fam in (:quad8, :tri6)
      r = compute_view_factors(plates[fam]; raytrace=true, n_rays=40_000, rng=Xoshiro(6),
                                verbose=false)
      @test isapprox(r.F_group[1, 2], F_ANALYTIC; rtol=0.05)
    end
    # The blocker is not radiating: it enters the scene only as extra geometry
    # (tagged 0), so rays that hit it are absorbed and never reach the top plate.
    mesh = facing_plates_with_blocker(half_width=0.5)
    r, out = capture_stdout() do
      compute_view_factors(mesh; raytrace=true, n_rays=2000, rng=Xoshiro(7),
                            radiating_groups=[1, 2], obstruction_groups=[3], verbose=true)
    end
    @test r.F_group[1, 2] == 0.0
    # (restrict_to_radiating already dropped the blocker's name, so it prints its tag)
    @test occursin("Extra (non-radiating) blocker groups: [\"tag 3\"]", out)
    @test occursin("(ray-shooting Monte Carlo)", out)
  end

  @testset "GPU dispatch plumbing" begin
    Asm = RadiativeViewFactor.Assembly
    mesh = plates[:quad4]
    dummy = compute_view_factors(mesh; verbose=false)
    seen = Ref{Any}(nothing)
    stub = (m, nquad, backend, FloatT, ArrayT; kw...) ->
             (seen[] = (nquad=nquad, backend=backend, FloatT=FloatT, ArrayT=ArrayT, kw=kw); dummy)
    original = Asm._GPU_HOOK_REF[]
    try
      Asm.register_gpu_hook!(stub)
      r = compute_view_factors(mesh; backend=RegisteredBackend(), nquad=5, n_rays=77,
                                obstruction_groups=Int[], verbose=false)
      @test r === dummy
      @test seen[].backend isa RegisteredBackend
      @test seen[].FloatT === Float32 && seen[].ArrayT === Array && seen[].nquad == 5
      @test seen[].kw[:n_rays] == 77 && seen[].kw[:raytrace] == false

      @test_logs (:warn, r"use_duffy is CPU-only") compute_view_factors(
          mesh; backend=RegisteredBackend(), use_duffy=true, verbose=false)

      # a backend *type* is instantiated; on the CPU backend it is not a GPU dispatch
      seen[] = nothing
      compute_view_factors(mesh; backend=KernelAbstractions.CPU, verbose=false)
      @test seen[] === nothing

      Asm._GPU_HOOK_REF[] = nothing
      @test_throws ErrorException compute_view_factors(mesh; backend=RegisteredBackend(), verbose=false)
    finally
      Asm.register_gpu_hook!(original)
    end
    @test_throws ErrorException compute_view_factors(mesh; backend=UnregisteredBackend(), verbose=false)
  end

  @testset ".re2: unmatched mid-side records are dropped with a warning; reverse_normals" begin
    corners = zeros(3, 8, 1)
    unit = [(0.0,0.0,0.0),(1.0,0.0,0.0),(1.0,1.0,0.0),(0.0,1.0,0.0),
            (0.0,0.0,1.0),(1.0,0.0,1.0),(1.0,1.0,1.0),(0.0,1.0,1.0)]
    for v in 1:8, d in 1:3; corners[d, v, 1] = unit[v][d]; end
    elem_nodes = reshape(collect(1:8), 8, 1)
    curves = [(2, (0.5, 0.5, 0.5)),      # element 2 does not exist
              (1, (9.0, 9.0, 9.0)),      # nowhere near any edge of element 1
              (1, (0.5, 0.0, -0.1))]     # bows the (0,0,0)-(1,0,0) edge: kept
    (mids, out) = capture_stdout() do
      @test_logs (:warn, r"2 of 3 mid-side records could not be matched") RadiativeViewFactor.MeshIO._re2_edge_midsides(
          corners, elem_nodes, curves, true)
    end
    @test collect(keys(mids)) == [(1, 2)]
    @test mids[(1, 2)] == (0.5, 0.0, -0.1)
    @test occursin("1 curved edges from mid-side ('m') records", out)

    allcodes = [(1,"HOT"),(2,"HOT"),(3,"HOT"),(4,"CLD"),(5,"CLD"),(6,"CLD")]
    f = _write_re2(tempname()*".re2", allcodes)
    m  = load_re2(f; verbose=false)
    mr, out = capture_stdout() do
      load_re2(f; reverse_normals=true, verbose=true)
    end
    rm(f)
    @test occursin("All normals reversed.", out)
    for (e, er) in zip(m.surface_elems, mr.surface_elems)
      n  = quad4_normal_and_area_element(m.coords,  e.nodes,  0.0, 0.0)[1]
      nr = quad4_normal_and_area_element(mr.coords, er.nodes, 0.0, 0.0)[1]
      @test isapprox(nr, -n; atol=1e-12)
    end
  end

  @testset "Gmsh loader: verbose Quad9 report, unsupported orders, empty meshes" begin
    f = write_plate_msh(tempname() * ".msh"; order=2, recombine=true)
    (m, out) = @test_logs (:info, r"Quad9 \(type 10\) found") match_mode=:any capture_stdout() do
      load_mesh(f; verbose=true)
    end
    @test all(e -> e.family === :quad && length(e.nodes) == 8, m.surface_elems)
    @test occursin("Element type Quad9:", out)
    @test occursin(r"Loaded \d+ surface elements \(quad8: \d+, quad4: 0, tri6: 0, tri3: 0\)", out)
    rm(f)

    # Quad8 proper (no centre node) and a 1st-order Tri3 mesh, verbose
    f = write_plate_msh(tempname() * ".msh"; order=2, recombine=true, second_order_incomplete=true)
    (_, out8) = capture_stdout() do
      load_mesh(f; verbose=true)
    end
    @test occursin("Element type Quad8:", out8)
    rm(f)

    # Third-order triangles (Gmsh type 21) are not supported
    f = write_plate_msh(tempname() * ".msh"; order=3)
    err = try load_mesh(f; verbose=false); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("No supported elements found", err.msg)
    @test occursin("Tri3 (2), Quad4 (3), Tri6 (9), Quad8 (16), Quad9 (10)", err.msg)
    rm(f)

    # No entities of the requested dimension at all (points only)
    f = tempname() * ".msh"
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("points")
    gmsh.model.geo.addPoint(0, 0, 0, 1, 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(0)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(f)
    gmsh.finalize()
    err = try load_mesh(f; verbose=false); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("No entities of dimension 2", err.msg)
    rm(f)
  end
end
