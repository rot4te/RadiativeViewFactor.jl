# test/curve_mesh_test.jl
#
# 2-D (curve mesh, `surface_dim=1`) workflow: Gmsh loading with Line2/Line3
# elements, normal orientation from the adjacent surface, per-unit-depth view
# factors (quadrature, Duffy fall-back, Monte Carlo), 2-D segment obstruction,
# and the option combinations that curve meshes reject. None of this was
# exercised before; analytic references are the classical 2-D results:
#   unit square enclosure:   F(opposite) = sqrt(2) - 1,  F(adjacent) = (2 - sqrt(2))/2
#   two unit strips 1 apart: F = sqrt(2) - 1   (Hottel crossed strings)

@testset "curve meshes (surface_dim=1)" begin

  # `surface_group=true`: the surface is a physical group; `false` + save_all:
  # no physical surface group, so the loader falls back to every surface entity.
  @testset "load: Line$(order == 1 ? 2 : 3) square, normals oriented toward the surface" for
      (order, fam) in ((1, :line2), (2, :line3)), surface_group in (true, false)
    f = tempname() * ".msh"
    tags = write_square_msh(f; order=order, surface_group=surface_group, save_all=!surface_group)
    m = load_mesh(f; surface_dim=1, verbose=false)
    rm(f)
    @test m.mesh_dim == 1
    @test all(e -> e.family === fam, m.surface_elems)
    @test length(m.surface_elems) == 16                  # 4 segments per side
    if surface_group   # with SaveAll Gmsh drops the physical-group names
      @test Set(values(m.group_tags)) == Set(["bottom", "right", "top", "left"])
    end
    # every normal points into the square, including the sides Gmsh drew backwards
    @test all(e -> points_inward(m, e), m.surface_elems)
    # segment soup for obstruction: (xyz, endpoint, segment) per group
    for (g, idxs) in m.group_elems
      @test size(m.group_tri_soup[g]) == (3, 2, length(idxs))
    end

    # reverse_normals=true is applied after orientation: all normals flip
    f2 = tempname() * ".msh"
    write_square_msh(f2; order=order, surface_group=surface_group, save_all=!surface_group)
    mr = load_mesh(f2; surface_dim=1, reverse_normals=true, verbose=false)
    rm(f2)
    @test !any(e -> points_inward(mr, e), mr.surface_elems)
  end

  @testset "load: verbose report, synthesized group, unsupported elements" begin
    f = tempname() * ".msh"
    write_square_msh(f; order=2, surface_group=true)
    m, out = capture_stdout() do
      load_mesh(f; surface_dim=1, verbose=true)
    end
    @test occursin("Element type Line3: 16 elements", out)
    @test occursin("Reoriented 8 Line3 element(s)", out)      # top + left sides
    @test occursin("Loaded 16 curve elements (line3: 16, line2: 0) in 4 physical group(s).", out)
    _, out_rev = capture_stdout() do
      load_mesh(f; surface_dim=1, reverse_normals=true, verbose=true)
    end
    @test occursin("All normals reversed.", out_rev)
    rm(f)

    # no physical groups at all: one synthetic "default" group over every curve
    f = tempname() * ".msh"
    write_square_msh(f; physical=false, surface_group=false, save_all=true)
    m_def = @test_logs (:info, r"synthetic \"default\" group") load_mesh(f; surface_dim=1)
    @test collect(values(m_def.group_tags)) == ["default"]
    @test all(e -> points_inward(m_def, e), m_def.surface_elems)
    rm(f)

    # third-order lines (Gmsh type 26) are not a supported family
    f = tempname() * ".msh"
    write_square_msh(f; order=3)
    err = try load_mesh(f; surface_dim=1, verbose=false); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("No supported elements found", err.msg)
    @test occursin("Line2 (1), Line3 (8)", err.msg)
    @test occursin("26", err.msg)                              # lists the type it did find
    rm(f)

    # missing file, and surface_dim outside {1,2} on a file that does exist
    @test_throws ErrorException load_mesh(f; surface_dim=1)
    write_square_msh(f)
    @test_throws ErrorException load_mesh(f; surface_dim=3)
    rm(f)
  end

  @testset "orientation warnings when no adjacent surface can be found" begin
    # Curves with no surface entity at all.
    f = tempname() * ".msh"
    write_strips_msh(f)
    m = @test_logs (:warn, r"shares no nodes with any surface") (:warn, r"shares no nodes with any surface") load_mesh(f; surface_dim=1, verbose=false)
    rm(f)
    @test all(e -> curve_normal(m.coords, e)[2] > 0, m.surface_elems[m.group_elems[1]])   # untouched
    # A surface entity exists but only its boundary was meshed ...
    f = tempname() * ".msh"
    write_square_msh(f; generate_dim=1, surface_group=true)
    m1 = @test_logs (:warn, r"shares no nodes with any surface") match_mode=:any load_mesh(f; surface_dim=1, verbose=false)
    rm(f)
    @test length(m1.surface_elems) == 16
    # ... or the surface mesh was not saved (no physical group, no SaveAll):
    # normals are then left exactly as Gmsh wrote them (top/left point outward).
    f = tempname() * ".msh"
    write_square_msh(f)
    m2 = @test_logs (:warn, r"shares no nodes with any surface") match_mode=:any load_mesh(f; surface_dim=1, verbose=false)
    rm(f)
    @test count(e -> points_inward(m2, e), m2.surface_elems) == 8
  end

  @testset "wall shared by two surfaces: normal points at the nearer one" for
      shared_down in (false, true), small_first in (true, false)
    # The wall x = 1 borders the small room (centroid x ~ 0.5) and the large one
    # (x ~ 2); whichever order the loader meets them in, the nearer one wins, so
    # every normal is -x however the wall was drawn.
    f = tempname() * ".msh"
    write_two_rooms_msh(f; shared_down=shared_down, small_first=small_first)
    m = load_mesh(f; surface_dim=1, verbose=false)
    rm(f)
    @test length(m.surface_elems) > 1
    @test all(e -> curve_normal(m.coords, e)[1] < -0.99, m.surface_elems)
  end

  @testset "square enclosure: analytic view factors, Line$(order == 1 ? 2 : 3)" for order in (1, 2)
    f = tempname() * ".msh"
    tags = write_square_msh(f; order=order, surface_group=true)
    m = load_mesh(f; surface_dim=1, verbose=false)
    rm(f)
    bottom, right, top, left = (tags[k] for k in ("bottom", "right", "top", "left"))
    function check(r; tol)
      g = r.group_tags
      idx(t) = findfirst(==(t), g)
      @test isapprox(r.F_group[idx(bottom), idx(top)],   F_OPPOSITE; atol=tol)
      @test isapprox(r.F_group[idx(left),   idx(right)], F_OPPOSITE; atol=tol)
      @test isapprox(r.F_group[idx(bottom), idx(left)],  F_ADJACENT; atol=tol)
      @test isapprox(r.F_group[idx(bottom), idx(right)], F_ADJACENT; atol=tol)
      @test all(x -> isapprox(x, 1.0; atol=tol), vec(sum(r.F_group, dims=2)))
      @test check_reciprocity(r)
    end
    check(compute_view_factors(m; nquad=6, verbose=false);              tol=5e-3)
    # Duffy is a 3-D construct: for curve meshes it must fall back to plain quadrature
    r_q = compute_view_factors(m; nquad=6, verbose=false)
    r_d = compute_view_factors(m; nquad=6, use_duffy=true, verbose=false)
    @test r_d.F_elem == r_q.F_elem
    # Monte Carlo (2-D kernel) with the near-pair Duffy patch skipped for curves
    check(compute_view_factors(m; monte_carlo=true, n_samples=4000, rng=Xoshiro(3),
                                verbose=false);                         tol=0.03)
    # ... and the facing cull must not change the answer, only the work done
    r_nc = compute_view_factors(m; nquad=6, facing_cull=false, verbose=false)
    @test isapprox(r_nc.F_elem, r_q.F_elem; atol=1e-12)
  end

  @testset "2-D obstruction by a blocking segment" begin
    load(kind) = begin
      f = tempname() * ".msh"
      tags = write_strips_msh(f; blocker=kind)
      m = @test_logs (:warn, r"shares no nodes") match_mode=:any load_mesh(f; surface_dim=1, verbose=false)
      rm(f)
      (m, tags)
    end
    idx(r, t) = findfirst(==(t), r.group_tags)

    m0, t0 = load(:none)
    open_r = compute_view_factors(m0; nquad=8, verbose=false)
    @test isapprox(open_r.F_group[idx(open_r, t0["bottom"]), idx(open_r, t0["top"])],
                   F_OPPOSITE; atol=2e-3)

    for kind in (:full, :partial), mc in (false, true)
      m, t = load(kind)
      kw = mc ? (monte_carlo=true, n_samples=3000, rng=Xoshiro(9)) : (nquad=8,)
      r = compute_view_factors(m; kw..., radiating_groups=[t["bottom"], t["top"]],
                                obstruction_groups=[t["blocker"]], verbose=false)
      F = r.F_group[idx(r, t["bottom"]), idx(r, t["top"])]
      F_open = open_r.F_group[1, 2]
      if kind === :full
        # Every chord crosses the blocker, so F must be exactly 0 -- including
        # chords that cross at a blocker mesh node, which used to slip between
        # the two adjoining segments (measured F = 0.0081 before the fix).
        @test isapprox(F, 0.0; atol=1e-12)
      else
        @test 0.0 < F < 0.9 * F_open                        # some chords are cut, not all
      end
      @test check_reciprocity(r)
    end
  end

  @testset "options that curve meshes reject" begin
    f = tempname() * ".msh"
    write_square_msh(f, surface_group=true)
    m = load_mesh(f; surface_dim=1, verbose=false)
    rm(f)
    @test_throws ErrorException compute_view_factors(m; raytrace=true, verbose=false)
    @test_throws ErrorException compute_view_factors(m; backend=NotCPU(), verbose=false)
  end
end
