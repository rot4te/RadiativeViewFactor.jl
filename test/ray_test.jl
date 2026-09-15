# test/ray_test.jl

@testset "BVH ray casting" begin
  # Single triangle in the z=1 plane: vertices at (0,0,1),(1,0,1),(0,1,1)
  # Soup layout is (xyz, vertex, tri), matching MeshIO._build_group_obs_soups.
  soup = zeros(Float64, 3, 3, 1)
  soup[:, 1, 1] = [0.0, 0.0, 1.0]
  soup[:, 2, 1] = [1.0, 0.0, 1.0]
  soup[:, 3, 1] = [0.0, 1.0, 1.0]

  bvh = build_bvh(soup)

  origin = SVector(0.25, 0.25, 0.0)
  direction = SVector(0.0, 0.0, 1.0)
  @test intersect_ray_bvh(bvh, origin, direction, 2.0)    # hits at t=1

  origin2 = SVector(2.0, 2.0, 0.0)
  @test !intersect_ray_bvh(bvh, origin2, direction, 2.0)  # misses
end

# ---------------------------------------------------------------------------

@testset "Visibility" begin
  # Triangle blocking the view — soup layout is (xyz, vertex, tri)
  soup = zeros(Float64, 3, 3, 1)
  soup[:, 1, 1] = [-1.0, -1.0, 1.0]
  soup[:, 2, 1] = [1.0, -1.0, 1.0]
  soup[:, 3, 1] = [0.0, 1.0, 1.0]
  bvh = build_bvh(soup)

  xi = SVector(0.0, 0.0, 0.0)
  xj = SVector(0.0, 0.0, 2.0)
  @test !is_visible(bvh, xi, xj)   # blocked

  xk = SVector(5.0, 0.0, 2.0)
  @test is_visible(bvh, xi, xk)   # unobstructed
end

# ---------------------------------------------------------------------------

@testset "Full blocker leaves no leaks (CPU + GPU traversal)" begin
  # Regression test for the (xyz, vertex, tri) soup layout: a plane at z=0.5
  # spanning [-1,2]² (two triangles) must block *every* ray between two unit
  # plates at z=0 and z=1.  A transposed vertex read leaks 25–70% of rays.
  GPUBVH = RadiativeViewFactor.GPUBVH

  bl=[-1.0,-1,0.5]; br=[2.0,-1,0.5]; tr=[2.0,2,0.5]; tl=[-1.0,2,0.5]
  soup = zeros(3, 3, 2)
  soup[:,1,1]=bl; soup[:,2,1]=br; soup[:,3,1]=tr
  soup[:,1,2]=bl; soup[:,2,2]=tr; soup[:,3,2]=tl

  bvh = build_bvh(soup)
  fb  = GPUBVH.build_flat_bvh(bvh, fill(Int32(9), 2), Float64, Array)

  rng = MersenneTwister(11)
  leaks_cpu = 0; leaks_gpu = 0
  for _ in 1:2000
    x1 = SVector(rand(rng), rand(rng), 0.0)
    x2 = SVector(rand(rng), rand(rng), 1.0)
    d  = x2 - x1;  L = sqrt(sum(abs2, d));  dh = d / L
    intersect_ray_bvh(bvh, x1, dh, L) || (leaks_cpu += 1)
    GPUBVH.gpu_intersect_bvh(fb.nodes_lo, fb.nodes_hi, fb.nodes_meta,
                             fb.tri_idx, fb.tri_verts, fb.tri_group,
                             x1[1], x1[2], x1[3], dh[1], dh[2], dh[3],
                             L, Int32(0), Int32(0)) || (leaks_gpu += 1)
  end
  @test leaks_cpu == 0
  @test leaks_gpu == 0

  # Same geometry through the MeshIO production path: group_tri_soup layout
  # must match what the traversals read.
  coords = [0.0 1 1 0  1 0 0 1;
            0.0 0 1 1  0 0 1 1;
            0.0 0 0 0  1 1 1 1]
  elems  = [SurfaceElement([1,2,3,4], 1, :quad4),
            SurfaceElement([5,6,7,8], 2, :quad4)]
  mesh = RadiativeViewFactor.MeshIO.MeshData(coords, elems,
      Dict(1=>"a", 2=>"b"), Dict(1=>[1], 2=>[2]),
      Dict(3=>soup), 2)
  r = compute_view_factors(mesh; obstruction_groups=[3],
                           monte_carlo=true, n_samples=2000,
                           rng=MersenneTwister(1), verbose=false)
  @test r.F_group[1, 2] == 0.0
end

# ---------------------------------------------------------------------------

@testset "MC pair estimator is unbiased for coarse elements" begin
  # Regression test for the stratum-pairing correlation: a single pair of
  # directly-opposed unit plates.  Index-paired strata sample only the
  # diagonal stratum blocks and give a ~2.4% systematic error that does not
  # shrink with n_samples; the shifted pairing must land within MC noise.
  MCK = RadiativeViewFactor.MCKernel

  coords = [0.0 1 1 0  1 0 0 1;
            0.0 0 1 1  0 0 1 1;
            0.0 0 0 0  1 1 1 1]
  ei = SurfaceElement([1,2,3,4], 1, :quad4)
  ej = SurfaceElement([5,6,7,8], 2, :quad4)

  rng = MersenneTwister(2)
  raw, Ai = MCK.element_pair_view_factor_mc(coords, ei, ej, 200_000,
                                            nothing, 2, rng)
  @test isapprox(raw / Ai, 0.19982; rtol=5e-3)
end
