# test/ray_test.jl

@testset "BVH ray casting" begin
  # Single triangle in the z=1 plane: vertices at (0,0,1),(1,0,1),(0,1,1)
  soup = zeros(Float64, 3, 3, 1)
  soup[1, :, 1] = [0.0, 0.0, 1.0]
  soup[2, :, 1] = [1.0, 0.0, 1.0]
  soup[3, :, 1] = [0.0, 1.0, 1.0]

  bvh = build_bvh(soup)

  origin = SVector(0.25, 0.25, 0.0)
  direction = SVector(0.0, 0.0, 1.0)
  @test intersect_ray_bvh(bvh, origin, direction, 2.0)    # hits at t=1

  origin2 = SVector(2.0, 2.0, 0.0)
  @test !intersect_ray_bvh(bvh, origin2, direction, 2.0)  # misses
end

# ---------------------------------------------------------------------------

@testset "Visibility" begin
  # Triangle blocking the view
  soup = zeros(Float64, 3, 3, 1)
  soup[1, :, 1] = [-1.0, -1.0, 1.0]
  soup[2, :, 1] = [1.0, -1.0, 1.0]
  soup[3, :, 1] = [0.0, 1.0, 1.0]
  bvh = build_bvh(soup)

  xi = SVector(0.0, 0.0, 0.0)
  xj = SVector(0.0, 0.0, 2.0)
  @test !is_visible(bvh, xi, xj)   # blocked

  xk = SVector(5.0, 0.0, 2.0)
  @test is_visible(bvh, xi, xk)   # unobstructed
end

# ---------------------------------------------------------------------------
