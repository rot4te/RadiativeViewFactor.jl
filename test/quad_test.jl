# test/quad_test.jl

# ---------------------------------------------------------------------------

@testset "Quadrature pre-tabulated" begin
  for n in 1:5
    pts, wts = gauss_legendre_1d(n)
    @test length(pts) == n
    @test isapprox(sum(wts), 2.0; atol=1e-12)

    rule = gauss_legendre_2d(n)
    @test size(rule.points, 2) == n^2
    @test isapprox(sum(rule.weights), 4.0; atol=1e-12)
  end

  # n-point GL integrates polynomials of degree ≤ 2n-1 exactly
  # ∫₋₁¹ x⁴ dx = 2/5  (requires n ≥ 3)
  pts3, wts3 = gauss_legendre_1d(3)
  @test isapprox(dot(wts3, pts3 .^ 4), 2/5; atol=1e-12)
end

# ---------------------------------------------------------------------------

@testset "Quadrature Golub-Welsch (n > 5)" begin
  for n in 6:8
    pts, wts = gauss_legendre_1d(n)
    @test length(pts) == n
    @test isapprox(sum(wts), 2.0; atol=1e-12)

    rule = gauss_legendre_2d(n)
    @test size(rule.points, 2) == n^2
    @test isapprox(sum(rule.weights), 4.0; atol=1e-12)
  end

  # ∫₋₁¹ x^10 dx = 2/11  (degree 10, requires n ≥ 6: 2*6-1=11 ≥ 10)
  pts6, wts6 = gauss_legendre_1d(6)
  @test isapprox(dot(wts6, pts6 .^ 10), 2/11; atol=1e-12)
end

# ---------------------------------------------------------------------------

@testset "Quad8 shape functions" begin
  # Partition of unity: Σ Nₐ = 1 everywhere
  for (ξ, η) in [(-0.5, 0.3), (0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)]
    N, _, _ = quad8_shape(Float64(ξ), Float64(η))
    @test isapprox(sum(N), 1.0; atol=1e-14)
  end

  # Nodal interpolation: N_a(ξ_b, η_b) = δ_{ab}
  ref = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.),
    (0., -1.), (1., 0.), (0., 1.), (-1., 0.)]
  for (a, (ξ_a, η_a)) in enumerate(ref)
    N, _, _ = quad8_shape(ξ_a, η_a)
    for b in 1:8
      @test isapprox(N[b], a==b ? 1.0 : 0.0; atol=1e-13)
    end
  end
end

# ---------------------------------------------------------------------------

# Quad8 node layout (Gmsh convention):
#
#   4---7---3
#   |       |
#   8       6
#   |       |
#   1---5---2
#
# For a 1×1 flat square in z=0: node 1=(0,0,0), 2=(1,0,0), 3=(1,1,0),
# 4=(0,1,0), 5=(0.5,0,0), 6=(1,0.5,0), 7=(0.5,1,0), 8=(0,0.5,0).

# ---------------------------------------------------------------------------

@testset "Quad8 physical geometry" begin
  coords = [0.0 1.0 1.0 0.0 0.5 1.0 0.5 0.0;
    0.0 0.0 1.0 1.0 0.0 0.5 1.0 0.5;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
  nodes = [1, 2, 3, 4, 5, 6, 7, 8]

  # Reference centre (ξ=η=0) maps to physical centre (0.5, 0.5, 0)
  x = quad8_physical_point(coords, nodes, 0.0, 0.0)
  @test isapprox(x, SVector(0.5, 0.5, 0.0); atol=1e-14)

  # Normal at centre is (0,0,1); area element = 0.25 (Jacobian of [-1,1]²→[0,1]²)
  n̂, dA = quad8_normal_and_area_element(coords, nodes, 0.0, 0.0)
  @test isapprox(n̂, SVector(0.0, 0.0, 1.0); atol=1e-14)
  @test isapprox(dA, 0.25; atol=1e-14)

  # ∫∫ dA over the 1×1 square = 1
  A = element_area(coords, nodes; nquad=4)
  @test isapprox(A, 1.0; atol=1e-12)
end

# ---------------------------------------------------------------------------
