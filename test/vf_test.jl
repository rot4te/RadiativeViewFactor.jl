# test/vf_test.jl

@testset "vf_kernel" begin
  # Points directly facing: cos_i = cos_j = 1, r = 1 → K = 1/π
  xi = SVector(0.0, 0.0, 0.0)
  ni = SVector(0.0, 0.0, 1.0)
  xj = SVector(0.0, 0.0, 1.0)
  nj = SVector(0.0, 0.0, -1.0)
  K = RadiativeViewFactor.ViewFactorKernel.vf_kernel(xi, ni, xj, nj)
  @test isapprox(K, 1/π; atol=1e-14)

  # Same-direction normals (backs facing each other) → K = 0
  K_back = RadiativeViewFactor.ViewFactorKernel.vf_kernel(xi, ni, xj, ni)
  @test K_back == 0.0

  # Coincident points → K = 0
  K_same = RadiativeViewFactor.ViewFactorKernel.vf_kernel(xi, ni, xi, nj)
  @test K_same == 0.0
end

# ---------------------------------------------------------------------------
# Two parallel 1×1 unit squares at z=0 (normal +z) and z=1 (normal -z).
# The top plate's node ordering is mirrored in x so that cross(dxdξ, dxdη)
# points downward.
# ---------------------------------------------------------------------------
@testset "element_pair_view_factor parallel plates" begin
  coords = zeros(Float64, 3, 16)

  # Bottom plate (z=0, normal +z): nodes 1-8
  coords[:, 1] = [0.0, 0.0, 0.0]   # corner 1
  coords[:, 2] = [1.0, 0.0, 0.0]   # corner 2
  coords[:, 3] = [1.0, 1.0, 0.0]   # corner 3
  coords[:, 4] = [0.0, 1.0, 0.0]   # corner 4
  coords[:, 5] = [0.5, 0.0, 0.0]   # mid 1-2
  coords[:, 6] = [1.0, 0.5, 0.0]   # mid 2-3
  coords[:, 7] = [0.5, 1.0, 0.0]   # mid 3-4
  coords[:, 8] = [0.0, 0.5, 0.0]   # mid 4-1

  # Top plate (z=1, normal -z): nodes 9-16, x-mirrored ordering
  coords[:, 9] = [1.0, 0.0, 1.0]
  coords[:, 10] = [0.0, 0.0, 1.0]
  coords[:, 11] = [0.0, 1.0, 1.0]
  coords[:, 12] = [1.0, 1.0, 1.0]
  coords[:, 13] = [0.5, 0.0, 1.0]  # mid 9-10
  coords[:, 14] = [0.0, 0.5, 1.0]  # mid 10-11
  coords[:, 15] = [0.5, 1.0, 1.0]  # mid 11-12
  coords[:, 16] = [1.0, 0.5, 1.0]  # mid 12-9

  elem_bot = SurfaceElement([1, 2, 3, 4, 5, 6, 7, 8], 1, :quad)
  elem_top = SurfaceElement([9, 10, 11, 12, 13, 14, 15, 16], 2, :quad)

  integ_ij, Ai = element_pair_view_factor(coords, elem_bot, elem_top, 4, nothing)
  integ_ji, Aj = element_pair_view_factor(coords, elem_top, elem_bot, 4, nothing)

  @test isapprox(Ai, 1.0; atol=1e-10)
  @test isapprox(Aj, 1.0; atol=1e-10)
  @test integ_ij > 0
  # Raw integrals are symmetric (reciprocity kernel is symmetric)
  @test isapprox(integ_ij, integ_ji; atol=1e-12)
  # View factor F_ij = integ_ij / Ai; for two unit squares at distance 1
  # the analytical value is ≈ 0.1998; check it's in a plausible range
  Fij = integ_ij / Ai
  @test 0.1 < Fij < 1.0
end

# ---------------------------------------------------------------------------
@testset "check_reciprocity and check_closure" begin
  # Perfect two-surface enclosure: equal areas, F_12 = F_21 = 1
  vfr = ViewFactorResult(
    Float64[0 1; 1 0],
    Float64[1.0, 1.0],
    Float64[0 1; 1 0],
    Float64[1.0, 1.0],
    [1, 2],
    ["surface1", "surface2"],
  )
  @test check_reciprocity(vfr)
  @test check_closure(vfr)

  # Unequal areas with correct reciprocity: A1=2, A2=1, F_12=0.5, F_21=1
  vfr2 = ViewFactorResult(
    Float64[0 0.5; 1 0],
    Float64[2.0, 1.0],
    Float64[0 0.5; 1 0],
    Float64[2.0, 1.0],
    [1, 2],
    ["surface1", "surface2"],
  )
  @test check_reciprocity(vfr2)
end

# ---------------------------------------------------------------------------
@testset "Linear shape functions (quad4, tri3, line2)" begin
  # Quad4: partition of unity and nodal interpolation N_a(ξ_b,η_b)=δ_ab
  for (ξ, η) in [(-0.3, 0.7), (0.0, 0.0), (1.0, -1.0), (0.5, 0.5)]
    N, _, _ = quad4_shape(Float64(ξ), Float64(η))
    @test isapprox(sum(N), 1.0; atol=1e-14)
  end
  quad4_ref = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
  for (b, (ξ, η)) in enumerate(quad4_ref)
    N, _, _ = quad4_shape(ξ, η)
    for a in 1:4
      @test isapprox(N[a], a == b ? 1.0 : 0.0; atol=1e-14)
    end
  end

  # Line2: partition of unity and endpoint interpolation
  for ξ in (-0.4, 0.0, 0.8)
    N, _ = line2_shape(ξ)
    @test isapprox(sum(N), 1.0; atol=1e-14)
  end
  N, _ = line2_shape(-1.0);
  @test isapprox(N[1], 1.0; atol=1e-14)
  N, _ = line2_shape(1.0);
  @test isapprox(N[2], 1.0; atol=1e-14)

  # Tri3 (internal to ViewFactorKernel): partition of unity + vertex interp
  tri3_shape = RadiativeViewFactor.ViewFactorKernel.tri3_shape
  for (ξ, η) in [(0.2, 0.3), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    N, _, _ = tri3_shape(ξ, η)
    @test isapprox(sum(N), 1.0; atol=1e-14)
  end
end

# ---------------------------------------------------------------------------
@testset "Triangle quadrature normalization (Dunavant)" begin
  # Regression: every rule must integrate constant 1 over the reference
  # triangle to its area 1/2.  (The degree-7 13-point rule previously
  # summed to ~0.797.)
  tri_quad_rule = RadiativeViewFactor.ViewFactorKernel.tri_quad_rule
  for n in 1:8
    rule = tri_quad_rule(n)
    # n=3 (degree-5) tabulated constants carry a ~4e-7 imprecision; the
    # rest are exact to machine precision.
    @test isapprox(sum(rule.weights), 0.5; atol=1e-6)
  end
  # Degree-7 rule integrates a degree-7 polynomial (x^3 y^4) exactly.
  rule = tri_quad_rule(4)
  approx = sum(rule.weights[k] * rule.points[1, k]^3 * rule.points[2, k]^4
               for k in 1:size(rule.points, 2))
  # ∬_T x^3 y^4 dA = 3! 4! / (3+4+2)! = 6*24/362880 = 1/2520
  @test isapprox(approx, 1/2520; atol=1e-10)
end

# ---------------------------------------------------------------------------
@testset "Linear elements & cross-order consistency" begin
  # Two coaxial unit squares at distance 1; corner nodes only.
  coords = zeros(Float64, 3, 8)
  coords[:, 1]=[0, 0, 0];
  coords[:, 2]=[1, 0, 0];
  coords[:, 3]=[1, 1, 0];
  coords[:, 4]=[0, 1, 0]
  coords[:, 5]=[1, 0, 1];
  coords[:, 6]=[0, 0, 1];
  coords[:, 7]=[0, 1, 1];
  coords[:, 8]=[1, 1, 1]
  bot4 = SurfaceElement([1, 2, 3, 4], 1, :quad4)
  top4 = SurfaceElement([5, 6, 7, 8], 2, :quad4)
  raw4, A4 = element_pair_view_factor(coords, bot4, top4, 8, nothing)
  @test isapprox(A4, 1.0; atol=1e-10)
  F4 = raw4 / A4
  @test isapprox(F4, 0.19982; atol=2e-4)   # analytic ≈ 0.19982

  # Cross-order: flat Quad8 plates (corner nodes coincide) must agree with Quad4.
  coords8 = zeros(Float64, 3, 16)
  coords8[:, 1:4] = coords[:, 1:4]
  coords8[:, 5]=[0.5, 0, 0];
  coords8[:, 6]=[1, 0.5, 0];
  coords8[:, 7]=[0.5, 1, 0];
  coords8[:, 8]=[0, 0.5, 0]
  coords8[:, 9:12] = coords[:, 5:8]
  coords8[:, 13]=[0.5, 0, 1];
  coords8[:, 14]=[0, 0.5, 1];
  coords8[:, 15]=[0.5, 1, 1];
  coords8[:, 16]=[1, 0.5, 1]
  bot8 = SurfaceElement([1, 2, 3, 4, 5, 6, 7, 8], 1, :quad)
  top8 = SurfaceElement([9, 10, 11, 12, 13, 14, 15, 16], 2, :quad)
  raw8, A8 = element_pair_view_factor(coords8, bot8, top8, 8, nothing)
  @test isapprox(raw8/A8, F4; atol=1e-6)
end

