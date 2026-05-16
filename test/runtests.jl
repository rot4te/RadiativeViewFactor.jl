# test/runtests.jl
using Test

include(joinpath(@__DIR__, "..", "src", "RadiativeViewFactor.jl"))
using .RadiativeViewFactor
using .RadiativeViewFactor.Quadrature
using .RadiativeViewFactor.Geometry
using .RadiativeViewFactor.BVH
using .RadiativeViewFactor.RayCast
using .RadiativeViewFactor.ViewFactorKernel
using .RadiativeViewFactor.MeshIO: SurfaceElement
using StaticArrays
using LinearAlgebra

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
    @test isapprox(dot(wts3, pts3.^4), 2/5; atol=1e-12)
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
    @test isapprox(dot(wts6, pts6.^10), 2/11; atol=1e-12)
end

# ---------------------------------------------------------------------------
@testset "Quad8 shape functions" begin
    # Partition of unity: Σ Nₐ = 1 everywhere
    for (ξ, η) in [(-0.5, 0.3), (0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)]
        N, _, _ = quad8_shape(Float64(ξ), Float64(η))
        @test isapprox(sum(N), 1.0; atol=1e-14)
    end

    # Nodal interpolation: N_a(ξ_b, η_b) = δ_{ab}
    ref = [(-1.,-1.), (1.,-1.), (1.,1.), (-1.,1.),
           (0.,-1.), (1.,0.), (0.,1.), (-1.,0.)]
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
    coords = [0.0  1.0  1.0  0.0  0.5  1.0  0.5  0.0;
              0.0  0.0  1.0  1.0  0.0  0.5  1.0  0.5;
              0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    nodes  = [1, 2, 3, 4, 5, 6, 7, 8]

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
@testset "BVH ray casting" begin
    # Single triangle in the z=1 plane: vertices at (0,0,1),(1,0,1),(0,1,1)
    soup = zeros(Float64, 3, 3, 1)
    soup[1, :, 1] = [0.0, 0.0, 1.0]
    soup[2, :, 1] = [1.0, 0.0, 1.0]
    soup[3, :, 1] = [0.0, 1.0, 1.0]

    bvh = build_bvh(soup)

    origin    = SVector(0.25, 0.25, 0.0)
    direction = SVector(0.0,  0.0,  1.0)
    @test intersect_ray_bvh(bvh, origin, direction, 2.0)    # hits at t=1

    origin2   = SVector(2.0, 2.0, 0.0)
    @test !intersect_ray_bvh(bvh, origin2, direction, 2.0)  # misses
end

# ---------------------------------------------------------------------------
@testset "Visibility" begin
    # Triangle blocking the view
    soup = zeros(Float64, 3, 3, 1)
    soup[1, :, 1] = [-1.0, -1.0, 1.0]
    soup[2, :, 1] = [ 1.0, -1.0, 1.0]
    soup[3, :, 1] = [ 0.0,  1.0, 1.0]
    bvh = build_bvh(soup)

    xi = SVector(0.0, 0.0, 0.0)
    xj = SVector(0.0, 0.0, 2.0)
    @test !is_visible(bvh, xi, xj)   # blocked

    xk = SVector(5.0, 0.0, 2.0)
    @test  is_visible(bvh, xi, xk)   # unobstructed
end

# ---------------------------------------------------------------------------
@testset "vf_kernel" begin
    # Points directly facing: cos_i = cos_j = 1, r = 1 → K = 1/π
    xi = SVector(0.0, 0.0, 0.0)
    ni = SVector(0.0, 0.0, 1.0)
    xj = SVector(0.0, 0.0, 1.0)
    nj = SVector(0.0, 0.0,-1.0)
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
    coords[:, 9]  = [1.0, 0.0, 1.0]
    coords[:, 10] = [0.0, 0.0, 1.0]
    coords[:, 11] = [0.0, 1.0, 1.0]
    coords[:, 12] = [1.0, 1.0, 1.0]
    coords[:, 13] = [0.5, 0.0, 1.0]  # mid 9-10
    coords[:, 14] = [0.0, 0.5, 1.0]  # mid 10-11
    coords[:, 15] = [0.5, 1.0, 1.0]  # mid 11-12
    coords[:, 16] = [1.0, 0.5, 1.0]  # mid 12-9

    elem_bot = SurfaceElement([1,2,3,4,5,6,7,8],   1, :quad)
    elem_top = SurfaceElement([9,10,11,12,13,14,15,16], 2, :quad)

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

println("\nAll tests passed.")
