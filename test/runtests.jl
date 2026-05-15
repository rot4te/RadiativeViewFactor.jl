# test/runtests.jl
using Test

include(joinpath(@__DIR__, "..", "src", "RadiativeViewFactor.jl"))
using .RadiativeViewFactor
using .RadiativeViewFactor.Quadrature
using .RadiativeViewFactor.Geometry
using .RadiativeViewFactor.BVH
using .RadiativeViewFactor.RayCast
using StaticArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
@testset "Quadrature" begin
    for n in 1:5
        pts, wts = gauss_legendre_1d(n)
        @test length(pts) == n
        @test isapprox(sum(wts), 2.0; atol=1e-12)

        rule = gauss_legendre_2d(n)
        @test size(rule.points, 2) == n^2
        @test isapprox(sum(rule.weights), 4.0; atol=1e-12)
    end

    # Exactness: n-point GL integrates polynomials of degree ≤ 2n-1 exactly
    # ∫₋₁¹ x⁴ dx = 2/5 = 0.4  (requires n≥3 for degree 4)
    pts3, wts3 = gauss_legendre_1d(3)
    @test isapprox(dot(wts3, pts3.^4), 2/5; atol=1e-12)
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
    # Same triangle blocking the view
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

println("\nAll tests passed.")
