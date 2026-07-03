# test/type_stable_test.jl

@testset "Type stability (hot paths)" begin
  MCK = RadiativeViewFactor.MCKernel
  DFK = RadiativeViewFactor.DuffyKernel

  # Planar (z=0) node set: 1–4 corners, 5–8 edge midpoints. Serves every
  # family — Quad8 uses 1–8, Tri6 1–6, Quad4/Tri3 the corners, lines 1–3.
  coords = [0.0 1.0 1.0 0.0 0.5 1.0 0.5 0.0;
    0.0 0.0 1.0 1.0 0.0 0.5 1.0 0.5;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]

  elems = Dict(
    :quad4 => (SurfaceElement([1, 2, 3, 4], 1, :quad4), 2),
    :quad => (SurfaceElement([1, 2, 3, 4, 5, 6, 7, 8], 1, :quad), 2),
    :tri3 => (SurfaceElement([1, 2, 3], 1, :tri3), 2),
    :tri => (SurfaceElement([1, 2, 3, 4, 5, 6], 1, :tri), 2),
    :line2 => (SurfaceElement([1, 2], 1, :line2), 1),
    :line3 => (SurfaceElement([1, 2, 3], 1, :line3), 1),
  )

  NdType = Tuple{SVector{3,Float64},Float64}

  # precompute_quad and the ElementQuad-based pair integrator, all families
  for (fam, (el, md)) in elems
    qi = @inferred precompute_quad(coords, el, 4, md)
    @test qi isa ElementQuad
    @test (@inferred element_pair_view_factor(qi, qi, nothing, md)) isa
          Tuple{Float64,Float64}
    # the convenience coords-based method must infer too
    @test (@inferred element_pair_view_factor(coords, el, el, 4, nothing, md)) isa
          Tuple{Float64,Float64}
    @test (@inferred element_is_2d(el, md)) isa Bool
  end

  # Monte Carlo hot path: per-element sampling + the ElementSamples pairing
  rng = Random.MersenneTwister(1)
  for (fam, (el, md)) in elems
    si = @inferred MCK.sample_element_mc(coords, el, 16, rng)
    @test si isa MCK.ElementSamples
    @test (@inferred MCK.element_pair_view_factor_mc(si, si, nothing, md)) isa
          Tuple{Float64,Float64}
  end

  # Duffy path (Quad8) — falls back to standard quadrature for this
  # non-adjacent pair but must still infer a concrete return type
  q8 = elems[:quad][1]
  @test (@inferred DFK.element_pair_view_factor_duffy(coords, q8, q8, 4, nothing, 2)) isa
        Tuple{Float64,Float64}

  # Quadrature rule constructors
  @test (@inferred gauss_legendre_1d(4)) isa Tuple{Vector{Float64},Vector{Float64}}
  @test (@inferred gauss_legendre_1d(8)) isa Tuple{Vector{Float64},Vector{Float64}}  # Golub–Welsch + cache
  @test (@inferred gauss_legendre_2d(4)) isa QuadRule2D

  # Geometry kernels
  @test (@inferred quad8_normal_and_area_element(coords, [1, 2, 3, 4, 5, 6, 7, 8], 0.1, 0.2)) isa NdType
  @test (@inferred quad4_normal_and_area_element(coords, [1, 2, 3, 4], 0.1, 0.2)) isa NdType
  @test (@inferred tri3_normal_and_area_element(coords, [1, 2, 3], 0.1, 0.2)) isa NdType
  @test (@inferred line2_normal_and_length_element(coords, [1, 2], 0.1)) isa NdType
  @test (@inferred line3_normal_and_length_element(coords, [1, 2, 3], 0.1)) isa NdType
end
