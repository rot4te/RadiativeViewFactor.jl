# test/gpu_families_test.jl
#
# The original GPU tests (GPU_test.jl) only ever fed the kernels Quad4
# meshes with no obstruction. This file runs every element family
# (Quad4/Tri3/Quad8/Tri6), the obstruction BVH, the pair-area Monte Carlo
# kernel and the error/warning paths of the GPU assembly on the KernelAbstractions
# CPU backend, and cross-checks the results against the analytic value and
# the independent CPU implementation.

using KernelAbstractions: CPU

@testset "GPU kernels (CPU backend): families, obstruction, Monte Carlo" begin
  GPUAssembly = RadiativeViewFactor.GPUAssembly
  GPUKernels  = RadiativeViewFactor.GPUKernels
  F_ANALYTIC  = 0.19982   # opposing unit squares one unit apart

  gpu(mesh; nquad=6, kw...) =
    GPUAssembly.compute_view_factors_gpu(mesh, nquad, CPU(), Float64, Array;
                                          verbose=false, kw...)

  plates = Dict(
    :quad4 => facing_plates_quad4(),
    :tri3  => to_tri3(facing_plates_quad4()),
    :quad8 => to_second_order(facing_plates_quad4()),
    :tri6  => to_second_order(to_tri3(facing_plates_quad4())),
  )

  @testset "_dunavant_rule: weights and polynomial exactness" begin
    exact(a, b) = factorial(a) * factorial(b) / factorial(a + b + 2)
    for (nq, degree) in ((1, 1), (2, 2), (3, 5), (4, 7))
      r = GPUKernels._dunavant_rule(nq, Float64)
      @test isapprox(sum(r.weights), 0.5; atol=1e-13)
      for a in 0:degree, b in 0:degree-a
        q = sum(r.weights[k] * r.points[1, k]^a * r.points[2, k]^b
                for k in eachindex(r.weights))
        @test isapprox(q, exact(a, b); atol=1e-13)
      end
    end
  end

  @testset "quadrature kernel, $fam plates: analytic value and CPU agreement" for fam in
      (:quad4, :tri3, :quad8, :tri6)
    mesh = plates[fam]
    r_gpu = gpu(mesh)
    @test isapprox(r_gpu.F_group[1, 2], F_ANALYTIC; atol=2e-3)
    r_cpu = compute_view_factors(mesh; nquad=6, verbose=false)
    @test isapprox(r_gpu.F_elem, r_cpu.F_elem; atol=1e-8)
    @test isapprox(r_gpu.A_elem, r_cpu.A_elem; rtol=1e-12)
  end

  @testset "quadrature kernel: mixed Quad4 / Tri3 / Quad8 / Tri6 in one mesh" begin
    # bottom plate Quad4, top plate two Tri6: every (family_i, family_j)
    # branch of the pair kernel is taken for a cross-family pair.
    q4 = plates[:quad4]
    t6 = plates[:tri6]
    nb = size(q4.coords, 2)
    coords = hcat(q4.coords, t6.coords[:, 5:end])   # top-plate nodes only
    top = [SurfaceElement(e.nodes .- 4 .+ nb, 2, :tri) for e in t6.surface_elems if e.group == 2]
    mesh = assemble_mesh(coords, vcat(q4.surface_elems[1:1], top),
                         Dict(1 => "bottom", 2 => "top"))
    r_gpu = gpu(mesh)
    r_cpu = compute_view_factors(mesh; nquad=6, verbose=false)
    @test isapprox(r_gpu.F_elem, r_cpu.F_elem; atol=1e-8)
    @test isapprox(r_gpu.F_group[1, 2], F_ANALYTIC; atol=2e-3)
  end

  @testset "quadrature kernel: low-order triangle rules (nquad = 1, 2, 3)" begin
    # 1-, 3- and 7-point rules: accuracy improves with rule order (measured:
    # 0.258, 0.1975, 0.19991 against 0.19982 on these two right triangles).
    tri = plates[:tri3]
    e1 = abs(gpu(tri; nquad=1).F_group[1, 2] - F_ANALYTIC)
    e2 = abs(gpu(tri; nquad=2).F_group[1, 2] - F_ANALYTIC)
    e3 = abs(gpu(tri; nquad=3).F_group[1, 2] - F_ANALYTIC)
    @test e3 < e2 < e1
    @test e2 < 5e-3
    @test e3 < 5e-4
  end

  @testset "obstruction BVH in the quadrature kernel" begin
    open_r = gpu(facing_plates_quad4())
    # Blocker covering only the middle: some rays pass beside it, some don't,
    # so both the "hit" and the "exhaust leaf / miss link" traversal paths run.
    part = facing_plates_with_blocker(half_width=0.25)
    r_gpu = gpu(part; obstruction_groups=[3])
    r_cpu = compute_view_factors(part; nquad=6, obstruction_groups=[3], verbose=false)
    @test 0.0 < r_gpu.F_elem[1, 2] < open_r.F_elem[1, 2]
    @test isapprox(r_gpu.F_elem, r_cpu.F_elem; atol=1e-6)

    # Same blocker built from a 6x6 grid: the BVH now has interior nodes, and
    # (identical geometry) the answer must not change.
    grid = facing_plates_with_grid_blocker(half_width=0.25)
    r_grid = gpu(grid; obstruction_groups=[3])
    @test isapprox(r_grid.F_elem[1:2, 1:2], r_gpu.F_elem[1:2, 1:2]; atol=1e-6)

    # Blocker as large as the plates: bottom cannot see top at all.
    full = facing_plates_with_blocker(half_width=0.5)
    r_full = gpu(full; obstruction_groups=[3])
    @test isapprox(r_full.F_elem[1, 2], 0.0; atol=1e-12)
    @test isapprox(r_full.F_elem[2, 1], 0.0; atol=1e-12)

    # An obstruction tag with no geometry warns and proceeds unobstructed.
    r_none = @test_logs (:warn, r"no triangle geometry") gpu(facing_plates_quad4();
                                                            obstruction_groups=[99])
    @test isapprox(r_none.F_elem, open_r.F_elem; atol=1e-12)
  end

  @testset "Monte Carlo kernel, $fam plates: unbiased, reciprocal" for fam in
      (:quad4, :tri3, :quad8, :tri6)
    Random.seed!(20260919)
    r = gpu(plates[fam]; monte_carlo=true, n_samples=40_000)
    @test isapprox(r.F_group[1, 2], F_ANALYTIC; rtol=0.03)
    @test check_reciprocity(r)
  end

  @testset "Monte Carlo kernel: obstruction, facing_cull=false, near-pair patch" begin
    Random.seed!(7)
    full = facing_plates_with_blocker(half_width=0.5)
    r = gpu(full; monte_carlo=true, n_samples=2000, obstruction_groups=[3],
            facing_cull=false)
    @test isapprox(r.F_elem[1, 2], 0.0; atol=1e-12)

    # Closed cube: every face pair is opposite or edge-adjacent, so the Duffy
    # patch after the GPU Monte Carlo pass is what makes the row sums close.
    Random.seed!(11)
    (r_cube, out) = capture_stdout() do
      GPUAssembly.compute_view_factors_gpu(unit_cube_quad4(), 8, CPU(), Float64, Array;
                                            monte_carlo=true, n_samples=4000, verbose=true)
    end
    @test all(x -> isapprox(x, 1.0; atol=0.03), vec(sum(r_cube.F_elem, dims=2)))
    @test occursin("Patching adjacent-pair singularities", out)
  end

  @testset "verbose progress output" begin
    mesh = facing_plates_with_blocker(half_width=0.25)
    for (kw, marker) in ((NamedTuple(), "nquad=6"),
                         ((monte_carlo=true, n_samples=500), "(Monte Carlo)"),
                         ((raytrace=true, n_rays=2000), "(ray-shooting Monte Carlo)"))
      r, out = capture_stdout() do
        GPUAssembly.compute_view_factors_gpu(mesh, 6, CPU(), Float64, Array;
                                              obstruction_groups=[3], verbose=true, kw...)
      end
      @test r isa RadiativeViewFactor.ViewFactorResult
      @test occursin("GPU compute_view_factors: 3 elements", out)
      @test occursin(marker, out)
      @test occursin("Row-sum check (group level)", out)
    end
    _, out = capture_stdout() do
      GPUAssembly.compute_view_factors_gpu(mesh, 6, CPU(), Float64, Array;
                                            obstruction_groups=[3], verbose=true)
    end
    @test occursin("Building obstruction BVH", out)
    @test occursin("kernel done", out)
  end

  @testset "ray-shooting kernel, $fam plates" for fam in (:tri3, :quad8, :tri6)
    Random.seed!(5)
    r = gpu(plates[fam]; raytrace=true, n_rays=60_000)
    @test isapprox(r.F_group[1, 2], F_ANALYTIC; rtol=0.03)
    @test r.F_elem[1, 2] * r.A_elem[1] ≈ r.F_elem[2, 1] * r.A_elem[2]
  end

  @testset "degenerate (zero-area) element is reported, not silently NaN'd" begin
    m = facing_plates_quad4()
    coords = hcat(m.coords, [0.3, 0.3, 0.3])            # one extra node
    bad = SurfaceElement([9, 9, 9, 9], 3, :quad4)        # all four corners coincide
    mesh = assemble_mesh(coords, vcat(m.surface_elems, [bad]),
                         Dict(1 => "bottom", 2 => "top", 3 => "degenerate"))
    @test_throws ErrorException gpu(mesh)
    @test_throws ErrorException gpu(mesh; monte_carlo=true, n_samples=100)
    @test_throws ErrorException gpu(mesh; raytrace=true, n_rays=100)
  end
end
