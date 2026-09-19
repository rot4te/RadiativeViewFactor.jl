# test/gpu_duffy_test.jl
#
# The Duffy transformation on the device (GPUDuffyKernels.jl), run on the
# KernelAbstractions CPU backend. The reference is the CPU implementation
# (DuffyKernel.jl): the device kernels reproduce its regions one for one, so in
# Float64 the two must agree to roundoff, not merely to quadrature accuracy.
# Also covered: the pair-list construction, all orderings of a shared edge,
# obstruction, the chunked launcher, Float32 type stability (Metal has no
# Float64) and the option plumbing.

using KernelAbstractions: CPU

# A backend the dispatch layer can allocate for, to check option forwarding
# without a GPU (the hook is stubbed in that testset).
struct DuffyStubBackend end
RadiativeViewFactor.Assembly._gpu_array_type(::DuffyStubBackend) = Array
RadiativeViewFactor.Assembly._gpu_float_type(::DuffyStubBackend) = Float64

# Floor (z = 0, normal +z) and wall (x = 0, normal +x) sharing the edge
# (0,0,0)-(0,1,0); perpendicular unit squares, F = 0.20004. `rot_a`/`rot_b`
# cyclically rotate each element's node list, which changes which local
# corners/edge the singularity classification sees while leaving the geometry
# and the normals alone.
function edge_pair_mesh(; rot_a=0, rot_b=0, baffle=false)
  coords = [0.0 1 1 0 0 0;
            0.0 0 1 1 0 1;
            0.0 0 0 0 1 1]
  a = circshift([1, 2, 3, 4], -rot_a)
  b = circshift([1, 4, 6, 5], -rot_b)
  elems = [SurfaceElement(a, 1, :quad4), SurfaceElement(b, 2, :quad4)]
  names = Dict(1 => "floor", 2 => "wall")
  if baffle   # the plane x = z bisects the dihedral angle: nothing can see across it
    coords = hcat(coords, [1.0, 1.0, 1.0], [1.0, 0.0, 1.0])
    push!(elems, SurfaceElement([1, 4, 7, 8], 3, :quad4))
    names[3] = "baffle"
  end
  return assemble_mesh(coords, elems, names)
end

# Floor and wall meeting only at the point (1,1,0) (wall normal -x, facing the floor).
function vertex_pair_mesh()
  coords = [0.0 1 1 0  1 1 1;
            0.0 0 1 1  2 2 1;
            0.0 0 0 0  0 1 1]
  elems = [SurfaceElement([1, 2, 3, 4], 1, :quad4),
           SurfaceElement([3, 7, 6, 5], 2, :quad4)]
  return assemble_mesh(coords, elems, Dict(1 => "floor", 2 => "wall"))
end

@testset "Duffy transformation on the device (CPU backend)" begin
  GPUAssembly = RadiativeViewFactor.GPUAssembly
  GPUKernels  = RadiativeViewFactor.GPUKernels
  GPUDuffy    = RadiativeViewFactor.GPUDuffyKernels
  D           = RadiativeViewFactor.DuffyKernel

  gpu(mesh, nquad; FloatT=Float64, kw...) =
    GPUAssembly.compute_view_factors_gpu(mesh, nquad, CPU(), FloatT, Array;
                                          verbose=false, kw...)

  @testset "touching_pairs = every pair singularity_type calls singular" begin
    cube4 = unit_cube_quad4()
    cube8 = to_second_order(cube4)
    # a Quad4 and a Quad8 sharing a node are not a Duffy pair (mixed families),
    # and triangles never are
    mixed = assemble_mesh(cube8.coords,
                          vcat(cube8.surface_elems[1:3],
                               [SurfaceElement(cube4.surface_elems[4].nodes, 1, :quad4)]),
                          Dict(1 => "cube"))
    for m in (cube4, cube8, mixed, to_tri3(cube4))
      brute = sort!(Tuple{Int,Int}[(i, j) for i in eachindex(m.surface_elems),
                              j in eachindex(m.surface_elems)
                              if i < j && D.singularity_type(m.surface_elems[i], m.surface_elems[j])[1] !== D.NONE])
      @test D.touching_pairs(m.surface_elems) == brute
    end
    @test length(D.touching_pairs(cube4.surface_elems)) == 12   # 6 faces, each edge-adjacent to 4
    @test isempty(D.touching_pairs(to_tri3(cube4).surface_elems))
    @test isempty(D.touching_pairs(facing_plates_quad4().surface_elems))
  end

  @testset "quadrature path: use_duffy reproduces the CPU Duffy values" for
      (name, cube) in (("Quad4", unit_cube_quad4()), ("Quad8", to_second_order(unit_cube_quad4()))),
      nq in (4, 10)
    r_cpu   = compute_view_factors(cube; nquad=nq, use_duffy=true, verbose=false)
    r_gpu   = gpu(cube, nq; use_duffy=true)
    r_plain = gpu(cube, nq)
    @test isapprox(r_gpu.F_elem, r_cpu.F_elem; atol=1e-12)
    @test check_reciprocity(r_gpu)
    # without the patch a closed cube's rows are ~24% over 1 (the shared-edge
    # singularity is unresolved); with it they close as far as nquad allows
    @test all(x -> x > 1.2, vec(sum(r_plain.F_elem, dims=2)))
    @test all(x -> isapprox(x, 1.0; atol = nq == 4 ? 2e-2 : 5e-4), vec(sum(r_gpu.F_elem, dims=2)))
  end

  @testset "all corner orderings of a shared edge, Quad4 and Quad8" begin
    F = Float64[]
    for rot_a in 0:3, rot_b in 0:3, second_order in (false, true)
      m = edge_pair_mesh(; rot_a=rot_a, rot_b=rot_b)
      second_order && (m = to_second_order(m))
      @test D.singularity_type(m.surface_elems[1], m.surface_elems[2])[1] === D.COMMON_EDGE
      r_gpu = gpu(m, 8; use_duffy=true)
      r_cpu = compute_view_factors(m; nquad=8, use_duffy=true, verbose=false)
      @test isapprox(r_gpu.F_elem, r_cpu.F_elem; atol=1e-12)
      push!(F, r_gpu.F_elem[1, 2])
    end
    # the same physical pair however its nodes are numbered
    @test maximum(F) - minimum(F) < 1e-5
    @test isapprox(F[1], 0.20004; atol=2e-3)      # perpendicular unit squares, common edge
  end

  @testset "vertex-only pair: device == CPU, Quad4 == Quad8" begin
    m4 = vertex_pair_mesh()
    m8 = to_second_order(m4)
    @test D.singularity_type(m4.surface_elems[1], m4.surface_elems[2])[1] === D.COMMON_VERTEX
    for m in (m4, m8)
      r_gpu = gpu(m, 8; use_duffy=true)
      r_cpu = compute_view_factors(m; nquad=8, use_duffy=true, verbose=false)
      @test r_gpu.F_elem[1, 2] > 0
      @test isapprox(r_gpu.F_elem, r_cpu.F_elem; atol=1e-12)
    end
    @test isapprox(gpu(m8, 8; use_duffy=true).F_elem, gpu(m4, 8; use_duffy=true).F_elem; atol=1e-9)
    # a vertex-touching pair the other way round (corner order swapped) agrees too
    swapped = assemble_mesh(m4.coords, reverse(m4.surface_elems) .|> (e -> SurfaceElement(e.nodes, 3 - e.group, e.family)),
                            Dict(1 => "floor", 2 => "wall"))
    @test isapprox(gpu(swapped, 8; use_duffy=true).F_elem[1, 2], gpu(m4, 8; use_duffy=true).F_elem[1, 2]; atol=1e-9)
  end

  @testset "obstruction inside the Duffy integrals" begin
    open_m = edge_pair_mesh()
    open_r = gpu(open_m, 6; use_duffy=true)
    @test open_r.F_elem[1, 2] > 0.19
    bm = edge_pair_mesh(baffle=true)
    r_gpu = gpu(bm, 6; use_duffy=true, obstruction_groups=[3])
    r_cpu = compute_view_factors(bm; nquad=6, use_duffy=true, obstruction_groups=[3], verbose=false)
    @test isapprox(r_cpu.F_elem[1, 2], 0.0; atol=1e-9)         # baffle cuts every chord
    @test isapprox(r_gpu.F_elem[1, 2], 0.0; atol=1e-9)
    @test isapprox(r_gpu.F_elem[2, 1], 0.0; atol=1e-9)
    # a vertex pair behind a baffle: same, through the 4-region integral
    vm = vertex_pair_mesh()
    vcoords = hcat(vm.coords, [1.0, 1.0, 0.5], [1.0, 2.0, 0.5], [0.0, 1.5, 0.5])   # nodes 8-10
    vbaffle = SurfaceElement([8, 9, 10], 3, :tri3)
    vb = assemble_mesh(vcoords, vcat(vm.surface_elems, [vbaffle]),
                       Dict(1 => "floor", 2 => "wall", 3 => "baffle"))
    rv_gpu = gpu(vb, 6; use_duffy=true, obstruction_groups=[3])
    rv_cpu = compute_view_factors(vb; nquad=6, use_duffy=true, obstruction_groups=[3], verbose=false)
    @test isapprox(rv_gpu.F_elem, rv_cpu.F_elem; atol=1e-7)
  end

  @testset "Monte Carlo path: the near-pair patch runs on the device" begin
    # In a cube every pair is opposite or edge-adjacent, hence within `factor`
    # of each other: the patch overwrites every entry, so the Monte Carlo result
    # is exactly the deterministic CPU Duffy result whatever the samples were.
    cube = unit_cube_quad4()
    r_cpu = compute_view_factors(cube; nquad=8, use_duffy=true, verbose=false)
    Random.seed!(3)
    r_gpu, out = capture_stdout() do
      GPUAssembly.compute_view_factors_gpu(cube, 8, CPU(), Float64, Array;
                                            monte_carlo=true, n_samples=50, verbose=true)
    end
    @test isapprox(r_gpu.F_elem, r_cpu.F_elem; atol=1e-12)
    @test occursin("Duffy transformation for 15 adjacent/near pair(s) (on device)", out)

    # Triangles, Quad8, mixed near pairs: no Duffy formula, so the patch falls
    # back to plain quadrature there -- again identical to the CPU.
    for m in (to_tri3(facing_plates_quad4()),
              to_second_order(facing_plates_quad4()),
              to_second_order(to_tri3(facing_plates_quad4())))
      Random.seed!(4)
      rg = gpu(m, 6; monte_carlo=true, n_samples=50)
      rc = compute_view_factors(m; nquad=6, verbose=false)
      @test isapprox(rg.F_elem, rc.F_elem; atol=1e-8)
    end

    # with obstruction and factor small enough that only touching pairs patch
    Random.seed!(5)
    bm = edge_pair_mesh(baffle=true)
    rb = gpu(bm, 6; monte_carlo=true, n_samples=200, obstruction_groups=[3], factor=0.5)
    @test isapprox(rb.F_elem[1, 2], 0.0; atol=1e-9)
  end

  @testset "launcher: chunking, empty lists, options that do nothing" begin
    cube = unit_cube_quad4()
    ga   = GPUKernels.build_gpu_arrays(cube, 6, Array, Float64)
    pairs = D.touching_pairs(cube.surface_elems)
    pi_, pj_ = Int32[p[1] for p in pairs], Int32[p[2] for p in pairs]
    raw_a, _ = GPUKernels.launch_vf_kernel!(ga, CPU())
    raw_b = copy(raw_a)
    GPUDuffy.launch_duffy_patch!(raw_a, ga, CPU(), pi_, pj_)
    for chunk in (1, 2, 5)
      raw_c = copy(raw_b)
      GPUDuffy.launch_duffy_patch!(raw_c, ga, CPU(), pi_, pj_; first_chunk=chunk)
      @test raw_c == raw_a
    end
    @test !(raw_b == raw_a)                                   # the patch did change something
    @test GPUDuffy.launch_duffy_patch!(raw_b, ga, CPU(), Int32[], Int32[]) === raw_b   # empty list: no-op

    # no touching pairs (facing plates): use_duffy changes nothing and launches nothing
    plates = facing_plates_quad4()
    r0, out0 = capture_stdout() do
      GPUAssembly.compute_view_factors_gpu(plates, 6, CPU(), Float64, Array; use_duffy=true, verbose=true)
    end
    @test r0.F_elem == gpu(plates, 6).F_elem
    @test !occursin("Duffy transformation for", out0)

    # raytrace ignores use_duffy on the device as it does on the CPU
    Random.seed!(6)
    rr = gpu(plates, 6; raytrace=true, n_rays=2000, use_duffy=true)
    @test isapprox(rr.F_group[1, 2], 0.19982; rtol=0.1)
  end

  @testset "Float32 (Metal): type-stable and close to Float64" begin
    for (m, label) in ((edge_pair_mesh(), "edge"), (vertex_pair_mesh(), "vertex"))
      ga = GPUKernels.build_gpu_arrays(m, 6, Array, Float32)
      @test eltype(ga.gl_pts01) === Float32 && eltype(ga.gl_wts01) === Float32
      dummy2 = zeros(Float32, 1, 1); dummy3 = zeros(Float32, 3, 3, 1)
      meta   = zeros(Int32, 1, 1);   idx = zeros(Int32, 1)
      args   = (ga.gl_pts01, ga.gl_wts01, Int32(1), Int32(2), dummy2, dummy2, meta, idx, dummy3, idx, false)
      fam = ga.elem_family[1]
      if label == "vertex"
        v = @inferred GPUDuffy._vertex_integral(ga.coords, ga.nodes_quad, fam, 1, 2, 3, 1, args...)
      else
        v = @inferred GPUDuffy._edge_integral(ga.coords, ga.nodes_quad, fam, 1, 2, 1, 4, 1, 2, args...)
      end
      @test v isa Float32
      @test v > 0
    end
    u = @inferred GPUDuffy._eval_uv(Float32.(edge_pair_mesh().coords),
                                     Int32[1 1; 2 4; 3 6; 4 5], Int8(2), 1, 0.3f0, 0.6f0)
    @test u[3] isa Float32
    for m in (unit_cube_quad4(), to_second_order(unit_cube_quad4()))
      r32 = gpu(m, 8; FloatT=Float32, use_duffy=true)
      r64 = gpu(m, 8; use_duffy=true)
      @test eltype(r32.A_elem) === Float64                    # promoted after the device
      @test isapprox(r32.F_elem, r64.F_elem; rtol=2e-4)
    end
  end

  @testset "the hook carries use_duffy, and a GPU backend no longer warns about it" begin
    Asm = RadiativeViewFactor.Assembly
    dummy = compute_view_factors(facing_plates_quad4(); verbose=false)
    seen = Ref{Any}(nothing)
    stub = (m, nquad, backend, FloatT, ArrayT; kw...) -> (seen[] = kw; dummy)
    original = Asm._GPU_HOOK_REF[]
    try
      Asm.register_gpu_hook!(stub)
      @test_logs compute_view_factors(facing_plates_quad4(); backend=DuffyStubBackend(),
                                       use_duffy=true, verbose=false)
      @test seen[][:use_duffy] === true
      compute_view_factors(facing_plates_quad4(); backend=DuffyStubBackend(), verbose=false)
      @test seen[][:use_duffy] === false
    finally
      Asm.register_gpu_hook!(original)
    end
  end
end
