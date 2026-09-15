# test/GPU_test.jl

@testset "GPU kernel (CPU backend) linear families" begin
  using KernelAbstractions: CPU
  GPUKernels = RadiativeViewFactor.GPUKernels
  MeshData = RadiativeViewFactor.MeshIO.MeshData

  coords = zeros(Float64, 3, 8)
  coords[:, 1]=[0, 0, 0];
  coords[:, 2]=[1, 0, 0];
  coords[:, 3]=[1, 1, 0];
  coords[:, 4]=[0, 1, 0]
  coords[:, 5]=[1, 0, 1];
  coords[:, 6]=[0, 0, 1];
  coords[:, 7]=[0, 1, 1];
  coords[:, 8]=[1, 1, 1]
  elems = [SurfaceElement([1, 2, 3, 4], 1, :quad4),
    SurfaceElement([5, 6, 7, 8], 2, :quad4)]
  mesh = MeshData(coords, elems, Dict(1=>"a", 2=>"b"),
    Dict(1=>[1], 2=>[2]), Dict{Int,Array{Float64,3}}(), 2)
  ga = GPUKernels.build_gpu_arrays(mesh, 6, Array, Float64)
  raw, area = GPUKernels.launch_vf_kernel!(ga, CPU())
  @test isapprox(Array(area)[1], 1.0; atol=1e-9)
  @test isapprox(Array(raw)[1, 2] / Array(area)[1], 0.19982; atol=2e-4)
end

@testset "GPU ray-shooting Monte Carlo (CPU backend)" begin
  using KernelAbstractions: CPU
  GPUBVH             = RadiativeViewFactor.GPUBVH
  GPUKernels         = RadiativeViewFactor.GPUKernels
  GPURayTraceKernels = RadiativeViewFactor.GPURayTraceKernels
  GPUAssembly        = RadiativeViewFactor.GPUAssembly
  MeshData           = RadiativeViewFactor.MeshIO.MeshData

  @testset "two facing unit squares: unbiased, low-level kernel calls" begin
    coords = [0.0 1 1 0  1 0 0 1;
              0.0 0 1 1  0 0 1 1;
              0.0 0 0 0  1 1 1 1]
    elems  = [SurfaceElement([1,2,3,4], 1, :quad4), SurfaceElement([5,6,7,8], 2, :quad4)]
    mesh   = MeshData(coords, elems, Dict(1=>"bottom",2=>"top"),
                      Dict(1=>[1],2=>[2]), Dict{Int,Array{Float64,3}}(), 2)

    ga = GPUKernels.build_gpu_arrays(mesh, 6, Array, Float64)
    area = Array(GPURayTraceKernels.launch_area_kernel!(ga, CPU()))
    @test isapprox(area[1], 1.0; atol=1e-9)

    scene = GPUBVH.build_flat_scene_bvh(mesh, Int[], Float64, Array)
    hits, escaped = GPURayTraceKernels.launch_raytrace_kernel!(
        ga, scene, CPU(); n_rays=200_000, seed=UInt64(1))
    hits = Array(hits); escaped = Array(escaped)
    # rtol chosen for ~3 sigma of headroom at n_rays=200_000 (predicted sigma
    # ~9e-4 absolute for p~0.2, i.e. ~0.45% relative) -- verified separately,
    # at n_rays=4e6, deviation stayed under 1 sigma with no systematic bias.
    @test isapprox(hits[1,2] / 200_000, 0.19982; rtol=1.5e-2)
    @test isapprox(hits[2,1] / 200_000, 0.19982; rtol=1.5e-2)
    @test all(x -> isapprox(x, 0.802; atol=0.01), escaped)   # open enclosure: F=0.19982 -> ~80% escapes
  end

  @testset "compute_view_factors_gpu(...; raytrace=true): reciprocity and closure" begin
    coords = [0.0 1 1 0  1 0 0 1;
              0.0 0 1 1  0 0 1 1;
              0.0 0 0 0  1 1 1 1]
    elems  = [SurfaceElement([1,2,3,4], 1, :quad4), SurfaceElement([5,6,7,8], 2, :quad4)]
    mesh   = MeshData(coords, elems, Dict(1=>"bottom",2=>"top"),
                      Dict(1=>[1],2=>[2]), Dict{Int,Array{Float64,3}}(), 2)
    r = GPUAssembly.compute_view_factors_gpu(mesh, 4, CPU(), Float64, Array;
                                              raytrace=true, n_rays=50_000, verbose=false)
    @test isapprox(r.F_group[1,2], 0.19982; rtol=1e-2)
    @test r.F_elem[1,2] * r.A_elem[1] == r.F_elem[2,1] * r.A_elem[2]   # exact by construction
  end

  @testset "radiating elements obstruct each other without obstruction_groups" begin
    # Same regression fixture as the CPU raytrace_test.jl: bottom/top plates
    # plus a third plate at the midplane, front wound toward bottom, never
    # listed in obstruction_groups.
    coords3 = [0.0 1 1 0  1 0 0 1  0.0 1 1 0;
               0.0 0 1 1  0 0 1 1  0.0 0 1 1;
               0.0 0 0 0  1 1 1 1  0.5 0.5 0.5 0.5]
    elems3  = [SurfaceElement([1,2,3,4], 1, :quad4), SurfaceElement([5,6,7,8], 2, :quad4),
               SurfaceElement([9,12,11,10], 3, :quad4)]
    mesh3   = MeshData(coords3, elems3, Dict(1=>"bottom",2=>"top",3=>"blocker"),
                       Dict(1=>[1],2=>[2],3=>[3]), Dict{Int,Array{Float64,3}}(), 2)
    r = GPUAssembly.compute_view_factors_gpu(mesh3, 6, CPU(), Float64, Array;
                                              raytrace=true, n_rays=50_000, verbose=false)
    @test isapprox(r.F_elem[1,2], 0.0; atol=1e-9)   # fully blocked
    raw_ref, Ai_ref = RadiativeViewFactor.ViewFactorKernel.element_pair_view_factor(
        coords3, elems3[1], elems3[3], 12, nothing, 2)
    @test isapprox(r.F_elem[1,3], raw_ref/Ai_ref; atol=0.01)   # ground truth ~0.415
  end

  @testset "closed cube: row-sum closure, chunked launch" begin
    cube_coords = [0.0 1 1 0 0 1 1 0;
                   0.0 0 1 1 0 0 1 1;
                   0.0 0 0 0 1 1 1 1]
    raw_faces = [ (1,2,3,4), (5,6,7,8), (1,2,6,5), (2,3,7,6), (3,4,8,7), (4,1,5,8) ]
    center = SVector(0.5, 0.5, 0.5)
    function orient_inward(coords, f)
      p(k) = SVector{3,Float64}(coords[1,k], coords[2,k], coords[3,k])
      a,b,c,d = f
      nvec = cross(p(b)-p(a), p(c)-p(a))
      centroid = (p(a)+p(b)+p(c)+p(d))/4
      dot(nvec, center - centroid) > 0 ? f : (a,d,c,b)
    end
    faces = [orient_inward(cube_coords, f) for f in raw_faces]
    cube_elems = [SurfaceElement(collect(f), 1, :quad4) for f in faces]
    cube_mesh = MeshData(cube_coords, cube_elems, Dict(1=>"cube"), Dict(1=>collect(1:6)),
                         Dict{Int,Array{Float64,3}}(), 2)
    # groupsize argument forces multiple chunks even for N=6, exercising the
    # row-block launcher's chunking logic rather than a single launch.
    r = GPUAssembly.compute_view_factors_gpu(cube_mesh, 6, CPU(), Float64, Array;
                                              raytrace=true, n_rays=30_000, verbose=false)
    rs = vec(sum(r.F_elem, dims=2))
    @test all(x -> isapprox(x, 1.0; atol=0.02), rs)
  end

  @testset "argument validation" begin
    coords = [0.0 1 1 0  1 0 0 1;
              0.0 0 1 1  0 0 1 1;
              0.0 0 0 0  1 1 1 1]
    elems  = [SurfaceElement([1,2,3,4], 1, :quad4), SurfaceElement([5,6,7,8], 2, :quad4)]
    mesh   = MeshData(coords, elems, Dict(1=>"bottom",2=>"top"),
                      Dict(1=>[1],2=>[2]), Dict{Int,Array{Float64,3}}(), 2)
    @test_throws ErrorException compute_view_factors(mesh; raytrace=true, monte_carlo=true, verbose=false)
    @test_throws ErrorException compute_view_factors(mesh; raytrace=true, self_vf=true, verbose=false)
  end

  @testset "type stability of GPU traversal/sampling helpers" begin
    GPUBVHm = RadiativeViewFactor.GPUBVH
    RTKm    = RadiativeViewFactor.GPURayTraceKernels
    coords = [0.0 1 1 0  1 0 0 1;
              0.0 0 1 1  0 0 1 1;
              0.0 0 0 0  1 1 1 1]
    elems  = [SurfaceElement([1,2,3,4], 1, :quad4), SurfaceElement([5,6,7,8], 2, :quad4)]
    mesh   = MeshData(coords, elems, Dict(1=>"bottom",2=>"top"),
                      Dict(1=>[1],2=>[2]), Dict{Int,Array{Float64,3}}(), 2)
    scene = GPUBVHm.build_flat_scene_bvh(mesh, Int[], Float64, Array)
    @test scene isa GPUBVHm.FlatBVH
    r = @inferred GPUBVHm.gpu_nearest_hit_bvh(
        scene.nodes_lo, scene.nodes_hi, scene.nodes_meta,
        scene.tri_idx, scene.tri_verts, scene.tri_group,
        0.5, 0.5, 0.0, 0.0, 0.0, 1.0, Int32(1))
    @test r isa Tuple{Bool,Int32,Bool}
    d = @inferred RTKm._cosine_dir_gpu(0.0, 0.0, 1.0, UInt32(12345))
    @test d isa Tuple{Float64,Float64,Float64,UInt32}
  end
end
