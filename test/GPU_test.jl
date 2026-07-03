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
