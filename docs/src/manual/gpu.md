# GPU Backends

RadiativeViewFactor.jl supports NVIDIA GPUs via CUDA.jl and Apple Silicon
GPUs via Metal.jl. Both backends are optional weak dependencies loaded
automatically when the user loads the corresponding package.

## NVIDIA (CUDA)

```julia
using CUDA
using RadiativeViewFactor

result = compute_view_factors(mesh; nquad=4, backend=CUDABackend())
```

CUDA runs in **Float64** throughout. All features except `use_duffy` and
`surface_dim=1` are supported.

## Apple Silicon (Metal)

```julia
using Metal
using RadiativeViewFactor

result = compute_view_factors(mesh; nquad=4, backend=MetalBackend())
```

Metal runs in **Float32** internally; results are promoted to Float64 before
aggregation and returned as `Float64`. Apple GPUs do not support Float64
natively. For geometries with near-grazing element pairs or coordinates
spanning many orders of magnitude, Float32 rounding (~1e-7 relative) may
introduce small errors in individual element-pair values that are generally
negligible after group-level aggregation.

## Monte Carlo (pair-area) on GPU

Both backends support pair-area Monte Carlo integration. Each GPU thread uses
an independent xorshift64 pseudo-random stream derived from a per-call global
seed and the thread index. The `rng` keyword is ignored on GPU backends.

```julia
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               backend=CUDABackend())
```

## Monte Carlo (ray-shooting) on GPU

Ray-shooting Monte Carlo (`raytrace=true`) has its own GPU kernel, one thread
per *element* rather than per element pair — each thread shoots `n_rays`
cosine-weighted rays from its element into a scene BVH covering the whole
radiating mesh, so obstruction among radiating elements is automatic:

```julia
result = compute_view_factors(mesh; raytrace=true, n_rays=10000,
                               backend=CUDABackend())
```

It is incompatible with `self_vf=true` and with `monte_carlo=true`, and is
3D-surface-mesh only. See [Integration Methods](@ref) for the method itself
and its known faceting-bias limitation on coarsely meshed curved bodies.

## Obstruction on GPU

Obstruction detection is fully supported on GPU. The BVH is built on the CPU,
flattened to typed arrays, and uploaded to the device once before the kernel
launch. The GPU kernel uses stackless BVH traversal (no per-thread stack
memory) for reduced register pressure.

```julia
result = compute_view_factors(mesh; nquad=4, backend=CUDABackend(),
                               obstruction_groups=[3, 4])
```

## Constraints

- `surface_dim=1` (curve meshes) is not supported on GPU; use `CPU()`
- `use_duffy=true` is CPU-only; it is silently ignored on GPU backends
- `self_vf=true` is CPU-only, and is also incompatible with `raytrace=true`
  on any backend

## Performance crossover

GPU backends outperform CPU (8 threads) approximately when:

| Method | Elements N |
|---|---|
| Quadrature | N ≳ 300–500 |
| Monte Carlo (pair-area) | N ≳ 200 |

Ray-shooting Monte Carlo has not had a CPU/GPU crossover point measured this
way; on CPU it already outperforms both other methods by 3–25× on the Howell
benchmark suite (see [Integration Methods](@ref)), so a GPU comparison would
need its own benchmark rather than reusing the pair-area numbers above.

For smaller meshes, kernel JIT compilation and host↔device data transfer
dominate the runtime. Metal is typically 2–5× slower than a comparable NVIDIA
GPU for this workload due to Float32 vs Float64 and lower compute throughput.
