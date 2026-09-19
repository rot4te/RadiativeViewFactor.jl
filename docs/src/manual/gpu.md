# GPU Backends

RadiativeViewFactor.jl supports NVIDIA GPUs via CUDA.jl and Apple Silicon
GPUs via Metal.jl. Both backends are optional weak dependencies loaded
automatically when the user loads the corresponding package.

All four surface element families (Quad4, Tri3, Quad8, Tri6) run on the GPU,
alone or mixed in one mesh.

## NVIDIA (CUDA)

```julia
using CUDA
using RadiativeViewFactor

result = compute_view_factors(mesh; nquad=4, backend=CUDABackend())
```

CUDA runs in **Float64** throughout. All features except `use_duffy`,
`self_vf`, and `surface_dim=1` are supported.

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

Long kernels are submitted in short chunks so they stay under macOS's GPU
watchdog. If a kernel still fails to complete, `compute_view_factors` raises an
error (a zero element area) rather than returning a NaN-filled matrix.

## Monte Carlo (pair-area) on GPU

Both backends support pair-area Monte Carlo integration, one thread per element
pair. Each thread seeds an independent stream from a per-call global seed and
its thread index (splitmix64), then draws from a fast inline 32-bit xorshift
generator. The `rng` keyword is ignored on GPU backends, and the global seed is
drawn at random on each call, so GPU Monte Carlo runs are not reproducible.

```julia
result = compute_view_factors(mesh; monte_carlo=true, n_samples=5000,
                               backend=CUDABackend())
```

As on the CPU, pairs sharing a vertex or edge (and other near pairs) are
patched afterwards with the deterministic Duffy transform, which runs on the
**CPU** on top of the GPU bulk.

Each thread also estimates the areas of its two elements from its own samples,
and every thread in a row writes the same element's area, so which estimate
survives is a race. Element areas therefore vary by around 1e-4 relative between
runs, and that noise propagates into `F = raw / A`; don't rely on GPU Monte Carlo
to better than roughly 0.01%. The quadrature and ray-shooting kernels do not
have this issue.

## Monte Carlo (ray-shooting) on GPU

Ray-shooting Monte Carlo (`raytrace=true`) has its own GPU kernel, one thread
per *element* rather than per element pair — each thread shoots `n_rays`
cosine-weighted rays from its element into a scene BVH covering the whole
radiating mesh, so obstruction among radiating elements is automatic:

```julia
result = compute_view_factors(mesh; raytrace=true, n_rays=10000,
                               backend=CUDABackend())
```

The scene BVH is built on the CPU and uploaded once; element areas come from a
quadrature kernel (`nquad`), and the reciprocity averaging and aggregation run on
the CPU afterwards. It is incompatible with `self_vf=true` and with
`monte_carlo=true`, and is 3D-surface-mesh only. See [Integration Methods](@ref)
for the method itself and its known faceting-bias limitation on coarsely meshed
curved bodies.

## Obstruction on GPU

Obstruction detection is fully supported on GPU. The BVH is built on the CPU,
flattened to typed arrays, and uploaded to the device once before the kernel
launch. The GPU kernel uses stackless BVH traversal (no per-thread stack
memory) for reduced register pressure.

```julia
result = compute_view_factors(mesh; nquad=4, backend=CUDABackend(),
                               obstruction_groups=[3, 4])
```

If `obstruction_groups` names groups with no geometry, a warning is issued and
the run proceeds without obstruction.

`facing_cull` and `radiating_groups` work on GPU as on CPU.

## Constraints

- `surface_dim=1` (curve meshes) is not supported on GPU; use `CPU()`
- `use_duffy=true` is CPU-only; on a GPU backend it is ignored with a warning
- `self_vf=true` is CPU-only and is silently ignored on GPU backends; with
  `raytrace=true` it is an error on any backend
- `rng` is ignored on GPU backends

## Performance crossover

GPU backends outperform CPU (8 threads) approximately when:

| Method | Elements N |
|---|---|
| Quadrature | N ≳ 300–500 |
| Monte Carlo (pair-area) | N ≳ 200 |

Ray-shooting Monte Carlo has not had a CPU/GPU crossover point measured this
way; on CPU it already runs about 3× faster than quadrature and 26× faster than
pair-area Monte Carlo on the Howell benchmark subset (see
[Integration Methods](@ref)), so a GPU comparison would need its own benchmark
rather than reusing the pair-area numbers above.

For smaller meshes, kernel JIT compilation and host↔device data transfer
dominate the runtime. Metal is typically 2–5× slower than a comparable NVIDIA
GPU for this workload due to Float32 vs Float64 and lower compute throughput.
