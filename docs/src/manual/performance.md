# Performance Guide

## Assembly cost and quadrature reuse

The view factor matrix is dense and assembly is O(N²) in the element count `N`.
To keep the per-pair work minimal, each element's quadrature points, normals,
and area elements (deterministic path) — or its stratified Monte Carlo samples
(MC path) — are **pre-evaluated once per element** and reused across every
element pair, instead of being rebuilt inside the pair loop.

Concretely, `compute_view_factors` builds one `ElementQuad` (or `ElementSamples`
for `monte_carlo=true`) per element up front, then feeds the cached data into
the pair integrator. This removes the O(N²) reconstruction of shape functions
and quadrature rules that a naive pairwise evaluation would incur, and for
`nquad > 5` it also avoids re-running the Golub–Welsch eigensolve per pair (the
rule is memoized).

The change is purely an evaluation-order optimization — results are numerically
identical, reciprocity holds to machine precision, and analytic view factors are
unchanged. Measured on an 8-core CPU it gives roughly:

| Path | Speedup | Allocation reduction |
|---|---|---|
| Deterministic quadrature | ~2.6–3.1× | ~60–80× |
| Monte Carlo | ~12–15× | ~350–450× |

The Monte Carlo path benefits more because it previously re-sampled *both*
elements on every pair. Reproducible benchmark scripts and the full
before/after tables live in the `benchmarks/` directory (`benchmarks/RESULTS.md`).

## Choosing `nquad`

For well-separated elements, `nquad=4` (16 quadrature points per element pair)
is a good default. Increase `nquad` when:

- Elements are large relative to their separation distance
- You need sub-percent accuracy on individual element-pair values
- Row sums deviate noticeably from the expected value

A convergence study is the most reliable guide:

```julia
for n in [2, 4, 6, 8, 12]
    result = compute_view_factors(mesh; nquad=n, verbose=false)
    i = findfirst(==("emitter"),  result.group_names)
    j = findfirst(==("receiver"), result.group_names)
    @printf("nquad=%2d  F12=%.7f\n", n, result.F_group[i,j])
end
```

## Choosing `n_samples` for Monte Carlo

The MC standard error scales as σ/√N where σ is the per-pair standard
deviation. For smooth geometries σ is small and `n_samples=10000` gives ~1%
relative error. For geometries with many obstructions or near-singular pairs,
σ is larger and `n_samples=100000` or more may be needed.

A rough guide for the parallel-plates benchmark (well-separated, no
obstructions):

| `n_samples` | Approximate error |
|---|---|
| 1000   | ~10% |
| 10000  | ~1%  |
| 100000 | ~0.3% |
| 1000000 | ~0.1% |

## Duffy vs high `nquad`

For inclined-plate geometries with a shared edge, `use_duffy=true` with
`nquad=4` typically outperforms `use_duffy=false` with `nquad=16` in both
accuracy and runtime. The Duffy path evaluates `5 × nquad⁴` points per shared
edge — with `nquad=4` that is 1280 per pair vs 65536 for `nquad=16` standard.

## Threading

The CPU path uses `Threads.@threads` over element rows. Set the thread count
before starting Julia:

```bash
julia --threads=8 script.jl
# or
JULIA_NUM_THREADS=8 julia script.jl
```

For Monte Carlo, one independent RNG is pre-generated per row (seeded from the
`rng` you pass), so results are reproducible and free of lock contention
regardless of how Julia schedules the threads.

## Memory

The `F_elem` matrix is dense with N² Float64 values. For N = 1000 elements
this is ~8 MB; for N = 10000 it is ~800 MB. If memory is a concern, consider
aggregating to group level and discarding `F_elem`:

```julia
result   = compute_view_factors(mesh; nquad=4)
F_groups = result.F_group   # keep this
result   = nothing           # allow F_elem to be garbage collected
GC.gc()
```

## GPU transfer cost

The host↔device transfer of mesh coordinates and node indices is a one-time
cost at the start of `compute_view_factors`. For very large meshes the BVH
(if used) is also transferred once. These costs are amortised over the
N(N-1)/2 element pair computations that follow.

For repeated computations on the same mesh (e.g. parameter sweeps), consider
keeping the mesh data resident on the device. This requires lower-level access
to `build_gpu_arrays` from `GPUKernels` — contact the package author for
guidance.
