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

For well-separated elements, `nquad=4` is a good default — `nquad²` = 16
quadrature points on *each* element, hence `nquad⁴` = 256 point-pairs per
element pair. Increase `nquad` when:

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

`n_samples` counts sample pairs **per element pair** (default 5000). The
Monte Carlo standard error scales as σ/√N, where σ is the per-pair standard
deviation, so quadrupling `n_samples` halves the noise.

The default of 5000 is the value used throughout `benchmarks/howell/`: across
115 Howell catalog points it landed 110 (96%) within 1% of the published value,
with a median error of 0.008%. At that sample count the estimator's own noise is
typically 10–1000× smaller than the mesh's discretization error against the
analytic answer, so raising it further mostly buys accuracy the mesh cannot use.
Lower it for a quick look at a large mesh; raise it only if an `n_samples`
sweep on your own geometry shows the noise still dominating. Geometries with
many obstructions or near-singular pairs have a larger σ and may need more.

Pairs that share a vertex or edge, or sit close to each other, are not sampled
at all: they are patched with the deterministic Duffy value, at `nquad` (and
within `factor` element diameters, default 3.0). Raising `factor` does not by
itself make those pairs more accurate — `nquad` is the more effective lever
for closure error on meshes with large or elongated elements.

## Ray-shooting Monte Carlo (`raytrace=true`)

For large or obstructed 3D meshes, `raytrace=true` is usually the fastest
option: on a 72-point subset of the Howell catalog benchmark it was 2.9×
faster than quadrature and 26.4× faster than pair-area Monte Carlo at
`n_samples=5000` (see `benchmarks/howell/RESULTS.md`). It scales as
O(N · n_rays · log N) rather than O(N²) in time, and — unlike pair-area Monte
Carlo — does not need a separate obstruction check per pair, since every
radiating element is already part of the one scene BVH each ray is cast
against. It is less accurate on that subset (median error 0.18% at
`n_rays=10000`, versus 0.005–0.008% for the other two on the full suite).

`n_rays` (default 10000) counts rays **per element**, not per element pair,
so it is not directly comparable to `n_samples` — tune it per case. Increase
it for smoother row sums on a closed enclosure (row sums converge to 1 as
ordinary MC noise shrinks with `n_rays`) or for finer per-pair accuracy;
decrease it for a quick estimate. On coarsely faceted curved bodies
(spheres, cylinders), increasing `n_rays` alone will not remove the small
systematic faceting bias described in [Integration Methods](@ref) — refine
the mesh instead.

## Duffy vs high `nquad`

For inclined-plate geometries with a shared edge, `use_duffy=true` with
`nquad=4` typically outperforms `use_duffy=false` with `nquad=16` in both
accuracy and runtime. The Duffy path evaluates `6 × nquad⁴` points per shared
edge (`4 × nquad⁴` per shared vertex) — with `nquad=4` that is 1536 per pair vs
65536 for `nquad=16` standard.

## Skipping work nobody asked for

Two options cut the number of element pairs that are actually integrated, and
neither changes the answer:

- **`facing_cull=true`** (the default) gives each element a conservative bound on
  its points and another on its normals (O(N)), and skips any pair whose kernel
  is provably zero at every point pair in O(1) — instead of discovering the zero
  one quadrature point, Monte Carlo sample, or BVH ray cast at a time. On closed
  convex bodies (tube bundles, pebble beds) most pairs face away from each other,
  so this is a large saving; the assembled matrix is bitwise identical with it on
  or off. On CPU, `verbose=true` reports how many pairs were skipped. It does
  not apply to `raytrace=true`.
- **`radiating_groups=[...]`** restricts the radiating set while every group
  still obstructs, turning an `O(N_total²)` problem into `O(N_radiating²)`. If
  you only need the view factors between a few surfaces in a mesh full of
  shadowing bodies, this is by far the larger saving (see
  [Obstruction Detection](@ref)). The enclosure is then open, so row sums do not
  close to 1.

## Threading

The CPU path uses `Threads.@threads` over element rows. Set the thread count
before starting Julia:

```bash
julia --threads=8 script.jl
# or
JULIA_NUM_THREADS=8 julia script.jl
```

For both Monte Carlo methods, one independent RNG is pre-generated per row
(seeded from the `rng` you pass), so results are reproducible and free of lock
contention regardless of how Julia schedules the threads.

## Memory

The `F_elem` matrix is dense with N² Float64 values. For N = 1000 elements
this is ~8 MB; for N = 10000 it is ~800 MB; for N ≈ 49000 it is ~19 GB. The
ray-shooting path builds `F_elem` in place, so its peak is a single N × N matrix
rather than several. If memory is a concern, restrict the radiating set with
`radiating_groups`, or consider aggregating to group level and discarding
`F_elem`:

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

`compute_view_factors` re-transfers the mesh on every call; there is no public
API for keeping mesh data resident on the device across calls. The internal
`GPUKernels.build_gpu_arrays` is what builds the device arrays.
