# Integration Methods

Four integration strategies are available, selected per call to
[`compute_view_factors`](@ref). They differ in accuracy, cost, and which
geometric configurations they handle well.

## Gauss–Legendre quadrature (default)

```julia
result = compute_view_factors(mesh; nquad=4)
```

The double surface integral is evaluated at a tensor product of `nquad` ×
`nquad` Gauss–Legendre points on each quadrilateral element (`nquad` points on
each curve element). This is the default method and is appropriate for most
geometries. Triangles (Tri3 and Tri6) use a Dunavant rule instead, with 1, 3, 7
or 13 points for `nquad` = 1, 2, 3 or ≥ 4.

**Convergence:** spectral — errors decrease as O(exp(-c·nquad)) for smooth
integrands. A good starting point is `nquad=4` — that is `nquad²` = 16 points
on *each* element, so `nquad⁴` = 256 point-pairs per element pair; increase to
`nquad=6` or `nquad=8` for finer accuracy or larger elements.

**Limitations:** convergence degrades near corner singularities where two
elements share a vertex or edge. In these cases use the Duffy transformation.

## Duffy transformation

```julia
result = compute_view_factors(mesh; nquad=6, use_duffy=true)
```

For same-order quad element pairs (Quad4–Quad4 or Quad8–Quad8) sharing a
vertex or edge, the `1/r²` singularity in the view factor kernel is
integrable but poorly resolved by standard quadrature. The Duffy
transformation introduces a radial coordinate whose Jacobian cancels the
singularity, leaving a smooth integrand that Gauss quadrature resolves
efficiently.

Singularity type is detected automatically:
- **No shared nodes** → standard quadrature (unchanged)
- **One shared corner node** → 4-region "biggest-coordinate" Duffy
  decomposition
- **Two shared corner nodes (one edge)** → 6-region decomposition (the shared
  edge split into two triangles, each further split 3 ways)

This is an elementary generalization of Duffy's original vertex-singularity
transformation, not the specific region formulas from Sauter & Schwab's boundary
element method (a related but different decomposition of the same
singularity) — see [Theory](@ref) for the exact construction.

**Constraints:**
- Also runs on GPU backends for 3-D meshes, as a device kernel over the
  touching pairs (see [Duffy transformation on GPU](@ref))
- Quad4–Quad4 and Quad8–Quad8 pairs only; triangle, curve, and mixed
  Quad4/Quad8 pairs always use standard quadrature
- Has no effect with `monte_carlo=true` (the pair-area Monte Carlo path always
  applies this same Duffy patch to near/touching pairs regardless of
  `use_duffy`) or with `raytrace=true` (which never evaluates `1/r²`)
- Not applicable for `surface_dim=1`: there is no 2-D Duffy transformation,
  and none is needed for correctness — at a shared endpoint the 2-D kernel
  stays bounded (the two cosines vanish as the points approach the corner), so
  plain quadrature converges there, just not as fast as on smooth pairs

**Cost:** `4 × nquad⁴` evaluations per vertex pair, `6 × nquad⁴` per edge pair,
versus `nquad⁴` for standard quadrature. Since only adjacent pairs trigger the
Duffy path, the overhead depends on mesh topology. A lower `nquad` (e.g. 4)
with `use_duffy=true` typically outperforms a higher `nquad` (e.g. 16) with
standard quadrature for inclined-plate geometries.

`.re2` meshes are all-Quad4 or, with curved faces (the default when the file
carries curved-side records), all-Quad8, so the Duffy path applies to every
pair. `use_duffy=true` is recommended whenever a structured hex-mesh boundary
has many edge-adjacent quad pairs — plain quadrature overestimates their view
factor. On the repository's `test/tall_cavity.re2` (2400 elongated Quad4 faces)
the largest element row sum is 2.11 with plain quadrature at `nquad=4`, 1.33
with `use_duffy=true` at `nquad=4`, and 1.11 with `use_duffy=true` at
`nquad=6`; it should be 1 for that closed enclosure, so raise `nquad` on
elongated elements (and see [`enforce_closure`](@ref)).

## Monte Carlo (pair-area sampling)

```julia
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000)
```

Each element pair is estimated from `n_samples` stratified sample pairs. Samples
are placed on a ⌊√N⌋ × ⌊√N⌋ grid of strata within the reference element, giving
O(1/N) variance convergence rather than O(1/√N) for plain Monte Carlo.

Because the `1/r²` kernel's variance diverges for touching or near-touching
pairs, `compute_view_factors` **always** patches every near pair (found via a
spatial grid, see `factor`) with a deterministic value once the Monte Carlo bulk
has finished: the Duffy integral for Quad4–Quad4 and Quad8–Quad8 pairs that
touch, plain quadrature for pairs that are merely close (and for triangles and
mixed families, which have no Duffy formula). On the CPU those pairs are skipped
in the sampling loop; on a GPU the bulk kernel samples them and a second device
kernel then overwrites them. `use_duffy` has no separate effect when
`monte_carlo=true`, since this patch is unconditional.

For efficiency, one independent stratified sample set is drawn **once per
element** and reused across that element's pairings (the diagonal self-pair,
with `self_vf=true`, draws a fresh second set). Within any pair the two
elements' samples are still independent, so each entry's estimate remains
unbiased with the stated variance; only estimates in the same row/column become
correlated. This reuse is what makes the MC path ~12–15× faster than
re-sampling both elements per pair — see the `benchmarks/` directory.

**When to use:**
- Near-singular pairs where MC variance is still finite (unlike the `1/r²`
  case where variance diverges — use Duffy instead)
- Rapid approximate estimates at low `n_samples`
- Very large meshes on GPU, where the O(N²) bulk parallelizes well

**Not** for obstructed geometries, despite the intuition that MC should win
there. Both paths apply the identical `K == 0` guard before calling
`is_visible`, so both ray-cast exactly once per kernel-positive point-pair —
MC has no structural advantage. It simply evaluates more point-pairs:
`n_samples` versus `nquad⁴` (256 at the default `nquad=4`), so `n_samples=5000`
issues ~20x the ray casts. Measured on a 4320-element obstructed reactor-pin
case, CPU: 1368 s at `n_samples=5000` versus 60 s at `nquad=4`, with the two
answers agreeing to five decimals (F = 0.380439 vs 0.380440). The GPU backend
does not rescue this — the per-sample BVH traversal is branch-divergent, and
the same case extrapolated to ~51 min on an M3 versus 22.8 min on 8 CPU
threads. For large obstructed meshes, ray-shooting Monte Carlo (below) is
faster on both CPU and GPU because it does not re-check obstruction per pair.

**Reproducibility:** pass an explicit RNG for deterministic results:

```julia
using Random
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               rng=MersenneTwister(42))
```

**GPU:** each thread (one per element pair) seeds an independent stream from
the global seed plus its index with splitmix64, then draws from a fast 32-bit
xorshift generator. The `rng` keyword is ignored on GPU; a random global seed is
generated on the host at each call, so GPU runs are not reproducible. See
[GPU Backends](@ref).

**Constraints:**
- `n_samples` applies **per element pair**, not to the whole geometry
- `use_duffy` has no separate effect (the near-pair patch is unconditional)
- Incompatible with `raytrace=true`

## Monte Carlo (ray-shooting)

```julia
result = compute_view_factors(mesh; raytrace=true, n_rays=10000)
```

A different Monte Carlo method from pair-area sampling above, not a faster
version of it. Instead of sampling point pairs on each element pair (O(N²)
pairs × `n_samples`), this shoots `n_rays` cosine-weighted rays per *element*
and tallies which element each first hits, using one BVH built over the
whole radiating mesh (O(N · n_rays · log N)). Because every radiating element
is already in that BVH, **radiating elements obstruct each other
automatically** — `obstruction_groups` is not needed for self-shadowing
within the radiating set and only adds *extra* non-radiating blocker
geometry here. There is also no adjacent-pair singularity (it never
evaluates `1/r²`), so `use_duffy` and the near-pair patch do not apply.

**Reciprocity is enforced by construction:** each unordered pair's two
independent ray-based estimates (one from each side) are averaged, so
results are exactly symmetric despite being stochastic. This is also why row
sums on a closed enclosure are only approximately 1 (ordinary MC noise,
shrinking with `n_rays`) rather than exact — every off-diagonal entry blends
in the *other* element's independent estimate too.

Element areas (`A_elem`) are computed by quadrature (`nquad`), not estimated
from the rays.

Runs on CPU or GPU (its own kernel, one thread per *element* on GPU, not per
pair):

```julia
using CUDA   # or Metal
result = compute_view_factors(mesh; raytrace=true, n_rays=10000,
                               backend=CUDABackend())
```

**Constraints:**
- 3D surface meshes only (`mesh.mesh_dim == 2`); not available for curve
  meshes
- Incompatible with `self_vf=true` (a ray is never tested against its own
  origin element) and with `monte_carlo=true` — pick one estimator
- `n_rays` counts rays **per element**, not per element pair, so it is not
  directly comparable to `n_samples`; tune per case

**Scene geometry:** rays are cast against a triangulation of the mesh. Quad8
elements are subdivided on a 3 × 3 parametric lattice through their curved map
(18 triangles each), so the scene follows the same surface the rays leave from;
Quad4 and Tri3 are split into 2 and 1 triangles, Tri6 is represented by its
corner triangle, and any extra `obstruction_groups` blockers are triangulated
from corner nodes only. The scene is therefore piecewise flat, and ray-shot
results on a curved mesh carry a small extra discretisation error that the
quadrature and pair-area Monte Carlo kernels (which integrate the curved map
directly) do not.

**Known limitation:** on coarsely faceted curved bodies (spheres, cylinders),
adjacent element facets can disagree slightly at their shared edge, letting a
near-grazing ray clip a neighbouring facet it geometrically shouldn't reach.
This produces a small, systematic bias (not ordinary MC noise — it does not
shrink with `n_rays`) that decreases with mesh refinement. See
`benchmarks/howell/RESULTS.md` for quantified examples.

**Accuracy**, on the same 72-point Howell subset at `n_rays=10000`: 59 of 72
points (82%) within 1% of the published value, median error 0.18% — looser than
quadrature or pair-area Monte Carlo, whose medians on the full 115-point suite
are 0.005% and 0.008%. The worst cases are curved bodies (concentric and
unequal spheres, sphere to disk, up to about 2.8%).

**Closure:** row sums on a closed enclosure carry Monte Carlo noise that falls
as `1/√n_rays`, so tightening them by sampling alone is expensive. For a closed
enclosure, [`enforce_closure`](@ref) restores exact row sums and reciprocity
afterwards.

**Speed**, on that same 72-point subset:

| Kernel | Total seconds | vs. ray-shooting |
|---|---:|---:|
| Ray-shooting | 193.4 | 1× |
| Quadrature | 555.9 | 2.9× slower |
| Pair-area Monte Carlo (`n_samples=5000`) | 5113.3 | 26.4× slower |

## Choosing a method

| Situation | Recommended method |
|---|---|
| Smooth geometry, well-separated surfaces | Quadrature (`nquad=4`–`8`) |
| Inclined plates with shared edge | Duffy (`nquad=4`–`6`) |
| Large or obstructed 3D mesh | Ray-shooting Monte Carlo (`n_rays=10000`) |
| Quick estimate, curve mesh, or self-view factors | Pair-area Monte Carlo (`n_samples=5000`) |
| GPU computation | Quadrature or either Monte Carlo variant |
| Closed enclosure feeding a radiosity solve | Any method, then [`enforce_closure`](@ref) |
| One pair of surfaces in a mesh of many shadowing bodies | Any method with `radiating_groups` (see [Obstruction Detection](@ref)) |
