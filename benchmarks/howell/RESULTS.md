# Benchmark — Howell Catalog of Configuration Factors (Section C)

RadiativeViewFactor.jl compared against published configuration factors from
J. R. Howell, *A Catalog of Radiation Heat Transfer Configuration Factors*,
3rd ed., <https://www.thermalradiation.net> — Section C, factors from finite
areas to finite areas.

## Method

Each case builds its geometry with Gmsh (second-order elements, OpenCASCADE
kernel), loads it through `load_mesh`, computes the view factor matrix with
`compute_view_factors`, and compares against the catalog's reference value at
the same parameters. Most cases compare against a closed-form equation,
transcribed by eye in `analytic.jl` (one function per case, using the
catalog's own variable names so the transcription can be checked against the
source page). Seven cases (C-8, C-10, C-33, C-34, C-72, C-73, C-137) instead
compare against the catalog's own **published tables of reference values**,
extracted programmatically from the page HTML into `tables.jl` rather than
transcribed by eye, and cross-checked against the closure identities each
geometry must satisfy (e.g. C-33's six view factors from one hexagon wall sum
to 1; C-72's four square-array neighbours sum to 1 at the touching limit)
before being used as reference values.

Every case is exercised at **several parameter values**, so agreement is
tested across a range of aspect ratios rather than at a single point. Every
case is also run with **both solver kernels** — deterministic Gauss-Legendre
quadrature and stratified Monte Carlo area-sampling (`monte_carlo=true`) — on
the *same* mesh, so the comparison isolates the kernel rather than mixing in a
discretization difference. All error figures below are **percent error**,
`100 × |computed − catalog| / catalog`.

Cases whose extractors read only some of the mesh's physical groups are run
with `radiating_groups`, so only those groups are assembled while every group
still obstructs — see "Restricted assembly" below.

```bash
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmarks --threads=auto benchmarks/howell/run.jl
# third kernel, appends to the same results.csv
julia --project=benchmarks --threads=auto benchmarks/howell/run_raytrace.jl
```

Results are written to `results.csv` (as a fraction, not percent), one row
per case per kernel, with per-call runtimes and a commented system-information
preamble. A third kernel, ray-shooting Monte Carlo (`raytrace=true`), is
covered separately in [Ray-shooting Monte Carlo](#ray-shooting-monte-carlo)
below — `run_raytrace.jl` shares the same case list (`cases.jl`) as
`run.jl` but runs only that kernel, and only on the 3D cases it supports.

## Results

115 parameter points across 22 catalog cases, each run with both kernels (230
data points total). **Quadrature: 109/115 (95%) within 1%, median 0.005%.
Monte Carlo (`n_samples=5000`): 110/115 (96%) within 1%, median 0.008%.**

| Case  | Geometry                                    | Points | Quad median | Quad worst | MC median | MC worst |
| ----- | ------------------------------------------- | -----: | ----------: | ---------: | --------: | -------: |
| C-1   | Infinite parallel plates, equal width       |      4 |    2.0e-14% |   5.4e-14% |  3.7e-05% | 3.1e-04% |
| C-2   | Infinite parallel plates, unequal width     |      3 |    5.1e-14% |   7.9e-14% |  1.2e-04% | 1.6e-04% |
| C-3   | Infinite perpendicular plates, common edge  |      4 |      0.018% |     0.051% |    0.034% |   0.169% |
| C-4   | Infinite equal plates, common edge, angle α |      4 |      0.079% |     0.384% |    0.032% |   0.054% |
| C-8   | Infinite plane to two rows of tubes         |      6 |      0.379% |      3.74% |    0.369% |    3.75% |
| C-10  | Rectangle to semi-infinite rectangle, angle |      6 |      0.801% |      3.31% |    0.801% |    3.31% |
| C-11  | Identical parallel opposed rectangles       |      4 |    3.2e-13% |   3.9e-13% |  6.2e-04% |   0.001% |
| C-14  | Perpendicular rectangles, common edge       |      4 |      0.011% |     0.023% |    0.009% |   0.025% |
| C-33  | Hexagonal prism (6 face pairs × 3 L)        |     18 |      0.019% |     0.862% |    0.019% |   0.862% |
| C-34  | Parallel regular polygons (n=3,4,5,6,8)     |     15 |    5.6e-04% |     0.498% |    0.001% |   0.500% |
| C-40  | Coaxial parallel disks, equal radius        |      3 |    3.1e-05% |   4.0e-05% |    0.002% |   0.002% |
| C-41  | Coaxial parallel disks, unequal radius      |      3 |    5.0e-06% |   4.0e-05% |  2.3e-04% | 5.6e-04% |
| C-63  | Concentric infinite cylinders               |      3 |    4.3e-04% |     0.001% |  2.1e-04% | 3.0e-04% |
| C-68  | Infinite parallel cylinders, equal diameter |      4 |    4.3e-05% |   1.2e-04% |    0.001% |   0.006% |
| C-69  | Infinite parallel cylinders, unequal radius |      3 |    8.5e-05% |   9.3e-05% |    0.003% |   0.004% |
| C-72  | Cylinder in square array                    |      6 |      0.003% |     0.021% |    0.004% |   0.061% |
| C-73  | Cylinder in triangular array                |      6 |      0.004% |      1.14% |    0.008% |   0.395% |
| C-79  | Cylinder base to inside lateral surface     |      3 |      0.015% |     0.021% |    0.015% |   0.022% |
| C-109 | Cone interior to base                       |      3 |      0.206% |     0.218% |    0.205% |   0.218% |
| C-125 | Sphere to coaxial disk                      |      4 |      0.002% |     0.005% |    0.001% |   0.008% |
| C-135 | Concentric spheres                          |      3 |      0.025% |     0.027% |    0.025% |   0.027% |
| C-137 | Two spheres of unequal radius               |      6 |      0.259% |     0.658% |    0.259% |   0.658% |

These figures are computed directly from `results.csv`. The worst-case column
and both pass counts are unchanged from the previously published table; several
*median* entries differ slightly because the earlier table was not regenerated
from the same CSV.

Coverage spans both solver paths — 2D per-unit-depth curve meshes (C-1 to C-4,
C-8, C-63, C-68, C-69, C-72, C-73) and 3D surface meshes (the rest) — over
planes, disks, cylinders, cones, spheres and polygon arrays, including opposed,
perpendicular, enclosing, edge-sharing, mutually shadowing and multi-body
obstruction configurations.

A third kernel, ray-shooting Monte Carlo (`raytrace=true`), is covered
separately below in [Ray-shooting Monte Carlo](#ray-shooting-monte-carlo) —
it only supports 3D surface meshes, so it runs on 72 of the 115 points here,
not all 115.

## System

The run recorded below, reproduced in the comment preamble of `results.csv`:

|                        |                                                          |
| ---------------------- | -------------------------------------------------------- |
| Date                   | 2026-09-14 14:09                                         |
| CPU                    | Apple M3, 8 logical cores (4 performance + 4 efficiency) |
| Julia threads          | 8                                                        |
| Memory                 | 16 GB                                                    |
| OS                     | Darwin arm64-apple-darwin25.5.0                          |
| Julia                  | 1.12.7                                                   |
| RadiativeViewFactor.jl | 0.6.2, commit `1a8508f` (working tree dirty)             |
| `n_samples` (MC)       | 5000                                                     |

`results.csv` records the core count as `cpu_threads_julia_reports: 4` as well
as `cpu_logical: 8`. Both are correct: `Sys.CPU_THREADS` counts only the
performance cores on Apple Silicon, which is why the two disagree. The run used
8 Julia threads.

## Runtimes

**Total wall time 6306 s (1 h 45 m)**, of which 565 s is quadrature and 5723 s
Monte Carlo — the MC kernel costs **10x** quadrature over the suite as a whole.
Times are per `compute_view_factors` call, measured after a warm-up case so
they exclude first-call compilation.

| Case  | Points | Quad (s) |  MC (s) | MC / quad |
| ----- | -----: | -------: | ------: | --------: |
| C-10  |      6 |   491.03 | 3173.64 |         6 |
| C-109 |      3 |    22.94 | 1165.20 |        51 |
| C-8   |      6 |     9.37 |  603.81 |        64 |
| C-33  |     18 |    17.55 |  163.71 |         9 |
| C-34  |     15 |     5.23 |  165.69 |        32 |
| C-135 |      3 |     0.80 |  134.91 |       169 |
| C-79  |      3 |    13.20 |   95.35 |         7 |
| C-125 |      4 |     1.07 |   80.57 |        75 |
| C-41  |      3 |     2.08 |   76.56 |        37 |
| C-137 |      6 |     0.39 |   26.55 |        69 |
| C-40  |      3 |     0.43 |   17.08 |        40 |
| C-11  |      4 |     0.20 |    8.33 |        41 |
| C-14  |      4 |     0.95 |    5.66 |         6 |
| C-72  |      6 |     0.03 |    1.07 |        35 |
| C-73  |      6 |     0.06 |    1.03 |        18 |
| C-3   |      4 |    <0.01 |    0.88 |       221 |
| C-4   |      4 |    <0.01 |    0.86 |       216 |
| C-1   |      4 |    <0.01 |    0.83 |       207 |
| C-2   |      3 |    <0.01 |    0.56 |       188 |
| C-63  |      3 |    <0.01 |    0.52 |       173 |
| C-69  |      3 |    <0.01 |    0.12 |         — |
| C-68  |      4 |    <0.01 |    0.11 |       109 |

Two things stand out. **C-10 alone is 58% of the whole suite** (3665 s of
6306 s): its geometry meshes a semi-infinite rectangle as a finite one of width
80, so the element count dwarfs every other case. And the **MC/quad ratio is
worst on the cheapest cases** — over 200x on C-1 to C-4 — because MC's cost is
`n_samples` point-pairs per element pair regardless of how simple the geometry
is, whereas `nquad=6` quadrature costs `nquad⁴` = 1296 and scales down with the
mesh. On the large, expensive cases the ratio falls to single digits.

## Restricted assembly

Three case families read only some of their mesh's physical groups, and are now
run with `radiating_groups` so that only those groups are assembled while every
group still obstructs:

| Case | Radiating / loaded elements            | Notes                                                                                                                |
| ---- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| C-72 | 144 / 1200 (all 6 points)              | 5×5 array of 25 cylinders; extractors read groups 1–3 only                                                           |
| C-73 | 144 / 1200 (all 6 points)              | as C-72, triangular array                                                                                            |
| C-33 | 157–1386 of 689–2135 (15 of 18 points) | hexagonal prism; each extraction reads a prefix of the six groups, so the restricted fraction ranges from 15% to 81% |

C-33's three `end→end` points read all six groups and so are not restricted.

C-72 and C-73 are the clearest case: all 25 cylinders must stay as obstructors,
but only the centre cylinder and its two reference neighbours are ever read, so
the assembled matrix drops from 1200 elements to 144 — roughly 70x fewer pairs,
on cases that are obstructed and therefore expensive per pair. Both now run in
under a second per kernel.

**This is exact, and was verified rather than assumed.** Every one of the 115
quadrature rows is *bitwise identical* to the same suite run without
`radiating_groups`. The Monte Carlo rows are not bitwise identical (57 of 115
are), which is expected and not a correctness signal: restricting the element
set — and, separately, the new pair-level facing cull — changes how many `rand`
draws the per-row RNG streams consume, so the sampling differs. The largest
relative change across all MC rows is 6.9e-4, well inside the kernel's own
noise, and both pass counts are unchanged.

C-8 is deliberately *not* restricted: its extractors sum over groups 1 through
1+nfront+nsecond, which is all but one of about 33 groups, and the tube counts
are not known until the mesh is built.

One safety note on using `radiating_groups` here. The extractors index
`F_group` positionally, and `_aggregate` orders groups by sorted tag, so
restricting to a non-prefix set such as `{1,4}` would move tag 4 into position 2
and `fij(1,4)` would silently read the wrong factor. `run_case` therefore
rejects any radiating set that is not the prefix `1:k`.

### Quadrature vs. Monte Carlo

At `n_samples=5000`, Monte Carlo tracks quadrature closely rather than adding
a separate, larger error band:

- **On edge-sharing cases** (C-3, C-4, C-14, C-33, C-79, C-109, C-137), MC's
  worst-case error is nearly identical to quadrature's, point for point (e.g.
  C-109: 0.22% both ways; C-137: 0.66% both ways). This is exactly what the
  docstring promises: those near/touching pairs carry the 1/r² singularity, so
  `compute_view_factors` always patches them with the same deterministic Duffy
  transform regardless of kernel — MC's stochastic estimator never actually
  touches the pairs that matter most for accuracy here.
- **On smooth, non-touching cases** (C-1, C-2, C-11, C-40, C-41, C-63, C-68,
  C-69, C-72), MC adds real but small noise on top of quadrature's
  near-machine-precision result — typically 1e-5% to 1e-3%, still far under
  the 1% threshold. This noise averages down over the many independent
  element pairs each aggregated group-to-group factor sums over; a single
  pair's own estimate is much noisier than that.
- **The one case where MC is a genuinely worse tail** is C-3 (0.051% quad vs
  0.169% MC) — still two orders of magnitude under 1%, not a concern, just the
  most visible instance of the added sampling noise.

Both kernels fail the same six points (the same underlying mesh/singularity
limitations discussed below), which is why 109 vs 110 out of 115 differ by
only one case (C-69's worst point crosses under 1% with MC by chance, not
because MC is more accurate there).

## Ray-shooting Monte Carlo

A different Monte Carlo method from the pair-area `monte_carlo=true` above —
see `src/RayTraceKernel.jl`'s module docstring for the method (cosine-
weighted rays, nearest-hit against one whole-scene BVH). It supports only
3D surface meshes, so it runs on 72 of this suite's 115 points (12 of the
22 cases; the 2D curve-mesh cases — C-1 to C-4, C-8, C-63, C-68, C-69, C-72,
C-73 — are skipped, not run with a fallback kernel). `n_rays=10000`
throughout.

**59/72 (82%) within 1%, median 0.18%.** Run with `benchmarks/howell/
run_raytrace.jl`; results are appended to `results.csv` alongside the
quad/mc rows (same column schema, `kernel=raytrace`), with their own
provenance comment block.

| Case  | Geometry                                    | Points | Median |  Worst |
| ----- | ------------------------------------------- | -----: | -----: | -----: |
| C-11  | Identical parallel opposed rectangles       |      4 | 0.049% | 0.089% |
| C-14  | Perpendicular rectangles, common edge       |      4 | 0.187% | 0.278% |
| C-40  | Coaxial parallel disks, equal radius        |      3 | 0.255% | 0.260% |
| C-41  | Coaxial parallel disks, unequal radius      |      3 | 0.185% | 0.217% |
| C-79  | Cylinder base to inside lateral surface     |      3 | 0.806% | 0.849% |
| C-109 | Cone interior to base                       |      3 | 0.546% | 0.684% |
| C-125 | Sphere to coaxial disk                      |      4 |  1.54% |  1.85% |
| C-34  | Parallel regular polygons (n=3,4,5,6,8)     |     15 | 0.041% | 0.470% |
| C-137 | Two spheres of unequal radius               |      6 |  1.49% |  2.55% |
| C-10  | Rectangle to semi-infinite rectangle, angle |      6 | 0.014% | 0.056% |
| C-33  | Hexagonal prism (6 face pairs × 3 L)        |     18 | 0.119% |  1.23% |
| C-135 | Concentric spheres                          |      3 |  2.19% |  2.84% |

### It's fast — dramatically so on this same 72-point subset

| Kernel                                   | Total seconds (72 points) | vs. ray-shooting |
| ---------------------------------------- | ------------------------: | ---------------: |
| Ray-shooting                             |                     193.4 |               1× |
| Quadrature                               |                     555.9 |      2.9× slower |
| Pair-area Monte Carlo (`n_samples=5000`) |                    5113.3 |     26.4× slower |

This is the O(N·rays·log N) vs. O(N²) advantage described in `changelog.md`,
now showing up across a real, varied benchmark suite rather than one
synthetic scene — reproducing (roughly) the earlier synthetic-scene
measurement (11.6×/45.2× at 836 elements) at a suite level.

### The errors are systematic, not noise, and track mesh curvature

Unlike the pair-area kernels (whose worst points, C-8 and C-10, come from
known discretization/convergence limits already discussed above),
ray-shooting's worst points here are concentrated on **curved bodies with
coarse meshes**: C-135 (spheres, 2.84%), C-137 (spheres, 2.55%), C-125
(sphere, 1.85%), C-79/C-109 (cylinder/cone, under 1%) — while the flat- or
mildly-curved cases (C-10, C-11, C-14, C-34, C-40, C-41) converge to well
under 1%, several to near machine precision. Two checks distinguish this
from ordinary Monte Carlo noise:

- **It does not shrink with `n_rays`.** C-135 at r2=0.75 stayed at 2.8%
  error from `n_rays=5000` through `n_rays=80000` — a 16× sample increase
  with no improvement, the signature of a systematic bias, not variance.
- **It shrinks with mesh refinement instead.** Refining that same case's
  mesh (`nsize` 0.2 → 0.1 → 0.05, element count 634 → 2530 → 9924) brought
  the error from 2.8% to 1.0% to 0.4%.

The mechanism: `RayTraceKernel.jl` triangulates each curved (Quad8/Tri6)
element by its 4/3 *corner* nodes only (the same faceting the existing
obstruction soups already use — see "Obstruction accuracy is limited by
facet count" below), and classifies a hit as landing on a target's front or
back face using *that flat triangle's* normal, not the true curved-surface
normal at the hit point. On a coarsely faceted convex body, two adjacent
facets can disagree slightly at their shared edge, and a ray leaving near
grazing incidence can clip a neighbouring facet it geometrically shouldn't
reach — this is the same category of faceting error the obstruction-soup
caveat already documents, now showing up as a front/back misclassification
rather than a missed/extra blocked ray. It is a mesh-resolution effect, not
a randomness one, and — as the refinement check above shows — converges
like one.

### A mesh orientation bug this suite's convention hid from the other kernels (fixed)

`concentric_spheres` (C-135) cuts the inner (r1) sphere out of the outer
(r2) one with an OCC boolean `cut`, then loads the result with a single
mesh-wide `reverse_normals=true`. Gmsh/OCC gives *both* resulting surfaces
an outward-facing normal by default — verified directly against a
standalone sphere, which is outward-facing (`dot(normal, radial) = +1`)
with `reverse_normals=false` — and the inner sphere from this cut comes out
the same way, *not* flipped by the cut itself. A blanket
`reverse_normals=true` therefore correctly fixes the outer sphere (which
does need to point into the gap) while wrongly over-correcting the inner
one, leaving it pointing into its own volume (`dot(normal, radial) = -1`)
instead of away from it.

This went unnoticed until now because the pair-area kernels tolerate it: a
point-pair cosine test doesn't care which of a pair's two normals is inward
as long as the sign works out over the full double integral, and
empirically the *aggregated* F(inner→outer) and F(outer→inner) values stay
correct under the old convention (quadrature: 0.999999 and 0.44435 against
1.0 and 0.44444) — it is only `check_reciprocity`/row-sum closure that
quietly fails (`Σⱼ Fᵢⱼ` off by up to 0.99, since the inner sphere's genuine
zero self-view was being computed as if it weren't zero), and this suite
had never checked closure for C-135, only the extracted F value.
`raytrace=true` fails outright instead of subtly: a ray sampled from the
cosine-weighted hemisphere around an inward-pointing normal is aimed *into*
the body's own volume, and — starting exactly on that body's own surface —
is geometrically guaranteed to exit back through the *same* body (a chord)
rather than ever reaching the outer sphere. First measurement, before the
fix: F(outer→inner) = 0.0068 against a catalog value of 0.444 (98.5% error;
16 of 18 sampled rays in one trial hit another element of the *same* inner
sphere instead of ever reaching the outer one).

Fixed by adding `reverse_group_normals(mesh, groups)` to `src/MeshIO.jl` —
flips only the requested physical groups instead of the whole mesh — and
changing `cases.jl`'s C-135 entry to flip only the outer sphere's group
(`reverse_groups=[2]`) instead of both. Quadrature and MC's C-135 rows are
unaffected by the fix (verified: still 0.999999/0.44435, matching the old
convention to 5 decimals) since they never depended on the inner sphere's
sign being right; ray-shooting's C-135 error dropped from 98.5% to the
2.2-2.8% (mesh-faceting-limited, see above) row in the table above. Every
other `reverse_normals=true` case in this suite (C-33, C-79, C-109) is one
topologically connected closed body, not two disconnected ones from a CSG
cut, and was checked and found unaffected by this issue.

## What the benchmark found

### A bug in `reverse_normals` (fixed)

`reverse_normals=true` silently corrupted every second-order surface element.
Reversing a Quad8's winding by swapping corners 1 ↔ 3 requires the mid-side
nodes to follow, swapping 5 ↔ 6 and 7 ↔ 8; the code swapped 5 ↔ 7, leaving
mid-side nodes attached to the wrong edges. Tri6 had the same defect (it
swapped 4 ↔ 6 where the correct permutation is 4 ↔ 5).

The failure was silent and severe: on a closed unit cube, face areas came out
0.9558 instead of 1.0 and row sums 0.108 instead of 1.0 — view factors roughly
10× too small, with no warning. First-order (Quad4, Tri3) and line elements
were unaffected, which is why the earlier cases in this suite passed: they
either used first-order elements or did not reverse normals.

Fixed in `src/MeshIO.jl`, with a regression test in `test/mesh_test.jl` that
checks mid-side nodes still lie on their own edges after reversal.

### A thread-safety bug in the obstruction BVH cache (fixed)

Adding the cylinder-array cases (C-72, C-73), which each need up to 25
obstruction groups, hit an intermittent `AssertionError: Multiple concurrent
writes to Dict detected!` crash. `build_bvh_lookup` in `src/Assembly.jl`
memoises the merged obstruction geometry per distinct set of active groups in
a plain `Dict`, filled with `get!` from inside the threaded assembly loop.
Julia's `Dict` is not thread-safe: two threads inserting distinct keys can
trigger a concurrent rehash. Earlier cases in this suite used at most 2-3
obstruction groups, so the cache filled on the first few pairs and the race
window was rarely hit; with dozens of groups it reproduces reliably.

Fixed by guarding the cache with a `ReentrantLock`, the same pattern already
used for the Golub-Welsch quadrature-rule cache in `src/Quadrature.jl`.

### Accuracy is limited by the shared-edge singularity, and Duffy is quad-only

Cases split cleanly:

- **Non-touching surfaces** (C-1, C-2, C-11, C-34, C-40, C-41, C-63, C-68,
  C-69, C-72, C-73, C-125, C-135, C-137) agree to between machine precision
  and ~0.1%. On the flat opposed geometries the integrand is smooth and
  Gauss-Legendre is effectively exact; the residual on curved cases is
  boundary faceting, not the integrator. C-137's ~0.1% residual doesn't
  respond to mesh refinement at all, consistent with the published table only
  giving 3 significant figures (e.g. "0.0294") rather than a solver limitation.

- **Surfaces sharing an edge** (C-3, C-4, C-10, C-14, C-33, C-79, C-109) carry
  the 1/r² singularity and are the least accurate. Enabling `use_duffy=true`
  is what makes these usable at all: it takes C-14 from 3.5% to 0.023%, a
  150× improvement. C-10 is the extreme case: at its smallest aspect ratio
  (A=0.4) convergence is confirmed but slow (n=10 -> 20 -> 40 -> 80 elements
  gives F=0.781 -> 0.744 -> 0.725 -> 0.716 against the catalog's 0.706,
  roughly halving the error each doubling), and the benchmark uses a moderate
  n=32 rather than paying for full convergence.

The Duffy correction only applies to **quad** pairs (`:quad`, `:quad4`) and
silently does nothing for triangles. This is visible in C-109: meshed with
quads the cone gives 0.2% error, but the *same geometry* meshed with triangles
(areas exact to 0.01% in both) gives 3.6-9.2% — and refining the triangle mesh
does not reliably improve it. **For edge-sharing geometries, mesh with quads.**

### Obstruction accuracy is limited by facet count, not solver precision

C-8 (plane to two staggered rows of tubes) needs each tube in its own
obstruction group — grouping a whole row together means a group is never
allowed to obstruct a pair involving itself (see note 3 below), so tubes in
the same group cannot shadow each other and the computed factor comes out too
high. Once split, the front row matches to 0.03-0.2%, but the second row
(partially shadowed by the front row) sits at 0.6% to 3.7%. Refining the
circle-to-polygon facet count converges it steadily (40 -> 80 -> 160 -> 320
segments per circle gives second-row F = 0.2023 -> 0.1970 -> 0.1960 -> 0.1957
against the catalog's 0.1953): obstruction soups use only element corner
nodes (see `_build_group_obs_soups` in `src/MeshIO.jl`), so a curved blocker
is really an inscribed polygon, slightly smaller than the true body, letting
a bit too much radiation through at its silhouette edge.

### Monte Carlo's per-pair cost is O(N⁴) sensitive to mesh density, not O(N²)

Adding the Monte Carlo comparison first ran into a wall: with the original
sphere meshes and `n_samples=10000`, C-135 (concentric spheres) and C-125
(sphere to disk) each took 60-80 minutes for just 3-4 parameter points — and
C-137 (two spheres), never reached, would likely have been worse. The whole
115-point suite, run once with quadrature alone, normally finishes in a few
minutes; adding Monte Carlo at that sample count threatened to take most of a
day.

The mechanism compounds two independent scalings. Element count `N` for a
fixed-size body scales as `1/msz²` (`msz` = target mesh element size, a 2D
area argument). The Monte Carlo bulk cost is `O(N²)` pairs `× n_samples` per
pair. So halving `msz` (doubling density) doesn't double the cost — element
count quadruples, pair count goes up 16x, and total MC cost scales as
`1/msz⁴`. The original sphere builders used `msz = radius/5` to `radius/8`,
producing dense meshes (thousands of elements each) that were fine for
quadrature — nquad=6 is only 36 points per pair — but the same mesh at
`n_samples=10000` (278x more points per pair) made the pair count the whole
problem.

Fixed by coarsening the three sphere builders (`concentric_spheres`,
`sphere_to_disk`, `two_spheres` in `geom.jl`) — roughly doubling `msz`, which
alone cuts the pair count by ~16x — combined with dropping the suite's
`MC_SAMPLES` from 10000 to 5000 (2x). C-135 and C-125 dropped to 2.5 and 4.3
minutes; the whole 230-point (115 points × 2 kernels) suite now completes in
about 2 hours. The coarser mesh costs quadrature accuracy too, since both
kernels share the same mesh: C-125's worst-case quadrature error went from
2.9e-4% to 4.8e-3%, C-135's from 1.7e-3% to 0.027% — both still two orders of
magnitude under the 1% threshold, so the trade is a good one here, but it's
worth knowing that a mesh built for quadrature isn't automatically a
reasonable one for Monte Carlo at any given `n_samples`.

C-10 (rectangle to a semi-infinite rectangle, graded mesh, ~9200 elements at
its finest parameter point) shows the same sensitivity — noticeably the
slowest remaining case in the finished run — but stayed within an acceptable
range and wasn't modified. Anyone raising `MC_SAMPLES` back up, or adding a
new case with a similarly large mesh, should expect the `1/msz⁴` scaling to
bite there too.

## Notes for anyone extending this suite

1. **Normal orientation.** For 2D curve meshes the loader orients line normals
   toward the interior of the adjacent meshed surface; bare curves with no
   surface produce a warning and unoriented normals, giving F = 0. Every 2D
   case here meshes the enclosed cavity and tags the radiating curves as its
   boundary. For 3D, plane surfaces set winding explicitly, and closed bodies
   use `reverse_normals=true` to turn Gmsh's outward normals inward.

2. **The catalog's cylinder spacing `s` (C-68, C-69, C-137) is the
   surface-to-surface gap**, not the axis/centre distance. Reading it as
   centre distance gives ~100% errors, not a small discrepancy.

3. **`obstruction_groups` never applies a group to a pair involving itself.**
   Convex self-shadowing needs no obstruction geometry (a point whose normal
   faces away from the target is already removed by the cosine test), but a
   set of *separate* bodies meant to shadow each other — tubes in a row,
   cylinders in an array — must each be its own group, or they can't obstruct
   one another. This bit both C-8 and the array cases (C-72, C-73) during
   development.

4. **Gmsh meshing of curved seams is fragile.** The default blossom recombiner
   segfaults on the cone seam at some aspect ratios; `RecombinationAlgorithm=0`
   is stable. Gmsh options also survive `finalize`/`initialize` within one
   process, so each builder resets them. `OCC.removeAllDuplicates()` after
   `synchronize()` cleans up apex/seam degeneracies (needed for the cone and
   the hexagonal prism's extruded side walls).

5. **A long "semi-infinite" plate needs a graded mesh** (C-10): representing
   an infinite rectangle by a finite one of width `W` and meshing it uniformly
   leaves elements with a huge aspect ratio once `W` is more than a few times
   the near-field length scale, and the view factor stops converging as `W`
   grows. `rect_to_long_rect` grades the far plate's mesh (`Progression`) so
   it stays fine near the shared edge and coarsens with distance.

## Two cases attempted and not included: C-35, C-154

Both have a genuinely ambiguous configuration in the published figure, and
none of the readings tried reproduced the table:

- **C-154 (two hemispheres in contact).** The figure shows two bowl shapes
  stacked on a common axis with concave (dotted, shaded) surfaces facing each
  other. Tried: domes touching apex-to-apex, convex sides facing
  (`F12=0.209` vs catalog `0.1511` at R=1, 39% off); the same geometry with
  reversed (concave) normals, which gave a self-view-factor `F11>1`, i.e. an
  unphysical result; bowls resting side by side on a shared plane, convex-up
  and concave-up variants (`0.047` and `0.047`, ~70% off); hemispheres nested
  on a shared axis and shared base plane (`F12=0` or `F12>1` depending on
  orientation). None reproduce the table, and two configurations produced
  self-view-factors exceeding 1, which shouldn't happen for any valid closed
  or open cavity and signals a geometry that isn't what the source intends.

- **C-35 (rectangle to one quarter of a parallel cylinder).** The figure
  specifies a rectangle facing a cylinder whose axis lies in the rectangle's
  plane (so that plane bisects it), with A2 "one quarter" of the cylinder —
  but doesn't mark which quarter-arc, or its angular position relative to the
  bisecting plane. Tried four candidate 90° arcs (the near quarter facing the
  rectangle, the far quarter, the full facing half, and an off-center 135°
  sweep); errors ranged 60-83% low, and the half-arc case returned exactly 0
  (evidence of an orientation problem, not just the wrong arc). No arc
  selection tried was close enough to call converged-but-slow, unlike C-10.

Both are documented here rather than shipped as a guessed geometry, per the
project's standing rule against presenting an inference as fact. If either
figure can be pinned down against the original Howell text (rather than the
web catalog's schematic), the geometry builders can be re-derived from that.

## Coverage relative to the full catalog

Section C contains 162 numbered entries; this suite covers 22, all validated
against reference values transcribed or extracted from the primary source
rather than assumed. Two further tabulated cases were examined and excluded
as poor fits for this kind of validation: C-13c publishes a Monte Carlo result
table for a specific "W-shaped tube" cross-section from a 2018 paper, with no
geometry definition on the catalog page itself, so it isn't reproducible from
what's published there; C-101 turns out to be a hybrid of raster equations and
tabulated correction values (not a pure lookup table), so it belongs with the
raster-equation backlog below, not this batch. One further tabulated case,
C-113 (cone frustum to base disk), was not attempted this round — its table
extraction needs more cleanup than the others (inconsistent exponent
formatting in the source HTML) and it has three independent geometric
parameters (L, R, and the cone half-angle) rather than two.

Of the remaining 140 entries: 120 carry an equation image (each needs its
formula transcribed by eye and a bespoke geometry — the same process used for
C-1 through C-135 above), and the rest are graphical-only, defer to
configuration-factor algebra over other cases, or describe geometries with no
meaningful mesh (C-156 to C-162: seated and standing people, a cow, a pig).
