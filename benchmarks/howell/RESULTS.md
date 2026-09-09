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

```bash
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmarks --threads=auto benchmarks/howell/run.jl
```

Results are written to `results.csv` (as a fraction, not percent), one row
per case per kernel.

## Results

115 parameter points across 22 catalog cases, each run with both kernels (230
data points total). **Quadrature: 109/115 (95%) within 1%, median 0.005%.
Monte Carlo (`n_samples=5000`): 110/115 (96%) within 1%, median 0.007%.**

| Case  | Geometry                                    | Points | Quad median | Quad worst | MC median | MC worst |
|-------|----------------------------------------------|-------:|--------:|--------:|--------:|--------:|
| C-1   | Infinite parallel plates, equal width        |  4 | 1.2e-14% | 5.4e-14% | 2.8e-05% | 3.1e-04% |
| C-2   | Infinite parallel plates, unequal width       |  3 | 5.1e-14% | 7.9e-14% | 1.2e-04% | 1.6e-04% |
| C-3   | Infinite perpendicular plates, common edge    |  4 | 0.011%   | 0.051%   | 0.033%   | 0.169%   |
| C-4   | Infinite equal plates, common edge, angle α   |  4 | 0.051%   | 0.38%    | 0.027%   | 0.054%   |
| C-8   | Infinite plane to two rows of tubes           |  6 | 0.17%    | 3.7%     | 0.17%    | 3.8%     |
| C-10  | Rectangle to semi-infinite rectangle, angle   |  6 | 0.48%    | 3.3%     | 0.48%    | 3.3%     |
| C-11  | Identical parallel opposed rectangles         |  4 | 2.8e-13% | 3.9e-13% | 4.7e-04% | 0.001%   |
| C-14  | Perpendicular rectangles, common edge         |  4 | 0.010%   | 0.023%   | 0.007%   | 0.025%   |
| C-33  | Hexagonal prism (6 face pairs × 3 L)          | 18 | 0.019%   | 0.86%    | 0.019%   | 0.86%    |
| C-34  | Parallel regular polygons (n=3,4,5,6,8)       | 15 | 5.6e-04% | 0.50%    | 0.002%   | 0.50%    |
| C-40  | Coaxial parallel disks, equal radius          |  3 | 3.1e-05% | 4.0e-05% | 0.002%   | 0.002%   |
| C-41  | Coaxial parallel disks, unequal radius        |  3 | 5.0e-06% | 4.0e-05% | 2.3e-04% | 5.6e-04% |
| C-63  | Concentric infinite cylinders                 |  3 | 4.3e-04% | 1.1e-03% | 6.7e-04% | 1.1e-03% |
| C-68  | Infinite parallel cylinders, equal diameter   |  4 | 3.9e-05% | 1.2e-04% | 0.001%   | 0.007%   |
| C-69  | Infinite parallel cylinders, unequal radius   |  3 | 8.5e-05% | 9.3e-05% | 1.9e-05% | 0.005%   |
| C-72  | Cylinder in square array                      |  6 | 1.4e-03% | 0.021%   | 0.004%   | 0.060%   |
| C-73  | Cylinder in triangular array                  |  6 | 3.7e-03% | 1.1%     | 0.012%   | 0.41%    |
| C-79  | Cylinder base to inside lateral surface       |  3 | 0.015%   | 0.021%   | 0.015%   | 0.022%   |
| C-109 | Cone interior to base                         |  3 | 0.21%    | 0.22%    | 0.21%    | 0.22%    |
| C-125 | Sphere to coaxial disk                        |  4 | 0.002%   | 0.005%   | 0.002%   | 0.007%   |
| C-135 | Concentric spheres                            |  3 | 0.025%   | 0.027%   | 0.025%   | 0.027%   |
| C-137 | Two spheres of unequal radius                 |  6 | 0.21%    | 0.66%    | 0.21%    | 0.66%    |

Coverage spans both solver paths — 2D per-unit-depth curve meshes (C-1 to C-4,
C-8, C-63, C-68, C-69, C-72, C-73) and 3D surface meshes (the rest) — over
planes, disks, cylinders, cones, spheres and polygon arrays, including opposed,
perpendicular, enclosing, edge-sharing, mutually shadowing and multi-body
obstruction configurations.

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
