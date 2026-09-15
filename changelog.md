# v0.6.2 changelog

Context: building a Claude Code skill to reproduce the nekRS `tall_cavity_vf_aurora`
Nek-VF radiation-case workflow for arbitrary geometries, using this package instead
of the paper's offline hemi-cube tool to compute view factors. That work surfaced
several real gaps/bugs in this package when applied to `.re2`-loaded meshes,
listed here in the order they were found and fixed. Every numerical claim below
was checked against either a hand-verified tiling/Jacobian calculation, a known
closed-form result (Modest, *Radiative Heat Transfer*), or the real
`tall_cavity.re2` fixture already in `test/`, not just unit-test pass/fail.

## 1. `SurfaceElement` now carries Nek5000 `(element, face)` identity

Added `eg::Int` and `iface::Int` fields to `SurfaceElement` (default `0` for
non-`.re2` meshes; 3-argument constructor unchanged and still works).
`load_re2` populates them from the `.re2` file's own boundary-condition
records. This is what makes it possible to write results back out keyed by
Nek's own numbering (see §4).

**Files**: `src/MeshIO.jl` (struct definition, `_re2_build_mesh`,
`_reverse_all_normals!`, `_orient_line3_normals!` to preserve the fields
through element reconstruction).

## 2. `load_re2` no longer drops periodic (`'P'`) boundary faces

**Bug**: `_RE2_INTERNAL_BC` treated `"P"` the same as `"E"`/blank (truly
internal element-element faces) and silently dropped periodic faces from
the loaded mesh.

**Why this is wrong**: Nek5000/NekRS's own view-factor module
(`view_factors.f`'s `vf_export_all_walls` / `vf_calculate_radiation_heat_flux`)
selects boundary faces with `cbc.ne.'E  '.and.cbc.ne.'   '` — i.e. it keeps
*every* non-internal face, periodic included, because the view-factor row-sum
closure (Σⱼ Fᵢⱼ = 1) requires the enclosure to be topologically closed.
Whether a face additionally *emits* (`cbc.eq.'W  '`) is a separate, runtime
decision made by the case's `.usr` file — not something derivable from the
`.re2` file's raw BC labels. Dropping `'P'` faces silently reproduced the
"open/leaky boundary" (`open_face_option=2`) case instead of the
default/expected closed enclosure, with no error or warning.

**Fix**: `_RE2_INTERNAL_BC = Set(["E", ""])` (was `Set(["E", "P", ""])`).
Periodic faces now load as their own group, named by their literal BC code
(e.g. `"P"`).

**Verified**: loading the real `test/tall_cavity.re2` now returns exactly
2400 boundary faces in 2 groups (`"MSH"`: 1600, `"P"`: 800) — matching the
2400-face header count in `nekRS/tall_cavity_vf_aurora/quads_file_tall_cavity`,
the case's own hemicube-pipeline wall-export file, computed independently via
the offline OpenGL hemi-cube tool. Total surface area of all 2400 faces is
1.15 (= the prism's true 2·(0.5·1.0 + 1.0·0.05 + 0.5·0.05)), vs. 1.05 for the
1600 wall faces alone.

**Behavior change**: any code relying on `load_re2` silently excluding
periodic faces will now see them as an extra group. `docs`/README updated to
describe this explicitly.

**Files**: `src/MeshIO.jl` (`_RE2_INTERNAL_BC`, docstrings).
**Tests**: `test/re2_test.jl`'s `tall_cavity.re2` testset expectations updated
(1600→2400 faces, `["MSH"]`→`["MSH","P"]`, area 1.05→1.15).

## 3. New: `write_nekrs_view_factors`

New function (`src/NekExport.jl`, new file) writing a `ViewFactorResult` +
`load_re2`-sourced `MeshData` to the exact list-directed ASCII format read by
Nek5000/NekRS's `vf_read_view_factors`:

```fortran
nwalls
iw ieg ifc nvwalls
  jw ieg ifc Fij   (nvwalls lines)
...
```

Confirmed against `view_factors.f`'s actual read loop (not just its README
summary) — critically, the *same* file format and content serves both
`open_face_option=1` and `=2` at runtime; the option only changes which
faces the radiosity iteration treats as emitting, not what's in the file,
so there's exactly one writer, not two.

Errors clearly (not silently) if `mesh` wasn't loaded via `load_re2` (i.e.
`eg`/`iface` are unset). Warns (mirroring the Fortran side's own non-fatal
behavior) if the written file would exceed `max_walls`/`max_visible_walls`
as currently sized in `view_factors/VIEW_FACTORS`.

**Files**: `src/NekExport.jl` (new), `src/RadiativeViewFactor.jl` (include +
export), `Project.toml` (added `Printf` dep — stdlib, needed for `@printf`).
**Tests**: `test/nek_export_test.jl` (new) — round-trips a computed result
through the writer and re-parses it, checking every `(ieg,ifc,Fij)` triple
against the source `ViewFactorResult` and `SurfaceElement`s, plus the
error path for a non-`.re2` mesh.

## 4. Fixed: the Duffy/Sauter-Schwab singular-pair transform was wrong

This is the significant one. Extending Duffy to Quad4 (needed because
`.re2`-loaded meshes are always Quad4, and structured hex-mesh boundaries
have many edge-adjacent Quad4 pairs) surfaced that the pre-existing
`_vertex_map`/`_edge_map` region formulas were **already badly wrong for
Quad8 too** — the bug predates this work and had never been caught, because
the only existing Duffy test (`type_stable_test.jl`) checked type inference
on a degenerate self-pair, never a value against any known-correct result.

**Symptom**: a fully closed unit cube (row sums must equal exactly 1.0)
gave 1.28 with plain quadrature (already too high — the shared-edge
singularity, present between *every* face pair in a cube, is underresolved)
and **37.9–45.1** with the old `use_duffy=true` — Duffy made it roughly
30× *worse*, identically whether the cube was built from Quad4 or Quad8
elements (confirming the bug was in the shared region-map math, not
Quad4-specific).

**Fix**: replaced the region maps entirely with a self-derived
"biggest-coordinate" Duffy decomposition, each half validated independently
before touching any element geometry:

- **Common vertex**: 4 regions (not 8) — split the shifted 4D offset space
  by which of (du,dv,ds,dt) is largest, Jacobian ρ³. Hand-verified: each
  region's ∂(du,dv,ds,dt)/∂(ρ,η₁,η₂,η₃) is lower-triangular with diagonal
  (1,ρ,ρ,ρ) ⟹ det = ρ³; 4×∫₀¹ρ³dρ = 1 = vol([0,1]⁴).
- **Common edge**: 6 regions (not 5) — split the (u,s) unit square into the
  two `u≥s`/`u<s` triangles via a ratio substitution (`s=u(1-w)`, Jacobian
  `u`; mirrored for the other triangle), then a 3-way "biggest of
  (w,v,t)" split within each (Jacobian ρ²). Also fixed a real, separate bug
  in the *old* edge-orientation handling: `_corners_to_edge` mapped a shared
  corner pair to one of 4 canonical edge numbers via `minmax(c1,c2)`,
  discarding which physical corner was first — meaning the two elements'
  edge-local `u`/`s` coordinates were never guaranteed to increase in the
  same physical direction along the shared edge. The rewrite builds each
  element's edge-local frame directly from `singularity_type`'s
  already-correspondence-ordered shared-corner pairs (`node(ci[k]) ==
  node(cj[k])` by construction), via an explicit 8-case
  corner-pair→reference-coordinate table (`_edge_local_to_ref`), so `u=0`
  and `s=0` are guaranteed the same physical point with no separate
  direction-tracking needed.
- Also removed ~30 lines of dead/duplicated Quad8-only shape-function
  evaluation (`_eval_quad_ref`, the old body of `_standard_integral`) that
  silently gave wrong results for Quad4 pairs reaching the `NONE`
  (non-adjacent) branch through the Duffy dispatch path; `_standard_integral`
  now just delegates to the already-correct, already-family-generic
  `element_pair_view_factor` from `ViewFactorKernel.jl`.

**Verified, in order**:

1. Pure tiling check (constant-function integral through each region
   decomposition, no element geometry involved) converges to 1.0 with
   finer quadrature (0.9995 at n=100 midpoint points) for both vertex and
   edge decompositions — confirms the Jacobians/regions are correct
   independent of any geometry bug.
2. Closed unit cube: `use_duffy=true` gives rowsum 1.008 at `nquad=6`,
   1.0003 at `nquad=10`, monotonically converging — vs. 1.28
   (plain quadrature) and the old 37.9 (broken Duffy). Reciprocity exact
   (`0.0` relative error) throughout, as expected regardless of accuracy.
3. Two independent closed-form cross-checks on that same cube (Modest,
   *Radiative Heat Transfer*): opposite unit squares 1 apart,
   F=0.19982 (matches, already covered by the pre-existing test) *and*
   perpendicular unit squares sharing a common edge, F=0.20004 — computed
   value 0.20006 at `nquad=16`.
4. Real mesh (`test/tall_cavity.re2`, 2400 faces): max rowsum improved from
   2.11 (no Duffy) to 1.33 (`nquad=4`) to 1.11 (`nquad=6`), continuing to
   tighten with `nquad` exactly as the cube case did.

**Files**: `src/DuffyKernel.jl` (`singularity_type`'s family guard widened
to accept matching Quad4/Quad4 or Quad8/Quad8 pairs, not just Quad8/Quad8;
`_eval_quad` and new `_eval_edge_quad`/`_edge_local_to_ref` dispatch on
`elem.family`; `_vertex_integral`/`_vertex_map` and `_edge_integral`/
`_edge_map` rewritten; `_corners_to_edge`/`_rotate_to_edge` removed).
**Tests**: `test/duffy_correctness_test.jl` (new) — the tiling check, the
closed-cube rowsum/reciprocity checks at multiple `nquad`, and both
closed-form value comparisons. `test/re2_test.jl`'s cube testset extended
with the same closure + adjacent-value checks; its real-file testset now
runs `use_duffy=true` and checks rowsums within 20% of 1.0 (loose bound
chosen for CI runtime at `nquad=6`, not accuracy — raise `nquad` for
tighter closure).

**Attribution correction**: the package's docs previously credited these
formulas to "Sauter & Schwab (2011), Eq. 5.3.11" specifically. The
replacement is a from-scratch, self-verified "biggest-coordinate"
generalization of Duffy's original single-simplex transform (Duffy, 1982) —
correct and validated as above, but *not* a reproduction of Sauter-Schwab's
specific region formulas (which were never available to check against in
this session). README/docstrings updated to stop attributing the
implementation to that specific reference; Duffy (1982) is the accurate
citation for the general technique.

## 5. Monte Carlo now auto-patches near/touching pairs with Duffy

Monte Carlo's stratified sampling has *unbounded* variance for pairs
sharing a vertex or edge (documented pre-existing behavior) — but testing
also showed it has *unreliable, non-shrinking* variance for pairs that are
merely close relative to their own size without sharing a node (e.g. two
elements a fraction of an element-width apart across a periodic seam,
which aren't mesh-adjacent by node-sharing but are geometrically close):
on the real 2400-face mesh, going from 2000 to 8000 samples/pair left the
worst row's error *unchanged* (max rowsum 2.13 → 2.17, not improving) —
a single lucky/unlucky sample pair landing close together dominates that
row's estimate regardless of how many samples are drawn elsewhere.

**Fix**: `compute_view_factors(mesh; monte_carlo=true, ...)` now
unconditionally finds every pair within `factor` (default 3.0) element-
diameters of each other — a strict superset of literally-touching pairs,
via a new spatial grid over element centroids, `near_pairs` (~O(N) for a
well-shaped mesh; verified to exactly match an O(n²) brute-force check on a
300-element random subset of the real mesh, `factor=3.0`: 3853/3853 pairs,
no misses) — and overwrites those O(N) pairs' raw integral with
`element_pair_view_factor_duffy` (which self-selects Duffy for touching
pairs and plain quadrature for merely-close ones) after the Monte Carlo
bulk finishes, **regardless of which backend (CPU or GPU) ran that bulk**.
This keeps the O(N²) well-separated majority — genuinely low-variance for
Monte Carlo, and where GPU throughput actually matters for large meshes —
on whatever backend was requested, while the O(N) correction is
asymptotically free at scale.

**Verified**: on the real mesh, `n_samples=2000` and `n_samples=8000` now
give near-identical results (max rowsum 1.111 vs. 1.111, mean 1.008 vs.
1.008) — matching the pure quadrature+Duffy result at the same `nquad=6`
almost exactly, confirming the residual error is now governed entirely by
the deterministic patch's `nquad`, not MC noise. The synthetic closed-cube
case closes to within 3% at `n_samples=2000, nquad=10`.

**API change**: `use_duffy` and `monte_carlo` are no longer mutually
exclusive (the old hard error is removed) — `use_duffy` simply has no
separate effect when `monte_carlo=true`, since the patch is now
unconditional. `nquad` is no longer fully ignored when `monte_carlo=true`;
it controls the near-pair patch's accuracy.

**Files**: `src/DuffyKernel.jl` (`near_pairs`, `patch_adjacent_pairs_duffy!`
— renamed/broadened from an earlier node-sharing-only `adjacent_pairs`
during this same session, before it reached a release), `src/Assembly.jl`
(CPU Monte Carlo path calls the patch; relaxed the `use_duffy`/
`monte_carlo` mutual exclusion), `src/GPUAssembly.jl` (GPU Monte Carlo path
calls the patch on the CPU after copying results back).
**Tests**: `test/duffy_correctness_test.jl` — MC+patch closure check on the
cube, and the grid-vs-brute-force `near_pairs` check on a real-mesh subset.

## 6. `SurfaceElement.phys_tag` + `split_groups_by_tag`

**Gap**: `load_re2` groups elements by Nek boundary-condition *code*
(`group`/`group_tags`) — `'W'`, `'P'`, or a generic placeholder like `"MSH"`
that `gmsh2nek` writes identically for *every* ordinary, non-periodic
surface, regardless of which named Gmsh `Physical Surface` it came from.
Two distinct surfaces sharing that placeholder (e.g. two concentric
spheres, or a pebble bed's pebble/duct-wall surfaces) were therefore
indistinguishable via `group` alone, making it impossible to pass just one
of them to `compute_view_factors`'s `obstruction_groups` — even though
`gmsh2nek` *does* separately write each face's original Gmsh Physical
Surface tag into the `.re2` boundary-condition record's 5th real parameter
(`bc(5,ifc,iel,1)`, the same value `usrdat2` reads as `bc_flag`), which
`load_re2` was reading and then silently discarding.

**Fix**: `_re2_parse` now keeps that 5th parameter; `SurfaceElement` gained
a `phys_tag::Int` field (`0` for non-`.re2` meshes) populated from it. New
exported `split_groups_by_tag(mesh)` returns a `MeshData` with
`group`/`group_tags`/`group_elems`/`group_tri_soup` refined to
`(code, phys_tag)` granularity, so a single named surface can be resolved
and passed to `obstruction_groups`.

**Files**: `src/MeshIO.jl` (`SurfaceElement` struct + docstring + outer
constructors, `_re2_reals`/`_re2_parse`/`_re2_build_mesh` threading the tag
through, `_reverse_all_normals!`/`_orient_line3_normals!` preserving it,
new `split_groups_by_tag`), `src/RadiativeViewFactor.jl` (import/export).

## 7. Fixed: 3-D ray/triangle obstruction was silently broken (CPU and GPU)

**Bug**: `BVH.jl`'s `intersect_ray_bvh` (CPU) and `GPUBVH.jl`'s
`gpu_intersect_bvh` (GPU) both read a triangle's three vertices from
`tri_soup`/`tri_verts` with the axes transposed relative to how
`_build_group_obs_soups` (MeshIO.jl) actually lays the array out — e.g. the
CPU code built `v0 = (tri_soup[1,1,t], tri_soup[1,2,t], tri_soup[1,3,t])`,
which is `(x of vertex1, x of vertex2, x of vertex3)`, not the xyz of one
vertex. `_triangle_aabb` and `_centroid` in the same file read the array
*correctly* (`(coord, vertex, tri)`), so the bug was an internal
inconsistency within `BVH.jl` itself, not a mismatch with `MeshIO.jl`.

**Effect**: essentially all 3-D obstruction queries (`obstruction_groups`
on a surface/`.re2` mesh) silently returned "unobstructed" for almost every
ray, only occasionally (and *less* often as the obstructing mesh got
finer — the opposite of a mere discretization error) reporting a correct
block near the exact antipodal direction. This was caught via the
concentric-spheres verification case (Nek-VF paper, Section 4.2): the
concave outer sphere's row sum came out ≈1.25 instead of the exact 1.0
(`F(outer→outer,self)` ≈1.0 instead of the true `1-(r1/r2)²=0.75`), and got
*worse*, not better, on a finer inner-sphere mesh — the tell that pointed
at a transposition bug in the traversal rather than a numerical-accuracy
issue. Segment-based (`mesh_dim=1`, 2-D) obstruction was unaffected — it
uses a separate, independently-correct function, `intersect_seg_bvh`.

**Fix**: corrected the vertex-extraction indexing in both
`intersect_ray_bvh` and `gpu_intersect_bvh` to `(coord, vertex, tri)`;
fixed the matching mislabeled axis-order comments in both files
(`BVHTree.tri_soup`'s docstring and `FlatBVH.tri_verts`'s doc comment had
the same "vertex, coord" order backwards).

**Verification**: after the fix, a direct sweep of the chord-vs-ball
blocking angle against a synthetic r=0.5 cubed-sphere triangle soup lands
exactly on the analytic boundary (`θ = 2·acos(r1) = 120°` for r1=0.5,
confirmed to within one 5° sweep step on both a coarse and a finer mesh);
the concentric-spheres end-to-end case now closes to `F(outer→outer) =
0.7502`, `F(outer→inner) = 0.2498` (analytic: 0.75, 0.25) with row-sum
closure error 0.017% (previously ≈25%, i.e. completely wrong).

**Files**: `src/BVH.jl` (`intersect_ray_bvh`, `BVHTree` docstring),
`src/GPUBVH.jl` (`gpu_intersect_bvh`, `FlatBVH` doc comment).
**Tests**: new `test/obstruction_test.jl` — direct `is_visible` sweep
across the analytic blocking boundary, plus an end-to-end
`compute_view_factors` check against the exact concentric-spheres closed
form.

## 8. Fixed: `_ray_aabb` produced `NaN` for axis-aligned rays (CPU)

**Bug**: found while building the regression test for §7. `_ray_aabb`
computed `t = (box.lo[i] - o[i]) * inv_d[i]` for each axis; when a ray's
direction component along axis `i` is exactly `0.0` (common for planar or
axis-aligned test/production geometry — e.g. two points in the same
`z=const` plane), `inv_d[i] = ±Inf`, and if the box also touches the ray's
origin coordinate on that axis (`box.lo[i] == o[i]` or `box.hi[i] ==
o[i]`), the product is `0 * Inf = NaN`. `NaN` compares `false` against
everything, so `tmin <= tmax2` silently evaluated to `false` — an
otherwise-valid AABB (and the real ray/triangle hit inside it) was
rejected outright, with no error or warning.

**Fix**: `_ray_aabb` now special-cases a zero direction component per
axis via a new `_slab_t` helper: a ray parallel to an axis passes that
axis's slab whenever its origin lies within `[lo, hi]` on that axis
(returned as an unconstrained `(-Inf, Inf)` interval), independent of the
degenerate `0 * Inf` formula.

**Files**: `src/BVH.jl` (`_ray_aabb`, new `_slab_t`; the one call site in
`intersect_ray_bvh` updated to pass `direction` through).
**Tests**: covered by the same `test/obstruction_test.jl` (its synthetic
sphere-sweep rays lie in a single plane, which is what surfaced this).

## Version

`0.6.1` → `0.6.2` (`Project.toml`) for changes §1-§5, done in an earlier
session. §6-§8 above were found and fixed in a later session (triggered by
actually generating and validating the concentric-spheres Nek-VF case) and
are **not** yet reflected in `Project.toml`'s version number — that bump
was deferred pending the maintainer's own decision on when/how to cut the
next release, so treat §6-§8 as already present in the `0.6.2`-tagged
source tree rather than as a `0.6.3` that doesn't exist yet.

## 9. Fixed: `factor` (near-pair Duffy-patch radius) was dead — never reached the CPU or GPU compute path

**Bug**: found while investigating a reported closure-error blowup (up to
147%) on meshes with large/elongated boundary elements, driven from the
`nek-vf-case` skill's `compute_view_factors.jl` CLI script. That script
parses `--factor` and prints it in its own error-tip message, but never
actually passes it to `compute_view_factors` — and `compute_view_factors`/
`_compute_cpu` didn't accept a `factor` keyword at all, so every
`patch_adjacent_pairs_duffy!` call silently used the hardcoded default of
`3.0` regardless of what the CLI flag said. A prior investigation that
raised `--factor` 3.0→20.0 to test whether the near-pair patch radius was
the cause of the closure blowup therefore never actually varied it.

**Root cause of the closure blowup itself**: separately confirmed (via a
synthetic reproduction, not committed here) that `factor` was never the
lever anyway — raising it with the plumbing bug fixed still made no
measurable difference. The real driver is `nquad`: every element gets a
fixed `nquad×nquad` Gauss-Legendre grid regardless of physical size
(`ViewFactorKernel.jl`/`DuffyKernel.jl`), so when a mesh's elements grow
large or elongated (e.g. widening a domain without scaling element count
proportionally), a low `nquad` under-resolves the `1/r²` kernel across
each element and closure degrades — this reproduces with pure deterministic
quadrature (no Monte Carlo, no randomness involved), and recovers cleanly
as `nquad` is raised (e.g. 10→20 cut closure error from ~10% to ~1.5% on a
10:1-elongated test mesh; MC mode with `nquad=30` on the same mesh: 0.26%,
still cheap since `nquad` there only governs the O(N) Duffy patch, not the
O(N²) Monte Carlo bulk). Not a code change here — this is a usage/config
finding for large-`nquad`-needing meshes, documented so it isn't
re-diagnosed as a `factor`-plumbing issue again.

**Fix**: added `factor::Float64 = 3.0` as a real keyword argument to
`compute_view_factors` (CPU and GPU dispatch), threaded through
`_compute_cpu` and `compute_view_factors_gpu` to their
`patch_adjacent_pairs_duffy!` calls. `nek-vf-case`'s
`compute_view_factors.jl` now passes `factor=factor` in its Monte Carlo
branch (the only path that uses it — the plain-quadrature/`use_duffy`
branch calls `element_pair_view_factor_duffy` directly per pair, not via
`near_pairs`, so `factor` doesn't apply there).

**Files**: `src/Assembly.jl` (`compute_view_factors`, `_compute_cpu`,
`_gpu_compute_hook`, docstring), `src/GPUAssembly.jl`
(`compute_view_factors_gpu`, docstring); outside this repo,
`~/.claude/skills/nek-vf-case/scripts/compute_view_factors.jl`.
**Tests**: full suite re-run after the change (`quad_test.jl`,
`vf_test.jl`, `mesh_test.jl`, `GPU_test.jl`, `vtk_test.jl`, `re2_test.jl`,
`duffy_correctness_test.jl` — which directly exercises
`patch_adjacent_pairs_duffy!` — `nek_export_test.jl`,
`obstruction_test.jl`, `type_stable_test.jl`: 274/274 passing, identical
to a baseline run on the unmodified code aside from the one known-broken
testset below).

## Known pre-existing, unrelated issues (not touched)

`test/mesh_test.jl`'s "Unstructured surface mesh end-to-end view factors"
testset already failed before any of the above changes (`e.family ===
:tri3`/`:tri` assertions on Gmsh-loaded meshes) — confirmed via a baseline
`Pkg.test()` run before touching any file. Left alone; out of scope for
this session, and unrelated to `.re2`/view-factor-export/Duffy/Monte Carlo/
obstruction. Because `Pkg.test()` aborts the whole `runtests.jl` `include`
chain on this failure, all new/changed tests above were additionally
verified by running `re2_test.jl`, `duffy_correctness_test.jl`,
`nek_export_test.jl`, `obstruction_test.jl`, and `type_stable_test.jl`
directly (230/230 passing) rather than through `Pkg.test()`.

`test/ray_test.jl`'s "Full blocker leaves no leaks (CPU + GPU traversal)"
testset's third assertion (`r.F_group[1, 2] == 0.0`) also already failed
before any of the §9 changes above — confirmed by stashing the §9 diff and
re-running `Pkg.test()` on the untouched tree (identical failure). Looks
real, not a test-infrastructure issue: the test places two directly-opposed
unit plates 1 apart, fully obstructed by a much larger blocking plane, and
expects `F_group[1,2] == 0.0`; it gets `0.19982` instead (the exact
unobstructed value) because the two plates are close enough to fall inside
`near_pairs`' default `factor=3.0` radius, and `patch_adjacent_pairs_duffy!`
does not check `obstruction_groups` for patched pairs at all — its
docstring assumes "nearby elements at this distance scale cannot have a
third surface positioned between them," which this test explicitly
contradicts. Left untouched — out of scope for the `factor`-plumbing fix
above and not something `factor` itself can address (the patch would need
to run the same obstruction check the bulk MC pairs get).

## 10. Fixed: `test/ray_test.jl`'s "Full blocker leaves no leaks" failure — two bugs, not one

Follow-up to the §9 "Known pre-existing" note above: `patch_adjacent_pairs_duffy!`
really did skip obstruction entirely for near-pairs, and fixing only that
uncovered a second, independent bug underneath it.

**Bug 1 — `patch_adjacent_pairs_duffy!` ignored `obstruction_groups`**:
confirmed by instrumenting `_compute_cpu`'s Monte Carlo loop — the bulk MC
pairs correctly detected the blocking plane (`nblocked == nchecked ==
n_samples`, raw contribution ≈0), but the near-pair Duffy patch that runs
immediately after unconditionally overwrote `raw_integral[i,j]` with
`element_pair_view_factor_duffy(..., nothing, mesh_dim)` — hardcoding
`bvh = nothing` — silently restoring the full unobstructed value for any
pair `near_pairs` flagged, regardless of `obstruction_groups`. Obstruction
geometry lives in `mesh.group_tri_soup`, entirely outside `elems`, so
proximity between two radiating elements never implies an unobstructed path
between them; the docstring's claim to the contrary was wrong.

**Fix 1**: factored the CPU path's inline obstruction-BVH-per-group-pair
closure out of `_compute_cpu` into a new shared, exported
`Assembly.build_bvh_lookup(mesh, obstruction_groups)`, and gave
`patch_adjacent_pairs_duffy!` a `bvh_for::Function` parameter (default
`(gi,gj)->nothing`, preserving old behavior when no obstruction is given)
that it now calls per patched pair. Both `_compute_cpu` (CPU) and
`GPUAssembly.compute_view_factors_gpu` (GPU — the near-pair patch always
runs on the CPU even for a GPU backend) now build one `get_bvh` lookup and
pass it to both the bulk pair loop and the Duffy patch, so obstruction is
checked consistently everywhere a pair is evaluated.

**Bug 2 — ray/triangle watertightness crack on shared coplanar edges**:
fixing Bug 1 alone only reduced `F_group[1,2]` from `0.19982` to `~0.0016`,
not to `0.0`. Root cause: the test's blocking plane is two triangles split
along its diagonal, and the two plates' bilinear Quad4 map sends symmetric
Gauss-Legendre node pairs (`ξ=η`) onto that same diagonal, so several
quadrature-point ray pairs pierce the blocker *exactly* on the shared edge
between its two triangles. Möller–Trumbore there computes `u` (or `v`) as a
tiny negative number — e.g. `-2.1e-17` — on both triangles from floating-point
rounding, and the strict `u < 0.0`/`v < 0.0` rejection in `_ray_triangle`
(`src/BVH.jl`) then misses the hit on *both* sides of the edge: a classic
watertightness gap, not specific to this test's geometry.

**Fix 2**: added a small negative tolerance (`_BARY_EPS = 1e-9`, CPU;
`_GPU_BARY_EPS = 1f-6`, GPU) to the barycentric bounds checks in
`BVH._ray_triangle` and the matching inline test in
`GPUBVH.gpu_intersect_bvh`, so a ray landing on (or within the tolerance of)
a shared triangle edge registers as a hit on at least one side instead of
neither.

**Files**: `src/Assembly.jl` (new `build_bvh_lookup`, `_compute_cpu` uses
it, passes it to `patch_adjacent_pairs_duffy!`), `src/DuffyKernel.jl`
(`patch_adjacent_pairs_duffy!` signature + docstring), `src/GPUAssembly.jl`
(builds and passes `get_bvh` to the Duffy patch), `src/BVH.jl`
(`_ray_triangle` tolerance), `src/GPUBVH.jl` (`gpu_intersect_bvh`
tolerance).

**Tests**: full suite via `Pkg.test()`: all testsets passing, including
`ray_test.jl`'s "Full blocker leaves no leaks (CPU + GPU traversal)" (now
3/3, previously 2/3) and `duffy_correctness_test.jl` (12/12, unaffected by
the tolerance widening).
---

# 2026-09-04 — Howell catalog validation benchmark

Added a benchmark suite validating the package against published closed-form
configuration factors from J. R. Howell, *A Catalog of Radiation Heat Transfer
Configuration Factors*, 3rd ed. (thermalradiation.net), Section C (finite area
to finite area). No package source was changed; this is validation only.

**Files added**

- `benchmarks/howell/geom.jl` — Gmsh builders for the benchmark geometries
  (parallel/perpendicular plates in 2D and 3D, coaxial disks, concentric and
  parallel cylinder cross-sections). Normal orientation is set explicitly:
  3D plane surfaces via node winding, 2D curves by meshing the enclosed cavity
  so line normals are oriented inward by `load_mesh`.
- `benchmarks/howell/analytic.jl` — the catalog's governing equations,
  transcribed one function per case using the catalog's own variable
  definitions (C-1, C-2, C-3, C-4, C-11, C-14, C-40, C-41, C-63, C-68, C-69).
- `benchmarks/howell/run.jl` — driver; sweeps each formula over several
  parameter values, compares to `compute_view_factors`, writes `results.csv`.
- `benchmarks/howell/RESULTS.md` — method, results table, and error analysis.

**Result**: 39 parameter points across 10 catalog cases all agree with the
published values, median relative error 4.0e-7, worst 3.8e-3. Non-touching
geometries land between machine precision and ~1e-5. Edge-sharing geometries
are the least accurate (1e-4 to 4e-3) because of the 1/r^2 kernel singularity
at the shared edge; enabling `use_duffy=true` improves the 3D shared-edge case
C-14 from 3.5e-2 to 2.3e-4.

**Note on catalog conventions** (both found by debugging wrong answers): the
cylinder spacing `s` in cases C-68/C-69 is the surface-to-surface gap, not the
axis distance; and 2D curve groups must bound a meshed surface or their normals
are never oriented and every view factor comes back 0.

Coverage is 15 of the 162 Section C entries. The catalog publishes each
governing equation as a raster image, so each further case needs its formula
transcribed by eye plus a bespoke geometry.

## Bug fix: `reverse_normals` corrupted 2nd-order surface elements

Found by this benchmark, while adding the cylinder/cone/sphere cases (the first
3D cases needing `reverse_normals=true`).

`_reverse_all_normals!` reversed element winding by swapping corner nodes but
permuted the mid-side nodes incorrectly, leaving them attached to the wrong
edges and silently corrupting the isoparametric map:

- **Quad8** (`:quad`): swapped nodes 5 <-> 7. With corners 1-4 and mid-sides
  5=(1,2), 6=(2,3), 7=(3,4), 8=(4,1), swapping corners 1 <-> 3 sends edge (1,2)
  to (3,2), whose mid-side node is the old 6. Correct permutation is
  5 <-> 6 and 7 <-> 8.
- **Tri6** (`:tri`): swapped nodes 4 <-> 6. Correct is 4 <-> 5, with node 6
  unchanged because edge (3,1) maps onto itself.

First-order (Quad4, Tri3) and line elements were handled correctly and are
unaffected.

**Impact**: silent and severe. Any second-order surface mesh loaded with
`reverse_normals=true` produced badly wrong view factors with no warning. On a
closed unit cube: face areas 0.9558 instead of 1.0, and row sums 0.108 instead
of 1.0 — view factors roughly 10x too small. Benchmark case C-79 (cylinder base
to inside surface) went from 90% error to 2e-4 after the fix.

**Files**: `src/MeshIO.jl` (`_reverse_all_normals!` and its docstring),
`test/mesh_test.jl` (new regression testset "reverse_normals preserves
2nd-order element geometry", checking that each mid-side node still lies at the
midpoint of its own edge after reversal, that the winding actually flips, and
that element area is unchanged).

Full existing test suite still passes.

## Benchmark extended to curved 3D bodies

Added cases C-79 (cylinder base to inside lateral surface), C-109 (cone
interior to base), C-125 (sphere to coaxial disk) and C-135 (concentric
spheres), taking the suite to 52 parameter points over 15 catalog cases, all
agreeing within 0.4% (median 9.3e-7).

Also recorded: the Duffy correction applies only to quad pairs and silently
skips triangles. Case C-109 shows the cost directly — the same cone geometry
gives 3.7e-4 meshed with quads but 3.6-9.2% meshed with triangles, and
refining the triangle mesh does not reliably help. Edge-sharing geometries
should be meshed with quads.

## Bug fix: obstruction BVH cache was not thread-safe

Found while adding the cylinder-array cases below (C-72, C-73), which each
need up to 25 obstruction groups and reproducibly hit:

```
AssertionError: Multiple concurrent writes to Dict detected!
```

`build_bvh_lookup` in `src/Assembly.jl` memoises the merged obstruction
geometry per distinct set of active groups in a plain `Dict`, filled with
`get!` from inside the threaded assembly loop. A Julia `Dict` is not
thread-safe: two threads inserting distinct keys can trigger a concurrent
rehash and corrupt it. Every case in this suite prior to the arrays used at
most 2-3 obstruction groups, so the cache filled within the first few pairs
and the race window was rarely hit in practice; dozens of groups reproduces it
reliably.

Fixed by guarding the cache with a `ReentrantLock`, matching the pattern
already used for the Golub-Welsch quadrature-rule cache in
`src/Quadrature.jl` (`_GL_LOCK`). **Files**: `src/Assembly.jl`
(`build_bvh_lookup`).

## Benchmark extended to tabulated-reference cases

Added seven cases whose reference values come from the catalog's own
published tables rather than a closed-form equation — C-8 (plane to two rows
of tubes), C-10 (rectangle to a semi-infinite rectangle at an angle), C-33
(hexagonal prism, 6 face-pair view factors), C-34 (parallel regular polygons,
n=3 to 8), C-72/C-73 (a cylinder in a square/triangular array, with real
multi-body obstruction), and C-137 (two spheres of unequal radius) — taking
the suite to 115 parameter points over 22 catalog cases, 109/115 (95%) within
1%, median 3.0e-5.

The reference values themselves are extracted programmatically from the
catalog's page HTML into `tables.jl` rather than transcribed by eye, and
cross-checked against closure identities each geometry must satisfy (e.g.
C-33's six view factors from one hexagon wall sum to 1) before being trusted.

Findings from this batch:

- **A group can't obstruct a pair involving itself.** C-8's tube rows and the
  cylinder arrays only shadow correctly once every individual tube/cylinder is
  its own obstruction group — grouping a whole row together silently disabled
  tube-on-tube shadowing and inflated the computed factor. This is documented
  in `RESULTS.md` note 3, since it's an easy trap for anyone building a
  multi-body obstruction case.
- **Obstruction accuracy is limited by facet count.** A curved blocker's
  obstruction soup uses only element corner nodes, so it's really an
  inscribed polygon, slightly smaller than the true body. C-8's second tube
  row (partially shadowed by the front row) converges steadily as the circle
  facet count increases (40 -> 320 segments takes it from 0.2023 to 0.1957
  against the catalog's 0.1953) but needs real refinement, not just a finer
  quadrature rule.
- **A "semi-infinite" plate needs a graded mesh.** Representing an infinite
  rectangle with a finite one of width `W`, meshed uniformly, leaves huge
  aspect-ratio elements once `W` exceeds a few times the near-field scale, and
  the view factor stops converging as `W` grows further. C-10's builder grades
  the far plate's mesh instead.

Two further tabulated cases were attempted and **not included**, both
because the published figure is genuinely ambiguous about which surface is
meant, and no reading tried reproduced the table (see `RESULTS.md` for the
specific configurations tried and their results):

- **C-154** (two hemispheres in contact): four candidate orientations tried,
  errors 39-83%, two of them producing a self-view-factor exceeding 1 (a
  physical impossibility, indicating those geometries aren't what the source
  intends).
- **C-35** (rectangle to one quarter of a parallel cylinder): four candidate
  90° arcs tried, errors 60-83%, one producing an exact 0.

Two more tabulated cases were reviewed and set aside as poor fits for this
kind of validation rather than attempted: C-13c publishes a Monte Carlo
result for a specific "W-shaped tube" cross-section with no geometry given on
the page; C-101 turns out to need raster equation images in addition to its
table, putting it with the equation-based cases rather than this batch. A
twelfth candidate, C-113 (cone frustum to base disk), was left for a future
pass — its table's exponent formatting needs cleanup and it has three
independent geometric parameters rather than two.

# 2026-09-09 — Benchmark extended to the Monte Carlo kernel

Every case in `benchmarks/howell/` now runs with both solver kernels —
deterministic quadrature and `monte_carlo=true` stratified sampling — on the
same mesh, so the comparison isolates the kernel rather than mixing in a
discretization difference. Quadrature: 109/115 (95%) within 1% of the
catalog value, median 0.005%. Monte Carlo (`n_samples=5000`): 110/115 (96%)
within 1%, median 0.007% — MC tracks quadrature closely rather than adding a
materially larger error band, including on the edge-sharing cases, where
`compute_view_factors` always Duffy-patches near/touching pairs regardless of
kernel (per its own docstring), so MC's worst-case error there matches
quadrature's almost exactly.

## Practical finding: Monte Carlo cost scales as 1/msz⁴, not 1/msz²

Adding the MC comparison first ran into a multi-hour stall: with the
original sphere meshes and `n_samples=10000`, C-135 (concentric spheres) and
C-125 (sphere to disk) each took 60-80 minutes for 3-4 parameter points,
against a few minutes for the whole 115-point suite under quadrature alone.

Two independent scalings compound. Element count `N` for a fixed-size body
scales as `1/msz²` (`msz` = target mesh element size). Monte Carlo's bulk
cost is `O(N²)` pairs `× n_samples` per pair. Halving `msz` doesn't double
the cost — it quadruples `N`, so pair count goes up 16x — meaning total MC
cost scales as `1/msz⁴`. The sphere builders' original `msz = radius/5` to
`radius/8` produced meshes fine for quadrature (nquad=6 is 36 points/pair)
but, at `n_samples=10000` (278x more points/pair), made the O(N²) pair count
the entire problem.

Fixed by coarsening the three sphere builders (`concentric_spheres`,
`sphere_to_disk`, `two_spheres` in `benchmarks/howell/geom.jl`) — roughly
doubling `msz`, cutting pair count ~16x alone — combined with dropping
`MC_SAMPLES` from 10000 to 5000 (2x). C-135 and C-125 dropped to 2.5 and 4.3
minutes; the full 230-point suite (115 points x 2 kernels) now completes in
about 2 hours. This did cost some quadrature accuracy on the same, now
coarser, mesh (C-125's worst case: 2.9e-4% to 4.8e-3%; C-135's: 1.7e-3% to
0.027%) — both still far under the 1% threshold, so the trade holds, but the
underlying lesson (a mesh sized for quadrature isn't automatically reasonable
for MC at a given `n_samples`) applies to any future case.

**Files**: `benchmarks/howell/run.jl` (kernel loop, `MC_SAMPLES`, per-kernel
CSV/summary output), `benchmarks/howell/geom.jl` (sphere mesh sizing),
`benchmarks/howell/RESULTS.md` (full quad-vs-MC comparison table and this
finding, written up in more detail there).

## 2026-09-14 — Documentation fix: `nquad` counts points per *element*, not per pair

**Bug (documentation only, no behaviour change)**: three places stated that
`nquad=4` gives "16 points per element pair". It gives `nquad²` = 16 points on
*each* element, so the pair integral in `_integrate_pair` costs `nquad⁴` = 256
point-*pairs* per element pair. `docs/src/manual/integration_methods.md`
already had it right for the Duffy path ("`nquad⁴` for standard quadrature"),
so the package contradicted itself.

**Why it matters**: `n_samples` counts point-pairs directly, so the natural
reading — compare `n_samples` against `nquad²` — overstates Monte Carlo's
relative cheapness by a factor of `nquad²`. The §"Monte Carlo cost scales as
1/msz⁴" entry above is itself affected: it reasons from "nquad=6 is 36
points/pair" and derives "at `n_samples=10000`, 278x more points/pair". With
the correct `nquad⁴` = 1296 the true ratio is 7.7x, not 278x. That entry's
remediation (coarsening the sphere meshes, halving `MC_SAMPLES`) was still
directionally right and its measured timings stand; only the stated factor
was wrong.

**Files**: `src/Assembly.jl` (`compute_view_factors` docstring, `nquad` and
`n_samples` entries), `docs/src/manual/integration_methods.md`,
`docs/src/manual/performance.md`.

## 2026-09-14 — Monte Carlo is the wrong kernel for obstructed geometries

Measured while computing view factors on a 2nd-order reactor fuel-assembly
slice (`FA_slice_2o.msh`, Quad9 → Quad8, pin surfaces + assembly wall) with
`obstruction_groups` set.

`compute_view_factors(...; monte_carlo=true, n_samples=5000)` on the
Central_Support ↔ Pin_23 case (N=4320, 6 groups, obstruction on) took
**1368 s**. The same case at `nquad=4` took **60.4 s** — 22.6x faster — and
the two agree to five decimals:

| Method | F(Central_Support→Pin_23) | F(Pin_23→Central_Support) | Time |
|---|---|---|---|
| MC, `n_samples=5000` | 0.380439 | 0.213580 | 1368 s |
| Quadrature, `nquad=4` | 0.380440 | 0.213580 | 60.4 s |
| Quadrature, `nquad=6` | 0.380403 | 0.213560 | 306 s |

Two compounding causes, both measured on an 8-core M3 (N=1440 fixture, no
obstruction):

1. **Point-pair count.** `n_samples=5000` versus `nquad=4`'s `nquad⁴` = 256 is
   19.5x more kernel evaluations — and, with obstruction on, 19.5x more BVH
   ray casts, because `_integrate_pair` and `element_pair_view_factor_mc`
   apply the identical `K == 0` guard before calling `is_visible`. The
   docs claimed MC wins on obstructed geometries because it "pays the BVH cost
   only for kernel-positive pairs"; quadrature does exactly the same, so there
   is no structural advantage. That guidance is now corrected.
2. **Per-evaluation cost.** At *matched* work (256 point-pairs each):
   quadrature 0.16 s, MC 1.09 s — MC is 6.8x slower per evaluation. Its loop
   reloads both sides per sample and gathers element-j data through a random
   index; quadrature's nested loop hoists element-i data out of the inner loop
   and keeps a ~1.8 kB working set in L1.

Net: 16.08 s (MC, 5000) versus 0.16 s (`nquad=4`) unobstructed — ~100x.
The obstructed ratio is smaller (22.6x) because the shared ray-cast cost
dominates both paths and compresses the gap toward the 19.5x count ratio.

The Metal GPU backend does not rescue the MC path: the same obstructed N=4320
case reached 25% of pairs in 761.7 s, extrapolating to ~51 min, against
22.8 min on 8 CPU threads. Per-sample BVH traversal is branch-divergent and
parallelizes badly. Not worth pursuing; Metal remains a `[weakdeps]`
extension and was not promoted.

**Files**: `docs/src/manual/integration_methods.md` ("When to use" and method
table), `src/Assembly.jl` (`n_samples` docstring).

## 2026-09-14 — Rejected optimization: cyclic-offset sample pairing in the MC kernel

`element_pair_view_factor_mc` draws `kj = rand(rng, 1:n)` per sample, which
reads element-j's sample arrays out of order. Replacing it with a single
random cyclic offset per pair (`kj = ((k-1+off) mod n)+1`) restores sequential
access and is **2.2x faster** at `n_samples=5000` (52.7 → 23.2 µs per pair;
1.05 vs 2.25 µs at n=256). It is also exactly unbiased: for fixed `k` a
uniform `off` makes `kj` uniform on `1:n`, so every term keeps the same
expectation.

**Rejected anyway.** Measured over 40 seeds on four facing Quad8 pairs from
the reactor-pin mesh, the means agree (|Δmean| ≈ 0.9–1.0 SEM, no detectable
bias) but the per-pair standard deviation is ~30x worse:

| Pair | mean (random idx) | sd | mean (cyclic) | sd |
|---|---|---|---|---|
| 1314-274 | 0.00148143 | 7.52e-6 | 0.00151512 | 2.35e-4 |
| 1246-287 | 0.00168612 | 9.98e-6 | 0.00173862 | 3.47e-4 |
| 1127-287 | 0.00110550 | 7.53e-6 | 0.00113845 | 2.04e-4 |
| 1195-235 | 0.00120976 | 6.47e-6 | 0.00124293 | 2.12e-4 |

The cause: one offset leaves a single random degree of freedom per pair
instead of `n`, so the `n` terms move coherently with it rather than
averaging down. Recovering that accuracy costs far more samples than the 2.2x
saved. Shuffling the sample arrays first does not help — the offset is still
one degree of freedom.

Hoisting a `Random.Sampler` out of the loop was also tried and does nothing
(52.7 → 49.2 µs), so the `rand` call itself is not the cost; the random memory
access is.

**Files**: `src/MCKernel.jl` — code unchanged; the `element_pair_view_factor_mc`
docstring and inner loop now carry a note recording this measurement, so the
random draw is not "optimized" away again.

## 2026-09-14 — Pair-level facing cull in all four kernels

New `src/ElementBounds.jl`, wired into the CPU quadrature, CPU Monte Carlo, GPU
quadrature and GPU Monte Carlo paths, behind
`compute_view_factors(...; facing_cull=true)` (the default; pass `false` to
disable).

**Why**: both pair integrators already returned zero for a *point* pair with
non-positive cosines (`K == 0 && continue`), but discovered that one point pair
at a time — after entering the pair and, with obstruction on, after a BVH ray
cast. On closed convex bodies most element pairs face away from each other
entirely, so that per-point test was paid `nquad⁴` (or `n_samples`) times per
pair to conclude nothing. This came out of a comparison with pyViewFactor,
which masks non-facing pairs wholesale before integrating.

**The bound**: each element gets an axis-aligned box for its points and another
for its unit normals, both sampled on a fixed grid and padded outward. For
p ∈ i, q ∈ j the separation lies in the Minkowski difference of the point
boxes, and maximising each component independently gives an upper bound on
dot(n, q-p); if it is ≤ 0 the kernel is zero at every point pair. Rejection is
therefore conservative and the assembled matrix is unchanged.

**Two earlier variants were measured and discarded**, on a 1440-element
reactor-pin fixture where 16.9% of pairs are genuinely nonzero (so 83.1% is the
ceiling):

| Bound | Pairs rejected |
|---|---|
| Bounding sphere + circular normal cone | 13.3% |
| Bounding sphere + normal box | 36.8% |
| Point box + normal box (adopted) | 62.0% |

The cone fails because its half-angle α contributes slack growing like
`|d|·sin α`, which swamps the test at range. On an extruded mesh the normals of
a cylindrical element have *no* axial component, but a circular cone inflates
that 1-D circumferential arc into a 2-D disc admitting axial tilt. The sphere
fails for the same reason in position space: it contributes `Rᵢ + Rⱼ` in every
direction, which badly over-bounds elements far taller than they are wide.
Both boxes represent the extruded case exactly.

**Verification**: on the 1440-element fixture and the 4320-element 6-group
reactor case, with and without obstruction, `F_elem` is **bitwise identical**
with the cull on and off, and a direct sweep of all N(N-1)/2 pairs confirms
**zero** pairs rejected that had a nonzero reference value. The GPU quadrature
kernel is likewise bitwise identical in both `raw_out` and `area_out`. Rejection
rates were 62.0% and 65.6%.

**GPU specifics**: the quadrature kernel skips the double loop but still writes
`area_out`, since an element all of whose pairs were rejected would otherwise
get no area and a NaN row. The MC kernel cannot skip its sample loops at all —
`area_out` is estimated from those very samples — so it culls only the
kernel/BVH evaluation, which is where the cost sits. Bounds are nudged one ulp
outward when narrowed to Float32 for Metal, so the device copy is never tighter
than the host one.

**Files**: `src/ElementBounds.jl` (new), `src/RadiativeViewFactor.jl` (include
and re-export), `src/Assembly.jl` (`facing_cull` kwarg, both CPU pair loops,
cull-rate diagnostic, GPU hook), `src/GPUKernels.jl` (`gpu_pair_can_see`,
`_pack_bounds`, kernel argument and guard, launcher), `src/GPUMCKernels.jl`
(kernel argument, guard at both `_vf_contribution` sites, launcher),
`src/GPUAssembly.jl` (plumbing).

## 2026-09-14 — Pre-existing: GPU Monte Carlo element areas are nondeterministic

Found while verifying the facing cull, and *not* caused by it.

`_mc_pair_kernel!` writes `area_out[i]` from every thread in row i. Each thread
seeds its RNG from its own `(i,j)` thread id, so every writer holds a
*different* Monte Carlo estimate of the same element area, and which one
survives depends on the race. Two runs with identical settings and an identical
`seed` produced areas differing by up to 8.3e-5 relative (max absolute
2.9e-6) on a 1440-element mesh.

Every value written is a valid estimate, so this is noise rather than error, but
it means GPU MC results are not bitwise reproducible even with a fixed seed, and
that ~1e-4 area noise propagates straight into `F = raw / A`. The CPU MC path
does not share the problem: it draws one sample set per element (O(N)) and takes
the area from that. Not fixed here — the GPU MC path is deprioritised after the
obstruction measurements above — but worth knowing before trusting a GPU MC run
to better than ~0.01%.

**Files**: none (diagnosis only).

## 2026-09-14 — `radiating_groups`: assemble a subset while everything still shadows

New `restrict_to_radiating(mesh, tags)` in `src/MeshIO.jl`, exposed as
`compute_view_factors(...; radiating_groups=Int[])`.

**Why**: assembly is dense over whatever it is handed, so asking for the view
factor between *one* pair of surfaces in a mesh that also contains two dozen
shadowing bodies cost `O(N_total²)` — and most of that work computed view
factors *between the shadowing bodies*, which nobody asked for. This was the
dominant term in a comparison against pyViewFactor, which lets obstruction
geometry stay out of the pair enumeration entirely; it was worth 9x to 37x in
pair count on the reactor cases, far more than the facing cull recovered.

**How**: the restriction happens at the `MeshData` level, before backend
dispatch, so both the CPU and GPU paths get it with no per-path plumbing.
`surface_elems`, `group_tags` and `group_elems` are cut to the nominated groups
and renumbered; `group_tri_soup` — the only thing the obstruction BVH reads — is
carried over **whole**, so occlusion is completely unaffected.

**Verified bitwise.** Each pair's integral depends only on its two elements and
the obstruction BVH, all identical between the two runs, so the expectation was
bitwise equality rather than mere agreement. That is what came out, on all three
reactor cases at `nquad=4` (obstruction on, subset compared against the
full-assembly value):

| Case | radiating / loaded elements | F, full assembly | F, subset | Bitwise |
|---|---|---|---|---|
| Central_Support → Pin_23 | 1440 / 4320 | 0.38043964100835853 | 0.38043964100835853 | yes |
| Pin_23 → Pin_11 | 1920 / 9120 | 0.03905464385428897 | 0.03905464385428897 | yes |
| FA_wall → Pin_11 | 3040 / 18400 | 0.0023661623551932535 | 0.0023661623551932535 | yes |

Group areas match bitwise too. Wall-clock, comparing against full-assembly
timings recorded in an earlier process: Pin_23 → Pin_11 223.9 s → 10.3 s
(21.7x, against a 22.6x pair-count ratio) and FA_wall → Pin_11 1358.0 s →
101.5 s (13.4x, against 36.6x — the excluded pin-pin pairs were
disproportionately cheap ones the facing cull already rejected). The
Central_Support → Pin_23 arm is not quoted: its full-assembly run absorbed
first-call compilation in that process and its 50x is not a fair number.

Peak memory falls with the same square: FA_wall → Pin_11 holds a 3040² matrix
rather than 18400², about 150 MB against 5.4 GB — which is what makes an
`nquad=6` run of that case feasible on a 16 GB machine at all.

**Caveat, documented in three places**: the enclosure is deliberately open. The
non-radiating bodies still absorb but are never assembled, so `Σⱼ Fᵢⱼ < 1` and
`check_closure` is not meaningful on such a result; `compute_view_factors`
prints a warning line when subsetting, and `check_closure`'s own docstring now
says so. Reciprocity is unaffected and remains the right validation.
`write_nekrs_view_factors` already refused a row-count mismatch; its error now
names this as a cause and points at `restrict_to_radiating`.

**Files**: `src/MeshIO.jl` (`restrict_to_radiating`, exported),
`src/RadiativeViewFactor.jl` (re-export), `src/Assembly.jl` (`radiating_groups`
kwarg, restriction before dispatch, docstring), `src/Results.jl`
(`check_closure` caveat), `src/NekExport.jl` (error message).

## 2026-09-14 — Howell benchmark suite re-run with `radiating_groups`, plus provenance

The Section-C suite now uses `radiating_groups` wherever a case's extractors
read only some of its mesh's physical groups, and `results.csv` records the
machine and the per-call runtimes alongside the errors.

**Restriction applied to three case families**:

| Case | Radiating / loaded elements | Points restricted |
|---|---|---|
| C-72 (cylinder, square array) | 144 / 1200 | 6 of 6 |
| C-73 (cylinder, triangular array) | 144 / 1200 | 6 of 6 |
| C-33 (hexagonal prism) | 157–1386 of 689–2135 | 15 of 18 |

C-72/C-73 are the clearest win: all 25 cylinders must remain obstructors but
only the centre cylinder and its two reference neighbours are ever read, so the
assembled matrix falls from 1200 elements to 144 — about 70x fewer pairs, on
cases that are obstructed and therefore expensive per pair. Both now complete in
under a second per kernel. C-8 was deliberately left alone: its extractors need
all but one of ~33 groups, and the tube counts are not known until the mesh is
built.

**Verified exact, not assumed.** All 115 quadrature rows are *bitwise identical*
to the same suite run without `radiating_groups`. Monte Carlo rows are not (57 of
115 match): restricting the element set, and separately the new facing cull,
change how many `rand` draws the per-row RNG streams consume, so the sampling
differs. Worst relative change across all MC rows is 6.9e-4 — inside the
kernel's own noise — and both pass counts are unchanged at 109/115 quad and
110/115 MC.

**A trap worth recording**: the case extractors index `F_group` positionally and
`_aggregate` orders groups by sorted tag, so restricting to a non-prefix set such
as `{1,4}` moves tag 4 into position 2 and `fij(1,4)` then silently reads the
wrong factor — wrong numbers, no error. `run_case` now rejects any radiating set
that is not the prefix `1:k`.

**Provenance and runtimes**: `results.csv` gains `n_radiating`, `restricted` and
`seconds` columns plus a commented preamble (date, CPU, core counts, Julia
threads, memory, OS, Julia and package versions, git commit with dirty flag,
`n_samples`, total wall time). Note `Sys.CPU_THREADS` reports 4 on this M3 —
it counts performance cores only — so the OS figure (`hw.logicalcpu` = 8) is
recorded beside it rather than leaving an ambiguous count. A warm-up case runs
before the timed loop so per-case times exclude first-call compilation, and
`flush(stdout)` after each case keeps progress visible (without it the entire
run's stdout sat in a buffer and the log showed only stderr warnings).

**Measured**: total 6306 s (1 h 45 m) on an Apple M3 with 8 Julia threads —
565 s quadrature, 5723 s Monte Carlo, so MC costs 10x quadrature over the suite.
Two patterns are visible. C-10 alone is 58% of the total (3665 s), because its
geometry meshes a semi-infinite rectangle as a finite one of width 80. And the
MC/quad ratio is *worst on the cheapest cases* — over 200x on C-1 to C-4 —
since MC pays `n_samples` point-pairs per element pair no matter how simple the
geometry, while `nquad=6` quadrature pays `nquad⁴` = 1296 and scales down with
the mesh.

**Files**: `benchmarks/howell/run.jl` (system info, warm-up, per-kernel timing,
`radiating` case option and prefix guard, extended CSV, stdout flush),
`benchmarks/howell/RESULTS.md` (regenerated tables, new System, Runtimes and
Restricted assembly sections), `benchmarks/howell/results.csv` (regenerated).

# 2026-09-15

Context: the Howell benchmark (above) showed the CPU Monte Carlo kernel
running 6x-221x slower than quadrature per case (see
`benchmarks/howell/RESULTS.md`, "Runtimes"), in contrast to MC's usual
speed advantage on complex geometry. Investigated where the MC assembly
time actually goes (per-case wall-time split, per-pair micro-benchmarks) and
made three changes that reduce it without changing what is estimated.

**Investigation** (no source changed at this stage; scratch scripts, not
committed): splitting `compute_view_factors(monte_carlo=true)`'s wall time
into element sampling, the O(N²) pair loop, and the (then-serial)
adjacent-pair Duffy patch showed the pair loop was 80-100% of MC time across
six Howell cases spanning Quad8, Tri6 and Line3 (obstructed and
unobstructed). A per-pair micro-benchmark on one Quad8 pair then showed the
per-sample `rand(rng, 1:n)` index in `element_pair_view_factor_mc`
(`src/MCKernel.jl`) cost about half the per-pair time (26.9 µs -> 13.0 µs
without it), because indexing the second element's samples out of order
defeats prefetching — matching the cost this same function's own docstring
had already measured and attributed to that index. Replacing it with a
same-index (sequential) pairing after storing each element's samples in a
random (`randperm`) order gave the same speedup *and*, measured over 400
trials, roughly 5x lower per-pair standard deviation (5.0e-4 -> 1.0e-4
relative) than the random-index version it replaced — not the single-offset
scheme the old docstring warned against (that preserves adjacency between
stratum indices and correlates nearby terms; a `randperm` does not).

## 1. `sample_element_mc` stores samples in random order; `element_pair_view_factor_mc` pairs them sequentially

Changed `sample_element_mc` to apply one `randperm(rng, n)` to the drawn
samples before storing them in `ElementSamples`. `element_pair_view_factor_mc`
now pairs same-index samples (`xs_i[k]` with `xs_j[k]`) instead of drawing a
fresh random index per sample. Because both arrays are independently
shuffled, this still pairs each `xᵢ[k]` with a uniformly random `xⱼ` — the
estimator is unchanged, only how the random matching is generated. `rng` is
still accepted by `element_pair_view_factor_mc` for API stability but is no
longer used inside it (all randomness was already spent when the samples
were drawn).

Verified on 5 assembly-level Howell cases (`benchmarks/howell/`, 6 seeds
each, `n_samples=5000`) spanning Quad8 (C-11, C-40), Tri6 (C-135), Line3
(C-68), and an obstructed Line3 case (C-8): group-level view-factor noise
(std. dev. across seeds) was equal to or lower than before the change in
every case, and wall time dropped 1.25x-2.8x on the unobstructed cases
(C-135: 7.8s -> 2.8s; C-40: 3.2s -> 2.0s; C-11: 1.2s -> 0.84s). C-8, which is
dominated by BVH ray casts rather than the pairing itself, was not
meaningfully affected by this change alone.

**Files**: `src/MCKernel.jl` (`sample_element_mc`,
`element_pair_view_factor_mc`, docstrings).

## 2. `n_samples` default lowered 10000 -> 5000

The MC estimator's own noise at `n_samples=5000` is typically 10x-1000x
smaller than the mesh's own discretization error against the analytic
answer (measured across the same 5 Howell cases above): e.g. C-135's noise
sd is 2.5e-6 against a 2.5e-4 systematic mesh error, C-8's is 9.3e-6 against
1.4e-2. The previous default of 10000 samples was mostly paying for
precision the mesh could not use. `n_samples=5000` is also the exact value
already validated across the full 115-point Howell catalog suite
(`benchmarks/howell/RESULTS.md`: 96% of points within 1% of the published
value, median error 0.008%), so this is not a new, unvalidated setting.

**Files**: `src/Assembly.jl` (`compute_view_factors` and `_compute_cpu`
keyword defaults, docstring).

## 3. Adjacent-pair Duffy patch is threaded; those pairs are skipped in the MC bulk loop

`compute_view_factors(monte_carlo=true)` always overwrites near/adjacent
element pairs (found by `near_pairs`) with a deterministic Duffy-transform
value after the MC bulk pass, because the 1/r² kernel has unbounded MC
variance there — so whatever the bulk loop computes for those O(N) pairs is
discarded. `patch_adjacent_pairs_duffy!` (`src/DuffyKernel.jl`) now applies
those patches with `Threads.@threads` instead of serially (each pair writes
distinct matrix entries, and the obstruction-BVH cache it reads through was
already lock-guarded — see the 2026-09-14 entry above). `_compute_cpu`
(`src/Assembly.jl`) now computes `near_pairs` once, before the MC bulk loop,
and uses it both to skip those pairs in the bulk loop (saving `n_samples`
wasted kernel evaluations, and BVH ray casts where obstruction is enabled,
per pair) and as the patch's input, instead of the patch recomputing the
same list afterward. Curve meshes (`mesh_dim == 1`), where the Duffy patch
is a no-op, get an empty pair list so nothing is skipped there — verified
this doesn't silently zero out 2-D adjacent pairs, since skipping without a
patch would leave them at their initialized 0.0 permanently.

**Files**: `src/DuffyKernel.jl` (`patch_adjacent_pairs_duffy!` threaded,
takes optional precomputed `pairs`), `src/Assembly.jl` (`_compute_cpu`
computes and reuses `near_pairs` once).

## Combined effect, measured head-to-head against the unmodified `v0.6.3` baseline

Same `n_samples=5000` on both sides (isolating the algorithmic changes from
the default-value change in §2 above), 8 Julia threads, Apple M3, each side
run alone (not concurrently with the other, to avoid CPU contention between
the two Julia processes skewing the numbers — an earlier concurrent run
showed exactly that: C-79 and C-109 looked artificially slow on the baseline
side because the branch's own C-8 run was still using all 8 threads).
Median of 5 seeds per case (3 for the obstructed case, which is much slower
per run):

| Case (Howell) | N elements | baseline median | branch median | speedup |
|---|---:|---:|---:|---:|
| C-11 (Quad8) | 288 | 1.223 s | 0.274 s | 4.5x |
| C-40 (Quad8) | 486 | 3.354 s | 0.631 s | 5.3x |
| C-135 (Tri6) | 996 | 8.526 s | 1.956 s | 4.4x |
| C-79 (Quad8) | 1169 | 13.589 s | 3.391 s | 4.0x |
| C-109 (Tri6+Quad8) | 2127 | 42.220 s | 11.290 s | 3.7x |
| C-68 (Line3) | 144 | 0.026 s | 0.017 s | 1.5x |
| C-8 (Line3, obstructed) | 1178 | 71.606 s | 66.609 s | 1.07x |

The combined speedup (3.7x-5.3x on the larger unobstructed cases) is well
above what §1 (pairing) alone gave (1.25-2.8x, measured earlier with 6
seeds): §3's near-pair skip turns out to matter more than expected on
meshes with many adjacent pairs — C-79 and C-109 have 46,960 and 109,048
near pairs respectively (`near_pairs(...; factor=3.0)`), each of which the
unmodified bulk loop spent a full `n_samples`-sample MC estimate computing
and then discarded when the Duffy patch overwrote it.

C-68 (a small, cheap case) speeds up much less in absolute terms because the
Duffy patch's own fixed cost (now threaded, but still nonzero) is a larger
fraction of its total time. C-8, the one obstructed case measured, is
dominated by BVH ray casts rather than the pairing or patch cost, so it
sees only a modest 1.07x from these three changes alone — its share of the
overall speedup for obstructed meshes comes mainly from §2's lower default
`n_samples`, not from this table (which holds `n_samples` fixed to isolate
§1 and §3).

Accuracy (group-level view-factor error against the Howell catalog, same
seeds) was unchanged to first order in every case above; per-case numbers
are in the investigation notes referenced below.

**Verified**: full test suite (`Pkg.test()`, 8 threads) passes unchanged,
including the two tests that exercise these paths directly — "MC pair
estimator is unbiased for coarse elements" (`test/ray_test.jl`) and "Monte
Carlo + near-pair Duffy patch closes the cube" (`test/duffy_correctness_test.jl`).
Branch: `mc_speedup`.

## New CPU kernel: ray-shooting Monte Carlo (`compute_view_factors(...; raytrace=true)`)

Context: even with the §1-§3 speedups above, the existing (pair-area
sampling) Monte Carlo kernel is architecturally O(N²) — it always pays a
point-pair visibility check per element pair, the same complexity class as
quadrature, just with a larger constant. Real ray-tracing view-factor codes
get their speed advantage on complex/obstructed geometry from a genuinely
different algorithm (O(N) rays per element against one whole-scene BVH, not
O(N²) pairs), which this package didn't have. Investigated and then
implemented that algorithm as a new, independent CPU kernel.

### The method

For each element i, draw a point on it (uniform by area) and a direction
from the cosine-weighted hemisphere around its normal (pdf ∝ cos θ), then
find the first surface that ray hits via one nearest-hit BVH query over the
*whole* radiating mesh. Because dA_j cos θ_j / r² = dΩ (the definition of
solid angle), cosine-weighted sampling's pdf exactly cancels the cos θ_i
factor in the view-factor kernel, so `E[1(first hit is j)] = F_{x→j}`
(the point-to-area form factor, occlusion included) — no explicit 1/r²,
cos θ_j, or separate visibility test needed; obstruction is a side effect
of the same nearest-hit query. This is the standard Monte Carlo
"ray-shooting"/"shooting rays" view-factor method (see e.g. Walker (1998),
*Review of methods used to compute view factors*; the underlying solid-angle
identity is standard, e.g. Cohen & Wallace, *Radiosity and Realistic Image
Synthesis*, or Siegel & Howell, *Thermal Radiation Heat Transfer*).

### Prototyped first, in scratch scripts, not committed

Before writing any `src/` code: implemented the method standalone, reusing
`BVH.build_bvh` and the package's own ray/triangle and ray/AABB tests by
qualified (non-exported) access rather than duplicating that fragile
geometry code. Validated against known cases, catching two real findings
before any of this reached `src/`:

- **A front/back-face bug.** The first version attributed *any* nearest hit
  to its owning element, with no check on which face was struck. On a
  synthetic 3-body test scene (two facing plates + a third plate at the
  midplane as a full blocker, never listed in `obstruction_groups`), this
  gave F(bottom→blocker) = 0.411 from ray-shooting vs 0.0 from the existing
  quadrature kernel on the same pair — traced to the blocker's normal
  facing away from the source (a genuine back-face hit), which must be
  geometrically blocking (an opaque surface blocks a ray incident on either
  face) but must *not* count as landing on that element — exactly the
  `cos θ_j <= 0 → K = 0` check `MCKernel.jl`'s pair kernel already makes,
  just needed here too, evaluated on whichever element the ray actually
  hit. Fixed by computing the hit triangle's own normal and checking
  `dot(normal, -direction) > 0`; reversing the test blocker's winding then
  brought quadrature to 0.415, matching ray-shooting within MC noise.
- **Quadrature itself was the less accurate reference, on a sharp
  obstruction shadow boundary.** A blocker covering 80% of the aperture
  (thin visible rim) gave ray-shooting 0.0192-0.0193 (tight, multi-seed:
  mean over 10 seeds 0.01925, sd(mean) 4.8e-5) against a coarse
  quadrature+obstruction reference of 0.0172 — a 43σ disagreement by
  ray-shooting's own noise estimate, i.e. not noise. Refining the
  quadrature mesh/order (n=6→24 elements/plate, nquad=6→10) moved
  quadrature's answer from 0.0172 to 0.0189 to 0.0193, converging onto
  ray-shooting's value — confirming the coarse quadrature was under-resolved
  (fixed-order polynomial quadrature integrates a sharp visibility
  discontinuity poorly; Monte Carlo hit-counting has no such difficulty),
  not a ray-shooting bug.
- **Speed**, on a synthetic obstructed 3-plate scene (bottom/top plates +
  partial blocker, `n_samples`/`n_rays`=5000 both sides): ray-shooting was
  12.6x faster than quadrature and 48x faster than the pair-area MC kernel
  at 836 elements, with cost barely growing (5x for 11x more elements)
  where the other two scale close to O(N²) — see the table below for the
  same comparison against the real `src/` implementation.

### The real implementation

- **`src/BVH.jl`**: new `nearest_hit_bvh(bvh, origin, direction; t_min,
  skip)` — closest-hit traversal (standard shrinking-tmax pruning),
  alongside the existing any-hit `intersect_ray_bvh`. No winding culling,
  same convention as `intersect_ray_bvh`. `skip(tri_idx)` excludes
  candidates (e.g. the ray's own origin element) without losing their
  AABB-pruning benefit; chosen over a `t_min` epsilon because a self-hit's
  `t` is only ever *near* zero by floating point, not reliably
  distinguishable from a genuine nearby hit at grazing incidence.
- **`src/RayTraceKernel.jl`** (new module): `SceneBVH` (the merged scene
  BVH + a triangle→element index map), `build_scene_bvh` (corner-triangulates
  every radiating element — same v1,v2,v3/v1,v3,v4 split
  `MeshIO._build_group_obs_soups` already uses, so this is consistent with
  the faceting the rest of the package already applies to curved 2nd-order
  elements for occlusion — plus any `obstruction_groups` geometry not
  already part of the radiating set, tagged 0 as an opaque non-target),
  `cosine_dir` (branchless orthonormal-basis cosine-weighted hemisphere
  sampling — Duff et al., *Building an Orthonormal Basis, Revisited*, JCGT
  2017), and `raytrace_element` (shoots `n_rays` from one element, tallying
  front-face hits per target element, escaped rays, and absorbed/back-face
  rays).
- **`src/Assembly.jl`**: new `raytrace::Bool=false` / `n_rays::Int=10000`
  keywords on `compute_view_factors`. CPU-only; 3-D meshes only
  (`mesh_dim=2`); errors if combined with `monte_carlo=true` or `self_vf=true`
  (a ray is never tested against its own origin element — self-view is not
  supported by this kernel). `use_duffy`/the near-pair patch/`factor`/
  `facing_cull` do not apply (no singular pairs, no O(N²) pair loop to
  cull). **Reciprocity by construction**: every unordered pair {i,j} gets
  two independent raw-integral estimates (from i's rays and from j's);
  `_compute_cpu_raytrace` averages them into one shared value before
  dividing back out to F, the same `raw[i,j] == raw[j,i]` convention the
  other CPU paths already use — this makes results exactly symmetric
  despite being stochastic, and is also lower-variance than either row's
  estimate alone (averaging two independent unbiased estimates). One
  consequence: row sums on a closed enclosure are only *approximately* 1
  (ordinary MC noise, ~1e-3 at `n_rays`=50000 in testing) rather than exact
  to machine precision — each element's own rays alone would close exactly,
  but every off-diagonal entry blends in the *other* element's independent
  estimate too.
- Element areas (`A_elem`) come from quadrature (`precompute_quad`), not
  from the rays — there is no ray-based area estimate, and this keeps areas
  exact rather than adding noise to a quantity that didn't need it.

### A real behavioural difference from the other two kernels, not a bug

Because every radiating element is already in the scene BVH, radiating
elements obstruct each other **automatically**, with no need to list them
in `obstruction_groups`. The other two kernels only apply an obstruction
check for groups explicitly listed there (see the existing note 3 in this
changelog: "a group never obstructs a pair involving itself"). This means
`raytrace=true` can give a genuinely different (more physically complete)
answer than the other kernels on the same mesh with the same
`obstruction_groups=[]`, if the user's mesh has radiating bodies that
occlude each other but were never told to. Documented prominently in both
the `compute_view_factors` docstring and `RayTraceKernel.jl`'s module
docstring.

### Verified

- New `test/raytrace_test.jl` (16 tests): unbiased against the analytic
  two-plate value (`n_rays=200000`, `rtol=5e-3`) with exact reciprocity;
  closed-cube row-sum closure (`atol=0.01`, see the row-sum caveat above)
  and agreement with quadrature; the front/back-face regression case from
  prototyping (a correctly-oriented blocker now matches quadrature to
  within MC noise); self-obstruction without `obstruction_groups` on a
  partial-blocker case (`rtol=0.05` against quadrature+`obstruction_groups`
  on the same mesh); argument validation
  (`monte_carlo=true`/`self_vf=true` combined with `raytrace=true` both
  error); `@inferred` type stability on `build_scene_bvh`,
  `raytrace_element`, `cosine_dir`, `nearest_hit_bvh`.
- Full test suite (`Pkg.test()`, 8 threads): all tests pass, including the
  new file.
- Speed, real `src/` implementation, same synthetic obstructed 3-plate scene
  as the prototype (`n_samples`/`n_rays`=5000, `nquad`=6, both quadrature and
  pair-area MC using `obstruction_groups=[3]`; ray-shooting using neither):

| Elements | quadrature | pair-area MC | ray-shooting | RT vs quad | RT vs pair-area MC |
|---:|---:|---:|---:|---:|---:|
| 76  | 0.052 s | 0.105 s  | 0.017 s | 3.1x  | 6.2x  |
| 304 | 0.340 s | 1.316 s  | 0.069 s | 4.9x  | 19.1x |
| 836 | 3.063 s | 11.927 s | 0.264 s | 11.6x | 45.2x |

(Reported group-level F values agreed within MC noise across all three
kernels at every size, 0.0985-0.0998.) Cost is barely growing for
ray-shooting (0.017 → 0.069 → 0.264 s, ~15x for 11x more elements) where the
other two scale close to O(N²) (quadrature ~59x, pair-area MC ~114x for the
same 11x) — the expected complexity advantage, reproduced (and slightly
exceeded) by the real `src/` implementation versus the scratch prototype.

**Files**: `src/BVH.jl` (`nearest_hit_bvh`), `src/RayTraceKernel.jl` (new),
`src/RadiativeViewFactor.jl` (include order), `src/Assembly.jl`
(`raytrace`/`n_rays` arguments, `_compute_cpu_raytrace`), `test/raytrace_test.jl`
(new), `test/runtests.jl` (registration), `README.md` (usage section, file
tree). Branch: `mc_speedup`.

### Not done / scope of this pass

No GPU ray-tracing kernel (CPU only). No 2-D curve-mesh (`mesh_dim=1`)
support. No `self_vf` support. Not run against the `benchmarks/howell/`
catalog suite (only the focused `test/raytrace_test.jl` cases above) — a
natural follow-up, and the more convincing accuracy evidence long-term.
Corner-triangulation faceting bias on curved 2nd-order elements (same
caveat the existing obstruction soups already carry) not separately
quantified for this kernel.

## Ray-shooting kernel wired into the GPU path (`raytrace=true, backend=CUDABackend()`/`MetalBackend()`)

Context: the CPU ray-shooting kernel above was left GPU-less. The existing
GPU pair kernels (`GPUMCKernels.jl`, `GPUKernels.jl`) are one-thread-per-pair
(`ndrange=(N,N)`); ray-shooting's whole appeal is *not* being O(N²), so it
needed its own thread mapping (one thread per *element*) and its own
nearest-hit BVH traversal (the existing `gpu_intersect_bvh` is any-hit only,
built for "is this pair blocked", not "what did this ray land on").

### New GPU-side pieces

- **`src/BVH.jl` / `src/GPUBVH.jl`**: `gpu_nearest_hit_bvh` — a stackless,
  miss-link-based nearest-hit traversal (same pruning idea as the CPU
  `nearest_hit_bvh` added earlier: shrink the AABB test's `t_max` to the
  best `t` found so far), alongside the existing any-hit
  `gpu_intersect_bvh`. Also computes the winning triangle's front/back
  classification inline (the sign of `dot(cross(e1,e2), -direction)` — no
  normalization needed, sign is scale-invariant), since the vertices are
  already in registers when a candidate becomes the new best hit. New
  `build_flat_scene_bvh(mesh, obstruction_groups, FloatT, ArrayT)` builds
  the GPU ray-tracing scene by calling the CPU-side
  `RayTraceKernel.build_scene_bvh` (reusing the triangulation/tagging logic
  rather than duplicating it a third time) and flattening the result with
  the existing `build_flat_bvh` — `FlatBVH`'s `tri_group` field is reused to
  carry a *per-triangle element index* here rather than a physical group
  tag (0 = non-radiating blocker), a different meaning from what the pair
  kernels' `build_flat_bvh_from_mesh` puts in that same field, so the two
  BVHs are not interchangeable even though they share a struct.
- **`src/GPURayTraceKernels.jl`** (new module): the ray-shooting `@kernel`
  (one thread per element, `n_rays` shot serially, same stratified-point +
  independent-cosine-direction sampling as the CPU kernel), a small
  quadrature-based element-area kernel (reusing `GPUKernels`'
  `_quad8_point_and_jac`/`_tri6_point_and_jac`/`_quad4_point_and_jac`/
  `_tri3_point_and_jac` — exact areas, not sampled, matching the CPU path's
  choice of `precompute_quad` over an MC area estimate), and a chunked
  host-side launcher (mirrors `GPUMCKernels.launch_mc_kernel!`'s
  adaptive-chunk-size pattern, needed for the same reason: macOS kills
  long-running Metal command buffers). Reuses existing GPU building blocks
  rather than reimplementing them: `GPUMCKernels._prepare_elem`/
  `_sample_prepared` (the pair-area MC kernel's per-thread-hoisted uniform
  point+normal sampler — exactly what a ray origin needs too) and
  `GPUMCKernels._xorshift32`/`_init_rng` (the inline PRNG; a 64-bit RNG
  would dominate kernel runtime on GPUs that emulate 64-bit integer math in
  software, e.g. Apple Silicon). Cosine-weighted direction sampling is the
  same branchless orthonormal-basis construction as the CPU kernel (Duff et
  al., *Building an Orthonormal Basis, Revisited*, JCGT 2017), written in
  scalar components to match this file's existing GPU style.
- **`src/GPUAssembly.jl`**: `compute_view_factors_gpu` gained
  `raytrace`/`n_rays` keywords, dispatching to a new
  `_compute_gpu_raytrace` that mirrors `Assembly._compute_cpu_raytrace`
  exactly once the kernel returns — same reciprocity-by-construction
  averaging of the two independent per-element raw estimates, same
  approximately-(not exactly-)1 row-sum caveat on closed enclosures, same
  verbose diagnostics.
- **`src/Assembly.jl`**: the `raytrace=true, backend isa CPU` restriction is
  gone; validation (mutually exclusive with `monte_carlo`/`self_vf`, no
  curve meshes) now happens before the CPU/GPU backend branch, and both
  branches pass `raytrace`/`n_rays` through (`_gpu_compute_hook`'s
  signature grew those two arguments).

### Verified

Real GPU hardware (CUDA/Metal) was not available in this environment; tested
the same way the existing (untested-elsewhere) GPU pair kernels are — via
KernelAbstractions' `CPU()` execution backend, which runs the *actual* GPU
kernel code (not a separate CPU implementation) through KernelAbstractions'
backend abstraction, just on CPU threads instead of a real device. This
validates kernel correctness and the whole dispatch path, but is **not** a
GPU speed measurement — `backend=CPU()` here is a correctness harness, not a
benchmark, and no claim is made about real CUDA/Metal performance.

New tests, appended to `test/GPU_test.jl` (14 tests, all passing): unbiased
against the analytic two-facing-plates value at the low level
(`launch_area_kernel!`/`launch_raytrace_kernel!` directly) and confirmed
bias-free at `n_rays`=4,000,000 (deviation under 1σ; the test itself uses a
looser tolerance sized for `n_rays`=200,000's larger single-seed noise);
`compute_view_factors_gpu(...; raytrace=true)` end-to-end with exact
reciprocity; the same front/back-face and self-obstruction-without-
`obstruction_groups` regression case as the CPU test suite, cross-checked
against quadrature; closed-cube row-sum closure exercising the chunked
launcher; argument validation; `@inferred` type stability on
`gpu_nearest_hit_bvh` and the cosine-direction sampler. Full test suite
(`Pkg.test()`, 8 threads) passes unchanged, including the new tests.

**Files**: `src/BVH.jl`/`src/GPUBVH.jl` (`gpu_nearest_hit_bvh`,
`build_flat_scene_bvh`), `src/GPURayTraceKernels.jl` (new),
`src/GPUAssembly.jl` (`raytrace`/`n_rays`, `_compute_gpu_raytrace`),
`src/Assembly.jl` (dispatch restructuring, `_gpu_compute_hook` signature),
`src/RadiativeViewFactor.jl` (include order), `test/GPU_test.jl` (new
testset), `README.md` (GPU usage examples, file tree). Branch: `mc_speedup`.

### Not done / scope of this pass

No real-hardware (CUDA/Metal) testing — only KernelAbstractions `CPU()`
backend correctness testing, as above. No chunking-boundary stress test
(the closed-cube test forces multiple chunks via a small mesh, but a
large-N/large-`n_rays` combination that would need many chunks on real
hardware wasn't run here). Same CPU-side scope gaps as before still apply
on GPU: no 2-D curve-mesh support, no `self_vf` support, not run against
the `benchmarks/howell/` catalog suite.

## Ray-shooting kernel run against the full Howell catalog suite — a real bug found and fixed along the way

Context: `raytrace=true` had only been checked against a handful of focused
`test/raytrace_test.jl` cases, not the 115-point Howell catalog suite the
other two kernels are validated against. Ran it on the 72 of those 115
points that are 3-D surface meshes (raytrace does not support the 43
2-D-curve-mesh points yet), sharing `benchmarks/howell/`'s existing case
list, and found — then fixed — a real, previously-invisible mesh-orientation
bug along the way.

### Shared the case list between `run.jl` and a new `run_raytrace.jl`

`benchmarks/howell/`'s 22-case, ~115-parameter-point case list (previously
inline in `run.jl`) is now `benchmarks/howell/cases.jl`, `include`d by both
drivers — avoided duplicating (and risking drift in) ~180 lines of delicate
geometry/parameter/solver-option combinations. Verified the extraction was
behavior-preserving before running anything expensive: case count (115),
per-case counts, and a spot-check of `run.jl`'s own quad/mc output on the
refactored file all matched the pre-refactor values.

`run_raytrace.jl` (new) mirrors `run.jl`'s structure (`Case`, `run_case`-style
driver, same CSV/provenance conventions) but runs only `raytrace=true`
(`n_rays=10000`) and appends its rows to the *existing* `results.csv`
(distinct `kernel=raytrace` value, its own provenance comment block) rather
than overwriting it.

### A real bug: `concentric_spheres`' blanket `reverse_normals` silently mis-orients the inner sphere

First full run found a catastrophic outlier: C-135 (concentric spheres)
gave F(outer→inner) = 0.0068 against a catalog value of 0.444 — 98.5% error,
not a fluke (reproduced across all 3 of its parameter points). Traced this
to the mesh, not the kernel: `concentric_spheres` builds the inner (r1)
sphere via an OCC boolean `cut`, then loads the result with a single
mesh-wide `reverse_normals=true`. Measured directly: a standalone sphere is
outward-facing by default (`dot(normal, radial) = +1` with
`reverse_normals=false`), and the inner sphere of this CSG cut comes out of
Gmsh the same way, *not* flipped by the cut itself — so the blanket
`reverse_normals=true` correctly fixes the outer sphere (which does need to
point into the gap) while wrongly over-correcting the inner one, leaving it
pointing into its own volume instead of away from it.

This is why it went unnoticed until now: **the pair-area kernels tolerate
it.** Their point-pair cosine test doesn't care which of a pair's two
normals is inward as long as the sign works out over the full double
integral, and empirically the *aggregated* F(inner→outer)/F(outer→inner)
values stay correct under the old (wrong) convention — quadrature gives
0.999999 and 0.44435 against analytic 1.0 and 0.44444, matching to 4-5
decimals. Only row-sum closure quietly fails (`Σⱼ Fᵢⱼ` off by up to 0.99),
and this suite had never checked closure for C-135, only the extracted F
value, so nothing flagged it. `raytrace=true` fails outright instead of
subtly, and this is *why* it's a genuinely useful validation exercise, not
just a stress test: a ray sampled from the cosine-weighted hemisphere
around an inward-pointing normal is aimed *into* the body's own volume,
and — starting exactly on that body's own surface — is geometrically
guaranteed to exit back through the *same* body (a chord) rather than ever
reaching the outer sphere. Confirmed directly before the fix: 16 of 18
sampled rays from one inner-sphere element, in one trial, hit *another*
inner-sphere element instead of ever reaching the outer sphere.

**Fixed** by adding `reverse_group_normals(mesh, groups)` to `src/MeshIO.jl`
— reverses only the requested physical group(s), reusing the same
per-family node-swap the existing mesh-wide `reverse_normals` flag already
uses, instead of flipping the whole mesh — and changing `cases.jl`'s C-135
entry to flip only the outer sphere's group (`reverse_groups=[2]`, a new
option `run_case`/`run_case_raytrace` both gained) instead of both. Verified
no regression: quadrature and MC's C-135 rows are unchanged to 5 decimals
under the corrected convention (they never depended on the inner sphere's
sign being right); `raytrace`'s C-135 error dropped from 98.5% to 2.2-2.8%
(now understood to be a separate, much smaller, mesh-faceting effect — see
below). Every other `reverse_normals=true` case in this suite (C-33, C-79,
C-109) is one topologically connected closed body, not two disconnected
ones from a CSG cut, and was checked and found unaffected.

**Files**: `src/MeshIO.jl` (`reverse_group_normals`, new, exported),
`src/RadiativeViewFactor.jl` (export), `benchmarks/howell/cases.jl` (C-135
entry), `benchmarks/howell/run.jl`/`run_raytrace.jl` (`reverse_groups`
option on `run_case`/`run_case_raytrace`), `test/mesh_test.jl` (new
testset, 9 checks: flips only the targeted group, accepts a single tag or a
collection, leaves the original `MeshData` untouched, is its own inverse).

### A second, smaller finding: ray-shooting has a real (small, quantified, mesh-resolution-limited) bias on coarsely faceted curved bodies

With the orientation bug fixed, 59/72 (82%) of raytrace's points are within
1% of the catalog (median 0.18%), but the worst points are concentrated on
curved bodies at this suite's default mesh coarseness: C-135 (spheres,
2.84%), C-137 (spheres, 2.55%), C-125 (sphere, 1.85%) — while flat/mildly-
curved cases (C-10, C-11, C-14, C-34, C-40, C-41) converge to well under 1%,
several to near machine precision. Two checks confirm this is a systematic,
mesh-resolution-limited bias, not ordinary MC noise: it does *not* shrink
with `n_rays` (C-135 stayed at ~2.8% error from `n_rays=5000` through
`n_rays=80000`, a 16× sample increase with no improvement), and it *does*
shrink with mesh refinement (halving element size twice, 634 → 2530 → 9924
elements, brought the same case from 2.8% to 1.0% to 0.4%).

Mechanism: `RayTraceKernel.jl` triangulates each curved (Quad8/Tri6)
element by its corner nodes only — the same faceting the existing
obstruction soups already use (see this file's own obstruction-accuracy
notes above) — and classifies a ray's hit as front- or back-facing using
*that flat triangle's* normal, not the true curved-surface normal at the
hit point. On a coarsely faceted convex body, adjacent facets can disagree
slightly at their shared edge, letting a near-grazing ray clip a
neighbouring facet it geometrically shouldn't reach. This is the same
category of faceting error the package's existing obstruction-soup caveat
already documents for occlusion; here it shows up as a front/back
misclassification instead of a missed/extra blocked ray. Not fixed in this
pass — noted as a known, quantified, mesh-resolution-dependent limitation
of the corner-triangulated scene approximation, worth keeping in mind for
anyone using `raytrace=true` on coarsely meshed curved 3-D geometry.

### Speed, on this same 72-point subset

| Kernel | Total seconds | vs. ray-shooting |
|---|---:|---:|
| Ray-shooting | 193.4 | 1× |
| Quadrature | 555.9 | 2.9× slower |
| Pair-area Monte Carlo (`n_samples=5000`) | 5113.3 | 26.4× slower |

Roughly reproduces the earlier synthetic-scene speed measurement
(11.6×/45.2× at 836 elements) at a full, varied-geometry suite level —
smaller ratios here since this mix includes many small/cheap cases where
fixed per-launch overhead matters more, and large cases (C-10, ~9200
elements) where quadrature itself is already fast.

**Files (results)**: `benchmarks/howell/results.csv` (72 new `kernel=raytrace`
rows appended, existing quad/mc rows untouched), `benchmarks/howell/RESULTS.md`
(new "Ray-shooting Monte Carlo" section: results table, speed table, the
orientation-bug writeup, the faceting-bias writeup).

**Verified**: full test suite (`Pkg.test()`, 8 threads) passes, including the
new `reverse_group_normals` tests. Branch: `mc_speedup`.

## Documentation pass: ray-shooting Monte Carlo, corrected Duffy region counts, and a real `load_vtu` export bug found along the way

The `docs/` (Documenter) site had not been updated since `raytrace=true` was
added (commits `f15c07c`, `272bed9`, `d0807a3`) — it still described three
integration methods with no mention of ray-shooting Monte Carlo anywhere.
Separately, `docs/src/theory.md` and `docs/src/manual/integration_methods.md`
described the Duffy transformation as an 8-region (vertex) / 5-region (edge)
Sauter–Schwab decomposition; the actual implementation (`src/DuffyKernel.jl`,
confirmed by reading the region-splitting code directly, not just its own
stale header comment) is a 4-region / 6-region "biggest-coordinate"
decomposition — the same discrepancy the top-of-file comment in
`DuffyKernel.jl` itself has (not corrected in this pass; flagged for a
follow-up since rewriting that comment block precisely was judged out of
scope for a docs-only pass).

**Changed**:
- `docs/src/index.md`, `docs/src/manual/integration_methods.md`,
  `docs/src/manual/gpu.md`, `docs/src/manual/obstruction.md`,
  `docs/src/manual/performance.md`: added ray-shooting Monte Carlo coverage
  (method, GPU kernel, obstruction-is-automatic behavior, `n_rays` guidance,
  the coarse-mesh faceting-bias limitation, and the 2.9×/26.4× Howell-suite
  speed comparison already recorded in `benchmarks/howell/RESULTS.md`).
- `docs/src/theory.md`, `docs/src/manual/integration_methods.md`: corrected
  the Duffy region counts (8→4 vertex, 5→6 edge) and reframed Sauter &
  Schwab (2011) as background/related work rather than the implemented
  method, matching how `README.md` already described it.
- `docs/src/references.md`, `docs/src/citing.md`: same Duffy reframing;
  added Cohen & Wallace, *Radiosity and Realistic Image Synthesis* (Academic
  Press, 1995) as the source for the solid-angle identity the ray-shooting
  estimator relies on (cited in `RayTraceKernel.jl`'s own module docstring).
- `CITATION.cff`, `docs/src/citing.md`: version bumped 0.5.0 → 0.6.3
  (matching `Project.toml`, last bumped 2026-09-14) — both were still
  pointing at a stale prior release.

**Bug found and fixed while checking the README's own examples against the
real exported API**: `load_vtu` was documented in `README.md` and
`docs/src/api.md` as directly callable after `using RadiativeViewFactor`,
but `src/RadiativeViewFactor.jl` never imported or exported it from
`MeshIO` — only `RadiativeViewFactor.MeshIO.load_vtu` actually resolved.
Confirmed with `isdefined(RadiativeViewFactor, :load_vtu) == false` before
the fix. Added `load_vtu` to the `using .MeshIO: ...` import list and the
top-level `export` list; verified `isdefined(...) == true` and a fresh
`using RadiativeViewFactor; load_vtu` resolves after the change. Every
other loader (`load_mesh`, `load_re2`) was already exported this way, so
this was a straightforward oversight rather than an intentional namespacing
choice.

**Follow-up (same day)**: rewrote `DuffyKernel.jl`'s own module header
comment (the "Reference elements" / decomposition-overview sections) and
the `element_pair_view_factor_duffy` docstring, which both still described
the old 8-region (vertex) / 5-region (edge) Sauter–Schwab scheme even though
the code beneath them was already correct. Now describes the actual 4-region
/ 6-region "biggest-coordinate" decomposition and points to the detailed,
already-accurate comments above `_vertex_integral` and `_edge_integral`
rather than re-deriving them. No code changed, comments only. **Verified**:
package loads and `Pkg.test()` passes (re-run twice; one earlier run in this
same session had one unrelated flaky failure in the stochastic GPU raytrace
reciprocity test, which passed on immediate rerun with no code changes in
between — see [[docs-known-issues]] memory).
