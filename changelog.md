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

```
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
