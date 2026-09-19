# Obstruction Detection

Obstruction detection determines whether the line of sight between two
quadrature (or sample) points is blocked by a third surface, setting
H_ij = 0 in the view factor integral for blocked pairs.

## Enabling obstruction

Pass the physical group tags of all surfaces that may block rays:

```julia
result = compute_view_factors(mesh; nquad=4, obstruction_groups=[3, 4])
```

The source and destination groups are **automatically excluded** from the
obstruction geometry for each element pair — a surface never blocks rays that
originate from or terminate on itself. You may safely include all group tags
in `obstruction_groups` without worrying about self-blocking:

```julia
all_tags = collect(keys(mesh.group_tags))
result   = compute_view_factors(mesh; nquad=4, obstruction_groups=all_tags)
```

Obstruction geometry is built from element **corner nodes only** — 2 triangles
per quadrilateral, 1 per triangle, 1 segment per curve element — so a curved
2nd-order blocker is treated as flat for visibility.

## Radiating only some groups

Assembly is dense over every radiating element, so a mesh that carries many
shadowing bodies costs `O(N_total²)` even if you only want one pair of surfaces
— and most of that work computes view factors *between* the shadowing bodies.
`radiating_groups` restricts the radiating set while every group still
obstructs:

```julia
all_tags = collect(keys(mesh.group_tags))
result = compute_view_factors(mesh; nquad=4,
                              radiating_groups   = [wall_tag, pin_tag],
                              obstruction_groups = all_tags)
```

The obstruction geometry is carried over whole, so occlusion is unchanged; the
pair count falls with the square of the element-count ratio (on a reactor
fuel-assembly slice, `FA_wall → Pin_11` with 19 shadowing pins loaded drops from
18400 radiating elements to 3040, a 37× reduction in pairs).

With `verbose=true` the run lists its obstruction groups; blockers outside the
radiating set are shown by tag (`"tag 3"`), since their names are dropped along
with their elements.

The returned `F_group` covers only the radiating groups. **The enclosure is
deliberately open**, so row sums no longer approach 1 and `check_closure` is not
meaningful — use `check_reciprocity`, which is unaffected. To write such a result
out with [`write_nekrs_view_factors`](@ref), pass
`restrict_to_radiating(mesh, tags)` as the mesh so the rows line up. See
[`restrict_to_radiating`](@ref).

## How it works

**CPU path:** for each unique set of active obstruction groups (after excluding
the source and destination groups), a BVH is built once from the merged triangle
(3D) or line-segment (2D) soups of those groups and reused for all element pairs
sharing that set. Ray–triangle intersection uses the Möller–Trumbore algorithm;
ray–segment intersection uses 2D Cramer's rule.

**GPU path:** the BVH is flattened to typed device arrays with miss-link pointers
for stackless traversal. Each GPU thread traverses the BVH independently with
no thread-local stack, eliminating register pressure from MVector storage.
Per-triangle group tags stored in the BVH allow each thread to skip triangles
belonging to the emitter or receiver group without host-side pre-filtering.

**Skipping non-facing pairs:** before integrating an element pair,
`compute_view_factors` (with `facing_cull=true`, the default) checks conservative
bounds on each element's points and normals and skips any pair whose kernel is
provably zero at every point pair — in O(1), rather than discovering that one
point pair (and one BVH ray cast) at a time. Results are unchanged. Set
`facing_cull=false` to disable it. It does not apply to `raytrace=true`, which
has no pair loop.

## Ray-shooting Monte Carlo is different

`raytrace=true` builds one scene BVH from every radiating element plus
`obstruction_groups`, so **radiating elements obstruct each other
automatically** without being listed in `obstruction_groups` — that argument
only adds *extra* non-radiating blocker geometry for this method. See
[Integration Methods](@ref).

## Compatibility

Obstruction detection works with all four integration methods, on CPU and GPU:

```julia
# Quadrature + obstruction
result = compute_view_factors(mesh; nquad=4, obstruction_groups=[3])

# Monte Carlo + obstruction
result = compute_view_factors(mesh; monte_carlo=true, n_samples=5000,
                               obstruction_groups=[3])

# Duffy + obstruction
result = compute_view_factors(mesh; nquad=6, use_duffy=true,
                               obstruction_groups=[3])

# Ray-shooting: group 3 is an extra blocker on top of the radiating mesh
result = compute_view_factors(mesh; raytrace=true, n_rays=10000,
                               obstruction_groups=[3])

# GPU + obstruction
using CUDA
result = compute_view_factors(mesh; nquad=4, backend=CUDABackend(),
                               obstruction_groups=[3])
```

The Monte Carlo near-pair Duffy patch (see [Integration Methods](@ref)) honours
`obstruction_groups` too.

## Performance

The BVH query is O(log N_tris) per ray. For meshes with many obstruction
triangles and high `nquad`, the BVH cost can dominate the integration cost.

Pair-area Monte Carlo does **not** win here, despite the intuition that it
should: both it and quadrature ray-cast once per kernel-positive point pair, and
Monte Carlo simply evaluates more point pairs (`n_samples` versus `nquad⁴`). For
large obstructed 3D meshes, prefer `raytrace=true`, and use `radiating_groups`
and `facing_cull` to avoid work on pairs nobody asked for. See
[Integration Methods](@ref) and [Performance Guide](@ref).
