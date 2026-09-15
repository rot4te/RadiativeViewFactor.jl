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

## Ray-shooting Monte Carlo is different

`raytrace=true` builds one scene BVH from every radiating element plus
`obstruction_groups`, so **radiating elements obstruct each other
automatically** without being listed in `obstruction_groups` — that argument
only adds *extra* non-radiating blocker geometry for this method. See
[Integration Methods](@ref).

## Compatibility

Obstruction detection works with quadrature, pair-area Monte Carlo, and
Duffy the same way:

```julia
# Quadrature + obstruction
result = compute_view_factors(mesh; nquad=4, obstruction_groups=[3])

# Monte Carlo + obstruction
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               obstruction_groups=[3])

# Duffy + obstruction
result = compute_view_factors(mesh; nquad=6, use_duffy=true,
                               obstruction_groups=[3])

# GPU + obstruction
using CUDA
result = compute_view_factors(mesh; nquad=4, backend=CUDABackend(),
                               obstruction_groups=[3])
```

## Performance

The BVH query is O(log N_tris) per ray. For meshes with many obstruction
triangles and high `nquad`, the BVH cost can dominate the integration cost.
Monte Carlo integration is often faster than quadrature when obstructions are
dense, because MC skips the BVH for pairs with K = 0 (back-facing pairs)
without evaluating the BVH at all.
