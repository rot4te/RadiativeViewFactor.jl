# Integration Methods

Three integration strategies are available, selected per call to
[`compute_view_factors`](@ref). They differ in accuracy, cost, and which
geometric configurations they handle well.

## Gauss–Legendre quadrature (default)

```julia
result = compute_view_factors(mesh; nquad=4)
```

The double surface integral is evaluated at a tensor product of `nquad` ×
`nquad` Gauss–Legendre points on each element pair. This is the default method
and is appropriate for most geometries.

**Convergence:** spectral — errors decrease as O(exp(-c·nquad)) for smooth
integrands. A good starting point is `nquad=4` (16 points per element pair);
increase to `nquad=6` or `nquad=8` for finer accuracy or larger elements.

**Limitations:** convergence degrades near corner singularities where two
elements share a vertex or edge. In these cases use the Duffy transformation.

## Duffy transformation

```julia
result = compute_view_factors(mesh; nquad=6, use_duffy=true)
```

For Quad8 element pairs sharing a vertex or edge, the `1/r²` singularity in
the view factor kernel is integrable but poorly resolved by standard quadrature.
The Duffy transformation introduces a radial coordinate whose Jacobian cancels
the singularity, leaving a smooth integrand that Gauss quadrature resolves
efficiently.

Singularity type is detected automatically:
- **No shared nodes** → standard quadrature (unchanged)
- **One shared corner node** → 8-region Sauter–Schwab decomposition
- **Two shared corner nodes (one edge)** → 5-region Sauter–Schwab decomposition

**Constraints:**
- CPU only (`use_duffy` is ignored on GPU backends with a warning)
- Quad8–Quad8 pairs only; Tri6 and Line3 pairs always use standard quadrature
- Incompatible with `monte_carlo=true`
- Not applicable for `surface_dim=1` (the 2D `1/r` singularity at shared
  endpoints is physically divergent and cannot be regularized)

**Cost:** `8 × nquad⁴` evaluations per vertex pair, `5 × nquad⁴` per edge pair,
versus `nquad⁴` for standard quadrature. Since only adjacent pairs trigger the
Duffy path, the overhead depends on mesh topology. A lower `nquad` (e.g. 4)
with `use_duffy=true` typically outperforms a higher `nquad` (e.g. 16) with
standard quadrature for inclined-plate geometries.

## Monte Carlo

```julia
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000)
```

Each element pair is estimated from `n_samples` stratified sample pairs. Samples
are placed on a ⌊√N⌋ × ⌊√N⌋ grid of strata within the reference element, giving
O(1/N) variance convergence rather than O(1/√N) for plain Monte Carlo.

For efficiency, one independent stratified sample set is drawn **once per
element** and reused across that element's pairings (the diagonal self-pair,
with `self_vf=true`, draws a fresh second set). Within any pair the two
elements' samples are still independent, so each entry's estimate remains
unbiased with the stated variance; only estimates in the same row/column become
correlated. This reuse is what makes the MC path ~12–15× faster than
re-sampling both elements per pair — see the `benchmarks/` directory.

**When to use:**
- Many obstructions (MC pays the BVH cost only for kernel-positive pairs)
- Near-singular pairs where MC variance is still finite (unlike the `1/r²`
  case where variance diverges — use Duffy instead)
- Rapid approximate estimates at low `n_samples`

**Reproducibility:** pass an explicit RNG for deterministic results:

```julia
using Random
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               rng=MersenneTwister(42))
```

**GPU:** each thread uses an independent xorshift64 pseudo-random stream seeded
from the global seed plus the thread index. The `rng` keyword is ignored on GPU;
a random global seed is generated on the host at each call.

**Constraints:**
- `n_samples` applies **per element pair**, not to the whole geometry
- Incompatible with `use_duffy=true`

## Choosing a method

| Situation | Recommended method |
|---|---|
| Smooth geometry, well-separated surfaces | Quadrature (`nquad=4`–`8`) |
| Inclined plates with shared edge | Duffy (`nquad=4`–`6`) |
| Many obstructions | Monte Carlo (`n_samples=50000`+) |
| Quick estimate | Monte Carlo (`n_samples=5000`) |
| GPU computation | Quadrature or Monte Carlo |
