# Benchmark Results — Quadrature-Point / Sample Pre-evaluation

This directory benchmarks the CPU assembly path before and after the
**pre-evaluation** optimization, in which each element's quadrature points
(deterministic path) or Monte Carlo samples (MC path) are computed **once**
per element — O(N) — and reused across every element pair, instead of being
re-derived inside the O(N²) pair loop.

## What changed

The assembly matrix is dense and O(N²) in the element count `N`. Previously,
`element_pair_view_factor(coords, elem_i, elem_j, …)` re-evaluated the shape
functions, the quadrature rule, and the physical points/normals for **both**
elements on **every** call. Element *i*'s data depends only on *i*, yet it was
rebuilt for all `N − i` partners; for `nquad > 5` this even re-ran a
Golub–Welsch eigensolve per pair.

The optimization:

- **Deterministic path** — `precompute_quad` builds one `ElementQuad` per
  element; the pair integrator consumes cached points
  (`src/ViewFactorKernel.jl`, `src/Assembly.jl`).
- **Monte Carlo path** — `sample_element_mc` draws one `ElementSamples` set per
  element, reused across the row/column; the diagonal self-pair draws a fresh
  independent set so `self_vf` stays correct (`src/MCKernel.jl`,
  `src/Assembly.jl`). Each per-entry estimate stays unbiased.
- **Rule memoization** — `gauss_legendre_1d` caches the Golub–Welsch rule for
  `nquad > 5` behind a lock (`src/Quadrature.jl`).

Because the math is unchanged (only the evaluation order), results are
numerically identical; reciprocity holds to machine precision and the
parallel-plate view factors match the analytic value.

## How to reproduce

```bash
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmarks --threads=auto benchmarks/quadrature_bench.jl
julia --project=benchmarks --threads=auto benchmarks/montecarlo_bench.jl
```

The "before" columns below were produced by running the same scripts against
the pre-optimization sources (`git stash` of the `src/` changes).

## Environment

| | |
|---|---|
| CPU | Apple M1 (8 logical cores) |
| Threads | 8 (`--threads=auto`) |
| Julia | 1.12.6 |
| Timing | minimum of 3 runs (warm) |

## Deterministic quadrature — two facing unit plates, `nquad=6`

Analytic F(bottom→top) ≈ 0.19982; every configuration reproduced it to a
relative error of 2.4×10⁻⁵.

| N (elements) | before (s) | after (s) | speedup | before alloc | after alloc | alloc ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 240  | 0.0100 | 0.0032 | 3.1× | 75.9 MiB   | 1.2 MiB  | 63× |
| 396  | 0.0243 | 0.0089 | 2.7× | 206.3 MiB  | 2.9 MiB  | 71× |
| 692  | 0.0708 | 0.0265 | 2.7× | 629.3 MiB  | 8.2 MiB  | 77× |
| 1088 | 0.1716 | 0.0658 | 2.6× | 1554.8 MiB | 19.5 MiB | 80× |

After the change, allocations are dominated by the two `N×N` result matrices
(constant per `N`) rather than per-pair temporaries.

## Monte Carlo — two facing unit plates, N = 484

| n_samples | before (s) | after (s) | speedup | before alloc | after alloc | alloc ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 1.189 | 0.096 | 12.4× | 16.2 GiB | 47.4 MiB  | 350× |
| 2000 | 2.475 | 0.205 | 12.1× | 32.3 GiB | 81.4 MiB  | 406× |
| 5000 | 6.865 | 0.464 | 14.8× | 86.0 GiB | 194.8 MiB | 452× |

The MC path re-sampled both elements on every pair, so the redundancy — and
therefore the speedup and allocation reduction — is even larger than for the
deterministic path. Estimated view factors matched the analytic value to a
relative error ≈ 1.5×10⁻⁵ across sample counts.

## Takeaways

- The optimization is a pure implementation change: identical results, no new
  approximations.
- Speedups are **~2.6–3.1× (deterministic)** and **~12–15× (Monte Carlo)** with
  **60–450× fewer allocations**, growing with `N` and `n_samples`.
- The remaining cost is the genuine O(N²) kernel work plus the dense result
  matrices; both are inherent to full-matrix assembly.
