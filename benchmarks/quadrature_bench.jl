# benchmarks/quadrature_bench.jl
# ---------------------------------------------------------------------------
# Benchmark the deterministic Gauss–Legendre / Dunavant assembly path.
#
# The assembly is O(N²) in the element count N, and the optimization being
# measured pre-evaluates each element's quadrature once (O(N)) instead of
# re-deriving it inside every pair. So we sweep the mesh refinement (N) at a
# fixed `nquad`, reporting wall-clock time, allocations, and the estimated
# view factor against the analytic value (≈ 0.19982) to keep accuracy visible.
#
# Run with threads for a representative number:
#     julia --project=benchmarks --threads=auto benchmarks/quadrature_bench.jl
# ---------------------------------------------------------------------------

using RadiativeViewFactor
using Printf
include(joinpath(@__DIR__, "common.jl"))

print_env()

const ANALYTIC = 0.19982   # two directly-opposed unit squares, separation 1
const NQUAD    = 6

println("Quadrature assembly — two facing unit plates, nquad = $NQUAD")
println("Analytic F(bottom→top) ≈ $ANALYTIC\n")

bi(r) = findfirst(==("bottom"), r.group_names)
ti(r) = findfirst(==("top"),    r.group_names)

@printf("%-8s %10s %12s %14s %12s %12s\n",
        "lc", "N", "time (s)", "alloc (MiB)", "F_bot→top", "rel.err")
for lc in (0.16, 0.12, 0.09, 0.07)
    f = tempname() * ".msh"
    make_two_plates_msh(f; lc=lc, order=1)
    mesh = load_mesh(f; verbose=false)
    N = length(mesh.surface_elems)
    t, a = best_time(() -> compute_view_factors(mesh; nquad=NQUAD, verbose=false))
    r    = compute_view_factors(mesh; nquad=NQUAD, verbose=false)
    F    = r.F_group[bi(r), ti(r)]
    @printf("%-8.2f %10d %12.4f %14.1f %12.6f %12.2e\n",
            lc, N, t, a/2^20, F, abs(F - ANALYTIC)/ANALYTIC)
    rm(f)
end
