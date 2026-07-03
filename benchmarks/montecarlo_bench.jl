# benchmarks/montecarlo_bench.jl
# ---------------------------------------------------------------------------
# Benchmark the stratified Monte Carlo assembly path.
#
# Measures full `compute_view_factors(monte_carlo=true)` wall-clock time and
# allocations on two facing unit plates, sweeping `n_samples`, and reports the
# estimated view factor against the analytic value (≈ 0.19982) so the accuracy
# of the sample-reuse optimization stays visible alongside the timing.
#
# Run with threads:
#     julia --project=benchmarks --threads=auto benchmarks/montecarlo_bench.jl
# ---------------------------------------------------------------------------

using RadiativeViewFactor
using Random
using Printf
include(joinpath(@__DIR__, "common.jl"))

print_env()

const ANALYTIC = 0.19982   # two directly-opposed unit squares, separation 1

f = tempname() * ".msh"
make_two_plates_msh(f; lc=0.1, order=1)
mesh = load_mesh(f; verbose=false)
N = length(mesh.surface_elems)
println("Monte Carlo assembly — two facing unit plates, N = $N elements")
println("Analytic F(bottom→top) ≈ $ANALYTIC\n")

bi(r) = findfirst(==("bottom"), r.group_names)
ti(r) = findfirst(==("top"),    r.group_names)

@printf("%-11s %12s %14s %12s %12s\n",
        "n_samples", "time (s)", "alloc (MiB)", "F_bot→top", "rel.err")
for ns in (1_000, 2_000, 5_000, 20_000)
    run() = compute_view_factors(mesh; monte_carlo=true, n_samples=ns,
                                 rng=MersenneTwister(1), verbose=false)
    t, a = best_time(run)
    r    = run()
    F    = r.F_group[bi(r), ti(r)]
    @printf("%-11d %12.4f %14.1f %12.6f %12.2e\n",
            ns, t, a/2^20, F, abs(F - ANALYTIC)/ANALYTIC)
end

rm(f)
