# benchmarks/howell/run_raytrace.jl
# Driver: runs the Howell catalog suite (benchmarks/howell/cases.jl, shared
# with run.jl) through *only* the ray-shooting Monte Carlo kernel
# (`compute_view_factors(...; raytrace=true)`), and appends the results to
# the existing results.csv (kernel="raytrace" rows, alongside run.jl's
# "quad"/"mc" rows) and to RESULTS.md.
#
# raytrace does not support curve meshes (mesh_dim=1) yet, so the 2D cases
# (C-1 to C-4, C-8, C-63, C-68, C-69, C-72, C-73 — 43 of 115 parameter
# points) are skipped entirely, not run with a fallback kernel; RESULTS.md
# and the summary below say so explicitly rather than silently covering
# fewer cases than the quad/mc table.
#
#   julia --project=benchmarks --threads=auto benchmarks/howell/run_raytrace.jl

using RadiativeViewFactor
using Random
using Printf

include("geom.jl")
include("analytic.jl")
include("tables.jl")

const WORKDIR = mktempdir()
const N_RAYS  = 10_000    # per element — see compute_view_factors' docstring;
                          # not directly comparable to run.jl's MC_SAMPLES
                          # (per element *pair*), so kept a separate constant

_try(f, default="unknown") = try f() catch; default end

function system_info()
  cpu = _try(() -> Sys.isapple() ?
                   strip(read(`sysctl -n machdep.cpu.brand_string`, String)) :
                   string(Sys.CPU_NAME), string(Sys.CPU_NAME))
  commit = _try(() -> strip(read(`git -C $(pkgdir(RadiativeViewFactor)) rev-parse --short HEAD`, String)))
  dirty  = _try(() -> !isempty(strip(read(`git -C $(pkgdir(RadiativeViewFactor)) status --porcelain`, String))), false)
  os_cpus = _try(() -> parse(Int, strip(read(`sysctl -n hw.logicalcpu`, String))),
                 Sys.CPU_THREADS)
  return (
    date        = Libc.strftime("%Y-%m-%d %H:%M:%S", time()),
    cpu         = cpu,
    cpu_logical = os_cpus,
    cpu_threads_julia_reports = Sys.CPU_THREADS,
    julia_threads = Threads.nthreads(),
    memory_gb   = round(Sys.total_memory() / 2^30, digits=1),
    os          = string(Sys.KERNEL, " ", Sys.MACHINE),
    julia       = string(VERSION),
    rvf_version = _try(() -> string(pkgversion(RadiativeViewFactor))),
    rvf_commit  = commit * (dirty === true ? "-dirty" : ""),
    n_rays      = N_RAYS,
    kernel      = "raytrace",
  )
end

const SYSINFO = system_info()

struct Case
  id       :: String
  title    :: String
  params   :: String
  build    :: Function
  extract  :: Function
  analytic :: Float64
  dim      :: Int
  opts     :: NamedTuple
end
Case(id, title, params, build, extract, analytic, dim) =
  Case(id, title, params, build, extract, analytic, dim, NamedTuple())

results = NamedTuple[]
rt_seed_counter = Ref(0)   # distinct, deterministic seed per case → reproducible runs

# Same shape as run.jl's run_case, but a single kernel (raytrace) instead of
# the quad+mc pair. `obstruct`/`use_duffy` are accepted (cases.jl's `opts`
# pass them for the quad/mc drivers) and simply have no effect here — no
# dim=2 case in cases.jl currently sets obstruct=true, and raytrace ignores
# use_duffy entirely (see compute_view_factors' docstring: no singular pairs,
# no Duffy patch for this kernel).
function run_case_raytrace(c::Case; nquad=6, reverse_normals=false, reverse_groups=Int[],
                           obstruct=false, radiating=Int[], kwargs...)
  path = joinpath(WORKDIR, replace(c.id, "-" => "_") * ".msh")
  c.build(path)
  mesh = load_mesh(path; surface_dim=c.dim, reverse_normals=reverse_normals,
                   verbose=false)
  isempty(reverse_groups) || (mesh = reverse_group_normals(mesh, reverse_groups))
  obs = obstruct ? collect(keys(mesh.group_elems)) : Int[]
  nelem = length(mesh.surface_elems)

  if !isempty(radiating)
    sort(radiating) == collect(1:length(radiating)) ||
      error("$(c.id): radiating set must be the prefix 1:k so that positional " *
            "F_group indices are unchanged; got $radiating")
  end
  rad_set = Set(radiating)
  nrad = isempty(radiating) ? nelem :
         count(e -> e.group in rad_set, mesh.surface_elems)

  rt_seed_counter[] += 1
  t_rt = @elapsed vf_rt =
    compute_view_factors(mesh; nquad=nquad, obstruction_groups=obs, radiating_groups=radiating,
                         verbose=false, raytrace=true, n_rays=N_RAYS,
                         rng=Xoshiro(rt_seed_counter[]), kwargs...)

  computed = c.extract(vf_rt)
  err = abs(computed - c.analytic) / abs(c.analytic)
  push!(results, (id=c.id, title=c.title, params=c.params, kernel=:raytrace,
                  computed=computed, analytic=c.analytic, relerr=err,
                  nelem=nelem, nrad=nrad, restricted=!isempty(radiating),
                  secs=t_rt))
  @printf("%-7s %-8s %-46s  RVF %.6f  cat %.6f  relerr %.2e  %7.2fs%s\n",
          c.id, "raytrace", first(c.title, 46), computed, c.analytic, err, t_rt,
          isempty(radiating) ? "" : "  [rad $nrad/$nelem]")
  flush(stdout)
  return nothing
end

cases = Case[]
include("cases.jl")   # shared with run.jl — see that file's header comment

n_total = length(cases)
filter!(c -> c.dim == 2, cases)   # raytrace: 3-D meshes only (mesh_dim=2)
n_2d = n_total - length(cases)
println("Skipping $n_2d of $n_total case points: 2-D curve meshes ",
        "(mesh_dim=1) are not supported by raytrace yet.")

# Warm up on a throwaway case so the per-case timings below exclude
# first-call compilation.
let warm = Case("warmup", "warmup", "", p -> rect_pair_parallel(p; a=1.0, b=1.0, c=1.0, n=4),
                f12, 1.0, 2)
  run_case_raytrace(warm)
  empty!(results); rt_seed_counter[] = 0
end

const T_START = time()

for c in cases
  try
    run_case_raytrace(c; c.opts...)
  catch e
    @printf("%-7s FAILED: %s\n", c.id, sprint(showerror, e))
    push!(results, (id=c.id, title=c.title, params=c.params, kernel=:raytrace,
                    computed=NaN, analytic=c.analytic, relerr=NaN, nelem=0,
                    nrad=0, restricted=false, secs=NaN))
  end
end

const T_TOTAL = time() - T_START

# ---------------------------------------------------------------------------
# Append to the existing results.csv — a comment block documenting this run's
# own provenance (distinct from run.jl's: different date, different sample
# parameter, possibly a different commit), then the new rows. Not
# re-emitting the header: this file is read by tools that only look at the
# first header line, and appended rows share the same column layout.
# ---------------------------------------------------------------------------
open(joinpath(@__DIR__, "results.csv"), "a") do io
  println(io, "# --- raytrace kernel appended below ---")
  for (k, v) in pairs(SYSINFO)
    println(io, "# ", k, ": ", v)
  end
  println(io, "# total_wall_seconds (raytrace only, $(length(cases)) of $n_total ",
              "points; 2-D curve-mesh points skipped, see above): ", round(T_TOTAL, digits=1))
  for r in results
    @printf(io, "%s,\"%s\",\"%s\",%s,%d,%d,%s,%.3f,%.8f,%.8f,%.3e\n",
            r.id, r.title, r.params, r.kernel, r.nelem, r.nrad, r.restricted,
            r.secs, r.computed, r.analytic, r.relerr)
  end
end

println("\nSystem:")
for (k, v) in pairs(SYSINFO)
  @printf("  %-14s %s\n", string(k) * ":", v)
end

ok = count(r -> !isnan(r.relerr) && r.relerr < 1e-2, results)
tk = sum(r -> isnan(r.secs) ? 0.0 : r.secs, results)
@printf("\n[raytrace] %d/%d cases within 1%% of the catalog value; worst = %.2e; %.1f s total\n",
        ok, length(results), maximum(r -> isnan(r.relerr) ? 0.0 : r.relerr, results), tk)
@printf("(%d of %d total Howell suite points skipped: 2-D curve meshes, mesh_dim=1)\n",
        n_2d, n_total)
@printf("\nTotal wall time (incl. meshing): %.1f s\n", T_TOTAL)
