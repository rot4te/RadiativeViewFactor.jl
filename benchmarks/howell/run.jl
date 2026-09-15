# benchmarks/howell/run.jl
# Driver: builds each catalog geometry, computes view factors with
# RadiativeViewFactor.jl, and compares against the published closed form.
#
#   julia --project=benchmarks --threads=auto benchmarks/howell/run.jl

using RadiativeViewFactor
using Random
using Printf

include("geom.jl")
include("analytic.jl")
include("tables.jl")

const WORKDIR = mktempdir()
const MC_SAMPLES = 5_000    # per element pair

# ---------------------------------------------------------------------------
# System information, recorded alongside the results so the runtimes below can
# be interpreted (and re-measured) later.
# ---------------------------------------------------------------------------
_try(f, default="unknown") = try f() catch; default end

function system_info()
  cpu = _try(() -> Sys.isapple() ?
                   strip(read(`sysctl -n machdep.cpu.brand_string`, String)) :
                   string(Sys.CPU_NAME), string(Sys.CPU_NAME))
  commit = _try(() -> strip(read(`git -C $(pkgdir(RadiativeViewFactor)) rev-parse --short HEAD`, String)))
  dirty  = _try(() -> !isempty(strip(read(`git -C $(pkgdir(RadiativeViewFactor)) status --porcelain`, String))), false)
  # Sys.CPU_THREADS counts only the performance cores on Apple Silicon (4 of
  # the 8 logical cores on an M3), so record the OS figure alongside it rather
  # than leaving an ambiguous core count in the provenance.
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
    mc_samples  = MC_SAMPLES,
  )
end

const SYSINFO = system_info()

struct Case
  id       :: String       # catalog case, e.g. "C-11"
  title    :: String
  params   :: String       # the concrete parameter values benchmarked
  build    :: Function     # path -> writes mesh
  extract  :: Function     # ViewFactorResult -> computed F
  analytic :: Float64
  dim      :: Int          # 1 = curve mesh (2D, per unit depth), 2 = surface
  opts     :: NamedTuple   # solver options (reverse_normals, obstruct, nquad)
end
Case(id, title, params, build, extract, analytic, dim) =
  Case(id, title, params, build, extract, analytic, dim, NamedTuple())

results = NamedTuple[]
mc_seed_counter = Ref(0)   # distinct, deterministic seed per case → reproducible runs

# Builds and loads the mesh once, then evaluates it with both kernels so the
# comparison isolates the kernel, not the discretization. `use_duffy` is a
# no-op under monte_carlo=true (near/touching pairs are always Duffy-patched
# in that path — see compute_view_factors' docstring), so it's safe to let it
# flow through `kwargs` unchanged for both calls.
function run_case(c::Case; nquad=6, reverse_normals=false, reverse_groups=Int[],
                  obstruct=false, radiating=Int[], kwargs...)
  path = joinpath(WORKDIR, replace(c.id, "-" => "_") * ".msh")
  c.build(path)
  mesh = load_mesh(path; surface_dim=c.dim, reverse_normals=reverse_normals,
                   verbose=false)
  isempty(reverse_groups) || (mesh = reverse_group_normals(mesh, reverse_groups))
  obs = obstruct ? collect(keys(mesh.group_elems)) : Int[]
  nelem = length(mesh.surface_elems)

  # `radiating` restricts the assembled matrix to the groups this case actually
  # reads, while every group still obstructs. The extractors index `F_group`
  # positionally and `_aggregate` orders groups by sorted tag, so the radiating
  # set must be the prefix 1:k or those indices would silently point elsewhere.
  if !isempty(radiating)
    sort(radiating) == collect(1:length(radiating)) ||
      error("$(c.id): radiating set must be the prefix 1:k so that positional " *
            "F_group indices are unchanged; got $radiating")
  end
  rad_set = Set(radiating)
  nrad = isempty(radiating) ? nelem :
         count(e -> e.group in rad_set, mesh.surface_elems)

  t_quad = @elapsed vf_quad =
    compute_view_factors(mesh; nquad=nquad, obstruction_groups=obs,
                         radiating_groups=radiating, verbose=false, kwargs...)
  mc_seed_counter[] += 1
  t_mc = @elapsed vf_mc =
    compute_view_factors(mesh; nquad=nquad, obstruction_groups=obs,
                         radiating_groups=radiating, verbose=false,
                         monte_carlo=true, n_samples=MC_SAMPLES,
                         rng=Xoshiro(mc_seed_counter[]), kwargs...)

  for (kernel, vf, secs) in ((:quad, vf_quad, t_quad), (:mc, vf_mc, t_mc))
    computed = c.extract(vf)
    err = abs(computed - c.analytic) / abs(c.analytic)
    push!(results, (id=c.id, title=c.title, params=c.params, kernel=kernel,
                    computed=computed, analytic=c.analytic, relerr=err,
                    nelem=nelem, nrad=nrad, restricted=!isempty(radiating),
                    secs=secs))
    @printf("%-7s %-4s %-46s  RVF %.6f  cat %.6f  relerr %.2e  %7.2fs%s\n",
            c.id, kernel, first(c.title, 46), computed, c.analytic, err, secs,
            isempty(radiating) ? "" : "  [rad $nrad/$nelem]")
    flush(stdout)   # otherwise progress sits in the buffer for the whole run
  end
  return nothing
end

cases = Case[]

# Case list (22 catalog cases, ~115 parameter points) lives in cases.jl so
# it can be shared with other drivers (e.g. run_raytrace.jl) without
# duplicating — and risking drift in — delicate geometry/parameter/solver
# combinations. It defines `f12`/`f21`/`fij`/`fsum` and appends to `cases`.
include("cases.jl")

# Warm up both kernels on a throwaway case so the per-case timings below
# measure computation rather than first-call compilation.
let warm = Case("warmup", "warmup", "", p -> parallel_plates_2d(p; w1=1.0, h=1.0, n=8),
                f12, 1.0, 1)
  run_case(warm)
  empty!(results); mc_seed_counter[] = 0
end

const T_START = time()

for c in cases
  try
    run_case(c; c.opts...)
  catch e
    @printf("%-7s FAILED: %s\n", c.id, sprint(showerror, e))
    for kernel in (:quad, :mc)
      push!(results, (id=c.id, title=c.title, params=c.params, kernel=kernel,
                      computed=NaN, analytic=c.analytic, relerr=NaN, nelem=0,
                      nrad=0, restricted=false, secs=NaN))
    end
  end
end

# ---------------------------------------------------------------------------
const T_TOTAL = time() - T_START

open(joinpath(@__DIR__, "results.csv"), "w") do io
  # Provenance for the runtimes below, as leading comment lines.
  for (k, v) in pairs(SYSINFO)
    println(io, "# ", k, ": ", v)
  end
  println(io, "# total_wall_seconds: ", round(T_TOTAL, digits=1))
  println(io, "# seconds is one compute_view_factors call for that kernel, ",
              "after warmup; n_radiating < n_elements means radiating_groups was used")
  println(io, "case,title,parameters,kernel,n_elements,n_radiating,restricted,",
              "seconds,F_rvf,F_catalog,rel_error")
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

for kernel in (:quad, :mc)
  rs = filter(r -> r.kernel == kernel, results)
  ok = count(r -> !isnan(r.relerr) && r.relerr < 1e-2, rs)
  tk = sum(r -> isnan(r.secs) ? 0.0 : r.secs, rs)
  @printf("\n[%-4s] %d/%d cases within 1%% of the catalog value; worst = %.2e; %.1f s total\n",
          kernel, ok, length(rs),
          maximum(r -> isnan(r.relerr) ? 0.0 : r.relerr, rs), tk)
end
@printf("\nTotal wall time (incl. meshing): %.1f s\n", T_TOTAL)
