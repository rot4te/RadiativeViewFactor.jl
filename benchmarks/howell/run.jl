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
function run_case(c::Case; nquad=6, reverse_normals=false, obstruct=false, kwargs...)
  path = joinpath(WORKDIR, replace(c.id, "-" => "_") * ".msh")
  c.build(path)
  mesh = load_mesh(path; surface_dim=c.dim, reverse_normals=reverse_normals,
                   verbose=false)
  obs = obstruct ? collect(keys(mesh.group_elems)) : Int[]
  nelem = length(mesh.surface_elems)

  vf_quad = compute_view_factors(mesh; nquad=nquad, obstruction_groups=obs,
                                 verbose=false, kwargs...)
  mc_seed_counter[] += 1
  vf_mc = compute_view_factors(mesh; nquad=nquad, obstruction_groups=obs,
                               verbose=false, monte_carlo=true,
                               n_samples=MC_SAMPLES,
                               rng=Xoshiro(mc_seed_counter[]), kwargs...)

  for (kernel, vf) in ((:quad, vf_quad), (:mc, vf_mc))
    computed = c.extract(vf)
    err = abs(computed - c.analytic) / abs(c.analytic)
    push!(results, (id=c.id, title=c.title, params=c.params, kernel=kernel,
                    computed=computed, analytic=c.analytic, relerr=err,
                    nelem=nelem))
    @printf("%-7s %-4s %-46s  RVF %.6f  cat %.6f  relerr %.2e\n",
            c.id, kernel, first(c.title, 46), computed, c.analytic, err)
  end
  return nothing
end

# F from group 1 to group 2 of the aggregated matrix.
f12(vf) = vf.F_group[1, 2]
f21(vf) = vf.F_group[2, 1]

cases = Case[]

# Each catalog formula is exercised at several parameter values, so the
# comparison probes a range of aspect ratios rather than one lucky point.

for h in (0.25, 0.5, 1.0, 2.0)          # C-1: H = h/w
  push!(cases, Case("C-1", "Infinite parallel plates, equal width",
    "w=1, h=$h", p -> parallel_plates_2d(p; w1=1.0, h=h, n=60),
    f12, c1(1.0, h), 1))
end

for (b, c) in ((0.5, 1.0), (1.0, 2.0), (2.0, 0.5))   # C-2: B=b/a, C=c/a, a=1
  push!(cases, Case("C-2", "Infinite parallel plates, unequal width",
    "a=1, b=$b, c=$c", p -> parallel_plates_2d(p; w1=b, w2=c, h=1.0, n=60),
    f12, c2(1.0, b, c), 1))
end

for hh in (0.3, 0.6, 1.0, 2.0)          # C-3: H = h/w
  push!(cases, Case("C-3", "Infinite perpendicular plates, common edge",
    "w=1, h=$hh", p -> wedge_plates_2d(p; w1=1.0, w2=hh, alpha=pi/2, n=60),
    f12, c3(1.0, hh), 1))
end

for a in (pi/6, pi/3, pi/2, 2pi/3)      # C-4: included angle α
  push!(cases, Case("C-4", "Infinite equal plates, common edge, angle α",
    "w=1, α=$(round(a, digits=3))",
    p -> wedge_plates_2d(p; w1=1.0, w2=1.0, alpha=a, n=60),
    f12, c4(a), 1))
end

for (a, b, c) in ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 1.0, 0.5), (2.0, 4.0, 1.0))
  push!(cases, Case("C-11", "Identical parallel directly opposed rectangles",
    "a=$a, b=$b, c=$c", p -> rect_pair_parallel(p; a=a, b=b, c=c, n=12),
    f12, c11(a, b, c), 2))
end

for (l, w, h) in ((1.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 0.5), (2.0, 1.0, 1.0))
  push!(cases, Case("C-14", "Perpendicular rectangles with common edge",
    "l=$l, w=$w, h=$h", p -> rect_pair_perp(p; l=l, w=w, h=h, n=12),
    f12, c14(l, w, h), 2, (use_duffy=true,)))
end

for L in (0.5, 1.0, 2.0)                # C-40: equal-radius coaxial disks
  push!(cases, Case("C-40", "Coaxial parallel disks, equal radius",
    "r=0.5, a=$L", p -> coax_disks(p; r1=0.5, r2=0.5, L=L, nrad=8),
    f12, c40(0.5, L), 2))
end

for (r1, r2, L) in ((0.5, 0.75, 1.5), (0.5, 1.0, 1.0), (1.0, 0.5, 1.0))
  push!(cases, Case("C-41", "Coaxial parallel disks, unequal radius",
    "r1=$r1, r2=$r2, a=$L", p -> coax_disks(p; r1=r1, r2=r2, L=L, nrad=8),
    f12, c41(r1, r2, L), 2))
end

for h in (0.5, 1.0, 2.0)                # C-79: cylinder base → inside surface
  push!(cases, Case("C-79", "Cylinder base to inside lateral surface",
    "r=0.5, h=$h", p -> cylinder_base_lateral(p; r=0.5, h=h),
    f12, c79(0.5, h), 2, (reverse_normals=true, use_duffy=true)))
end

for h in (0.5, 1.0, 2.0)                # C-109: cone interior → base
  push!(cases, Case("C-109", "Cone interior to base",
    "r=0.5, h=$h", p -> cone_interior_base(p; r=0.5, h=h),
    f12, c109(0.5, h), 2, (reverse_normals=true, use_duffy=true)))
end

for r2 in (0.75, 1.0, 2.0)              # C-135: concentric spheres, outer→inner
  push!(cases, Case("C-135", "Concentric spheres (outer→inner)",
    "r1=0.5, r2=$r2", p -> concentric_spheres(p; r1=0.5, r2=r2),
    f21, c135_21(0.5, r2), 2, (reverse_normals=true,)))
end

for r in (0.25, 0.5, 1.0, 2.0)          # C-125: sphere → coaxial disk
  push!(cases, Case("C-125", "Sphere to coaxial disk",
    "rs=0.3, r=$r, a=1", p -> sphere_to_disk(p; rs=0.3, r=r, a=1.0),
    f12, c125(r, 1.0), 2))
end

# --- Cases published as tables of reference values -------------------------

fij(i, j) = vf -> vf.F_group[i, j]

for (L, n, F) in C34                    # C-34: parallel regular polygons
  L in (0.4, 1.0, 2.0) || continue
  push!(cases, Case("C-34", "Parallel regular polygons (n=$n)",
    "n=$n, L=$L", p -> parallel_polygons(p; n=n, l=L, h=1.0),
    f12, F, 2))
end

for (R, S, F) in C137                   # C-137: two spheres of unequal radius
  (R in (0.4, 1.0, 3.0) && S in (0.2, 1.0)) || continue
  push!(cases, Case("C-137", "Two spheres of unequal radius",
    "R=$R, S=$S", p -> two_spheres(p; r1=R, r2=1.0, s=S),
    f12, F, 2))
end

fsum(i, js) = vf -> sum(vf.F_group[i, j] for j in js)

for (P, F12, F13) in C72                # C-72: cylinder square array
  P in (1.2, 1.5, 2.0) || continue
  push!(cases, Case("C-72", "Cylinder in square array (nearest neighbour)",
    "P=$P", p -> cylinder_array_2d(p; P=P, square=true),
    f12, F12, 1, (reverse_normals=true, obstruct=true)))
  push!(cases, Case("C-72", "Cylinder in square array (diagonal neighbour)",
    "P=$P", p -> cylinder_array_2d(p; P=P, square=true),
    fij(1, 3), F13, 1, (reverse_normals=true, obstruct=true)))
end

for (P, F12, F13) in C73                # C-73: cylinder triangular array
  P in (1.2, 1.5, 2.0) || continue
  push!(cases, Case("C-73", "Cylinder in triangular array (nearest neighbour)",
    "P=$P", p -> cylinder_array_2d(p; P=P, square=false),
    f12, F12, 1, (reverse_normals=true, obstruct=true)))
  push!(cases, Case("C-73", "Cylinder in triangular array (next neighbour)",
    "P=$P", p -> cylinder_array_2d(p; P=P, square=false),
    fij(1, 3), F13, 1, (reverse_normals=true, obstruct=true)))
end

for (R, Ff, Fs) in C8                   # C-8: plane to two rows of tubes
  R in (2.0, 3.0, 5.0) || continue
  local nf_
  push!(cases, Case("C-8", "Infinite plane to front row of tubes",
    "R=$R", p -> begin nf_ = plane_tube_rows_2d(p; R=R, ntube=31).nfront; end,
    vf -> sum(vf.F_group[1, 2:1+nf_]), Ff, 1, (obstruct=true,)))
end
for (R, Ff, Fs) in C8
  R in (2.0, 3.0, 5.0) || continue
  local nf_, ns_
  push!(cases, Case("C-8", "Infinite plane to second row of tubes",
    "R=$R", p -> begin info = plane_tube_rows_2d(p; R=R, ntube=31); nf_ = info.nfront; ns_ = info.nsecond; end,
    vf -> sum(vf.F_group[1, 2+nf_:1+nf_+ns_]), Fs, 1, (obstruct=true,)))
end

for (A, phi, F) in C10                  # C-10: rectangles at angle, one semi-infinite
  (A in (0.4, 1.0, 2.0) && phi in (45.0, 120.0)) || continue
  push!(cases, Case("C-10", "Rectangle to semi-infinite rectangle at angle",
    "A=$A, φ=$(phi)°", p -> rect_to_long_rect(p; a=A, b=1.0, phi=deg2rad(phi), W=80.0, n=32),
    f12, F, 2, (use_duffy=true,)))
end

for (L, F12, F13, F14, FH1, F1H, FHH) in C33   # C-33: hexagonal prism
  L in (0.5, 1.0, 2.0) || continue
  for (lbl, ex, ref) in (("wall→adjacent wall", fij(1, 2), F12),
                         ("wall→next wall",     fij(1, 3), F13),
                         ("wall→opposite wall", fij(1, 4), F14),
                         ("end→wall",           fij(5, 1), FH1),
                         ("wall→end",           fij(1, 5), F1H),
                         ("end→end",            fij(5, 6), FHH))
    push!(cases, Case("C-33", "Hexagonal prism, $lbl",
      "L=$L", p -> hex_prism(p; l=L, h=1.0),
      ex, ref, 2, (reverse_normals=true, use_duffy=true)))
  end
end

for r2 in (0.75, 1.0, 2.0)              # C-63: concentric cylinders, outer→inner
  push!(cases, Case("C-63", "Concentric infinite cylinders (outer→inner)",
    "r1=0.5, r2=$r2", p -> concentric_cylinders_2d(p; r1=0.5, r2=r2, narc=80),
    f21, c63_21(1.0, 2r2), 1))
end

for s in (0.5, 1.0, 2.0, 4.0)           # C-68: equal cylinders, surface gap s
  push!(cases, Case("C-68", "Infinite parallel cylinders, equal diameter",
    "r=0.5, s=$s", p -> parallel_cylinders_2d(p; r1=0.5, r2=0.5, s=s, narc=72),
    f12, c68(0.5, s), 1, (reverse_normals=true,)))
end

for (r1, r2, s) in ((0.5, 0.75, 1.0), (0.5, 1.5, 0.5), (1.0, 0.25, 2.0))
  push!(cases, Case("C-69", "Infinite parallel cylinders, unequal radius",
    "r1=$r1, r2=$r2, s=$s",
    p -> parallel_cylinders_2d(p; r1=r1, r2=r2, s=s, narc=72),
    f12, c69(r1, r2, s), 1, (reverse_normals=true,)))
end

for c in cases
  try
    run_case(c; c.opts...)
  catch e
    @printf("%-7s FAILED: %s\n", c.id, sprint(showerror, e))
    for kernel in (:quad, :mc)
      push!(results, (id=c.id, title=c.title, params=c.params, kernel=kernel,
                      computed=NaN, analytic=c.analytic, relerr=NaN, nelem=0))
    end
  end
end

# ---------------------------------------------------------------------------
open(joinpath(@__DIR__, "results.csv"), "w") do io
  println(io, "case,title,parameters,kernel,n_elements,F_rvf,F_catalog,rel_error")
  for r in results
    @printf(io, "%s,\"%s\",\"%s\",%s,%d,%.8f,%.8f,%.3e\n",
            r.id, r.title, r.params, r.kernel, r.nelem, r.computed, r.analytic, r.relerr)
  end
end

for kernel in (:quad, :mc)
  rs = filter(r -> r.kernel == kernel, results)
  ok = count(r -> !isnan(r.relerr) && r.relerr < 1e-2, rs)
  @printf("\n[%-4s] %d/%d cases within 1%% of the catalog value; worst = %.2e\n",
          kernel, ok, length(rs),
          maximum(r -> isnan(r.relerr) ? 0.0 : r.relerr, rs))
end
