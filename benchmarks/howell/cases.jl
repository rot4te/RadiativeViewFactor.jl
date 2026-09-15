# benchmarks/howell/cases.jl
# The Howell catalog case list, extracted from run.jl so it can be shared
# with other drivers (e.g. run_raytrace.jl) without duplicating — and
# risking drift in — 22 cases' worth of delicate geometry/parameter/solver-
# option combinations. Needs `Case` (the struct) already defined by the
# includer; builds and appends to `cases::Vector{Case}` (must also already
# exist). Requires geom.jl, analytic.jl and tables.jl already included too.

# F from group 1 to group 2 of the aggregated matrix.
f12(vf) = vf.F_group[1, 2]
f21(vf) = vf.F_group[2, 1]

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
  # `concentric_spheres` cuts the inner (r1) sphere out of the outer (r2)
  # one; Gmsh/OCC already gives *both* resulting surfaces an outward-facing
  # normal by default (verified directly against a standalone sphere, which
  # is +1 out of the box), so only the outer one (group 2) actually needs
  # flipping to point into the annular gap — a blanket `reverse_normals`
  # over-corrects the inner sphere (group 1), leaving it pointing into its
  # own volume instead of away from it. The pair-area kernels tolerate that
  # (their point-pair cosine test doesn't care which of a pair's two
  # normals is inward, as long as the sign works out over the full double
  # integral) enough that the *aggregated* F values stay correct even under
  # the old blanket flip — but `raytrace=true` cannot: a ray sampled from
  # the cosine-weighted hemisphere around an inward-pointing normal is
  # aimed into that body's own volume and is geometrically guaranteed to
  # exit back through the *same* body rather than ever reaching the other
  # one. See `reverse_group_normals`'s docstring and changelog.md.
  push!(cases, Case("C-135", "Concentric spheres (outer→inner)",
    "r1=0.5, r2=$r2", p -> concentric_spheres(p; r1=0.5, r2=r2),
    f21, c135_21(0.5, r2), 2, (reverse_groups=[2],)))
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
    f12, F12, 1, (reverse_normals=true, obstruct=true, radiating=[1,2,3])))
  push!(cases, Case("C-72", "Cylinder in square array (diagonal neighbour)",
    "P=$P", p -> cylinder_array_2d(p; P=P, square=true),
    fij(1, 3), F13, 1, (reverse_normals=true, obstruct=true, radiating=[1,2,3])))
end

for (P, F12, F13) in C73                # C-73: cylinder triangular array
  P in (1.2, 1.5, 2.0) || continue
  push!(cases, Case("C-73", "Cylinder in triangular array (nearest neighbour)",
    "P=$P", p -> cylinder_array_2d(p; P=P, square=false),
    f12, F12, 1, (reverse_normals=true, obstruct=true, radiating=[1,2,3])))
  push!(cases, Case("C-73", "Cylinder in triangular array (next neighbour)",
    "P=$P", p -> cylinder_array_2d(p; P=P, square=false),
    fij(1, 3), F13, 1, (reverse_normals=true, obstruct=true, radiating=[1,2,3])))
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
  # Each extraction reads only a prefix of the six groups, so the rest need not
  # be assembled (this geometry is unobstructed, so dropping them is exact).
  for (lbl, ex, ref, rad) in (("wall→adjacent wall", fij(1, 2), F12, [1,2]),
                              ("wall→next wall",     fij(1, 3), F13, [1,2,3]),
                              ("wall→opposite wall", fij(1, 4), F14, [1,2,3,4]),
                              ("end→wall",           fij(5, 1), FH1, [1,2,3,4,5]),
                              ("wall→end",           fij(1, 5), F1H, [1,2,3,4,5]),
                              ("end→end",            fij(5, 6), FHH, Int[]))
    push!(cases, Case("C-33", "Hexagonal prism, $lbl",
      "L=$L", p -> hex_prism(p; l=L, h=1.0),
      ex, ref, 2, (reverse_normals=true, use_duffy=true, radiating=rad)))
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
