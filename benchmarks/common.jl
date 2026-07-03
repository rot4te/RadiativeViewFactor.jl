# benchmarks/common.jl
# ---------------------------------------------------------------------------
# Shared mesh generators and helpers for the benchmark scripts.
# Requires Gmsh (see benchmarks/Project.toml).
# ---------------------------------------------------------------------------

import Gmsh: gmsh

"""
    make_box_msh(path; lc=0.09, order=1)

Write a closed unit-cube surface mesh (6 physical surfaces, one group) to
`path`. Smaller `lc` → more elements. `order=1` gives Quad4/Tri3, `order=2`
gives Quad8/Tri6.
"""
function make_box_msh(path; lc=0.09, order=1)
    gmsh.initialize(); gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("box")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    ptag = gmsh.model.addPhysicalGroup(2, collect(1:6))
    gmsh.model.setPhysicalName(2, ptag, "box")
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()
    return path
end

"""
    make_two_plates_msh(path; lc=0.1, order=1)

Two coaxial unit-square plates a distance 1 apart, wound to face each other so
the analytic bottom→top view factor is ≈ 0.19982. Physical groups "bottom" and
"top". Uses Gmsh's default unstructured triangulation.
"""
function make_two_plates_msh(path; lc=0.1, order=1)
    gmsh.initialize(); gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("plates")
    # Bottom plate (z=0), outward normal +z (toward the top plate)
    gmsh.model.geo.addPoint(0,0,0, lc, 1); gmsh.model.geo.addPoint(1,0,0, lc, 2)
    gmsh.model.geo.addPoint(1,1,0, lc, 3); gmsh.model.geo.addPoint(0,1,0, lc, 4)
    for (i,(a,b)) in enumerate([(1,2),(2,3),(3,4),(4,1)]); gmsh.model.geo.addLine(a,b,i); end
    gmsh.model.geo.addCurveLoop([1,2,3,4], 1); gmsh.model.geo.addPlaneSurface([1], 1)
    # Top plate (z=1), wound clockwise so its outward normal points -z (downward)
    gmsh.model.geo.addPoint(0,0,1, lc, 5); gmsh.model.geo.addPoint(1,0,1, lc, 6)
    gmsh.model.geo.addPoint(1,1,1, lc, 7); gmsh.model.geo.addPoint(0,1,1, lc, 8)
    for (i,(a,b)) in enumerate([(5,6),(6,7),(7,8),(8,5)]); gmsh.model.geo.addLine(a,b,i+4); end
    gmsh.model.geo.addCurveLoop([-8,-7,-6,-5], 2); gmsh.model.geo.addPlaneSurface([2], 2)
    gmsh.model.geo.synchronize()
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [1]), "bottom")
    gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [2]), "top")
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()
    return path
end

"""
    best_time(f; samples=3)

Run `f()` once to warm up, then return the minimum wall-clock time (seconds)
and the allocated bytes over `samples` runs. Minimum time is the standard
choice for microbenchmarks — it is the run least perturbed by the OS / GC.
"""
function best_time(f; samples=3)
    f()                                   # warm up / compile
    GC.gc()
    t = Inf; a = 0
    for _ in 1:samples
        GC.gc()
        b = @allocated f()
        s = @elapsed f()
        t = min(t, s)
        a = b
    end
    return t, a
end

"Print the machine / thread configuration used for a benchmark run."
function print_env()
    println("─"^64)
    println("CPU     : ", Sys.CPU_NAME, "  (", Sys.CPU_THREADS, " logical cores)")
    println("Machine : ", Sys.MACHINE)
    println("Julia   : ", VERSION, "   threads=", Threads.nthreads())
    println("─"^64)
end
