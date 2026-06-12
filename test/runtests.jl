# test/runtests.jl
using Test

include(joinpath(@__DIR__, "..", "src", "RadiativeViewFactor.jl"))
using .RadiativeViewFactor
using .RadiativeViewFactor.Quadrature
using .RadiativeViewFactor.Geometry
using .RadiativeViewFactor.BVH
using .RadiativeViewFactor.RayCast
using .RadiativeViewFactor.ViewFactorKernel
using .RadiativeViewFactor.MeshIO: SurfaceElement
using StaticArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
@testset "Quadrature pre-tabulated" begin
    for n in 1:5
        pts, wts = gauss_legendre_1d(n)
        @test length(pts) == n
        @test isapprox(sum(wts), 2.0; atol=1e-12)

        rule = gauss_legendre_2d(n)
        @test size(rule.points, 2) == n^2
        @test isapprox(sum(rule.weights), 4.0; atol=1e-12)
    end

    # n-point GL integrates polynomials of degree ≤ 2n-1 exactly
    # ∫₋₁¹ x⁴ dx = 2/5  (requires n ≥ 3)
    pts3, wts3 = gauss_legendre_1d(3)
    @test isapprox(dot(wts3, pts3.^4), 2/5; atol=1e-12)
end

# ---------------------------------------------------------------------------
@testset "Quadrature Golub-Welsch (n > 5)" begin
    for n in 6:8
        pts, wts = gauss_legendre_1d(n)
        @test length(pts) == n
        @test isapprox(sum(wts), 2.0; atol=1e-12)

        rule = gauss_legendre_2d(n)
        @test size(rule.points, 2) == n^2
        @test isapprox(sum(rule.weights), 4.0; atol=1e-12)
    end

    # ∫₋₁¹ x^10 dx = 2/11  (degree 10, requires n ≥ 6: 2*6-1=11 ≥ 10)
    pts6, wts6 = gauss_legendre_1d(6)
    @test isapprox(dot(wts6, pts6.^10), 2/11; atol=1e-12)
end

# ---------------------------------------------------------------------------
@testset "Quad8 shape functions" begin
    # Partition of unity: Σ Nₐ = 1 everywhere
    for (ξ, η) in [(-0.5, 0.3), (0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)]
        N, _, _ = quad8_shape(Float64(ξ), Float64(η))
        @test isapprox(sum(N), 1.0; atol=1e-14)
    end

    # Nodal interpolation: N_a(ξ_b, η_b) = δ_{ab}
    ref = [(-1.,-1.), (1.,-1.), (1.,1.), (-1.,1.),
           (0.,-1.), (1.,0.), (0.,1.), (-1.,0.)]
    for (a, (ξ_a, η_a)) in enumerate(ref)
        N, _, _ = quad8_shape(ξ_a, η_a)
        for b in 1:8
            @test isapprox(N[b], a==b ? 1.0 : 0.0; atol=1e-13)
        end
    end
end

# ---------------------------------------------------------------------------
# Quad8 node layout (Gmsh convention):
#
#   4---7---3
#   |       |
#   8       6
#   |       |
#   1---5---2
#
# For a 1×1 flat square in z=0: node 1=(0,0,0), 2=(1,0,0), 3=(1,1,0),
# 4=(0,1,0), 5=(0.5,0,0), 6=(1,0.5,0), 7=(0.5,1,0), 8=(0,0.5,0).
# ---------------------------------------------------------------------------
@testset "Quad8 physical geometry" begin
    coords = [0.0  1.0  1.0  0.0  0.5  1.0  0.5  0.0;
              0.0  0.0  1.0  1.0  0.0  0.5  1.0  0.5;
              0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0]
    nodes  = [1, 2, 3, 4, 5, 6, 7, 8]

    # Reference centre (ξ=η=0) maps to physical centre (0.5, 0.5, 0)
    x = quad8_physical_point(coords, nodes, 0.0, 0.0)
    @test isapprox(x, SVector(0.5, 0.5, 0.0); atol=1e-14)

    # Normal at centre is (0,0,1); area element = 0.25 (Jacobian of [-1,1]²→[0,1]²)
    n̂, dA = quad8_normal_and_area_element(coords, nodes, 0.0, 0.0)
    @test isapprox(n̂, SVector(0.0, 0.0, 1.0); atol=1e-14)
    @test isapprox(dA, 0.25; atol=1e-14)

    # ∫∫ dA over the 1×1 square = 1
    A = element_area(coords, nodes; nquad=4)
    @test isapprox(A, 1.0; atol=1e-12)
end

# ---------------------------------------------------------------------------
@testset "BVH ray casting" begin
    # Single triangle in the z=1 plane: vertices at (0,0,1),(1,0,1),(0,1,1)
    soup = zeros(Float64, 3, 3, 1)
    soup[1, :, 1] = [0.0, 0.0, 1.0]
    soup[2, :, 1] = [1.0, 0.0, 1.0]
    soup[3, :, 1] = [0.0, 1.0, 1.0]

    bvh = build_bvh(soup)

    origin    = SVector(0.25, 0.25, 0.0)
    direction = SVector(0.0,  0.0,  1.0)
    @test intersect_ray_bvh(bvh, origin, direction, 2.0)    # hits at t=1

    origin2   = SVector(2.0, 2.0, 0.0)
    @test !intersect_ray_bvh(bvh, origin2, direction, 2.0)  # misses
end

# ---------------------------------------------------------------------------
@testset "Visibility" begin
    # Triangle blocking the view
    soup = zeros(Float64, 3, 3, 1)
    soup[1, :, 1] = [-1.0, -1.0, 1.0]
    soup[2, :, 1] = [ 1.0, -1.0, 1.0]
    soup[3, :, 1] = [ 0.0,  1.0, 1.0]
    bvh = build_bvh(soup)

    xi = SVector(0.0, 0.0, 0.0)
    xj = SVector(0.0, 0.0, 2.0)
    @test !is_visible(bvh, xi, xj)   # blocked

    xk = SVector(5.0, 0.0, 2.0)
    @test  is_visible(bvh, xi, xk)   # unobstructed
end

# ---------------------------------------------------------------------------
@testset "vf_kernel" begin
    # Points directly facing: cos_i = cos_j = 1, r = 1 → K = 1/π
    xi = SVector(0.0, 0.0, 0.0)
    ni = SVector(0.0, 0.0, 1.0)
    xj = SVector(0.0, 0.0, 1.0)
    nj = SVector(0.0, 0.0,-1.0)
    K = RadiativeViewFactor.ViewFactorKernel.vf_kernel(xi, ni, xj, nj)
    @test isapprox(K, 1/π; atol=1e-14)

    # Same-direction normals (backs facing each other) → K = 0
    K_back = RadiativeViewFactor.ViewFactorKernel.vf_kernel(xi, ni, xj, ni)
    @test K_back == 0.0

    # Coincident points → K = 0
    K_same = RadiativeViewFactor.ViewFactorKernel.vf_kernel(xi, ni, xi, nj)
    @test K_same == 0.0
end

# ---------------------------------------------------------------------------
# Two parallel 1×1 unit squares at z=0 (normal +z) and z=1 (normal -z).
# The top plate's node ordering is mirrored in x so that cross(dxdξ, dxdη)
# points downward.
# ---------------------------------------------------------------------------
@testset "element_pair_view_factor parallel plates" begin
    coords = zeros(Float64, 3, 16)

    # Bottom plate (z=0, normal +z): nodes 1-8
    coords[:, 1] = [0.0, 0.0, 0.0]   # corner 1
    coords[:, 2] = [1.0, 0.0, 0.0]   # corner 2
    coords[:, 3] = [1.0, 1.0, 0.0]   # corner 3
    coords[:, 4] = [0.0, 1.0, 0.0]   # corner 4
    coords[:, 5] = [0.5, 0.0, 0.0]   # mid 1-2
    coords[:, 6] = [1.0, 0.5, 0.0]   # mid 2-3
    coords[:, 7] = [0.5, 1.0, 0.0]   # mid 3-4
    coords[:, 8] = [0.0, 0.5, 0.0]   # mid 4-1

    # Top plate (z=1, normal -z): nodes 9-16, x-mirrored ordering
    coords[:, 9]  = [1.0, 0.0, 1.0]
    coords[:, 10] = [0.0, 0.0, 1.0]
    coords[:, 11] = [0.0, 1.0, 1.0]
    coords[:, 12] = [1.0, 1.0, 1.0]
    coords[:, 13] = [0.5, 0.0, 1.0]  # mid 9-10
    coords[:, 14] = [0.0, 0.5, 1.0]  # mid 10-11
    coords[:, 15] = [0.5, 1.0, 1.0]  # mid 11-12
    coords[:, 16] = [1.0, 0.5, 1.0]  # mid 12-9

    elem_bot = SurfaceElement([1,2,3,4,5,6,7,8],   1, :quad)
    elem_top = SurfaceElement([9,10,11,12,13,14,15,16], 2, :quad)

    integ_ij, Ai = element_pair_view_factor(coords, elem_bot, elem_top, 4, nothing)
    integ_ji, Aj = element_pair_view_factor(coords, elem_top, elem_bot, 4, nothing)

    @test isapprox(Ai, 1.0; atol=1e-10)
    @test isapprox(Aj, 1.0; atol=1e-10)
    @test integ_ij > 0
    # Raw integrals are symmetric (reciprocity kernel is symmetric)
    @test isapprox(integ_ij, integ_ji; atol=1e-12)
    # View factor F_ij = integ_ij / Ai; for two unit squares at distance 1
    # the analytical value is ≈ 0.1998; check it's in a plausible range
    Fij = integ_ij / Ai
    @test 0.1 < Fij < 1.0
end

# ---------------------------------------------------------------------------
@testset "check_reciprocity and check_closure" begin
    # Perfect two-surface enclosure: equal areas, F_12 = F_21 = 1
    vfr = ViewFactorResult(
        Float64[0 1; 1 0],
        Float64[1.0, 1.0],
        Float64[0 1; 1 0],
        Float64[1.0, 1.0],
        [1, 2],
        ["surface1", "surface2"],
    )
    @test check_reciprocity(vfr)
    @test check_closure(vfr)

    # Unequal areas with correct reciprocity: A1=2, A2=1, F_12=0.5, F_21=1
    vfr2 = ViewFactorResult(
        Float64[0 0.5; 1 0],
        Float64[2.0, 1.0],
        Float64[0 0.5; 1 0],
        Float64[2.0, 1.0],
        [1, 2],
        ["surface1", "surface2"],
    )
    @test check_reciprocity(vfr2)
end

# ---------------------------------------------------------------------------
@testset "Linear shape functions (quad4, tri3, line2)" begin
    # Quad4: partition of unity and nodal interpolation N_a(ξ_b,η_b)=δ_ab
    for (ξ, η) in [(-0.3, 0.7), (0.0, 0.0), (1.0, -1.0), (0.5, 0.5)]
        N, _, _ = quad4_shape(Float64(ξ), Float64(η))
        @test isapprox(sum(N), 1.0; atol=1e-14)
    end
    quad4_ref = [(-1.,-1.), (1.,-1.), (1.,1.), (-1.,1.)]
    for (b, (ξ, η)) in enumerate(quad4_ref)
        N, _, _ = quad4_shape(ξ, η)
        for a in 1:4
            @test isapprox(N[a], a == b ? 1.0 : 0.0; atol=1e-14)
        end
    end

    # Line2: partition of unity and endpoint interpolation
    for ξ in (-0.4, 0.0, 0.8)
        N, _ = line2_shape(ξ)
        @test isapprox(sum(N), 1.0; atol=1e-14)
    end
    N, _ = line2_shape(-1.0); @test isapprox(N[1], 1.0; atol=1e-14)
    N, _ = line2_shape( 1.0); @test isapprox(N[2], 1.0; atol=1e-14)

    # Tri3 (internal to ViewFactorKernel): partition of unity + vertex interp
    tri3_shape = RadiativeViewFactor.ViewFactorKernel.tri3_shape
    for (ξ, η) in [(0.2, 0.3), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        N, _, _ = tri3_shape(ξ, η)
        @test isapprox(sum(N), 1.0; atol=1e-14)
    end
end

# ---------------------------------------------------------------------------
@testset "Triangle quadrature normalization (Dunavant)" begin
    # Regression: every rule must integrate constant 1 over the reference
    # triangle to its area 1/2.  (The degree-7 13-point rule previously
    # summed to ~0.797.)
    tri_quad_rule = RadiativeViewFactor.ViewFactorKernel.tri_quad_rule
    for n in 1:8
        rule = tri_quad_rule(n)
        # n=3 (degree-5) tabulated constants carry a ~4e-7 imprecision; the
        # rest are exact to machine precision.
        @test isapprox(sum(rule.weights), 0.5; atol=1e-6)
    end
    # Degree-7 rule integrates a degree-7 polynomial (x^3 y^4) exactly.
    rule = tri_quad_rule(4)
    approx = sum(rule.weights[k] * rule.points[1,k]^3 * rule.points[2,k]^4
                 for k in 1:size(rule.points, 2))
    # ∬_T x^3 y^4 dA = 3! 4! / (3+4+2)! = 6*24/362880 = 1/2520
    @test isapprox(approx, 1/2520; atol=1e-10)
end

# ---------------------------------------------------------------------------
@testset "Linear elements & cross-order consistency" begin
    # Two coaxial unit squares at distance 1; corner nodes only.
    coords = zeros(Float64, 3, 8)
    coords[:,1]=[0,0,0]; coords[:,2]=[1,0,0]; coords[:,3]=[1,1,0]; coords[:,4]=[0,1,0]
    coords[:,5]=[1,0,1]; coords[:,6]=[0,0,1]; coords[:,7]=[0,1,1]; coords[:,8]=[1,1,1]
    bot4 = SurfaceElement([1,2,3,4], 1, :quad4)
    top4 = SurfaceElement([5,6,7,8], 2, :quad4)
    raw4, A4 = element_pair_view_factor(coords, bot4, top4, 8, nothing)
    @test isapprox(A4, 1.0; atol=1e-10)
    F4 = raw4 / A4
    @test isapprox(F4, 0.19982; atol=2e-4)   # analytic ≈ 0.19982

    # Cross-order: flat Quad8 plates (corner nodes coincide) must agree with Quad4.
    coords8 = zeros(Float64, 3, 16)
    coords8[:,1:4]  = coords[:,1:4]
    coords8[:,5]=[0.5,0,0]; coords8[:,6]=[1,0.5,0]; coords8[:,7]=[0.5,1,0]; coords8[:,8]=[0,0.5,0]
    coords8[:,9:12] = coords[:,5:8]
    coords8[:,13]=[0.5,0,1]; coords8[:,14]=[0,0.5,1]; coords8[:,15]=[0.5,1,1]; coords8[:,16]=[1,0.5,1]
    bot8 = SurfaceElement([1,2,3,4,5,6,7,8], 1, :quad)
    top8 = SurfaceElement([9,10,11,12,13,14,15,16], 2, :quad)
    raw8, A8 = element_pair_view_factor(coords8, bot8, top8, 8, nothing)
    @test isapprox(raw8/A8, F4; atol=1e-6)
end

# ---------------------------------------------------------------------------
@testset "load_mesh: 1st & 2nd order + group fallback" begin
    import Gmsh: gmsh

    # Build a single unit square plate, mesh at a given element order, write .msh
    function make_plate_msh(path::String, order::Int; physical::Bool=true)
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("plate")
        gmsh.model.geo.addPoint(0,0,0, 0.5, 1)
        gmsh.model.geo.addPoint(1,0,0, 0.5, 2)
        gmsh.model.geo.addPoint(1,1,0, 0.5, 3)
        gmsh.model.geo.addPoint(0,1,0, 0.5, 4)
        for (i,(a,b)) in enumerate([(1,2),(2,3),(3,4),(4,1)])
            gmsh.model.geo.addLine(a,b,i)
        end
        gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        if physical
            ptag = gmsh.model.addPhysicalGroup(2, [1])
            gmsh.model.setPhysicalName(2, ptag, "plate")
        end
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.model.mesh.generate(2)
        gmsh.write(path)
        gmsh.finalize()
    end

    # Order 1 → linear elements (Tri3/Quad4)
    f1 = tempname() * ".msh"
    make_plate_msh(f1, 1)
    m1 = load_mesh(f1; verbose=false)
    @test !isempty(m1.surface_elems)
    @test all(e -> e.family in (:tri3, :quad4), m1.surface_elems)
    A1 = sum(element_pair_view_factor(m1.coords, e, e, 4, nothing)[2]
             for e in m1.surface_elems)
    @test isapprox(A1, 1.0; atol=1e-9)   # total plate area

    # Order 2 → quadratic elements (Tri6/Quad8)
    f2 = tempname() * ".msh"
    make_plate_msh(f2, 2)
    m2 = load_mesh(f2; verbose=false)
    @test all(e -> e.family in (:tri, :quad), m2.surface_elems)
    A2 = sum(element_pair_view_factor(m2.coords, e, e, 4, nothing)[2]
             for e in m2.surface_elems)
    @test isapprox(A2, 1.0; atol=1e-9)

    # Group fallback: a mesh with no physical groups gets a synthetic "default"
    f3 = tempname() * ".msh"
    make_plate_msh(f3, 1; physical=false)
    m3 = load_mesh(f3; verbose=false)
    @test !isempty(m3.surface_elems)
    @test collect(values(m3.group_tags)) == ["default"]

    foreach(rm, (f1, f2, f3))
end

# ---------------------------------------------------------------------------
@testset "Unstructured surface mesh end-to-end view factors" begin
    import Gmsh: gmsh

    # Two coaxial unit-square plates a distance 1 apart, each discretized with
    # Gmsh's DEFAULT (unstructured) triangulation — no transfinite/structured
    # meshing. Physical groups "bottom" and "top". Analytic F(bottom→top) for
    # two directly-opposed unit squares at separation 1 is ≈ 0.19982.
    function make_two_plates_msh(path::String, order::Int, h::Float64)
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("plates")
        # Bottom plate (z=0)
        gmsh.model.geo.addPoint(0,0,0, h, 1); gmsh.model.geo.addPoint(1,0,0, h, 2)
        gmsh.model.geo.addPoint(1,1,0, h, 3); gmsh.model.geo.addPoint(0,1,0, h, 4)
        for (i,(a,b)) in enumerate([(1,2),(2,3),(3,4),(4,1)]); gmsh.model.geo.addLine(a,b,i); end
        gmsh.model.geo.addCurveLoop([1,2,3,4], 1); gmsh.model.geo.addPlaneSurface([1], 1)
        # Top plate (z=1). Wind the loop clockwise (viewed from +z) so its
        # outward normal points DOWN (-z), i.e. toward the bottom plate.
        gmsh.model.geo.addPoint(0,0,1, h, 5); gmsh.model.geo.addPoint(1,0,1, h, 6)
        gmsh.model.geo.addPoint(1,1,1, h, 7); gmsh.model.geo.addPoint(0,1,1, h, 8)
        for (i,(a,b)) in enumerate([(5,6),(6,7),(7,8),(8,5)]); gmsh.model.geo.addLine(a,b,i+4); end
        gmsh.model.geo.addCurveLoop([-8,-7,-6,-5], 2); gmsh.model.geo.addPlaneSurface([2], 2)
        gmsh.model.geo.synchronize()
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [1]), "bottom")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [2]), "top")
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.model.mesh.generate(2)
        gmsh.write(path)
        gmsh.finalize()
    end

    analytic = 0.19982

    # First-order unstructured triangle mesh (Tri3)
    f1 = tempname() * ".msh"
    make_two_plates_msh(f1, 1, 0.2)
    m1 = load_mesh(f1; verbose=false)
    @test all(e -> e.family === :tri3, m1.surface_elems)   # genuinely unstructured tris
    @test length(m1.group_tags) == 2
    r1 = compute_view_factors(m1; nquad=4, verbose=false)
    b = findfirst(==("bottom"), r1.group_names)
    t = findfirst(==("top"),    r1.group_names)
    @test isapprox(r1.F_group[b, t], analytic; atol=5e-3)
    @test check_reciprocity(r1)

    # Second-order unstructured triangle mesh (Tri6) — cross-order agreement
    f2 = tempname() * ".msh"
    make_two_plates_msh(f2, 2, 0.2)
    m2 = load_mesh(f2; verbose=false)
    @test all(e -> e.family === :tri, m2.surface_elems)
    r2 = compute_view_factors(m2; nquad=4, verbose=false)
    b2 = findfirst(==("bottom"), r2.group_names)
    t2 = findfirst(==("top"),    r2.group_names)
    @test isapprox(r2.F_group[b2, t2], analytic; atol=5e-3)

    foreach(rm, (f1, f2))
end

# ---------------------------------------------------------------------------
@testset "GPU kernel (CPU backend) linear families" begin
    using KernelAbstractions: CPU
    GPUKernels = RadiativeViewFactor.GPUKernels
    MeshData   = RadiativeViewFactor.MeshIO.MeshData

    coords = zeros(Float64, 3, 8)
    coords[:,1]=[0,0,0]; coords[:,2]=[1,0,0]; coords[:,3]=[1,1,0]; coords[:,4]=[0,1,0]
    coords[:,5]=[1,0,1]; coords[:,6]=[0,0,1]; coords[:,7]=[0,1,1]; coords[:,8]=[1,1,1]
    elems = [SurfaceElement([1,2,3,4], 1, :quad4),
             SurfaceElement([5,6,7,8], 2, :quad4)]
    mesh = MeshData(coords, elems, Dict(1=>"a",2=>"b"),
                    Dict(1=>[1], 2=>[2]), Dict{Int,Array{Float64,3}}(), 2)
    ga = GPUKernels.build_gpu_arrays(mesh, 6, Array, Float64)
    raw, area = GPUKernels.launch_vf_kernel!(ga, CPU())
    @test isapprox(Array(area)[1], 1.0; atol=1e-9)
    @test isapprox(Array(raw)[1,2] / Array(area)[1], 0.19982; atol=2e-4)
end

# ---------------------------------------------------------------------------
@testset "VTK routing (sniffer + ReadVTK guard)" begin
    _is_xml_vtk = RadiativeViewFactor.MeshIO._is_xml_vtk

    # XML .vtu → detected as XML VTK
    f_vtu = tempname() * ".vtu"
    write(f_vtu, "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\">")
    @test _is_xml_vtk(f_vtu)

    # XML-form .vtk → detected via header sniff
    f_xvtk = tempname() * ".vtk"
    write(f_xvtk, "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\">")
    @test _is_xml_vtk(f_xvtk)

    # Legacy .vtk → NOT XML (routed to Gmsh instead)
    f_legacy = tempname() * ".vtk"
    write(f_legacy, "# vtk DataFile Version 3.0\nmesh\nASCII\n")
    @test !_is_xml_vtk(f_legacy)

    # Non-VTK extension → false
    f_msh = tempname() * ".msh"
    write(f_msh, "\$MeshFormat\n")
    @test !_is_xml_vtk(f_msh)

    # Without `using ReadVTK`, load_vtu must raise a helpful error (the
    # extension method is not installed in this include-based test context).
    @test_throws ErrorException load_vtu(f_vtu; verbose=false)

    foreach(rm, (f_vtu, f_xvtk, f_legacy, f_msh))
end

println("\nAll tests passed.")
