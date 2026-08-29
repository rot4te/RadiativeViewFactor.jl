# test/re2_test.jl
# Round-trip tests for the Nek5000/NekRS .re2 loader. A synthetic writer emits a
# single unit-cube hex with per-face boundary-condition labels, matching the
# byte layout the loader expects, so the header/endianness/word-size handling,
# boundary-face extraction, BC grouping, and outward normal orientation are all
# exercised end to end. (Validation against a real Nek .re2 file is separate.)

# Write one unit hex [0,1]³ to `path`. `codes` is a vector of (iside, label)
# boundary records. `swap` emits big-/foreign-endian; `wd` is the real word size.
function _write_re2(path::AbstractString, codes::Vector{Tuple{Int,String}};
                    swap::Bool=false, wd::Int=8)
    corners = [(0.0,0.0,0.0),(1.0,0.0,0.0),(1.0,1.0,0.0),(0.0,1.0,0.0),
               (0.0,0.0,1.0),(1.0,0.0,1.0),(1.0,1.0,1.0),(0.0,1.0,1.0)]
    T  = wd == 8 ? Float64 : Float32
    io = IOBuffer()
    hdr = "#v001" * lpad("1",9) * lpad("3",3) * lpad("1",9)   # nelgt=1 ndim=3 nelgv=1
    write(io, rpad(hdr, 80))
    wf(x, ty) = begin
        b = collect(reinterpret(UInt8, [ty(x)]))
        swap && reverse!(b)
        write(io, b)
    end
    wf(6.54321, Float32)                        # endian tag (always real*4)
    wf(0.0, T)                                  # igroup
    for d in 1:3, v in 1:8; wf(corners[v][d], T); end   # x8, y8, z8
    wf(0.0, T)                                  # ncurve = 0
    wf(Float64(length(codes)), T)               # nbc (single BC field)
    for (iside, code) in codes
        wf(1.0, T); wf(Float64(iside), T)       # element, face
        for _ in 1:5; wf(0.0, T); end           # 5 bc params
        cb = zeros(UInt8, 8)                     # char*8 code slot (not swapped)
        for (i, c) in enumerate(code); i <= 8 && (cb[i] = UInt8(c)); end
        write(io, cb)
    end
    write(path, take!(io))
    return path
end

@testset ".re2 Nek5000 loader" begin
    allcodes = [(1,"HOT"),(2,"HOT"),(3,"HOT"),(4,"CLD"),(5,"CLD"),(6,"CLD")]

    @testset "boundary faces, BC grouping, orientation" begin
        f = _write_re2(tempname()*".re2", allcodes)
        m = load_re2(f; verbose=false)

        @test m.mesh_dim == 2
        @test length(m.surface_elems) == 6
        @test all(e -> e.family === :quad4, m.surface_elems)
        @test sort(collect(values(m.group_tags))) == ["CLD","HOT"]
        for (t, _) in m.group_tags
            @test length(m.group_elems[t]) == 3          # 3 faces per label
        end

        # Every element carries its Nek (global element, local face) identity;
        # for this single-hex fixture eg is always 1 and iface is the BC
        # record's face index 1-6.
        @test all(e -> e.eg == 1, m.surface_elems)
        @test sort([e.iface for e in m.surface_elems]) == 1:6

        # total surface area of the unit cube = 6
        A = sum(element_pair_view_factor(m.coords, e, e, 2, nothing)[2]
                for e in m.surface_elems)
        @test isapprox(A, 6.0; atol=1e-9)

        # every face normal points inward (toward the cube centroid = into the
        # radiating cavity), the convention load_re2 assigns
        cube_c = SVector(0.5, 0.5, 0.5)
        for e in m.surface_elems
            n̂, _ = quad4_normal_and_area_element(m.coords, e.nodes, 0.0, 0.0)
            fc   = sum(SVector{3,Float64}(m.coords[:, i]) for i in e.nodes) / 4
            @test dot(n̂, fc - cube_c) < 0
        end

        # Physics check: the two faces perpendicular to z are directly-opposed
        # unit squares a distance 1 apart → F ≈ 0.19982 (Hottel/Modest, no
        # shared edge). Reciprocity is exact regardless of nquad/use_duffy.
        r  = compute_view_factors(m; nquad=6, verbose=false)
        @test check_reciprocity(r)
        zc = [sum(m.coords[3, e.nodes]) / 4 for e in m.surface_elems]
        i0 = findfirst(z -> isapprox(z, 0.0; atol=1e-9), zc)
        i1 = findfirst(z -> isapprox(z, 1.0; atol=1e-9), zc)
        @test isapprox(r.F_elem[i0, i1], 0.19982; atol=5e-3)
        @test all(>(0), vec(sum(r.F_elem, dims=2)))   # every face sees the cavity

        # Full-cube closure requires resolving the shared-edge singularity
        # between every pair of adjacent Quad4 faces (every face pair in a
        # cube is either opposite or edge-adjacent — there are no
        # non-adjacent, non-opposite pairs). Plain quadrature overestimates
        # this badly (rowsum ≈1.28 at nquad=6); use_duffy=true must bring it
        # within a fraction of a percent of the true rowsum=1.0, and the
        # adjacent-pair value must match Modest's closed-form perpendicular
        # common-edge-rectangles result F≈0.20004.
        r_plain = compute_view_factors(m; nquad=6, verbose=false)
        rs_plain = vec(sum(r_plain.F_elem, dims=2))
        @test all(x -> x > 1.2, rs_plain)   # the known-bad baseline, guards regressions

        r_duffy = compute_view_factors(m; nquad=10, use_duffy=true, verbose=false)
        @test check_reciprocity(r_duffy)
        rs_duffy = vec(sum(r_duffy.F_elem, dims=2))
        @test all(x -> isapprox(x, 1.0; atol=1e-3), rs_duffy)
        adjacent_j = first(j for j in 1:6 if j != i0 && j != i1)   # any face adjacent to i0
        @test isapprox(r_duffy.F_elem[i0, adjacent_j], 0.20004; atol=2e-3)

        rm(f)
    end

    @testset "load_mesh routing dispatches to .re2" begin
        f = _write_re2(tempname()*".re2", allcodes)
        m = load_mesh(f; verbose=false)
        @test length(m.surface_elems) == 6
        rm(f)
    end

    @testset "byte-swapped (foreign endian)" begin
        f = _write_re2(tempname()*".re2", allcodes; swap=true)
        m = load_re2(f; verbose=false)
        @test length(m.surface_elems) == 6
        @test sort(collect(values(m.group_tags))) == ["CLD","HOT"]
        rm(f)
    end

    @testset "4-byte reals auto-detected" begin
        f = _write_re2(tempname()*".re2", allcodes; wd=4)
        m = load_re2(f; verbose=false)
        @test length(m.surface_elems) == 6
        @test sort(collect(values(m.group_tags))) == ["CLD","HOT"]
        rm(f)
    end

    @testset "topological fallback when all faces are internal" begin
        f = _write_re2(tempname()*".re2", [(i, "E") for i in 1:6])
        m = load_re2(f; verbose=false)
        @test collect(values(m.group_tags)) == ["default"]
        @test length(m.surface_elems) == 6          # all 6 faces are on the boundary
        rm(f)
    end

    @testset "real Nek file: tall_cavity.re2" begin
        # A real #v004 file: interior of a rectangular prism (0.5 × 1.0 × 0.05),
        # 8000 hexes, periodic in x. gmsh2nek labels the four solid walls "MSH"
        # and the two x-faces "P" (periodic). The loader must auto-detect the
        # v004/8-byte/little-endian layout and keep *all* 2400 non-internal
        # faces (1600 "MSH" + 800 "P") — periodic faces are boundary faces of
        # the radiation enclosure too (Nek5000/NekRS's own view-factor module
        # keeps every cbc≠'E'/blank face for closure; only 'W' emits, decided
        # at Nek runtime, not by this loader). This is also the exact case
        # documented in nekRS/tall_cavity_vf_aurora/ — its hemicube-derived
        # reference view-factor file (vf_tall_cavity) has the same 2400-face
        # header count.
        path = joinpath(@__DIR__, "tall_cavity.re2")
        if isfile(path)
            m = load_re2(path; verbose=false)
            @test m.mesh_dim == 2
            @test length(m.surface_elems) == 2400
            @test all(e -> e.family === :quad4, m.surface_elems)
            @test sort(collect(values(m.group_tags))) == ["MSH", "P"]
            @test length(m.group_elems[first(k for (k,v) in m.group_tags if v=="MSH")]) == 1600
            @test length(m.group_elems[first(k for (k,v) in m.group_tags if v=="P")]) == 800
            @test all(e -> e.eg > 0 && 1 <= e.iface <= 6, m.surface_elems)

            # Full box surface area: 2*(Lx*Ly + Ly*Lz + Lx*Lz) with
            # Lx=0.5, Ly=1.0, Lz=0.05 → 1.15 (wall 1.05 + periodic 0.10).
            A = sum(element_pair_view_factor(m.coords, e, e, 2, nothing)[2]
                    for e in m.surface_elems)
            @test isapprox(A, 1.15; atol=1e-6)

            # all normals point into the cavity, walls and periodic faces alike
            cen = SVector(sum(m.coords[1,:]), sum(m.coords[2,:]),
                          sum(m.coords[3,:])) / size(m.coords, 2)
            @test all(m.surface_elems) do e
                n̂, _ = quad4_normal_and_area_element(m.coords, e.nodes, 0.0, 0.0)
                fc   = sum(SVector{3,Float64}(m.coords[:, i]) for i in e.nodes) / 4
                dot(n̂, fc - cen) < 0
            end

            # End-to-end solve on the real, fully-closed enclosure: reciprocity
            # is exact regardless of nquad; use_duffy=true is required for
            # closure (row sums close to 1) since every face has several
            # edge-adjacent neighbors in this structured mesh — without it,
            # rowsum errors exceed 100% (see DuffyKernel.jl).
            r  = compute_view_factors(m; nquad=6, use_duffy=true, verbose=false)
            rs = vec(sum(r.F_elem, dims=2))
            @test check_reciprocity(r)
            @test isapprox(sum(rs) / length(rs), 1.0; atol=0.02)
            @test all(x -> 0.9 < x < 1.2, rs)
        else
            @info "tall_cavity.re2 not present; skipping real-file .re2 test."
        end
    end

    @testset "helpful errors" begin
        # 2D .re2 is rejected
        bad = tempname()*".re2"
        write(bad, rpad("#v001" * lpad("1",9) * lpad("2",3) * lpad("1",9), 80) *
                   String(collect(reinterpret(UInt8, [Float32(6.54321)]))))
        @test_throws ErrorException load_re2(bad; verbose=false)
        rm(bad)

        # surface_dim=1 unsupported
        f = _write_re2(tempname()*".re2", allcodes)
        @test_throws ErrorException load_re2(f; surface_dim=1, verbose=false)
        rm(f)
    end
end
