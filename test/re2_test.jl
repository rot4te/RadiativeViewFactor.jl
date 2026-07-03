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
        # unit squares a distance 1 apart → F ≈ 0.19982 (this pair has no shared
        # edge, so it is well resolved by plain quadrature). Reciprocity is exact
        # regardless. (A full cube's closure needs singular-edge integration.)
        r  = compute_view_factors(m; nquad=6, verbose=false)
        @test check_reciprocity(r)
        zc = [sum(m.coords[3, e.nodes]) / 4 for e in m.surface_elems]
        i0 = findfirst(z -> isapprox(z, 0.0; atol=1e-9), zc)
        i1 = findfirst(z -> isapprox(z, 1.0; atol=1e-9), zc)
        @test isapprox(r.F_elem[i0, i1], 0.19982; atol=5e-3)
        @test all(>(0), vec(sum(r.F_elem, dims=2)))   # every face sees the cavity

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
        # v004/8-byte/little-endian layout, keep the 1600 wall faces, and drop
        # the 800 periodic faces.
        path = joinpath(@__DIR__, "tall_cavity.re2")
        if isfile(path)
            m = load_re2(path; verbose=false)
            @test m.mesh_dim == 2
            @test length(m.surface_elems) == 1600
            @test all(e -> e.family === :quad4, m.surface_elems)
            @test collect(values(m.group_tags)) == ["MSH"]    # "P" faces excluded

            # wall area of the cavity minus the two periodic x-faces:
            # 2·(0.5×0.05) + 2·(0.5×1.0) = 1.05
            A = sum(element_pair_view_factor(m.coords, e, e, 2, nothing)[2]
                    for e in m.surface_elems)
            @test isapprox(A, 1.05; atol=1e-6)

            # all wall normals point into the cavity
            cen = SVector(sum(m.coords[1,:]), sum(m.coords[2,:]),
                          sum(m.coords[3,:])) / size(m.coords, 2)
            @test all(m.surface_elems) do e
                n̂, _ = quad4_normal_and_area_element(m.coords, e.nodes, 0.0, 0.0)
                fc   = sum(SVector{3,Float64}(m.coords[:, i]) for i in e.nodes) / 4
                dot(n̂, fc - cen) < 0
            end

            # end-to-end solve: reciprocity is exact. The mean row sum is below 1
            # because radiation escapes through the two periodic openings (a few
            # individual faces exceed 1 from edge-singularity overcounting where
            # walls meet — the same Quad4 limitation as a closed cube).
            r  = compute_view_factors(m; nquad=3, verbose=false)
            rs = vec(sum(r.F_elem, dims=2))
            @test check_reciprocity(r)
            @test sum(rs) / length(rs) < 1.0          # net leakage through openings
            @test all(>(0), rs)                       # every wall sees the cavity
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
