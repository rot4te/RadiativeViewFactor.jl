# test/nek_export_test.jl
# write_nekrs_view_factors round-trips the (eg,iface) identity load_re2
# records, in the exact list-directed format Nek5000/NekRS's
# vf_read_view_factors expects.

@testset "write_nekrs_view_factors" begin
    allcodes = [(1,"HOT"),(2,"HOT"),(3,"HOT"),(4,"CLD"),(5,"CLD"),(6,"CLD")]
    f = _write_re2(tempname()*".re2", allcodes)
    m = load_re2(f; verbose=false)
    rm(f)
    r = compute_view_factors(m; nquad=10, use_duffy=true, verbose=false)

    out = tempname()
    n = write_nekrs_view_factors(out, r, m; verbose=false)
    @test n == 6

    lines = readlines(out)
    @test parse(Int, lines[1]) == 6

    # Re-parse the file with the same list-directed structure
    # vf_read_view_factors uses, and check it's self-consistent with `r`.
    idx = 2
    seen_iw = Int[]
    for expected_iw in 1:6
        hdr = split(lines[idx]); idx += 1
        iw, ieg, ifc, nvw = parse.(Int, hdr)
        @test iw == expected_iw
        e = m.surface_elems[iw]
        @test ieg == e.eg
        @test ifc == e.iface
        push!(seen_iw, iw)

        expected_nvw = count(j -> j != iw && r.F_elem[iw, j] > 0.0, 1:6)
        @test nvw == expected_nvw

        for _ in 1:nvw
            row = split(lines[idx]); idx += 1
            jw = parse(Int, row[1]); jeg = parse(Int, row[2])
            jfc = parse(Int, row[3]); fij = parse(Float64, row[4])
            ej = m.surface_elems[jw]
            @test jeg == ej.eg
            @test jfc == ej.iface
            @test isapprox(fij, r.F_elem[iw, jw]; rtol=1e-6)
        end
    end
    @test idx - 1 == length(lines)   # no leftover/missing lines
    @test seen_iw == 1:6

    rm(out)

    # Requires a load_re2-sourced mesh (eg/iface populated); a plain
    # SurfaceElement mesh (eg=iface=0) must error, not silently write a
    # nonsense file.
    bad_mesh = RadiativeViewFactor.MeshData(
        m.coords,
        [RadiativeViewFactor.SurfaceElement(e.nodes, e.group, e.family) for e in m.surface_elems],
        m.group_tags, m.group_elems, m.group_tri_soup, m.mesh_dim,
    )
    @test_throws ErrorException write_nekrs_view_factors(tempname(), r, bad_mesh; verbose=false)
end
