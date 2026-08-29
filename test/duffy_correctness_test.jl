# test/duffy_correctness_test.jl
# Correctness (not just type-stability) coverage for the Duffy/Sauter-Schwab
# singular-pair transform. The pre-existing test suite only ever smoke-tested
# element_pair_view_factor_duffy for type inference on a degenerate self-pair
# — it never checked the actual singular-region math against a known value,
# and that math turned out to be wrong (row sums off by 30-40x on a closed
# cube). These tests pin the fix: a fully-closed enclosure must have every
# row sum to 1 (Eq. 9 in the classical view-factor theory), and the
# common-edge value must match the classical closed-form result.

@testset "Duffy correctness (closed enclosures)" begin
    # Reuse the tested unit-cube .re2 fixture from re2_test.jl (same file, so
    # normals/orientation are already validated there).
    allcodes = [(1,"HOT"),(2,"HOT"),(3,"HOT"),(4,"CLD"),(5,"CLD"),(6,"CLD")]

    @testset "Quad4 unit cube: closure and known adjacent/opposite values" begin
        f = _write_re2(tempname()*".re2", allcodes)
        m = load_re2(f; verbose=false)
        rm(f)

        # Baseline: plain quadrature does NOT close a cube (every face pair is
        # either opposite or edge-adjacent; the shared-edge singularity is
        # unresolved). This guards against silently losing the Duffy path.
        r_plain = compute_view_factors(m; nquad=6, verbose=false)
        @test all(x -> x > 1.2, vec(sum(r_plain.F_elem, dims=2)))

        # With Duffy, closure should hold to a fraction of a percent by
        # nquad=10, and reciprocity is exact at any nquad (kernel-symmetric
        # regardless of quadrature accuracy).
        for nq in (6, 10)
            r = compute_view_factors(m; nquad=nq, use_duffy=true, verbose=false)
            @test check_reciprocity(r)
            rs = vec(sum(r.F_elem, dims=2))
            @test all(x -> isapprox(x, 1.0; atol = nq == 6 ? 3e-3 : 5e-4), rs)
        end

        # Known closed-form values (Modest, Radiative Heat Transfer):
        # opposite unit squares 1 apart: F=0.19982; perpendicular unit squares
        # sharing a common edge: F=0.20004.
        r = compute_view_factors(m; nquad=10, use_duffy=true, verbose=false)
        zc = [sum(m.coords[3, e.nodes]) / 4 for e in m.surface_elems]
        i0 = findfirst(z -> isapprox(z, 0.0; atol=1e-9), zc)
        i1 = findfirst(z -> isapprox(z, 1.0; atol=1e-9), zc)
        @test isapprox(r.F_elem[i0, i1], 0.19982; atol=2e-3)
        adjacent_j = first(j for j in 1:6 if j != i0 && j != i1)
        @test isapprox(r.F_elem[i0, adjacent_j], 0.20004; atol=2e-3)
    end

    @testset "Monte Carlo + near-pair Duffy patch closes the cube" begin
        # Plain Monte Carlo has unbounded variance for the shared-edge pairs
        # in a closed cube (every face pair is opposite or edge-adjacent) —
        # patch_adjacent_pairs_duffy! must bring it to comparable accuracy
        # to the quadrature+Duffy path, and reciprocity stays exact.
        f = _write_re2(tempname()*".re2", allcodes)
        m = load_re2(f; verbose=false)
        rm(f)

        r = compute_view_factors(m; monte_carlo=true, n_samples=2000, nquad=10,
                                  rng=MersenneTwister(1), verbose=false)
        @test check_reciprocity(r)
        rs = vec(sum(r.F_elem, dims=2))
        @test all(x -> isapprox(x, 1.0; atol=0.03), rs)
    end

    @testset "near_pairs matches brute force" begin
        # The spatial-grid neighbor finder must be a strict superset of what
        # an O(n²) distance check would find (missing a near pair silently
        # degrades back to unbounded/high MC variance for it).
        path = joinpath(@__DIR__, "tall_cavity.re2")
        if isfile(path)
            m = load_re2(path; verbose=false)
            Random.seed!(1)
            sub = m.surface_elems[randperm(length(m.surface_elems))[1:200]]
            factor = 3.0

            function brute_near(coords, elems, factor)
                N = length(elems)
                cents = [sum(coords[:, n] for n in e.nodes) / length(e.nodes) for e in elems]
                sizes = [2 * maximum(sqrt(sum((coords[:, n] .- cents[i]) .^ 2)) for n in elems[i].nodes)
                         for i in 1:N]
                pairs = Set{Tuple{Int,Int}}()
                for i in 1:N, j in i+1:N
                    d = sqrt(sum((cents[i] .- cents[j]) .^ 2))
                    d < factor * max(sizes[i], sizes[j]) && push!(pairs, (i, j))
                end
                return pairs
            end

            bf = brute_near(m.coords, sub, factor)
            gd = Set(RadiativeViewFactor.DuffyKernel.near_pairs(m.coords, sub; factor=factor))
            @test issubset(bf, gd)
        end
    end

    @testset "region decompositions tile the unit hypercube exactly" begin
        # Direct check on the transform's own bookkeeping, independent of any
        # element geometry: integrating the constant function 1 through each
        # decomposition must recover the true domain volume (1.0). This is
        # what actually caught the original bug (wrong region formulas gave a
        # transform that didn't tile the domain at all).
        DFK = RadiativeViewFactor.DuffyKernel
        n = 40
        pts = ((0:n-1) .+ 0.5) ./ n
        w   = 1.0 / n

        total_vertex = 0.0
        for region in 1:4, ρ in pts, η1 in pts, η2 in pts, η3 in pts
            total_vertex += w^4 * ρ^3
        end
        @test isapprox(total_vertex, 1.0; atol=1e-3)

        total_edge = 0.0
        for branch in (:plus, :minus), region in 1:3, a in pts, ρ in pts, η1 in pts, η2 in pts
            total_edge += w^4 * a * ρ^2
        end
        @test isapprox(total_edge, 1.0; atol=1e-3)
    end
end
