# test/raytrace_test.jl
#
# Ray-shooting Monte Carlo kernel (RayTraceKernel.jl / `compute_view_factors(
# ...; raytrace=true)`). See RayTraceKernel.jl's module docstring for the
# method and changelog.md for the validation history, including a real
# front/back-face bug found while prototyping this (§ below reproduces the
# regression case that caught it) and a case where the *quadrature* result
# turned out to be the less accurate one (a sharp, under-resolved obstruction
# shadow boundary) -- included here as a documented, not just narrated, check.

@testset "Ray-shooting Monte Carlo (raytrace=true)" begin

    @testset "two facing unit squares: unbiased against the analytic value" begin
        # Same fixture as the "MC pair estimator is unbiased" regression test
        # in ray_test.jl -- single Quad4 pair, analytic F=0.19982.
        coords = [0.0 1 1 0  1 0 0 1;
                  0.0 0 1 1  0 0 1 1;
                  0.0 0 0 0  1 1 1 1]
        elems  = [SurfaceElement([1,2,3,4], 1, :quad4), SurfaceElement([5,6,7,8], 2, :quad4)]
        mesh   = MeshData(coords, elems, Dict(1=>"bottom",2=>"top"),
                          Dict(1=>[1],2=>[2]), Dict{Int,Array{Float64,3}}(), 2)

        r = compute_view_factors(mesh; raytrace=true, n_rays=200_000,
                                  rng=Xoshiro(1), verbose=false)
        @test isapprox(r.F_group[1,2], 0.19982; rtol=5e-3)
        # Reciprocity is exact by construction, not merely approximate.
        @test r.F_elem[1,2] * r.A_elem[1] == r.F_elem[2,1] * r.A_elem[2]
        @test check_reciprocity(r)
    end

    @testset "closed cube: row-sum closure and agreement with quadrature" begin
        cube_coords = [0.0 1 1 0 0 1 1 0;
                       0.0 0 1 1 0 0 1 1;
                       0.0 0 0 0 1 1 1 1]
        raw_faces = [ (1,2,3,4), (5,6,7,8), (1,2,6,5), (2,3,7,6), (3,4,8,7), (4,1,5,8) ]
        center = SVector(0.5, 0.5, 0.5)
        function orient_inward(coords, f)
            p(k) = SVector{3,Float64}(coords[1,k], coords[2,k], coords[3,k])
            a,b,c,d = f
            nvec = cross(p(b)-p(a), p(c)-p(a))
            centroid = (p(a)+p(b)+p(c)+p(d))/4
            dot(nvec, center - centroid) > 0 ? f : (a,d,c,b)
        end
        faces = [orient_inward(cube_coords, f) for f in raw_faces]
        cube_elems = [SurfaceElement(collect(f), 1, :quad4) for f in faces]
        cube_mesh = MeshData(cube_coords, cube_elems, Dict(1=>"cube"), Dict(1=>collect(1:6)),
                             Dict{Int,Array{Float64,3}}(), 2)

        r = compute_view_factors(cube_mesh; raytrace=true, n_rays=50_000,
                                  rng=Xoshiro(2), verbose=false)
        rs = vec(sum(r.F_elem, dims=2))
        # In a closed convex enclosure no ray can ever escape or hit a
        # backface, so *each row's own* hit-count estimate sums to exactly
        # 1.0 -- but the reciprocity-averaging step (see Assembly.jl
        # `_compute_cpu_raytrace`) blends in the *other* element's
        # independent estimate for every off-diagonal entry, which reintroduces
        # ordinary MC noise into the row sum (~1e-3 at this n_rays, measured).
        # This is expected and documented, not a closure bug.
        @test all(x -> isapprox(x, 1.0; atol=0.01), rs)

        r_quad = compute_view_factors(cube_mesh; nquad=8, use_duffy=true, verbose=false)
        @test isapprox(r.F_elem[1,2], r_quad.F_elem[1,2]; atol=0.01)   # adjacent faces, analytic 0.20004
        @test isapprox(r.F_elem[1,4], r_quad.F_elem[1,4]; atol=0.01)   # opposite faces, analytic 0.19982
    end

    @testset "radiating elements obstruct each other without obstruction_groups" begin
        # Bottom/top unit squares 1 apart, plus a third unit square at the
        # midplane, wound so its front faces bottom -- a full blocker never
        # listed in `obstruction_groups`. Regression case for the front/back
        # -face bug found while prototyping: an un-oriented or wrongly-wound
        # blocker attributes back-face hits to F(bottom->blocker), which an
        # earlier version of this kernel did not check for.
        coords3 = [0.0 1 1 0  1 0 0 1  0.0 1 1 0;
                   0.0 0 1 1  0 0 1 1  0.0 0 1 1;
                   0.0 0 0 0  1 1 1 1  0.5 0.5 0.5 0.5]
        elems3  = [SurfaceElement([1,2,3,4], 1, :quad4), SurfaceElement([5,6,7,8], 2, :quad4),
                   SurfaceElement([9,12,11,10], 3, :quad4)]   # front wound toward bottom
        mesh3   = MeshData(coords3, elems3, Dict(1=>"bottom",2=>"top",3=>"blocker"),
                           Dict(1=>[1],2=>[2],3=>[3]), Dict{Int,Array{Float64,3}}(), 2)

        r = compute_view_factors(mesh3; raytrace=true, n_rays=50_000,
                                  rng=Xoshiro(3), verbose=false)
        @test isapprox(r.F_elem[1,2], 0.0; atol=1e-9)   # fully blocked

        raw_ref, Ai_ref = RadiativeViewFactor.ViewFactorKernel.element_pair_view_factor(
            coords3, elems3[1], elems3[3], 12, nothing, 2)
        @test isapprox(r.F_elem[1,3], raw_ref/Ai_ref; atol=0.01)   # ground truth ~0.415
    end

    @testset "partial obstruction matches quadrature+obstruction_groups" begin
        # A blocker covering the central 50% of the aperture (blocker_half=0.25):
        # moderate obstruction, resolved well by both methods at this mesh/nquad,
        # so this checks agreement rather than exercising the discontinuity
        # sensitivity documented in changelog.md.
        function three_body_mesh(; blocker_half::Float64, n::Int=6)
            function plate(z)
                idx(i,j) = (i)*(n+1) + j + 1
                pts = Matrix{Float64}(undef, 3, (n+1)^2)
                for i in 0:n, j in 0:n; pts[:, idx(i,j)] = [i/n, j/n, z]; end
                quads = NTuple{4,Int}[]
                for i in 0:n-1, j in 0:n-1
                    push!(quads, (idx(i,j), idx(i+1,j), idx(i+1,j+1), idx(i,j+1)))
                end
                pts, quads
            end
            p1, q1 = plate(0.0); p2, q2 = plate(1.0)
            off1 = 0; off2 = size(p1,2)
            coords = hcat(p1, p2)
            elems = SurfaceElement[]
            for (a,b,c,d) in q1; push!(elems, SurfaceElement([a+off1,b+off1,c+off1,d+off1], 1, :quad4)); end
            for (a,b,c,d) in q2; push!(elems, SurfaceElement([d+off2,c+off2,b+off2,a+off2], 2, :quad4)); end
            lo = 0.5 - blocker_half; hi = 0.5 + blocker_half
            bcoords = [lo hi hi lo; lo lo hi hi; 0.5 0.5 0.5 0.5]
            coords = hcat(coords, bcoords)
            boff = size(coords,2) - 4
            push!(elems, SurfaceElement([boff+1,boff+2,boff+3,boff+4], 3, :quad4))
            group_elems = Dict(1=>collect(1:length(q1)), 2=>collect(length(q1)+1:length(q1)+length(q2)),
                               3=>[length(elems)])
            soups = RadiativeViewFactor.MeshIO._build_group_obs_soups(coords, elems, group_elems, 2)
            MeshData(coords, elems, Dict(1=>"bottom",2=>"top",3=>"blocker"), group_elems, soups, 2)
        end
        m = three_body_mesh(; blocker_half=0.25, n=6)
        r_ref = compute_view_factors(m; nquad=6, obstruction_groups=[3], use_duffy=true, verbose=false)
        r_rt  = compute_view_factors(m; raytrace=true, n_rays=20_000, rng=Xoshiro(4), verbose=false)
        @test isapprox(r_rt.F_group[1,2], r_ref.F_group[1,2]; rtol=0.05)

        # The blocker itself is never listed in `obstruction_groups` for the
        # ray-traced call above, and the result still shows obstruction --
        # confirming radiating elements obstruct each other automatically.
        @test r_rt.F_group[1,2] < 0.19982   # strictly less than the unblocked value
    end

    @testset "raytrace argument validation" begin
        coords = [0.0 1 1 0  1 0 0 1;
                  0.0 0 1 1  0 0 1 1;
                  0.0 0 0 0  1 1 1 1]
        elems  = [SurfaceElement([1,2,3,4], 1, :quad4), SurfaceElement([5,6,7,8], 2, :quad4)]
        mesh   = MeshData(coords, elems, Dict(1=>"bottom",2=>"top"),
                          Dict(1=>[1],2=>[2]), Dict{Int,Array{Float64,3}}(), 2)
        @test_throws ErrorException compute_view_factors(mesh; raytrace=true, monte_carlo=true, verbose=false)
        @test_throws ErrorException compute_view_factors(mesh; raytrace=true, self_vf=true, verbose=false)
    end

    @testset "type stability" begin
        RTK = RadiativeViewFactor.RayTraceKernel
        coords = [0.0 1.0 1.0 0.0 0.5 1.0 0.5 0.0;
                  0.0 0.0 1.0 1.0 0.0 0.5 1.0 0.5;
                  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
        elems  = [SurfaceElement([1,2,3,4], 1, :quad4), SurfaceElement([5,6,7,8], 2, :quad4)]
        mesh   = MeshData(coords, elems, Dict(1=>"a",2=>"b"), Dict(1=>[1],2=>[2]),
                          Dict{Int,Array{Float64,3}}(), 2)
        scene = @inferred RTK.build_scene_bvh(coords, elems, mesh.group_tri_soup, Int[])
        @test scene isa RTK.SceneBVH
        rng = Xoshiro(1)
        @test (@inferred RTK.raytrace_element(coords, elems[1], 1, scene, 8, 2, rng)) isa
              Tuple{Vector{Int},Int,Int}
        n = SVector(0.0, 0.0, 1.0)
        @test (@inferred RTK.cosine_dir(n, rng)) isa SVector{3,Float64}
        d = RTK.cosine_dir(n, rng)
        @test (@inferred nearest_hit_bvh(scene.bvh, SVector(0.5,0.5,0.0), d)) isa Tuple{Int,Float64}
    end
end
