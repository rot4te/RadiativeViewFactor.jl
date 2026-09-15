# test/obstruction_test.jl
#
# Regression test for a severe, previously-undiscovered bug in the 3-D
# ray/triangle obstruction path: BVH.jl's `intersect_ray_bvh` (CPU) and
# GPUBVH.jl's `gpu_intersect_bvh` (GPU) both read each triangle's vertex
# coordinates from `tri_soup`/`tri_verts` with the (coord, vertex) axes
# transposed relative to how `_build_group_obs_soups` (MeshIO.jl) actually
# lays the array out -- e.g. CPU's `v0` was built as
# `(tri_soup[1,1,t], tri_soup[1,2,t], tri_soup[1,3,t])`, which is
# `(x of vertex1, x of vertex2, x of vertex3)`, not the xyz of one vertex.
# This silently broke essentially all 3-D obstruction queries: two points
# on a sphere would report visible=true for nearly all separations, only
# occasionally (and increasingly rarely as the obstructor mesh got finer)
# reporting the correct blocked=true near the exact antipodal direction.
#
# Found via the concentric-spheres verification case (paper Section 4.2 of
# the Nek-VF paper): the outer sphere's row sum came out ~1.25 instead of
# the analytically exact 1.0 (F(outer->outer,self) computed as ~1.0 instead
# of the true 1-(r1/r2)^2 = 0.75), and got *worse*, not better, when the
# obstructing inner sphere's mesh was refined -- the opposite of what a mere
# discretization/quadrature error would do, which is what pointed at a
# transposition bug in the BVH traversal rather than a numerical-accuracy
# issue. Segment-based (mesh_dim=1) obstruction was unaffected: it uses a
# separate, independently-correct function (`intersect_seg_bvh`).

@testset "3-D ray/triangle obstruction (BVH)" begin

    @testset "direct is_visible: chord-vs-ball blocking threshold" begin
        # A triangle soup approximating a solid ball of radius r1=0.5 at the
        # origin. Winding/normal orientation is irrelevant here -- ray/
        # triangle intersection doesn't cull by winding -- only whether the
        # soup's geometry sits where it should.
        r1 = 0.5
        n  = 12  # divisions per cubed-sphere face edge
        function cubed_sphere_tris(r, n)
            verts = Dict{NTuple{3,Int},SVector{3,Float64}}()
            key(p) = (round(Int, p[1]*1e8), round(Int, p[2]*1e8), round(Int, p[3]*1e8))
            get_vertex(face, i, j) = begin
                u = -1.0 + 2.0*i/n;  v = -1.0 + 2.0*j/n
                p = face == 1 ? SVector(u, v, -1.0) :
                    face == 2 ? SVector(u, v,  1.0) :
                    face == 3 ? SVector(u, -1.0, v) :
                    face == 4 ? SVector(u,  1.0, v) :
                    face == 5 ? SVector(-1.0, u, v) :
                                SVector( 1.0, u, v)
                pn = p ./ norm(p) .* r
                get!(verts, key(pn), pn)
            end
            tris = SVector{3,Float64}[]
            for face in 1:6, i in 0:n-1, j in 0:n-1
                a = get_vertex(face,i,j);   b = get_vertex(face,i+1,j)
                c = get_vertex(face,i+1,j+1); d = get_vertex(face,i,j+1)
                append!(tris, (a,b,c, a,c,d))
            end
            soup = Array{Float64,3}(undef, 3, 3, length(tris) ÷ 3)
            for (t, k) in enumerate(1:3:length(tris))
                soup[:,1,t] .= tris[k];  soup[:,2,t] .= tris[k+1];  soup[:,3,t] .= tris[k+2]
            end
            soup
        end
        soup = cubed_sphere_tris(r1, n)
        bvh  = build_bvh(soup)

        p1 = SVector(1.0, 0.0, 0.0)
        # Chord between two points at angular separation θ on a unit circle
        # passes at perpendicular distance cos(θ/2) from the center; it
        # enters the r1 ball exactly when cos(θ/2) < r1, i.e. θ > 2*acos(r1).
        θ_boundary = 2*acos(r1)   # = 120° for r1 = 0.5
        for deg in (30, 60, 90, θ_boundary*180/pi - 10)
            pth = SVector(cosd(deg), sind(deg), 0.0)
            @test is_visible(bvh, p1, pth)   # well outside the ball: unobstructed
        end
        for deg in (θ_boundary*180/pi + 10, 150, 170, 179.9)
            pth = SVector(cosd(deg), sind(deg), 0.0)
            @test !is_visible(bvh, p1, pth)  # well inside the ball: blocked
        end
    end

    @testset "compute_view_factors: concentric spheres (analytic)" begin
        # End-to-end check against the exact closed-form result: for
        # concentric spheres of radius r1 < r2,
        #   F(inner->outer) = 1                     (convex, no self-view)
        #   F(outer->inner) = (r1/r2)^2              (reciprocity)
        #   F(outer->outer) = 1 - (r1/r2)^2          (conservation)
        r1, r2, n = 0.5, 1.0, 8
        function cubed_sphere_quad4(r, n)
            verts = Dict{NTuple{3,Int},Int}()
            coords = Vector{Float64}[]
            key(p) = (round(Int, p[1]*1e8), round(Int, p[2]*1e8), round(Int, p[3]*1e8))
            function node(face, i, j)
                u = -1.0 + 2.0*i/n;  v = -1.0 + 2.0*j/n
                p = face == 1 ? (u, v, -1.0) :
                    face == 2 ? (u, v,  1.0) :
                    face == 3 ? (u, -1.0, v) :
                    face == 4 ? (u,  1.0, v) :
                    face == 5 ? (-1.0, u, v) :
                                ( 1.0, u, v)
                nm = sqrt(p[1]^2+p[2]^2+p[3]^2)
                pn = (p[1]/nm*r, p[2]/nm*r, p[3]/nm*r)
                get!(verts, key(pn)) do
                    push!(coords, [pn[1], pn[2], pn[3]]); length(coords)
                end
            end
            quads = NTuple{4,Int}[]
            for face in 1:6, i in 0:n-1, j in 0:n-1
                push!(quads, (node(face,i,j), node(face,i+1,j),
                              node(face,i+1,j+1), node(face,i,j+1)))
            end
            coords, quads
        end
        # Orient each quad's node order so its cross-product normal points
        # *into the enclosure* (RadiativeViewFactor.jl's convention -- see
        # MeshIO.jl's `_re2_orient_inward`): for the inner sphere that's
        # radially outward (away from its own center, into the annular
        # gap); for the outer sphere that's radially inward (toward the
        # origin, into the same gap from the other side).
        function orient!(coords, quads; away_from_origin::Bool)
            out = NTuple{4,Int}[]
            for (a,b,c,d) in quads
                pa, pb, pc = coords[a], coords[b], coords[c]
                nvec = cross(pb.-pa, pc.-pa)
                center = (pa .+ pb .+ pc) ./ 3
                pointing_away = dot(nvec, center) > 0
                (pointing_away != away_from_origin) ?
                    push!(out, (a,d,c,b)) : push!(out, (a,b,c,d))
            end
            out
        end
        c1, q1 = cubed_sphere_quad4(r1, n);  q1 = orient!(c1, q1; away_from_origin=true)
        c2, q2 = cubed_sphere_quad4(r2, n);  q2 = orient!(c2, q2; away_from_origin=false)
        n1 = length(c1)
        coords = zeros(3, n1 + length(c2))
        for (i,p) in enumerate(c1); coords[:,i] .= p; end
        for (i,p) in enumerate(c2); coords[:,n1+i] .= p; end
        elems = SurfaceElement[]
        for (a,b,c,d) in q1; push!(elems, SurfaceElement([a,b,c,d], 1, :quad4, 0, 0, 1)); end
        for (a,b,c,d) in q2; push!(elems, SurfaceElement([n1+a,n1+b,n1+c,n1+d], 2, :quad4, 0, 0, 2)); end
        group_tags  = Dict(1=>"inner", 2=>"outer")
        group_elems = Dict(1=>collect(1:length(q1)), 2=>collect(length(q1)+1:length(q1)+length(q2)))
        soups = RadiativeViewFactor.MeshIO._build_group_obs_soups(coords, elems, group_elems, 2)
        mesh  = MeshData(coords, elems, group_tags, group_elems, soups, 2)

        r = compute_view_factors(mesh; nquad=4, monte_carlo=false, use_duffy=false,
                                  obstruction_groups=[1], verbose=false)
        F = r.F_elem
        inner_idx, outer_idx = group_elems[1], group_elems[2]

        F_io = sum(F[inner_idx, outer_idx]) / length(inner_idx)
        F_oi = sum(F[outer_idx, inner_idx]) / length(outer_idx)
        F_oo = sum(F[outer_idx, outer_idx]) / length(outer_idx)

        @test isapprox(F_io, 1.0; atol=1e-2)
        @test isapprox(F_oi, (r1/r2)^2; atol=1e-2)
        @test isapprox(F_oo, 1 - (r1/r2)^2; atol=2e-2)
    end
end