# ext/RadiativeViewFactorMakieExt.jl
# ---------------------------------------------------------------------------
# Makie visualisation extension for RadiativeViewFactor.jl.
# Loaded automatically when the user loads any Makie backend:
#
#   using GLMakie    # desktop window
#   using CairoMakie # vector/raster file output
#   using WGLMakie   # browser / Jupyter
#
# Provides:
#   plot_mesh_normals(mesh; kwargs...) -> Figure
# ---------------------------------------------------------------------------

module RadiativeViewFactorMakieExt

using Makie
using LinearAlgebra

import RadiativeViewFactor: MeshData, SurfaceElement, plot_mesh_normals,
                             quad8_physical_point, quad8_normal_and_area_element,
                             line3_physical_point, line3_normal_and_length_element

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    plot_mesh_normals(mesh;
                      normal_scale  = nothing,
                      group_colors  = nothing,
                      show_nodes    = false,
                      show_indices  = false,
                      backend_3d    = false) -> Figure

Visualise the mesh stored in `mesh` with surface/curve elements drawn in
their physical group colours and outward normal arrows at each element centre.

Keyword arguments
-----------------
- `normal_scale`  : length of normal arrows in mesh units. Defaults to 15% of
                    the mean element size (estimated from bounding box / √N).
- `group_colors`  : `Dict{Int,Any}` mapping physical group tag → Makie colour.
                    Overrides the automatic palette for specified groups.
- `show_nodes`    : if `true`, scatter-plot all element nodes.
- `show_indices`  : if `true`, annotate each element with its index number.
- `backend_3d`    : if `true`, use `Axis3` regardless of mesh dimension.
                    Defaults to `false`; automatically set to `true` for surface
                    meshes (`mesh.mesh_dim == 2`).

Returns the `Figure` object so it can be further modified or saved:

    fig = plot_mesh_normals(mesh; normal_scale=0.05)
    save("normals.png", fig)
"""
function plot_mesh_normals(mesh::MeshData;
                            normal_scale ::Union{Float64,Nothing} = nothing,
                            group_colors ::Union{Dict,Nothing}    = nothing,
                            show_nodes   ::Bool                   = false,
                            show_indices ::Bool                   = false,
                            backend_3d   ::Bool                   = (mesh.mesh_dim == 2))

    # ---- Automatic normal scale ----
    if normal_scale === nothing
        # Estimate from full 3D bounding box diagonal / √(N elements).
        # Using all three axes ensures a sensible scale regardless of which
        # plane the geometry lies in (e.g. xz-plane plates where y≈0).
        ranges = [maximum(mesh.coords[d,:]) - minimum(mesh.coords[d,:])
                  for d in 1:3]
        diag   = sqrt(sum(r^2 for r in ranges))
        # Guard against degenerate cases (all points in a plane gives one
        # zero range — use the mean of the non-zero ranges instead)
        nonzero = filter(r -> r > eps(), ranges)
        if isempty(nonzero)
            normal_scale = 0.1   # absolute fallback
        else
            char_len     = sqrt(sum(r^2 for r in nonzero))
            normal_scale = char_len / sqrt(length(mesh.surface_elems)) * 0.5
        end
    end

    # ---- Group colour palette ----
    groups    = sort(collect(keys(mesh.group_tags)))
    palette   = Makie.wong_colors()
    group_col = Dict{Int,Any}(g => palette[mod1(i, length(palette))]
                               for (i,g) in enumerate(groups))
    group_colors !== nothing && merge!(group_col, group_colors)

    # ---- Figure and axis ----
    fig = Figure(size=(1000, 750))
    if backend_3d
        ax = Axis3(fig[1,1],
                   xlabel="x", ylabel="y", zlabel="z",
                   title="Mesh elements and normals")
    else
        ax = Axis(fig[1,1], aspect=DataAspect(),
                  xlabel="x", ylabel="y",
                  title="Mesh elements and normals")
    end

    # ---- Draw elements ----
    for (i, el) in enumerate(mesh.surface_elems)
        col = group_col[el.group]

        if el.family === :line3
            _draw_line3!(ax, el, mesh.coords, col,
                          normal_scale, show_nodes, show_indices, i, backend_3d)
        elseif el.family === :quad
            _draw_quad8!(ax, el, mesh.coords, col,
                          normal_scale, show_nodes, show_indices, i, backend_3d)
        elseif el.family === :tri
            _draw_tri6!(ax, el, mesh.coords, col,
                         normal_scale, show_nodes, show_indices, i, backend_3d)
        end
    end

    # ---- Legend ----
    legend_elements = [LineElement(color=group_col[g], linewidth=2.0)
                        for g in groups]
    legend_labels   = [string(mesh.group_tags[g], " (tag ", g, ")")
                        for g in groups]
    Legend(fig[1,2], legend_elements, legend_labels, "Physical groups";
           framevisible=true)

    display(fig)
    return fig
end

# ---------------------------------------------------------------------------
# Per-family drawing helpers
# ---------------------------------------------------------------------------

function _draw_line3!(ax, el::SurfaceElement, coords, col,
                       scale, show_nodes, show_indices, idx, use_3d)
    # Sample the quadratic curve at 25 points for a smooth line
    xpts = Float64[]; ypts = Float64[]; zpts = Float64[]
    for ξ in range(-1.0, 1.0; length=25)
        p = line3_physical_point(coords, el.nodes, ξ)
        push!(xpts, p[1]); push!(ypts, p[2]); push!(zpts, p[3])
    end

    if use_3d
        lines!(ax, xpts, ypts, zpts; color=col, linewidth=1.5)
    else
        lines!(ax, xpts, ypts; color=col, linewidth=1.5)
    end

    # Normal arrow at midpoint (ξ=0 → physically node 3)
    mid = line3_physical_point(coords, el.nodes, 0.0)
    n, _  = line3_normal_and_length_element(coords, el.nodes, 0.0)
    _draw_arrow!(ax, mid, n, scale, col, use_3d)

    if show_nodes
        if use_3d
            scatter!(ax, coords[1,el.nodes], coords[2,el.nodes], coords[3,el.nodes];
                     color=col, markersize=6)
        else
            scatter!(ax, coords[1,el.nodes], coords[2,el.nodes];
                     color=col, markersize=6)
        end
    end

    if show_indices
        _annotate!(ax, mid, string(idx), use_3d)
    end
end

function _draw_quad8!(ax, el::SurfaceElement, coords, col,
                       scale, show_nodes, show_indices, idx, use_3d)
    # Draw the four edges by sampling each edge at several points
    # Edge node pairs in Gmsh Quad8 order: (1,5,2), (2,6,3), (3,7,4), (4,8,1)
    edge_nodes = [(1,5,2), (2,6,3), (3,7,4), (4,8,1)]
    for (a,m,b) in edge_nodes
        # Sample a quadratic edge: map to reference coords and evaluate
        xp = [coords[1,el.nodes[a]], coords[1,el.nodes[m]], coords[1,el.nodes[b]]]
        yp = [coords[2,el.nodes[a]], coords[2,el.nodes[m]], coords[2,el.nodes[b]]]
        zp = [coords[3,el.nodes[a]], coords[3,el.nodes[m]], coords[3,el.nodes[b]]]
        if use_3d
            lines!(ax, xp, yp, zp; color=col, linewidth=1.0)
        else
            lines!(ax, xp, yp; color=col, linewidth=1.0)
        end
    end

    # Normal arrow at element centre (ξ=0, η=0)
    mid = quad8_physical_point(coords, el.nodes, 0.0, 0.0)
    n, _ = quad8_normal_and_area_element(coords, el.nodes, 0.0, 0.0)
    _draw_arrow!(ax, mid, n, scale, col, use_3d)

    if show_nodes
        if use_3d
            scatter!(ax, coords[1,el.nodes], coords[2,el.nodes], coords[3,el.nodes];
                     color=col, markersize=6)
        else
            scatter!(ax, coords[1,el.nodes], coords[2,el.nodes];
                     color=col, markersize=6)
        end
    end

    show_indices && _annotate!(ax, mid, string(idx), use_3d)
end

function _draw_tri6!(ax, el::SurfaceElement, coords, col,
                      scale, show_nodes, show_indices, idx, use_3d)
    # Draw three edges: (1,4,2), (2,5,3), (3,6,1)
    edge_nodes = [(1,4,2), (2,5,3), (3,6,1)]
    for (a,m,b) in edge_nodes
        xp = [coords[1,el.nodes[a]], coords[1,el.nodes[m]], coords[1,el.nodes[b]]]
        yp = [coords[2,el.nodes[a]], coords[2,el.nodes[m]], coords[2,el.nodes[b]]]
        zp = [coords[3,el.nodes[a]], coords[3,el.nodes[m]], coords[3,el.nodes[b]]]
        if use_3d
            lines!(ax, xp, yp, zp; color=col, linewidth=1.0)
        else
            lines!(ax, xp, yp; color=col, linewidth=1.0)
        end
    end

    # Normal at centroid of reference triangle (ξ=η=1/3)
    mid = _tri6_physical_point_centroid(coords, el.nodes)
    n   = _tri6_normal_centroid(coords, el.nodes)
    _draw_arrow!(ax, mid, n, scale, col, use_3d)

    if show_nodes
        if use_3d
            scatter!(ax, coords[1,el.nodes], coords[2,el.nodes], coords[3,el.nodes];
                     color=col, markersize=6)
        else
            scatter!(ax, coords[1,el.nodes], coords[2,el.nodes];
                     color=col, markersize=6)
        end
    end

    show_indices && _annotate!(ax, mid, string(idx), use_3d)
end

# ---------------------------------------------------------------------------
# Geometry helpers (inline, no import needed)
# ---------------------------------------------------------------------------

@inline function _tri6_physical_point_centroid(coords, nodes)
    # Centroid of Tri6 element (average of corner nodes is a good approximation)
    x = (coords[1,nodes[1]] + coords[1,nodes[2]] + coords[1,nodes[3]]) / 3
    y = (coords[2,nodes[1]] + coords[2,nodes[2]] + coords[2,nodes[3]]) / 3
    z = (coords[3,nodes[1]] + coords[3,nodes[2]] + coords[3,nodes[3]]) / 3
    return [x, y, z]
end

@inline function _tri6_normal_centroid(coords, nodes)
    # Normal from cross product of two edges (corner nodes only, sufficient for normal)
    v1 = coords[:, nodes[2]] - coords[:, nodes[1]]
    v2 = coords[:, nodes[3]] - coords[:, nodes[1]]
    n  = [v1[2]*v2[3]-v1[3]*v2[2], v1[3]*v2[1]-v1[1]*v2[3], v1[1]*v2[2]-v1[2]*v2[1]]
    nlen = sqrt(n[1]^2+n[2]^2+n[3]^2)
    return nlen > 0 ? n/nlen : n
end

# ---------------------------------------------------------------------------
# Arrow and annotation helpers
# ---------------------------------------------------------------------------

function _draw_arrow!(ax, origin, normal, scale, col, use_3d)
    ox, oy, oz = origin[1], origin[2], origin[3]
    nx, ny, nz = normal[1]*scale, normal[2]*scale, normal[3]*scale
    # arrowhead is 40% of total arrow length; linewidth scaled to be visible
    head_size = scale * 0.4
    if use_3d
        arrows!(ax, [ox], [oy], [oz], [nx], [ny], [nz];
                color=col, linewidth=2.5,
                arrowsize=Vec3f(head_size, head_size, head_size * 1.2),
                normalize=false)
    else
        arrows!(ax, [ox], [oy], [nx], [ny];
                color=col, linewidth=2.5,
                arrowsize=head_size,
                normalize=false)
    end
end

function _annotate!(ax, pos, label, use_3d)
    if use_3d
        text!(ax, pos[1], pos[2], pos[3]; text=label,
              fontsize=10, align=(:center,:center))
    else
        text!(ax, pos[1], pos[2]; text=label,
              fontsize=10, align=(:center,:center))
    end
end

end # module RadiativeViewFactorMakieExt
