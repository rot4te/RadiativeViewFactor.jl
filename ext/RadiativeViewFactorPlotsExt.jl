# ext/RadiativeViewFactorPlotsExt.jl
# ---------------------------------------------------------------------------
# Plots.jl visualisation extension for RadiativeViewFactor.jl.
# Loaded automatically when the user does `using Plots`.
#
# Provides:
#   plot_mesh_normals(mesh; kwargs...) -> Plots.Plot
#
# Focus: 2D curve meshes (surface_dim=1). For 3D surface meshes a basic
# xy-projection is shown; use Gmsh's built-in normal visualisation for
# full 3D inspection.
# ---------------------------------------------------------------------------

module RadiativeViewFactorPlotsExt

using Plots
using LinearAlgebra

import RadiativeViewFactor: MeshData, SurfaceElement, plot_mesh_normals,
                             quad8_physical_point, quad8_normal_and_area_element,
                             line3_physical_point, line3_normal_and_length_element

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    plot_mesh_normals(mesh;
                      normal_scale = nothing,
                      group_colors = nothing,
                      show_nodes   = false,
                      show_indices = false) -> Plots.Plot

Visualise mesh elements with normal arrows coloured by physical group.

For 2D curve meshes (`mesh.mesh_dim == 1`) elements are drawn as smooth
quadratic curves and normals as arrows using `quiver!`.

For 3D surface meshes (`mesh.mesh_dim == 2`) element edges and normals are
projected onto the xy-plane as a quick sanity check. For full 3D inspection
use Gmsh's built-in normal visualisation (View → Mesh → Normals).

# Arguments
- `normal_scale`  : arrow length in mesh units. Auto-estimated from the
                    bounding box if omitted.
- `group_colors`  : `Dict{Int,Any}` mapping physical group tag → any colour
                    accepted by Plots.jl (`:red`, `"#FF0000"`, etc.).
                    Overrides the automatic palette for specified groups.
- `show_nodes`    : scatter-plot all element nodes.
- `show_indices`  : annotate each element centre with its index number.

# Returns
A `Plots.Plot` object. Display it with `display(fig)` or save with
`savefig(fig, "normals.pdf")`.

# Example
```julia
using Plots
using RadiativeViewFactor

mesh = load_mesh("planar.msh"; surface_dim=1)
fig  = plot_mesh_normals(mesh; normal_scale=0.05)
savefig(fig, "normals.pdf")
```
"""
function plot_mesh_normals(mesh::MeshData;
                            normal_scale ::Union{Float64,Nothing} = nothing,
                            group_colors ::Union{Dict,Nothing}    = nothing,
                            show_nodes   ::Bool                   = false,
                            show_indices ::Bool                   = false)

    # ---- Automatic normal scale ----
    if normal_scale === nothing
        xs = mesh.coords[1,:]; ys = mesh.coords[2,:]; zs = mesh.coords[3,:]
        ranges    = [maximum(xs)-minimum(xs),
                     maximum(ys)-minimum(ys),
                     maximum(zs)-minimum(zs)]
        nonzero   = filter(r -> r > eps(), ranges)
        char_len  = isempty(nonzero) ? 1.0 : sqrt(sum(r^2 for r in nonzero))
        normal_scale = char_len / sqrt(length(mesh.surface_elems)) * 0.5
    end

    # ---- Group colour palette ----
    # Plots default discrete colours
    default_palette = [
        :steelblue, :darkorange, :green3, :firebrick, :mediumpurple,
        :sienna, :deeppink, :teal, :goldenrod, :slategray,
    ]
    groups    = sort(collect(keys(mesh.group_tags)))
    group_col = Dict{Int,Any}(g => default_palette[mod1(i, length(default_palette))]
                               for (i,g) in enumerate(groups))
    group_colors !== nothing && merge!(group_col, group_colors)

    # ---- Initialise plot ----
    plt = plot(; aspect_ratio=:equal,
                 xlabel="x", ylabel="y",
                 title="Mesh elements and normals",
                 legend=:outerright,
                 grid=false,
                 size=(900, 650))

    # Track which group tags have been added to the legend
    legend_added = Set{Int}()

    # ---- Draw elements ----
    for (idx, el) in enumerate(mesh.surface_elems)
        col   = group_col[el.group]
        label = el.group ∈ legend_added ? "" :
                string(mesh.group_tags[el.group], " (tag ", el.group, ")")

        if el.family === :line3
            _draw_line3!(plt, el, mesh.coords, col, label,
                          normal_scale, show_nodes, show_indices, idx)
        elseif el.family === :line2
            _draw_line2!(plt, el, mesh.coords, col, label,
                          normal_scale, show_nodes, show_indices, idx)
        elseif el.family === :quad
            _draw_quad8_2d!(plt, el, mesh.coords, col, label,
                             normal_scale, show_nodes, show_indices, idx)
        elseif el.family === :quad4
            _draw_poly_2d!(plt, el, mesh.coords, col, label,
                            normal_scale, show_nodes, show_indices, idx, 4)
        elseif el.family === :tri
            _draw_tri6_2d!(plt, el, mesh.coords, col, label,
                            normal_scale, show_nodes, show_indices, idx)
        elseif el.family === :tri3
            _draw_poly_2d!(plt, el, mesh.coords, col, label,
                            normal_scale, show_nodes, show_indices, idx, 3)
        end

        push!(legend_added, el.group)
    end

    return plt
end

# ---------------------------------------------------------------------------
# Per-family drawing helpers
# ---------------------------------------------------------------------------

function _draw_line3!(plt, el::SurfaceElement, coords, col, label,
                       scale, show_nodes, show_indices, idx)
    # Smooth curve: 25 sample points along the quadratic element
    xpts = Float64[]; ypts = Float64[]
    for ξ in range(-1.0, 1.0; length=25)
        p = line3_physical_point(coords, el.nodes, ξ)
        push!(xpts, p[1]); push!(ypts, p[2])
    end
    plot!(plt, xpts, ypts; color=col, linewidth=1.5, label=label)

    # Normal arrow at midpoint (ξ=0, physically node 3)
    mid    = line3_physical_point(coords, el.nodes, 0.0)
    n, _   = line3_normal_and_length_element(coords, el.nodes, 0.0)
    _arrow2d!(plt, mid[1], mid[2], n[1]*scale, n[2]*scale, col)

    if show_nodes
        scatter!(plt, coords[1,el.nodes], coords[2,el.nodes];
                 color=col, markersize=3, markerstrokewidth=0, label="")
    end
    if show_indices
        mid_x = coords[1, el.nodes[3]]
        mid_y = coords[2, el.nodes[3]]
        annotate!(plt, mid_x, mid_y, text(string(idx), 7, col, :center))
    end
end

function _draw_quad8_2d!(plt, el::SurfaceElement, coords, col, label,
                           scale, show_nodes, show_indices, idx)
    # Draw four edges via midpoint nodes: (1,5,2),(2,6,3),(3,7,4),(4,8,1)
    edge_triples = [(1,5,2),(2,6,3),(3,7,4),(4,8,1)]
    for (k, (a,m,b)) in enumerate(edge_triples)
        lbl = k == 1 ? label : ""
        plot!(plt,
              [coords[1,el.nodes[a]], coords[1,el.nodes[m]], coords[1,el.nodes[b]]],
              [coords[2,el.nodes[a]], coords[2,el.nodes[m]], coords[2,el.nodes[b]]];
              color=col, linewidth=1.0, label=lbl)
    end
    # Normal at element centre (ξ=0, η=0), projected to xy
    mid    = quad8_physical_point(coords, el.nodes, 0.0, 0.0)
    n, _   = quad8_normal_and_area_element(coords, el.nodes, 0.0, 0.0)
    _arrow2d!(plt, mid[1], mid[2], n[1]*scale, n[2]*scale, col)

    if show_nodes
        scatter!(plt, coords[1,el.nodes], coords[2,el.nodes];
                 color=col, markersize=3, markerstrokewidth=0, label="")
    end
    if show_indices
        cx = mean(coords[1,el.nodes[k]] for k in 1:4)
        cy = mean(coords[2,el.nodes[k]] for k in 1:4)
        annotate!(plt, cx, cy, text(string(idx), 7, col, :center))
    end
end

function _draw_tri6_2d!(plt, el::SurfaceElement, coords, col, label,
                          scale, show_nodes, show_indices, idx)
    edge_triples = [(1,4,2),(2,5,3),(3,6,1)]
    for (k, (a,m,b)) in enumerate(edge_triples)
        lbl = k == 1 ? label : ""
        plot!(plt,
              [coords[1,el.nodes[a]], coords[1,el.nodes[m]], coords[1,el.nodes[b]]],
              [coords[2,el.nodes[a]], coords[2,el.nodes[m]], coords[2,el.nodes[b]]];
              color=col, linewidth=1.0, label=lbl)
    end
    # Normal from corner cross product, projected to xy
    v1 = coords[:,el.nodes[1]]; v2 = coords[:,el.nodes[2]]; v3 = coords[:,el.nodes[3]]
    e1 = v2-v1; e2 = v3-v1
    n  = [e1[2]*e2[3]-e1[3]*e2[2], e1[3]*e2[1]-e1[1]*e2[3], e1[1]*e2[2]-e1[2]*e2[1]]
    nn = norm(n); n = nn > 0 ? n/nn : n
    cx = (v1[1]+v2[1]+v3[1])/3;  cy = (v1[2]+v2[2]+v3[2])/3
    _arrow2d!(plt, cx, cy, n[1]*scale, n[2]*scale, col)

    if show_nodes
        scatter!(plt, coords[1,el.nodes], coords[2,el.nodes];
                 color=col, markersize=3, markerstrokewidth=0, label="")
    end
    if show_indices
        annotate!(plt, cx, cy, text(string(idx), 7, col, :center))
    end
end

# Linear curve element (Line2): straight segment between the two endpoints.
function _draw_line2!(plt, el::SurfaceElement, coords, col, label,
                       scale, show_nodes, show_indices, idx)
    x1, y1 = coords[1,el.nodes[1]], coords[2,el.nodes[1]]
    x2, y2 = coords[1,el.nodes[2]], coords[2,el.nodes[2]]
    plot!(plt, [x1, x2], [y1, y2]; color=col, linewidth=1.5, label=label)

    # Normal at midpoint: 90° CCW rotation of the unit tangent in xy
    mx, my = 0.5*(x1+x2), 0.5*(y1+y2)
    tx, ty = x2-x1, y2-y1
    tl = hypot(tx, ty)
    if tl > 0
        _arrow2d!(plt, mx, my, -ty/tl*scale, tx/tl*scale, col)
    end

    if show_nodes
        scatter!(plt, coords[1,el.nodes], coords[2,el.nodes];
                 color=col, markersize=3, markerstrokewidth=0, label="")
    end
    show_indices && annotate!(plt, mx, my, text(string(idx), 7, col, :center))
end

# Linear surface element (Tri3 / Quad4): straight edges between `nc` corners.
function _draw_poly_2d!(plt, el::SurfaceElement, coords, col, label,
                         scale, show_nodes, show_indices, idx, nc::Int)
    xs = [coords[1, el.nodes[mod1(k, nc)]] for k in 1:nc+1]
    ys = [coords[2, el.nodes[mod1(k, nc)]] for k in 1:nc+1]
    plot!(plt, xs, ys; color=col, linewidth=1.0, label=label)

    # Normal from first-corner cross product, projected to xy
    v1 = coords[:,el.nodes[1]]; v2 = coords[:,el.nodes[2]]; v3 = coords[:,el.nodes[3]]
    e1 = v2-v1; e2 = v3-v1
    n  = [e1[2]*e2[3]-e1[3]*e2[2], e1[3]*e2[1]-e1[1]*e2[3], e1[1]*e2[2]-e1[2]*e2[1]]
    nn = norm(n); n = nn > 0 ? n/nn : n
    cx = mean(coords[1,el.nodes[k]] for k in 1:nc)
    cy = mean(coords[2,el.nodes[k]] for k in 1:nc)
    _arrow2d!(plt, cx, cy, n[1]*scale, n[2]*scale, col)

    if show_nodes
        scatter!(plt, coords[1,el.nodes], coords[2,el.nodes];
                 color=col, markersize=3, markerstrokewidth=0, label="")
    end
    show_indices && annotate!(plt, cx, cy, text(string(idx), 7, col, :center))
end

# ---------------------------------------------------------------------------
# Arrow helper
# ---------------------------------------------------------------------------

"""
Draw a 2D arrow using a line + a small filled triangle arrowhead.
Plots.jl's `quiver!` arrowhead size is poorly controllable across backends;
this gives consistent appearance.
"""
function _arrow2d!(plt, x0, y0, dx, dy, col)
    x1 = x0 + dx;  y1 = y0 + dy
    # Shaft
    plot!(plt, [x0, x1], [y0, y1]; color=col, linewidth=1.5, label="",
          arrow=(:closed, :head, 0.4, 0.3))
end

end # module RadiativeViewFactorPlotsExt
