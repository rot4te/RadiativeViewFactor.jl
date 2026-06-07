# src/MeshIO.jl
module MeshIO

using LinearAlgebra
using Statistics: mean
using StaticArrays
import Gmsh: gmsh



# ---------------------------------------------------------------------------
# Supported element types
# ---------------------------------------------------------------------------
# Dimension 1 (curves, planar 2D view factors per unit depth):
#   Type  8 — Line3  (3-node 2nd-order line)
# Dimension 2 (surfaces, 3D view factors):
#   Type  9 — Tri6   (6-node 2nd-order triangle)
#   Type 16 — Quad8  (8-node serendipity quadrilateral)   ← preferred
#   Type 10 — Quad9  (9-node Lagrange quad) → centre node dropped → Quad8

const ELEM_INFO = Dict{Int, NamedTuple}(
     8 => (n_nodes=3, n_corners=2, family=:line3),
     9 => (n_nodes=6, n_corners=3, family=:tri),
    16 => (n_nodes=8, n_corners=4, family=:quad),
    10 => (n_nodes=9, n_corners=4, family=:quad),
)

struct SurfaceElement
    nodes  :: Vector{Int}
    group  :: Int
    family :: Symbol        # :line3 | :tri | :quad
end; export SurfaceElement

"""
    MeshData

Fields
------
- `coords`         : (3 × N_nodes) coordinate matrix
- `surface_elems`  : vector of SurfaceElement (all radiating curves/surfaces)
- `group_tags`     : Dict tag → name
- `group_elems`    : Dict tag → element indices (into surface_elems)
- `group_tri_soup` : Dict tag → obstruction geometry for that group.
                     For surface meshes (mesh_dim=2): (3, 3, N_tris) triangle soup.
                     For curve meshes  (mesh_dim=1): (3, 2, N_segs) segment soup,
                       axis 1 = xyz, axis 2 = endpoint (1 or 2), axis 3 = segment.
- `mesh_dim`       : 1 for curve meshes, 2 for surface meshes
"""
struct MeshData
    coords         :: Matrix{Float64}
    surface_elems  :: Vector{SurfaceElement}
    group_tags     :: Dict{Int, String}
    group_elems    :: Dict{Int, Vector{Int}}
    group_tri_soup :: Dict{Int, Array{Float64,3}}
    mesh_dim       :: Int
end; export MeshData

"""
    load_mesh(filename; surface_dim=2, verbose=true) -> MeshData

Load a Gmsh `.msh` file and extract all 2nd-order elements belonging to
named physical groups.

`surface_dim=2` (default) — surface mesh, 3D view factors.
`surface_dim=1`            — planar curve mesh, 2D view factors per unit depth.
                             Physical Curve groups are expected; elements must be
                             Line3 (Gmsh type 8, requires `Mesh.ElementOrder = 2`).
"""
function load_mesh(filename::AbstractString;
                   surface_dim    ::Int  = 2,
                   reverse_normals::Bool = false,
                   verbose        ::Bool = true)::MeshData
    isfile(filename) || error("Mesh file not found: $filename")
    surface_dim ∈ (1, 2) || error("surface_dim must be 1 or 2, got $surface_dim")
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(filename)
    try
        coords, tag2idx = _read_nodes()
        group_tags      = _read_physical_groups(surface_dim)
        surface_elems, group_elems =
            _read_surface_elements(surface_dim, group_tags, tag2idx, verbose)
        if surface_dim == 1
            _orient_line3_normals!(surface_elems, group_elems, coords, verbose)
        end
        if reverse_normals
            _reverse_all_normals!(surface_elems, surface_dim)
            verbose && println("  All normals reversed.")
        end
        group_tri_soup  = _build_group_obs_soups(coords, surface_elems,
                                                  group_elems, surface_dim)
        if verbose
            counts = Dict{Symbol,Int}(:quad=>0, :tri=>0, :line3=>0)
            for e in surface_elems; counts[e.family] += 1; end
            if surface_dim == 2
                println("Loaded $(length(surface_elems)) surface elements ",
                        "(quad: $(counts[:quad]), tri: $(counts[:tri])) ",
                        "in $(length(group_tags)) physical group(s).")
            else
                println("Loaded $(length(surface_elems)) curve elements ",
                        "(line3: $(counts[:line3])) ",
                        "in $(length(group_tags)) physical group(s).")
            end
        end
        return MeshData(coords, surface_elems, group_tags, group_elems,
                        group_tri_soup, surface_dim)
    finally
        gmsh.finalize()
    end
end; export load_mesh

function _read_nodes()
    node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
    N      = length(node_tags)
    coords = Matrix{Float64}(reshape(coords_flat, 3, N))
    tag2idx = Dict{Int,Int}(Int(t) => i for (i,t) in enumerate(node_tags))
    return coords, tag2idx
end

function _read_physical_groups(dim::Int)::Dict{Int,String}
    groups = Dict{Int,String}()
    for (d, tag) in gmsh.model.getPhysicalGroups()
        d == dim || continue
        groups[Int(tag)] = gmsh.model.getPhysicalName(d, tag)
    end
    isempty(groups) && @warn "No physical groups of dimension $dim found in mesh."
    return groups
end

function _read_surface_elements(dim, group_tags, tag2idx, verbose)
    surface_elems = SurfaceElement[]
    group_elems   = Dict{Int,Vector{Int}}(tag => Int[] for tag in keys(group_tags))
    type_counts   = Dict{Int,Int}()

    for (gtag, _) in group_tags
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)
        for ent in entities
            elem_types, elem_tags, node_tags_per_elem =
                gmsh.model.mesh.getElements(dim, ent)
            for (etype, etags, ntags) in zip(elem_types, elem_tags, node_tags_per_elem)
                itype = Int(etype)
                haskey(ELEM_INFO, itype) || continue
                info      = ELEM_INFO[itype]
                n_nodes   = info.n_nodes
                n_elems   = length(etags)
                ntags_mat = reshape(ntags, n_nodes, n_elems)
                type_counts[itype] = get(type_counts, itype, 0) + n_elems
                for k in 1:n_elems
                    raw = ntags_mat[:, k]
                    if info.family === :quad
                        node_idx = [tag2idx[Int(raw[a])] for a in 1:8]
                        push!(surface_elems, SurfaceElement(node_idx, gtag, :quad))
                    elseif info.family === :tri
                        node_idx = [tag2idx[Int(raw[a])] for a in 1:6]
                        push!(surface_elems, SurfaceElement(node_idx, gtag, :tri))
                    else  # :line3
                        node_idx = [tag2idx[Int(raw[a])] for a in 1:3]
                        push!(surface_elems, SurfaceElement(node_idx, gtag, :line3))
                    end
                    push!(group_elems[gtag], length(surface_elems))
                end
            end
        end
    end

    if isempty(surface_elems)
        all_types = Set{Int}()
        for (d, ent) in gmsh.model.getEntities(dim)
            etypes, _, _ = gmsh.model.mesh.getElements(d, ent)
            union!(all_types, Int.(etypes))
        end
        supported = dim == 1 ? "Line3 (8)" : "Tri6 (9), Quad8 (16), Quad9 (10)"
        error("""
No supported 2nd-order elements found in physical groups (dimension $dim).
Supported types: $supported
Element types present at dimension $dim: $(sort(collect(all_types)))
Common causes:
  • Mesh is 1st order — re-run with `Mesh.ElementOrder = 2` or `-order 2`
  • Physical groups defined on wrong dimension (expected $dim)
  • Entities not included in a Physical Group
""")
    end

    if verbose
        type_names = Dict(8=>"Line3", 9=>"Tri6", 16=>"Quad8", 10=>"Quad9")
        for (t, n) in sort(collect(type_counts))
            t == 10 && @info "Quad9 (type 10) found — centre node dropped, treated as Quad8."
            println("  Element type $(get(type_names,t,string(t))): $n elements")
        end
    end

    return surface_elems, group_elems
end

# ---------------------------------------------------------------------------
# Line3 normal orientation
# ---------------------------------------------------------------------------



"""
    _build_curve_to_surface_centroid(all_curve_entities, gmshtag_to_pos, coords)
        -> Dict{Int, SVector{3,Float64}}

Build a map from curve entity tag to the centroid of the adjacent surface,
determined purely from mesh connectivity (shared nodes) rather than from
CAD topology. This works correctly when loading a `.msh` file, where
`gmsh.model.getAdjacencies` returns nothing because CAD topology is not
stored in mesh files.

Algorithm
---------
1. Collect the set of Gmsh node tags on each curve entity.
2. For every 2-D surface entity in the mesh, collect its node tags.
3. A surface is adjacent to a curve if they share at least one node.
4. For each curve, find all adjacent surfaces and compute the centroid of
   the first one's node coordinates.
"""
function _build_curve_to_surface_centroid(
        all_curve_entities::Set{Int},
        gmshtag_to_pos    ::Dict{Int,Int},
        coords            ::Matrix{Float64})::Dict{Int, SVector{3,Float64}}

    # Collect node tag sets for each curve entity
    curve_nodes = Dict{Int, Set{Int}}()
    for ent in all_curve_entities
        node_tags, _, _ = gmsh.model.mesh.getNodes(1, ent)
        curve_nodes[ent] = Set{Int}(Int(t) for t in node_tags)
    end

    # Build surface node sets and centroids from element connectivity.
    #
    # gmsh.model.mesh.getNodes(2, ent) only returns nodes *classified* on
    # that surface entity — i.e. interior nodes — and explicitly excludes
    # nodes shared with boundary curves. This means curve nodes never appear
    # in a surface's node set, making the shared-node adjacency test fail.
    #
    # The correct approach is to collect nodes via element connectivity:
    # getElements(2, ent) returns the node tags of all 2-D elements on that
    # entity, including their boundary/corner nodes which are shared with
    # adjacent curves. A curve is then adjacent to a surface if any of the
    # curve's nodes appear in the surface's element node list.
    surf_node_sets = Dict{Int, Set{Int}}()
    surf_centroids = Dict{Int, SVector{3,Float64}}()

    # Collect surface entities from physical groups (works in both v2.2 and v4)
    surf_entities = Set{Int}()
    for (dim, ptag) in gmsh.model.getPhysicalGroups(2)
        for ent_raw in gmsh.model.getEntitiesForPhysicalGroup(2, ptag)
            push!(surf_entities, Int(ent_raw))
        end
    end
    # Fallback: try getEntities if no physical surface groups found
    if isempty(surf_entities)
        for (dim, ent_raw) in gmsh.model.getEntities(2)
            push!(surf_entities, Int(ent_raw))
        end
    end

    for ent in surf_entities
        # Use getElements to get node tags from element connectivity —
        # this includes corner/boundary nodes shared with adjacent curves
        etypes, _, enode_lists = gmsh.model.mesh.getElements(2, ent)
        isempty(etypes) && continue

        node_set = Set{Int}()
        for nlist in enode_lists
            for t in nlist
                push!(node_set, Int(t))
            end
        end
        isempty(node_set) && continue
        surf_node_sets[ent] = node_set

        # Compute centroid from coordinates of those nodes
        positions = [gmshtag_to_pos[t] for t in node_set
                     if haskey(gmshtag_to_pos, t)]
        isempty(positions) && continue
        surf_centroids[ent] = SVector{3,Float64}(
            mean(coords[1, p] for p in positions),
            mean(coords[2, p] for p in positions),
            mean(coords[3, p] for p in positions),
        )
    end

    # For each curve entity, find all adjacent surfaces (those sharing nodes).
    # If multiple surfaces are adjacent (e.g. an internal boundary between a
    # transfinite layer and a background mesh), pick the one whose centroid is
    # closest to the curve's own centroid — this is the surface the curve
    # actually borders rather than a distant surface that merely shares a corner.
    result = Dict{Int, SVector{3,Float64}}()
    for ent in all_curve_entities
        cnodes = curve_nodes[ent]

        # Compute curve centroid from its nodes
        curve_positions = [gmshtag_to_pos[t] for t in cnodes
                           if haskey(gmshtag_to_pos, t)]
        isempty(curve_positions) && continue
        curve_centroid = SVector{3,Float64}(
            mean(coords[1, p] for p in curve_positions),
            mean(coords[2, p] for p in curve_positions),
            mean(coords[3, p] for p in curve_positions),
        )

        # Collect all adjacent surfaces and their centroids
        adjacent = Tuple{Int, SVector{3,Float64}}[]
        for (surf, snodes) in surf_node_sets
            if !isempty(intersect(cnodes, snodes))
                push!(adjacent, (surf, surf_centroids[surf]))
            end
        end

        if isempty(adjacent)
            @warn "Curve entity $ent shares no nodes with any surface entity. " *
                  "Normal orientation for elements on this curve will not be corrected."
            continue
        end

        # Pick the adjacent surface whose centroid is closest to the curve.
        # For a curve on the boundary of a transfinite layer, the transfinite
        # surface centroid is much closer than any distant background surface.
        best_surf_centroid = adjacent[1][2]
        best_dist = sum((curve_centroid - adjacent[1][2]).^2)
        for (_, sc) in adjacent[2:end]
            d = sum((curve_centroid - sc).^2)
            if d < best_dist
                best_dist = d
                best_surf_centroid = sc
            end
        end
        result[ent] = best_surf_centroid
    end
    return result
end

"""
    _orient_line3_normals!(surface_elems, group_elems, coords, verbose)

For each Line3 element, determine the correct normal direction element-wise:

1. Find the curve entity that the element belongs to via Gmsh adjacency queries.
2. Compute the centroid of the adjacent transfinite surface — the interior that
   the normal should point toward.
3. Evaluate the element's actual normal at its midpoint (ξ=0).
4. If the actual normal points away from the surface interior, flip the element
   by swapping its two endpoint nodes.

This works correctly for circular or arbitrarily curved physical groups where
a single group-level reference normal would cancel out.

Must be called while the Gmsh API session is still open.
"""
function _orient_line3_normals!(surface_elems::Vector{SurfaceElement},
                                 group_elems  ::Dict{Int,Vector{Int}},
                                 coords       ::Matrix{Float64},
                                 verbose      ::Bool)
    # Build a map from node index → curve entity tag so we can look up
    # which Gmsh curve entity each element belongs to.
    # We query all curve entities that appear in any physical group.
    all_curve_entities = Set{Int}()
    for (gtag, idxs) in group_elems
        isempty(idxs) && continue
        surface_elems[idxs[1]].family === :line3 || continue
        for ent_raw in gmsh.model.getEntitiesForPhysicalGroup(1, gtag)
            push!(all_curve_entities, Int(ent_raw))
        end
    end

    # Map: node global index → curve entity tag
    # (a node can belong to multiple entities at corners; first match wins)
    node_to_curve = Dict{Int,Int}()
    for ent in all_curve_entities
        node_tags, _, _ = gmsh.model.mesh.getNodes(1, ent)
        for t in node_tags
            get!(node_to_curve, Int(t), ent)
        end
    end

    # We also need a map from global node index to tag2idx-style index.
    # We already have coords indexed by position; use the node at el.nodes[1]
    # to look up the curve entity. Since el.nodes stores 1-based position
    # indices (not Gmsh tags), we need the reverse: position → Gmsh tag.
    # Build it from the same getNodes call.
    pos_to_gmshtag = Dict{Int,Int}()
    for ent in all_curve_entities
        node_tags, coords_flat, _ = gmsh.model.mesh.getNodes(1, ent)
        for t in node_tags
            # Find which position index this Gmsh tag corresponds to by
            # matching coordinates — but that's expensive. Instead, rebuild
            # from the global node list which we already parsed in _read_nodes.
            # We'll pass tag2idx as a separate query.
        end
    end
    # Simpler: re-query the global node list to get tag→position map
    all_node_tags, _, _ = gmsh.model.mesh.getNodes()
    gmshtag_to_pos = Dict{Int,Int}(Int(t) => i for (i,t) in enumerate(all_node_tags))

    # Map: position index → curve entity
    pos_to_curve = Dict{Int,Int}()
    for ent in all_curve_entities
        node_tags, _, _ = gmsh.model.mesh.getNodes(1, ent)
        for t in node_tags
            pos = get(gmshtag_to_pos, Int(t), 0)
            pos > 0 && get!(pos_to_curve, pos, ent)
        end
    end

    # Build curve→surface centroid map from mesh connectivity (not CAD topology)
    # so this works correctly when loading a .msh file without CAD data.
    surf_centroid_cache = _build_curve_to_surface_centroid(
        all_curve_entities, gmshtag_to_pos, coords)

    n_flipped = 0
    for (gtag, idxs) in group_elems
        isempty(idxs) && continue
        surface_elems[idxs[1]].family === :line3 || continue

        for idx in idxs
            el = surface_elems[idx]
            el.family === :line3 || continue

            # Find curve entity for this element via its first node
            curve_ent = get(pos_to_curve, el.nodes[1], 0)
            if curve_ent == 0
                # Fallback: try node 2
                curve_ent = get(pos_to_curve, el.nodes[2], 0)
            end
            if curve_ent == 0
                @warn "Cannot find curve entity for element $idx in group $gtag; skipping."
                continue
            end

            # Get surface centroid for this curve entity
            surf_c = get(surf_centroid_cache, curve_ent, nothing)
            if surf_c === nothing
                # No adjacent surface found — skip orientation correction
                continue
            end

            # Evaluate element midpoint in physical space
            # At ξ=0: N = (0, 0, 1) so midpoint = coords of node 3 (the midpoint node)
            el_mid = SVector{3,Float64}(
                coords[1, el.nodes[3]],
                coords[2, el.nodes[3]],
                coords[3, el.nodes[3]],
            )

            # Evaluate actual normal at ξ=0:
            # dN/dξ at ξ=0: dN₁=-0.5, dN₂=0.5, dN₃=0
            # dx/dξ = 0.5*(x₂ - x₁) — tangent pointing from node 1 to node 2
            dx = coords[1, el.nodes[2]] - coords[1, el.nodes[1]]
            dy = coords[2, el.nodes[2]] - coords[2, el.nodes[1]]
            tlen = sqrt(dx^2 + dy^2)
            tlen < eps() && continue
            # CCW normal
            actual_n = SVector{3,Float64}(-dy/tlen, dx/tlen, 0.0)

            # Does the actual normal point toward the surface interior?
            toward_surface = surf_c - el_mid
            if dot(actual_n, toward_surface) < 0.0
                nodes = copy(el.nodes)
                nodes[1], nodes[2] = nodes[2], nodes[1]
                surface_elems[idx] = SurfaceElement(nodes, el.group, el.family)
                n_flipped += 1
            end
        end
    end

    verbose && n_flipped > 0 &&
        println("  Reoriented $n_flipped Line3 element(s) to point toward transfinite surface interior.")
end

# ---------------------------------------------------------------------------
# Normal reversal
# ---------------------------------------------------------------------------

"""
    _reverse_all_normals!(surface_elems, surface_dim)

Reverse the normal of every element by swapping the node ordering:
  - Line3  (:line3) : swap nodes 1 ↔ 2 (endpoints); node 3 (midpoint) unchanged
  - Quad8  (:quad)  : swap nodes 1 ↔ 3 and 5 ↔ 7 (reverses winding)
  - Tri6   (:tri)   : swap nodes 1 ↔ 3 and 4 ↔ 6 (reverses winding)
"""
function _reverse_all_normals!(surface_elems::Vector{SurfaceElement},
                                surface_dim  ::Int)
    for (i, el) in enumerate(surface_elems)
        nodes = copy(el.nodes)
        if el.family === :line3
            nodes[1], nodes[2] = nodes[2], nodes[1]
        elseif el.family === :quad
            nodes[1], nodes[3] = nodes[3], nodes[1]
            nodes[5], nodes[7] = nodes[7], nodes[5]
        elseif el.family === :tri
            nodes[1], nodes[3] = nodes[3], nodes[1]
            nodes[4], nodes[6] = nodes[6], nodes[4]
        end
        surface_elems[i] = SurfaceElement(nodes, el.group, el.family)
    end
end

# ---------------------------------------------------------------------------
# Obstruction geometry soups
# ---------------------------------------------------------------------------

"""
Build per-group obstruction geometry.

Surface meshes (dim=2): triangle soup (3, 3, N_tris).
  dim 1 = vertex index (1,2,3), dim 2 = xyz (1,2,3), dim 3 = triangle.

Curve meshes (dim=1): segment soup (3, 2, N_segs).
  dim 1 = xyz (1,2,3), dim 2 = endpoint (1 or 2), dim 3 = segment.
  Only corner nodes are used; the midpoint is irrelevant for obstruction.
"""
function _build_group_obs_soups(coords     ::Matrix{Float64},
                                 elems      ::Vector{SurfaceElement},
                                 group_elems::Dict{Int,Vector{Int}},
                                 dim        ::Int)
    soups = Dict{Int, Array{Float64,3}}()

    for (gtag, idxs) in group_elems
        if dim == 2
            n_tris = sum(elems[i].family === :quad ? 2 : 1 for i in idxs)
            soup   = Array{Float64,3}(undef, 3, 3, n_tris)
            t = 0
            for i in idxs
                el = elems[i]; c = el.nodes
                v1 = @view coords[:, c[1]]
                v2 = @view coords[:, c[2]]
                v3 = @view coords[:, c[3]]
                t += 1
                soup[:, 1, t] .= v1; soup[:, 2, t] .= v2; soup[:, 3, t] .= v3
                if el.family === :quad
                    v4 = @view coords[:, c[4]]
                    t += 1
                    soup[:, 1, t] .= v1; soup[:, 2, t] .= v3; soup[:, 3, t] .= v4
                end
            end
            soups[gtag] = soup
        else  # dim == 1: segment soup
            n_segs = length(idxs)
            soup   = Array{Float64,3}(undef, 3, 2, n_segs)
            for (s, i) in enumerate(idxs)
                el = elems[i]; c = el.nodes
                soup[:, 1, s] .= @view coords[:, c[1]]   # first corner
                soup[:, 2, s] .= @view coords[:, c[2]]   # second corner
            end
            soups[gtag] = soup
        end
    end
    return soups
end

end # module MeshIO