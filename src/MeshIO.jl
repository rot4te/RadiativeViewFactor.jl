# src/MeshIO.jl
module MeshIO

using LinearAlgebra
using Statistics: mean
using StaticArrays
import Gmsh: gmsh



# ---------------------------------------------------------------------------
# Supported element types
# ---------------------------------------------------------------------------
# Both 1st-order (linear) and 2nd-order (quadratic) elements are supported.
#
# Dimension 1 (curves, planar 2D view factors per unit depth):
#   Type  1 — Line2  (2-node 1st-order line)
#   Type  8 — Line3  (3-node 2nd-order line)
# Dimension 2 (surfaces, 3D view factors):
#   Type  2 — Tri3   (3-node 1st-order triangle)
#   Type  3 — Quad4  (4-node 1st-order quadrilateral)
#   Type  9 — Tri6   (6-node 2nd-order triangle)
#   Type 16 — Quad8  (8-node serendipity quadrilateral)   ← preferred 2nd order
#   Type 10 — Quad9  (9-node Lagrange quad) → centre node dropped → Quad8

const ELEM_INFO = Dict{Int, NamedTuple}(
     1 => (n_nodes=2, n_corners=2, family=:line2),
     8 => (n_nodes=3, n_corners=2, family=:line3),
     2 => (n_nodes=3, n_corners=3, family=:tri3),
     9 => (n_nodes=6, n_corners=3, family=:tri),
     3 => (n_nodes=4, n_corners=4, family=:quad4),
    16 => (n_nodes=8, n_corners=4, family=:quad),
    10 => (n_nodes=9, n_corners=4, family=:quad),
)

"""
    SurfaceElement(nodes, group, family, eg=0, iface=0, phys_tag=0)

`eg` and `iface` are the Nek5000/NekRS global element number and local face
index (1-6) this element corresponds to, populated by [`load_re2`](@ref) so
that per-element view factors can be written back out in Nek's own
`(element, face)` bookkeeping (see [`write_nekrs_view_factors`](@ref)). They
default to `0` for elements loaded from any other mesh format, where the
concept doesn't apply.

`phys_tag` is the original numeric Gmsh `Physical Surface` tag for this
face (Nek's `bc(5,ifc,iel,1)`, the same value `usrdat2` reads), also
populated by `load_re2` (`0` otherwise). It exists because `group`/
`group_tags` are keyed by the Nek boundary-condition *code* ('W', 'P', or a
generic placeholder like "MSH" that `gmsh2nek` writes for every ordinary,
non-periodic surface) — meshes with several differently-named Physical
Surfaces that all end up with the same generic code (e.g. two concentric
spheres, or a pebble bed's pebble/duct-wall surfaces) are otherwise
indistinguishable by `group` alone, even though `usrdat2` can tell them
apart. Use [`split_groups_by_tag`](@ref) to regroup a mesh at
`(code, phys_tag)` granularity when you need to select one such surface as
an `obstruction_groups` entry.
"""
struct SurfaceElement
    nodes    :: Vector{Int}
    group    :: Int
    family   :: Symbol        # :line3 | :tri | :quad
    eg       :: Int           # Nek5000 global element number (.re2 only; 0 otherwise)
    iface    :: Int           # Nek5000 local face index 1-6 (.re2 only; 0 otherwise)
    phys_tag :: Int           # original Gmsh Physical Surface tag (.re2 only; 0 otherwise)
end
SurfaceElement(nodes, group, family) = SurfaceElement(nodes, group, family, 0, 0, 0)
SurfaceElement(nodes, group, family, eg, iface) = SurfaceElement(nodes, group, family, eg, iface, 0)
export SurfaceElement

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

Load a mesh and extract all supported 1st- or 2nd-order elements belonging to
named physical groups.

Any file format that the Gmsh API can open is supported — the format is
inferred from the extension. This includes Gmsh `.msh` (all versions), and via
Gmsh's importers `.stl`, `.step`/`.stp`, `.bdf`/`.nas` (Nastran), `.med`,
`.vtk` (legacy ASCII unstructured grids), and others.

Supported element families (any mix in one mesh):
- Surfaces (`surface_dim=2`): Tri3, Quad4 (1st order); Tri6, Quad8/Quad9 (2nd order).
- Curves   (`surface_dim=1`): Line2 (1st order); Line3 (2nd order).

`surface_dim=2` (default) — surface mesh, 3D view factors.
`surface_dim=1`            — planar curve mesh, 2D view factors per unit depth.

Named physical groups are used to partition the radiating surfaces. Formats
that cannot carry physical groups (e.g. STL) have no named groups; in that case
a single synthetic group named `"default"` covering every entity of
`surface_dim` is created so the rest of the pipeline behaves uniformly.

XML VTK files (`.vtu` and XML-form `.vtk`) are detected automatically and read
through ReadVTK.jl (a weak dependency — `using ReadVTK` must be in scope).
Legacy ASCII/binary `.vtk` files are not handled by ReadVTK and fall through to
the Gmsh importer. See [`load_vtu`](@ref) for VTK-specific options.

Nek5000/NekRS `.re2` binary meshes are detected by extension and read by a
dedicated in-tree parser (Gmsh cannot open them). The 3D hex volume mesh's
boundary faces become radiating Quad4 surfaces grouped by Nek boundary-condition
label. See [`load_re2`](@ref).
"""
function load_mesh(filename::AbstractString;
                   surface_dim    ::Int  = 2,
                   reverse_normals::Bool = false,
                   verbose        ::Bool = true)::MeshData
    isfile(filename) || error("Mesh file not found: $filename")
    surface_dim ∈ (1, 2) || error("surface_dim must be 1 or 2, got $surface_dim")
    # Nek5000/NekRS binary meshes (.re2) are read by a dedicated in-tree parser;
    # Gmsh cannot open them.
    if _is_re2(filename)
        return load_re2(filename; surface_dim, reverse_normals, verbose)
    end
    # XML VTK (.vtu / XML .vtk) is read via the ReadVTK extension; everything
    # else goes through Gmsh (which also reads legacy .vtk partially).
    if _is_xml_vtk(filename)
        return load_vtu(filename; surface_dim, reverse_normals, verbose)
    end
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(filename)
    try
        coords, tag2idx = _read_nodes()
        group_tags      = _read_physical_groups(surface_dim)
        if isempty(group_tags)
            group_tags = _synthesize_default_group(surface_dim, verbose)
        end
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
            counts = Dict{Symbol,Int}(:quad=>0, :quad4=>0, :tri=>0, :tri3=>0,
                                      :line3=>0, :line2=>0)
            for e in surface_elems; counts[e.family] += 1; end
            if surface_dim == 2
                println("Loaded $(length(surface_elems)) surface elements ",
                        "(quad8: $(counts[:quad]), quad4: $(counts[:quad4]), ",
                        "tri6: $(counts[:tri]), tri3: $(counts[:tri3])) ",
                        "in $(length(group_tags)) physical group(s).")
            else
                println("Loaded $(length(surface_elems)) curve elements ",
                        "(line3: $(counts[:line3]), line2: $(counts[:line2])) ",
                        "in $(length(group_tags)) physical group(s).")
            end
        end
        return MeshData(coords, surface_elems, group_tags, group_elems,
                        group_tri_soup, surface_dim)
    finally
        gmsh.finalize()
    end
end; export load_mesh

# ---------------------------------------------------------------------------
# XML VTK (.vtu) support via the ReadVTK extension
# ---------------------------------------------------------------------------

"""
    _is_xml_vtk(filename) -> Bool

Detect XML-form VTK files (`.vtu` and `.vtk` written in XML form) by sniffing
the header. Legacy VTK files begin with `# vtk DataFile Version` and return
`false` (they are routed to Gmsh instead).
"""
function _is_xml_vtk(filename::AbstractString)::Bool
    ext = lowercase(splitext(filename)[2])
    ext in (".vtu", ".pvtu", ".vts", ".vtr", ".vti", ".vtp") && return true
    ext == ".vtk" || return false
    # Ambiguous .vtk: peek at the first non-whitespace bytes.
    head = open(filename, "r") do io
        String(read(io, min(512, filesize(filename))))
    end
    return occursin(r"<\?xml|<VTKFile", head)
end

# Hook implemented by RadiativeViewFactorReadVTKExt (weak dep on ReadVTK).
# Defined here with no methods so the extension can add one and so `load_mesh`
# can detect whether ReadVTK is loaded.
function _load_vtu_impl end

"""
    load_vtu(filename; surface_dim=2, reverse_normals=false, verbose=true,
             group_field=nothing) -> MeshData

Load an XML VTK unstructured grid (`.vtu`) through ReadVTK.jl. Requires
`using ReadVTK` to be in scope (ReadVTK is a weak dependency).

VTK has no native concept of physical groups. If `group_field` names a
per-cell integer data array it is used to partition elements into groups;
otherwise (or if the named array is absent) a single `"default"` group is
created. Common region arrays (`CellEntityIds`, `gmsh:physical`, `RegionId`,
`MaterialIds`) are tried automatically when `group_field === nothing`.

Cell types are mapped to element families as:
Line(3)→line2, QuadraticEdge(21)→line3, Triangle(5)→tri3,
QuadraticTriangle(22)→tri6, Quad(9)→quad4, QuadraticQuad(23)→quad8.
"""
function load_vtu(filename::AbstractString;
                  surface_dim    ::Int  = 2,
                  reverse_normals::Bool = false,
                  verbose        ::Bool = true,
                  group_field           = nothing)::MeshData
    if isempty(methods(_load_vtu_impl))
        error("""
        Reading XML VTK files (.vtu) requires ReadVTK.jl.
        Add it and bring it into scope:
            import Pkg; Pkg.add("ReadVTK")
            using ReadVTK
        Then call load_mesh / load_vtu again.
        """)
    end
    return _load_vtu_impl(filename, surface_dim, reverse_normals, verbose, group_field)
end; export load_vtu

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
        name = gmsh.model.getPhysicalName(d, tag)
        # Unnamed physical groups return ""; fall back to a tag-based label.
        groups[Int(tag)] = isempty(name) ? "group_$(Int(tag))" : name
    end
    return groups
end

"""
    _synthesize_default_group(dim, verbose) -> Dict{Int,String}

For formats that carry no physical groups (e.g. STL), create a single physical
group of dimension `dim` containing every entity of that dimension, so the rest
of the loader can treat the whole mesh as one radiating group named "default".
Must be called while the Gmsh session is open.
"""
function _synthesize_default_group(dim::Int, verbose::Bool)::Dict{Int,String}
    ent_tags = Int[Int(tag) for (d, tag) in gmsh.model.getEntities(dim)]
    isempty(ent_tags) &&
        error("No entities of dimension $dim found in mesh; nothing to load.")
    ptag = Int(gmsh.model.addPhysicalGroup(dim, ent_tags))
    gmsh.model.setPhysicalName(dim, ptag, "default")
    verbose && @info "No physical groups of dimension $dim found; " *
                     "created a synthetic \"default\" group over all $(length(ent_tags)) entities."
    return Dict{Int,String}(ptag => "default")
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
                # Quad9 (type 10) carries a 9th centre node we drop → treat as Quad8.
                n_keep = info.family === :quad && n_nodes == 9 ? 8 : n_nodes
                for k in 1:n_elems
                    raw      = ntags_mat[:, k]
                    node_idx = [tag2idx[Int(raw[a])] for a in 1:n_keep]
                    push!(surface_elems, SurfaceElement(node_idx, gtag, info.family))
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
        supported = dim == 1 ? "Line2 (1), Line3 (8)" :
                               "Tri3 (2), Quad4 (3), Tri6 (9), Quad8 (16), Quad9 (10)"
        error("""
No supported elements found in physical groups (dimension $dim).
Supported types (1st or 2nd order): $supported
Element types present at dimension $dim: $(sort(collect(all_types)))
Common causes:
  • Physical groups defined on wrong dimension (expected $dim)
  • Entities not included in a Physical Group
""")
    end

    if verbose
        type_names = Dict(1=>"Line2", 8=>"Line3", 2=>"Tri3", 9=>"Tri6",
                          3=>"Quad4", 16=>"Quad8", 10=>"Quad9")
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

# True for curve element families (Line2, Line3).
@inline _is_line(family::Symbol) = family === :line3 || family === :line2
# True for quadrilateral element families (Quad4, Quad8).
@inline _is_quad(family::Symbol) = family === :quad || family === :quad4

"""
    _orient_line3_normals!(surface_elems, group_elems, coords, verbose)

For each Line2/Line3 element, determine the correct normal direction element-wise:

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
        _is_line(surface_elems[idxs[1]].family) || continue
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
        _is_line(surface_elems[idxs[1]].family) || continue

        for idx in idxs
            el = surface_elems[idx]
            _is_line(el.family) || continue

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

            # Evaluate element midpoint in physical space.
            # Line3: at ξ=0, N=(0,0,1) → midpoint is node 3 (the midpoint node).
            # Line2: no midpoint node → average the two endpoints.
            el_mid = if el.family === :line3
                SVector{3,Float64}(
                    coords[1, el.nodes[3]],
                    coords[2, el.nodes[3]],
                    coords[3, el.nodes[3]],
                )
            else  # :line2
                SVector{3,Float64}(
                    0.5*(coords[1, el.nodes[1]] + coords[1, el.nodes[2]]),
                    0.5*(coords[2, el.nodes[1]] + coords[2, el.nodes[2]]),
                    0.5*(coords[3, el.nodes[1]] + coords[3, el.nodes[2]]),
                )
            end

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
                surface_elems[idx] = SurfaceElement(nodes, el.group, el.family, el.eg, el.iface, el.phys_tag)
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
  - Line2  (:line2) : swap nodes 1 ↔ 2 (endpoints)
  - Line3  (:line3) : swap nodes 1 ↔ 2 (endpoints); node 3 (midpoint) unchanged
  - Quad4  (:quad4) : swap nodes 1 ↔ 3 (reverses winding)
  - Quad8  (:quad)  : swap nodes 1 ↔ 3, then 5 ↔ 6 and 7 ↔ 8
  - Tri3   (:tri3)  : swap nodes 1 ↔ 3 (reverses winding)
  - Tri6   (:tri)   : swap nodes 1 ↔ 3, then 4 ↔ 5; node 6 unchanged

Second-order elements need their mid-side nodes permuted to follow the new
corner order, or the mid-side nodes end up attached to the wrong edges and the
isoparametric map is silently corrupted. With corners numbered 1-4 and
mid-sides 5=(1,2), 6=(2,3), 7=(3,4), 8=(4,1), swapping corners 1 ↔ 3 sends
edge (1,2) → (3,2), whose mid-side node is the old 6 — hence 5 ↔ 6, and
likewise 7 ↔ 8. The Tri6 case (mid-sides 4=(1,2), 5=(2,3), 6=(3,1)) leaves
node 6 in place because edge (3,1) maps onto itself.
"""
function _reverse_all_normals!(surface_elems::Vector{SurfaceElement},
                                surface_dim  ::Int)
    for (i, el) in enumerate(surface_elems)
        nodes = copy(el.nodes)
        if el.family === :line3 || el.family === :line2
            nodes[1], nodes[2] = nodes[2], nodes[1]
        elseif el.family === :quad
            nodes[1], nodes[3] = nodes[3], nodes[1]
            nodes[5], nodes[6] = nodes[6], nodes[5]
            nodes[7], nodes[8] = nodes[8], nodes[7]
        elseif el.family === :quad4
            nodes[1], nodes[3] = nodes[3], nodes[1]
        elseif el.family === :tri
            nodes[1], nodes[3] = nodes[3], nodes[1]
            nodes[4], nodes[5] = nodes[5], nodes[4]
        elseif el.family === :tri3
            nodes[1], nodes[3] = nodes[3], nodes[1]
        end
        surface_elems[i] = SurfaceElement(nodes, el.group, el.family, el.eg, el.iface, el.phys_tag)
    end
end

# ---------------------------------------------------------------------------
# Obstruction geometry soups
# ---------------------------------------------------------------------------

"""
Build per-group obstruction geometry.

Surface meshes (dim=2): triangle soup (3, 3, N_tris).
  dim 1 = xyz (1,2,3), dim 2 = vertex index (1,2,3), dim 3 = triangle.

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
            n_tris = sum(_is_quad(elems[i].family) ? 2 : 1 for i in idxs)
            soup   = Array{Float64,3}(undef, 3, 3, n_tris)
            t = 0
            for i in idxs
                el = elems[i]; c = el.nodes
                v1 = @view coords[:, c[1]]
                v2 = @view coords[:, c[2]]
                v3 = @view coords[:, c[3]]
                t += 1
                soup[:, 1, t] .= v1; soup[:, 2, t] .= v2; soup[:, 3, t] .= v3
                if _is_quad(el.family)
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

# ---------------------------------------------------------------------------
# Nek5000 / NekRS  .re2  binary meshes
# ---------------------------------------------------------------------------
# A .re2 file stores a 2D (quad) or 3D (hex) spectral-element *volume* mesh:
#
#   [ 80-byte ASCII header:  "#vNNN" nelgt ndim nelgv (fixed-width ints) ]
#   [ 4-byte float endian-test tag = 6.54321                            ]
#   [ geometry: per element  igroup + corner coords                     ]
#   [ curved-side block:      ncurve, then ncurve records               ]
#   [ boundary-condition block: per field  nbc, then nbc records        ]
#
# For radiative view factors we take the *boundary faces* of the 3D hex mesh as
# the radiating Quad4 surfaces, grouped by their Nek boundary-condition label.
# All binary reals are `wdsize`-byte (8 in modern files, 4 in older ones) and
# may be byte-swapped; both are auto-detected. A curve/BC record is laid out as
# 7 reals (element, face, 5 params) followed by an 8-byte character code, so a
# record is `7*wdsize + 8` bytes. The whole layout is cross-checked against the
# file size, which pins down `wdsize` and the number of BC fields.
#
# Nek hex corner order (symmetric preprocessor convention):
#   1:(-,-,-) 2:(+,-,-) 3:(+,+,-) 4:(-,+,-) 5:(-,-,+) 6:(+,-,+) 7:(+,+,+) 8:(-,+,+)
# and local face → corner map (BC `iside` is 1-based into this tuple):
const _RE2_HEX_FACE = ((1,2,6,5), (2,3,7,6), (3,4,8,7),
                       (4,1,5,8), (1,2,3,4), (5,6,7,8))
# Boundary-condition codes that denote genuinely *internal* connections, not
# part of the domain's topological boundary: 'E' (conforming element-element
# face) and blank (unset). 'P' (periodic) is *not* included here — although
# periodic faces aren't solid walls, Nek5000/NekRS's own view-factor module
# (`view_factors.f`'s `vf_export_all_walls`/`vf_calculate_radiation_heat_flux`,
# selecting on `cbc.ne.'E  '.and.cbc.ne.'   '`) keeps every non-'E'/blank face
# — including periodic and inlet/outlet faces — in the radiation enclosure so
# that view factors close (each face's row sums to 1); only solid walls (`cbc.eq.
# 'W  '`) additionally *emit*, decided at Nek5000 runtime, not by this loader.
# Dropping 'P' faces here would silently reproduce the "open/leaky boundary"
# (`open_face_option=2`) case instead of the closed-enclosure default.
const _RE2_INTERNAL_BC = Set(["E", ""])

_is_re2(filename::AbstractString)::Bool =
    lowercase(splitext(filename)[2]) == ".re2"

# Read `n` reals of element type `T` from `bytes` at 0-based `off`; returns the
# values as Float64 and the new offset. `swap` byte-swaps for foreign endianness.
function _re2_reals(bytes::Vector{UInt8}, off::Int, n::Int, ::Type{T},
                    swap::Bool) where {T<:AbstractFloat}
    w   = sizeof(T)
    nb  = n * w
    raw = bytes[off+1 : off+nb]
    if swap                              # reverse each w-byte group in place
        @inbounds for i in 0:n-1
            reverse!(view(raw, i*w+1 : (i+1)*w))
        end
    end
    out = Float64.(reinterpret(T, raw))
    return out, off + nb
end

# Parse the full .re2 payload for a given word size / endianness. Returns
# `(ok, corners, bc)` where `corners` is (3, 8, nelgt) hex corner coordinates
# and `bc` is a vector of (element, face, code) boundary records. `ok` is false
# (without throwing) when the layout does not consume the file exactly, so the
# caller can try a different `wdsize`.
function _re2_parse(bytes::Vector{UInt8}, nelgt::Int, ndim::Int,
                    wdsize::Int, swap::Bool)
    T       = wdsize == 8 ? Float64 : Float32
    nvert   = 2^ndim                       # 8 corners (3D), 4 (2D)
    ncoord  = ndim * nvert
    recsize = 7 * wdsize + 8               # curve / bc record: 7 reals + char*8
    total   = length(bytes)
    off     = 84                           # 80-byte header + 4-byte endian tag

    corners = Array{Float64,3}(undef, 3, nvert, nelgt)
    for e in 1:nelgt
        vals, off = _re2_reals(bytes, off, 1 + ncoord, T, swap)   # igroup + coords
        off > total && return (false, corners, Tuple{Int,Int,String,Int}[])
        @inbounds for v in 1:nvert
            corners[1, v, e] = vals[1 + v]                 # x block
            corners[2, v, e] = vals[1 + nvert + v]         # y block
            corners[3, v, e] = ndim == 3 ? vals[1 + 2nvert + v] : 0.0
        end
    end

    # curved-side block: count, then ncurve records (skipped — corners suffice)
    off + wdsize > total && return (false, corners, Tuple{Int,Int,String,Int}[])
    cval, off = _re2_reals(bytes, off, 1, T, swap)
    ncurve    = round(Int, cval[1])
    (ncurve < 0 || off + ncurve*recsize > total) &&
        return (false, corners, Tuple{Int,Int,String,Int}[])
    off += ncurve * recsize

    # boundary-condition block: one or more fields, each `nbc` then nbc records
    bc = Tuple{Int,Int,String,Int}[]
    while off < total
        off + wdsize > total && return (false, corners, bc)
        nval, off = _re2_reals(bytes, off, 1, T, swap)
        nbc       = round(Int, nval[1])
        (nbc < 0 || nbc > 6*nelgt || off + nbc*recsize > total) &&
            return (false, corners, bc)
        for _ in 1:nbc
            r, off = _re2_reals(bytes, off, 7, T, swap)   # elem, face, 5 params
            code   = rstrip(String(bytes[off+1 : off+3]), [' ', '\0'])  # char*8 slot
            off   += 8
            # r[7] is bc(5,ifc,iel,1) -- gmsh2nek writes the original Gmsh
            # Physical Surface tag there for every non-periodic boundary
            # face (see SurfaceElement.phys_tag's docstring).
            push!(bc, (round(Int, r[1]), round(Int, r[2]), code, round(Int, r[7])))
        end
    end

    return (off == total, corners, bc)
end

"""
    load_re2(filename; surface_dim=2, reverse_normals=false, verbose=true) -> MeshData

Load a Nek5000/NekRS `.re2` binary mesh. The 3D hex volume mesh's boundary
faces become radiating Quad4 surfaces, grouped by their Nek boundary-condition
label. Only genuinely internal faces (`cbc = 'E'` or blank) are skipped;
periodic (`'P'`) faces are kept as their own group, matching Nek5000/NekRS's
own view-factor convention of including every non-internal boundary face in
the radiation enclosure so that view factors close (row sums to 1) — solid
walls vs. other boundaries is a runtime (`cbc.eq.'W  '`) distinction made by
the case's `.usr` file, not something this loader can determine from `.re2`
labels alone. Each element's Nek global element number and local face index
(1-6) are recorded in `SurfaceElement.eg`/`.iface` for round-tripping view
factors back into Nek5000's `(element, face)` bookkeeping — see
[`write_nekrs_view_factors`](@ref). Word size (4- or 8-byte reals) and byte
order are auto-detected. Only `surface_dim=2` (3D → surfaces) is supported.
"""
function load_re2(filename::AbstractString;
                  surface_dim    ::Int  = 2,
                  reverse_normals::Bool = false,
                  verbose        ::Bool = true)::MeshData
    surface_dim == 2 ||
        error(".re2 loading supports 3D hex volume meshes → surfaces only " *
              "(surface_dim=2); got surface_dim=$surface_dim.")

    bytes = read(filename)
    length(bytes) >= 84 || error("File too small to be a valid .re2: $filename")

    header = String(bytes[1:80])
    startswith(header, "#v") ||
        error("Not a .re2 file (missing \"#v\" version header): $filename")
    fields = split(strip(header[6:end]))
    length(fields) >= 2 ||
        error("Malformed .re2 header: \"$(strip(header))\"")
    nelgt = parse(Int, fields[1])
    ndim  = parse(Int, fields[2])
    ndim == 3 ||
        error(".re2 file is $(ndim)D; only 3D hex meshes are supported for " *
              "surface view factors.")

    # Endian test tag: real*4 = 6.54321.
    tagbytes = bytes[81:84]
    tag_native  = reinterpret(Float32, tagbytes)[1]
    tag_swapped = reinterpret(Float32, reverse(tagbytes))[1]
    swap = if abs(tag_native - 6.54321f0) < 1f-3
        false
    elseif abs(tag_swapped - 6.54321f0) < 1f-3
        true
    else
        error("Unrecognised .re2 endian test tag ($tag_native); file may be corrupt.")
    end

    # Auto-detect word size by which layout consumes the file exactly.
    corners = bc = nothing
    for wd in (8, 4)
        ok, c, b = _re2_parse(bytes, nelgt, ndim, wd, swap)
        if ok
            corners, bc = c, b
            verbose && println("  .re2: $(nelgt) hex elements, " *
                               "$(wd)-byte reals, $(swap ? "byte-swapped" : "native") endian")
            break
        end
    end
    corners === nothing &&
        error("Could not parse .re2 layout (word-size/field-count mismatch). " *
              "Please share the file — its byte layout may differ from the " *
              "assumed Nek5000 format.")

    return _re2_build_mesh(corners, bc, reverse_normals, verbose)
end; export load_re2

# Deduplicate hex corners into a global node list; returns the (3, N) coords
# matrix and an (8, nelgt) array of global node indices per element.
function _re2_dedup_nodes(corners::Array{Float64,3})
    nvert, nelgt = size(corners, 2), size(corners, 3)
    extent = maximum(abs, corners; init = 0.0)
    tol    = max(extent * 1e-8, 1e-12)
    keyof(x) = (round(Int, x[1]/tol), round(Int, x[2]/tol), round(Int, x[3]/tol))

    index = Dict{NTuple{3,Int}, Int}()
    coords_cols = Vector{NTuple{3,Float64}}()
    elem_nodes  = Array{Int,2}(undef, nvert, nelgt)
    for e in 1:nelgt, v in 1:nvert
        x = (corners[1,v,e], corners[2,v,e], corners[3,v,e])
        k = keyof(x)
        elem_nodes[v, e] = get!(index, k) do
            push!(coords_cols, x)
            length(coords_cols)
        end
    end

    coords = Matrix{Float64}(undef, 3, length(coords_cols))
    for (j, c) in enumerate(coords_cols)
        coords[1,j] = c[1]; coords[2,j] = c[2]; coords[3,j] = c[3]
    end
    return coords, elem_nodes
end

# Orient the four face node indices so the Quad4 normal points *into* the fluid
# domain — i.e. toward the owner hex's centroid. The hex volume is the radiating
# cavity, so boundary walls must face inward to exchange radiation across it.
# (Use `reverse_normals=true` for the opposite convention.)
function _re2_orient_inward(coords::Matrix{Float64}, face::NTuple{4,Int},
                            elem_centroid::SVector{3,Float64})
    p(i) = SVector{3,Float64}(coords[1,i], coords[2,i], coords[3,i])
    v1, v2, v3, v4 = p(face[1]), p(face[2]), p(face[3]), p(face[4])
    nrm = cross(v2 - v1, v4 - v1)
    fc  = (v1 + v2 + v3 + v4) / 4
    return dot(nrm, fc - elem_centroid) > 0 ?          # points outward → flip
           (face[1], face[4], face[3], face[2]) : face
end

function _re2_build_mesh(corners::Array{Float64,3},
                         bc::Vector{Tuple{Int,Int,String,Int}},
                         reverse_normals::Bool, verbose::Bool)::MeshData
    nelgt = size(corners, 3)
    coords, elem_nodes = _re2_dedup_nodes(corners)

    # element centroids (for outward orientation)
    centroid(e) = SVector{3,Float64}(
        sum(@view corners[1, :, e]) / 8,
        sum(@view corners[2, :, e]) / 8,
        sum(@view corners[3, :, e]) / 8)

    # Radiating boundary faces come from non-internal BC records, grouped by code.
    radiating = filter(r -> !(r[3] in _RE2_INTERNAL_BC), bc)

    surface_elems = SurfaceElement[]
    group_tags    = Dict{Int,String}()
    group_elems   = Dict{Int,Vector{Int}}()
    code_to_tag   = Dict{String,Int}()

    function tag_for(code::String)
        get!(code_to_tag, code) do
            t = length(code_to_tag) + 1
            group_tags[t]  = code
            group_elems[t] = Int[]
            t
        end
    end

    if !isempty(radiating)
        for (eg, iside, code, ptag) in radiating
            (1 <= eg <= nelgt && 1 <= iside <= 6) || continue
            fnodes = ntuple(k -> elem_nodes[_RE2_HEX_FACE[iside][k], eg], 4)
            fnodes = _re2_orient_inward(coords, fnodes, centroid(eg))
            gtag   = tag_for(code)
            push!(surface_elems, SurfaceElement(collect(fnodes), gtag, :quad4, eg, iside, ptag))
            push!(group_elems[gtag], length(surface_elems))
        end
        verbose && println("  .re2: $(length(surface_elems)) boundary faces in " *
                           "$(length(group_tags)) BC group(s): " *
                           join(sort(collect(values(group_tags))), ", "))
    else
        # No usable BC labels — fall back to topological boundary extraction:
        # faces referenced by exactly one element are on the boundary.
        verbose && @info ".re2: no boundary-condition labels found; extracting " *
                         "the topological boundary as a single \"default\" group."
        face_count = Dict{NTuple{4,Int}, Tuple{Int,Int}}()  # sorted key → (eg,iside)
        seen       = Dict{NTuple{4,Int}, Int}()
        for e in 1:nelgt, f in 1:6
            fn  = ntuple(k -> elem_nodes[_RE2_HEX_FACE[f][k], e], 4)
            key = Tuple(sort(collect(fn)))
            seen[key] = get(seen, key, 0) + 1
            haskey(face_count, key) || (face_count[key] = (e, f))
        end
        gtag = tag_for("default")
        for (key, cnt) in seen
            cnt == 1 || continue
            e, f   = face_count[key]
            fnodes = ntuple(k -> elem_nodes[_RE2_HEX_FACE[f][k], e], 4)
            fnodes = _re2_orient_inward(coords, fnodes, centroid(e))
            push!(surface_elems, SurfaceElement(collect(fnodes), gtag, :quad4, e, f))
            push!(group_elems[gtag], length(surface_elems))
        end
        verbose && println("  .re2: $(length(surface_elems)) topological " *
                           "boundary faces.")
    end

    isempty(surface_elems) &&
        error("No radiating boundary faces found in .re2 mesh.")

    if reverse_normals
        _reverse_all_normals!(surface_elems, 2)
        verbose && println("  All normals reversed.")
    end

    group_tri_soup = _build_group_obs_soups(coords, surface_elems, group_elems, 2)
    return MeshData(coords, surface_elems, group_tags, group_elems,
                    group_tri_soup, 2)
end

"""
    split_groups_by_tag(mesh::MeshData) -> MeshData

Return a new `MeshData` with `group`/`group_tags`/`group_elems`/
`group_tri_soup` refined to `(code, phys_tag)` granularity instead of just
`code`. `load_re2`'s default grouping is by Nek boundary-condition *code*
('W', 'P', or a generic placeholder like "MSH"), which `gmsh2nek` writes
identically for every ordinary, non-periodic surface regardless of which
named `Physical Surface` it came from — so two concentric spheres, or a
pebble bed's pebble/duct-wall surfaces, land in the same group and can't be
told apart via `group` alone even though `usrdat2` distinguishes them via
`bc(5,ifc,iel,1)` (exposed here as [`SurfaceElement`](@ref)'s `phys_tag`).

Call this once after `load_re2` when you need to pass one particular named
surface to `compute_view_factors`'s `obstruction_groups` — e.g. the inner
sphere blocking the outer sphere's concave self-view, or a pebble blocking
view between two duct-wall patches. Elements with `phys_tag == 0` (any mesh
not loaded from a `.re2` file) keep their original group unchanged, so this
is a no-op for non-Nek meshes.

```julia
mesh  = load_re2("case.re2")
mesh2 = split_groups_by_tag(mesh)
# mesh2.group_tags now has one entry per (code, phys_tag) pair, e.g.
# 1 => "MSH", 2 => "MSH#2", ... instead of a single merged "MSH" group.
inner = findfirst(t -> occursin("#1", t), collect(values(mesh2.group_tags)))
compute_view_factors(mesh2; obstruction_groups=[inner], ...)
```
"""
function split_groups_by_tag(mesh::MeshData)::MeshData
    elems = mesh.surface_elems
    key(e) = e.phys_tag == 0 ? (e.group, 0) : (e.group, e.phys_tag)

    newgroup_of  = Dict{Tuple{Int,Int}, Int}()
    new_tags     = Dict{Int,String}()
    new_elems_by = Dict{Int,Vector{Int}}()
    new_elems    = Vector{SurfaceElement}(undef, length(elems))

    for (idx, e) in enumerate(elems)
        k = key(e)
        g = get!(newgroup_of, k) do
            length(newgroup_of) + 1
        end
        if !haskey(new_tags, g)
            base = mesh.group_tags[e.group]
            new_tags[g] = e.phys_tag == 0 ? base : "$(base)#$(e.phys_tag)"
            new_elems_by[g] = Int[]
        end
        push!(new_elems_by[g], idx)
        new_elems[idx] = SurfaceElement(e.nodes, g, e.family, e.eg, e.iface, e.phys_tag)
    end

    soups = _build_group_obs_soups(mesh.coords, new_elems, new_elems_by, mesh.mesh_dim)
    return MeshData(mesh.coords, new_elems, new_tags, new_elems_by, soups, mesh.mesh_dim)
end
export split_groups_by_tag

end # module MeshIO