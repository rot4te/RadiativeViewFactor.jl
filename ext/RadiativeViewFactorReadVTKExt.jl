# ext/RadiativeViewFactorReadVTKExt.jl
# ---------------------------------------------------------------------------
# ReadVTK.jl loading extension for RadiativeViewFactor.jl.
# Loaded automatically when the user does `using ReadVTK`.
#
# Reads XML VTK unstructured grids (.vtu, XML-form .vtk) into a MeshData,
# mapping VTK cell types to first/second-order element families. VTK has no
# physical-group concept, so groups are taken from a per-cell integer data
# array if present, else a single "default" group is synthesized.
#
# NOTE: ReadVTK does not read legacy (non-XML) .vtk files — those are routed
# to the Gmsh importer by load_mesh instead.
# ---------------------------------------------------------------------------

module RadiativeViewFactorReadVTKExt

using ReadVTK
import RadiativeViewFactor

const MeshIO = RadiativeViewFactor.MeshIO

# VTK cell type id → (family, surface_dim, n_nodes). Quadratic VTK cells use
# corners-then-edge-midpoints ordering, identical to the Gmsh ordering we use,
# so no per-node remapping is required.
const VTK_FAMILY = Dict{Int,Tuple{Symbol,Int,Int}}(
     3 => (:line2, 1, 2),   # VTK_LINE
    21 => (:line3, 1, 3),   # VTK_QUADRATIC_EDGE
     5 => (:tri3,  2, 3),   # VTK_TRIANGLE
    22 => (:tri,   2, 6),   # VTK_QUADRATIC_TRIANGLE
     9 => (:quad4, 2, 4),   # VTK_QUAD
    23 => (:quad,  2, 8),   # VTK_QUADRATIC_QUAD
)

# Per-cell region arrays tried automatically when group_field === nothing.
const GROUP_FIELD_CANDIDATES =
    ("CellEntityIds", "gmsh:physical", "RegionId", "MaterialIds", "region", "group")

# Resolve a per-cell integer group array, or `nothing` for a single group.
function _resolve_groups(vtk, group_field, ncells::Int, verbose::Bool)
    cd = try
        get_cell_data(vtk)
    catch
        return nothing
    end
    names = group_field === nothing ? GROUP_FIELD_CANDIDATES : (string(group_field),)
    for nm in names
        arr = try
            get_data(cd[nm])
        catch
            continue
        end
        if length(arr) == ncells
            verbose && println("  Using per-cell group array \"$nm\".")
            return arr
        end
    end
    group_field !== nothing && @warn(
        "Requested group_field \"$group_field\" not found as a per-cell array; " *
        "using a single \"default\" group.")
    return nothing
end

# Node index range (1-based, into connectivity) for cell k, robust to whether
# `offsets` carries a leading zero (length ncells+1) or not (length ncells).
@inline function _cell_range(offs, k::Int, ncells::Int)
    if length(offs) == ncells + 1
        return (Int(offs[k]) + 1):Int(offs[k+1])
    else
        lo = k == 1 ? 1 : Int(offs[k-1]) + 1
        return lo:Int(offs[k])
    end
end

function MeshIO._load_vtu_impl(filename, surface_dim::Int, reverse_normals::Bool,
                                verbose::Bool, group_field)
    vtk    = VTKFile(filename)
    coords = Matrix{Float64}(get_points(vtk))   # 3 × N
    cells  = get_cells(vtk)                      # connectivity (1-based), offsets, types
    conn   = cells.connectivity
    offs   = cells.offsets
    types  = cells.types
    ncells = length(types)

    group_vals = _resolve_groups(vtk, group_field, ncells, verbose)

    surface_elems = MeshIO.SurfaceElement[]
    group_elems   = Dict{Int,Vector{Int}}()
    skipped = 0

    for k in 1:ncells
        info = get(VTK_FAMILY, Int(types[k]), nothing)
        if info === nothing || info[2] != surface_dim
            info === nothing && (skipped += 1)
            continue
        end
        fam, _, nn = info
        rng   = _cell_range(offs, k, ncells)
        nodes = Int[Int(conn[i]) for i in rng]
        length(nodes) == nn || error(
            "VTK cell $k has $(length(nodes)) nodes, expected $nn for $fam.")
        g = group_vals === nothing ? 1 : Int(round(Int, group_vals[k]))
        push!(surface_elems, MeshIO.SurfaceElement(nodes, g, fam))
        push!(get!(group_elems, g, Int[]), length(surface_elems))
    end

    isempty(surface_elems) && error(
        "No VTK cells of dimension $surface_dim found in $filename " *
        "(found $skipped cell(s) of unsupported/other-dimension types).")

    group_tags = Dict{Int,String}(
        g => (group_vals === nothing ? "default" : "group_$g")
        for g in keys(group_elems))

    if surface_dim == 1
        @warn "VTK curve meshes are loaded without Gmsh-based normal " *
              "orientation; relying on node winding (use reverse_normals if needed)."
    end
    reverse_normals && MeshIO._reverse_all_normals!(surface_elems, surface_dim)

    soups = MeshIO._build_group_obs_soups(coords, surface_elems, group_elems, surface_dim)

    if verbose
        println("Loaded $(length(surface_elems)) VTK cell(s) (dimension $surface_dim) ",
                "in $(length(group_tags)) group(s)",
                skipped > 0 ? "; skipped $skipped cell(s) of other types." : ".")
    end
    return MeshIO.MeshData(coords, surface_elems, group_tags, group_elems,
                           soups, surface_dim)
end

end # module RadiativeViewFactorReadVTKExt
