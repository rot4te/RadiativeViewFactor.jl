# src/NekExport.jl
# ---------------------------------------------------------------------------
# Write view factors back out in the ASCII format read by Nek5000/NekRS's
# view-factor radiation module (`view_factors.f`'s `vf_read_view_factors`,
# as used by the Nek-VF surface-to-surface radiation solver). This is the
# counterpart to `load_re2`: `load_re2` reads a Nek5000/NekRS mesh and
# records each boundary face's Nek `(global element, local face)` identity in
# `SurfaceElement.eg`/`.iface`; this module writes `compute_view_factors`'s
# results back out keyed by that same identity, so a case's `.usr` file can
# load them directly with `call vf_read_view_factors(...)`.
# ---------------------------------------------------------------------------

module NekExport

using Printf

import ..MeshIO:  MeshData, SurfaceElement
import ..Results: ViewFactorResult

export write_nekrs_view_factors

"""
    write_nekrs_view_factors(path, result, mesh;
                              tol=0.0, max_visible_walls=10_000,
                              max_walls=4_000_000, verbose=true) -> Int

Write `result.F_elem` to `path` in the list-directed ASCII format read by
Nek5000/NekRS's `vf_read_view_factors` (in the Nek-VF `view_factors.f`
module):

```
nwalls
iw₁ ieg₁ ifc₁ nvwalls₁
  jw ieg ifc Fij        (nvwalls₁ lines)
iw₂ ieg₂ ifc₂ nvwalls₂
  ⋮
```

`iw` runs `1:N` in `mesh.surface_elems` order (Nek5000 does not require any
particular enclosure-face numbering — it rebuilds its own `(eg,ifc) → iw`
map from this file's `ieg`/`ifc` columns at load time). `ieg`/`ifc` are read
from `SurfaceElement.eg`/`.iface`, so `mesh` must come from
[`load_re2`](@ref) — an error is raised otherwise.

Only pairs with `F_elem[i,j] > tol` are written as visible faces per wall,
matching how Nek5000's hemi-cube view-factor calculator only records
genuinely visible (rasterized) pairs; `tol=0.0` (default) keeps every
strictly-positive entry, which for the quadrature/Monte Carlo/Duffy kernels
here already excludes back-facing and obstructed pairs (`vf_kernel` returns
exactly `0.0` for those). Raise `tol` to shrink the file for dense meshes at
the cost of a small closure error.

The same file is used for both `open_face_option=1` and `=2` in
`vf_calculate_radiation_heat_flux` — the option only changes which faces
*emit* at runtime (`cbc.eq.'W  '`, decided by the case's `.usr` file), not
which faces appear in this file. This function therefore writes every
element in `mesh.surface_elems`, walls and non-walls (e.g. periodic) alike —
see the `load_re2` docstring for why non-wall boundary faces must stay in
the enclosure.

Prints (does not throw) a warning if any row's visible-face count would
exceed `max_visible_walls`, or if `N` exceeds `max_walls` — these mirror the
`parameter`s of the same name in `view_factors/VIEW_FACTORS` and only
*warn*, not abort, on the Fortran side too; increase those parameters if you
see this warning and re-run the Nek5000/NekRS case.

Returns `N`, the number of walls written.
"""
function write_nekrs_view_factors(path::AbstractString,
                                   result::ViewFactorResult,
                                   mesh::MeshData;
                                   tol::Float64           = 0.0,
                                   max_visible_walls::Int = 10_000,
                                   max_walls::Int         = 4_000_000,
                                   verbose::Bool          = true)::Int
    elems = mesh.surface_elems
    N     = length(elems)
    N == size(result.F_elem, 1) ||
        error("result.F_elem ($(size(result.F_elem))) does not match " *
              "mesh.surface_elems ($N elements); pass the same `mesh` that " *
              "was used to compute `result`.")
    all(e -> e.eg > 0 && 1 <= e.iface <= 6, elems) ||
        error("write_nekrs_view_factors requires a mesh loaded with " *
              "load_re2 (every SurfaceElement needs a Nek (eg,iface) " *
              "identity); got elements with eg=0/iface=0.")

    N > max_walls &&
        @warn "N=$N boundary faces exceeds max_walls=$max_walls; increase " *
              "the `max_walls` parameter in view_factors/VIEW_FACTORS " *
              "before running the NekRS case, or this file will silently " *
              "overflow Nek5000's arrays."

    F = result.F_elem
    max_row_visible = 0

    open(path, "w") do io
        println(io, N)
        for i in 1:N
            ei = elems[i]
            js = [j for j in 1:N if j != i && F[i, j] > tol]
            max_row_visible = max(max_row_visible, length(js))
            println(io, i, " ", ei.eg, " ", ei.iface, " ", length(js))
            for j in js
                ej = elems[j]
                @printf(io, "%d %d %d %.8e\n", j, ej.eg, ej.iface, F[i, j])
            end
        end
    end

    max_row_visible > max_visible_walls &&
        @warn "Largest per-wall visible-face count ($max_row_visible) " *
              "exceeds max_visible_walls=$max_visible_walls; increase the " *
              "`max_visible_walls` parameter in view_factors/VIEW_FACTORS " *
              "before running the NekRS case, or raise `tol` here to prune " *
              "small view factors, otherwise Nek5000's arrays will " *
              "silently overflow."

    verbose && println("Wrote $N walls to $path " *
                        "(max $max_row_visible visible faces/wall).")
    return N
end

end # module NekExport
