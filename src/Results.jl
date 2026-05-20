# src/Results.jl
# ---------------------------------------------------------------------------
# Shared result types and aggregation logic used by both Assembly (CPU path)
# and GPUAssembly (GPU path).  Kept in its own module so neither Assembly nor
# GPUAssembly need to import each other, breaking the circular dependency.
# ---------------------------------------------------------------------------

module Results

using LinearAlgebra

import ..MeshIO: MeshData

export ViewFactorResult, _aggregate, aggregate_by_group,
       check_reciprocity, check_closure

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

"""
    ViewFactorResult

Stores element-level and group-level view factor matrices produced by
`compute_view_factors`.

Fields
------
- `F_elem`      : (N_elem × N_elem) element-level view factor matrix
- `A_elem`      : N_elem-vector of element areas (or lengths for curve meshes)
- `F_group`     : (N_group × N_group) group-level view factor matrix
- `A_group`     : N_group-vector of group areas/lengths
- `group_tags`  : sorted physical group tags (row/column labels for F_group)
- `group_names` : physical group names in the same order
"""
struct ViewFactorResult
    F_elem      :: Matrix{Float64}
    A_elem      :: Vector{Float64}
    F_group     :: Matrix{Float64}
    A_group     :: Vector{Float64}
    group_tags  :: Vector{Int}
    group_names :: Vector{String}
end

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

"""
    _aggregate(mesh, F_elem, A_elem) -> (group_tags, group_names, F_group, A_group)

Aggregate element-level view factors to physical-group level using the
area-weighted formula:

    Aᵍ · F_{g→h} = Σᵢ∈g  Aᵢ · Σⱼ∈h  Fᵢⱼ
"""
function _aggregate(mesh::MeshData,
                     F_elem::Matrix{Float64},
                     A_elem::Vector{Float64})
    gtags  = sort(collect(keys(mesh.group_tags)))
    gnames = [mesh.group_tags[t] for t in gtags]
    G      = length(gtags)

    A_group = zeros(Float64, G)
    for (k, tag) in enumerate(gtags)
        for ei in mesh.group_elems[tag]
            A_group[k] += A_elem[ei]
        end
    end

    F_group = zeros(Float64, G, G)
    for (gi, tagi) in enumerate(gtags)
        for ei in mesh.group_elems[tagi]
            for (gj, tagj) in enumerate(gtags)
                Σ = sum(F_elem[ei, ej] for ej in mesh.group_elems[tagj])
                F_group[gi, gj] += A_elem[ei] * Σ
            end
        end
        F_group[gi, :] ./= A_group[gi]
    end

    return gtags, gnames, F_group, A_group
end

"""
    aggregate_by_group(result, mesh) -> (F_group, A_group, tags, names)

Re-aggregate element-level results to group level (useful after modifying
`result.F_elem`).
"""
function aggregate_by_group(result::ViewFactorResult, mesh::MeshData)
    tags, names, Fg, Ag = _aggregate(mesh, result.F_elem, result.A_elem)
    return Fg, Ag, tags, names
end

# ---------------------------------------------------------------------------
# Post-processing checks
# ---------------------------------------------------------------------------

"""
    check_reciprocity(result; tol=1e-4) -> Bool

Verify Aᵢ Fᵢⱼ ≈ Aⱼ Fⱼᵢ for all element pairs.
Prints the maximum relative error and returns `true` if it is below `tol`.
"""
function check_reciprocity(result::ViewFactorResult; tol::Float64=1e-4)::Bool
    F = result.F_elem; A = result.A_elem; N = size(F, 1)
    max_err = 0.0
    for i in 1:N, j in i+1:N
        err = abs(A[i]*F[i,j] - A[j]*F[j,i]) / max(A[i]*F[i,j], 1e-30)
        max_err = max(max_err, err)
    end
    println("Reciprocity max relative error: $max_err")
    return max_err < tol
end

"""
    check_closure(result; tol=1e-3) -> Bool

Verify that no row of `F_elem` sums to more than 1 + `tol`.
Prints the row-sum range and returns `true` if the check passes.
"""
function check_closure(result::ViewFactorResult; tol::Float64=1e-3)::Bool
    row_sums = vec(sum(result.F_elem, dims=2))
    println("Row sums: min=$(round(minimum(row_sums),digits=6)), " *
            "max=$(round(maximum(row_sums),digits=6))")
    return maximum(row_sums) <= 1.0 + tol
end

end # module Results
