# src/Results.jl
# ---------------------------------------------------------------------------
# Shared result types and aggregation logic used by both Assembly (CPU path)
# and GPUAssembly (GPU path).  Kept in its own module so neither Assembly nor
# GPUAssembly need to import each other, breaking the circular dependency.
# ---------------------------------------------------------------------------

module Results

using LinearAlgebra

import ..MeshIO: MeshData



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
end; export ViewFactorResult

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
end; export _aggregate

"""
    aggregate_by_group(result, mesh) -> (F_group, A_group, tags, names)

Re-aggregate element-level results to group level (useful after modifying
`result.F_elem`).
"""
function aggregate_by_group(result::ViewFactorResult, mesh::MeshData)
    tags, names, Fg, Ag = _aggregate(mesh, result.F_elem, result.A_elem)
    return Fg, Ag, tags, names
end; export aggregate_by_group

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
end; export check_reciprocity

"""
    enforce_closure(result, mesh; iters=200, tol=1e-12, verbose=true)
        -> ViewFactorResult

Return a copy of `result` whose `F_elem` satisfies both

    Σⱼ Fᵢⱼ = 1            (closure, every row)
    Aᵢ Fᵢⱼ = Aⱼ Fⱼᵢ        (reciprocity, with `result.A_elem`)

to machine precision, by alternating a symmetrisation of `Aᵢ Fᵢⱼ` with a
rescaling of each row — each step perturbs the other, so they are iterated
(at most `iters` times, stopping once both residuals are below `tol`).

# Why a converged calculation still needs this

Net radiative flux is a *difference* of radiosities, `qᵢ = Jᵢ - Σⱼ FᵢⱼJⱼ`, and
in a nearly-isothermal enclosure that difference is orders of magnitude
smaller than `J` itself: in the single-pebble NekRS case this function was
written for, `J ≈ 5.9e4 W/m²` and `q ≈ 2.5e2 W/m²`. A row-sum error of `ε`
therefore leaks `ε·J` straight into `q` — 0.3 % of closure becomes 70 % of the
answer. Exact row sums remove that channel entirely: with `Σⱼ Fᵢⱼ = 1` an
isothermal enclosure gives `q = 0` identically, whatever the remaining
per-pair errors, so only genuine temperature differences drive a flux.

Reaching that by sampling alone is hopeless — Monte Carlo closure error falls
as `1/√n_rays`, so `ε = 1e-5` would need about `1e10` rays per element — and
quadrature on curved elements plateaus around `1e-3`. Enforcing the two
identities the view factors must satisfy anyway is the cheap route.

# When not to use it

Only for a **closed** enclosure, where every row genuinely sums to 1. A result
from `compute_view_factors(...; radiating_groups=...)`, or any geometry open
to the surroundings, has rows that legitimately sum to less than 1 (see
[`check_closure`](@ref)), and forcing them to 1 would invent radiation that
is not there.
"""
function enforce_closure(result::ViewFactorResult, mesh::MeshData;
                         iters::Int=200, tol::Float64=1e-12,
                         verbose::Bool=true)::ViewFactorResult
    F = copy(result.F_elem)
    A = result.A_elem
    row_err = rec_err = NaN
    for _ in 1:iters
        M = A .* F                        # Aᵢ Fᵢⱼ
        M = 0.5 .* (M .+ transpose(M))    # reciprocity
        F = M ./ A
        rows = vec(sum(F, dims=2))
        F  ./= rows                       # closure
        row_err = maximum(abs.(rows .- 1.0))
        M2      = A .* F
        rec_err = maximum(abs.(M2 .- transpose(M2))) / max(maximum(M2), 1e-30)
        (row_err < tol && rec_err < tol) && break
    end
    verbose && println("enforce_closure: row-sum residual $(row_err), " *
                       "reciprocity residual $(rec_err)")
    tags, names, Fg, Ag = _aggregate(mesh, F, A)
    return ViewFactorResult(F, A, Fg, Ag, tags, names)
end; export enforce_closure

"""
    check_closure(result; tol=1e-3) -> Bool

Verify that no row of `F_elem` sums to more than 1 + `tol`.
Prints the row-sum range and returns `true` if the check passes.

Only meaningful for a closed enclosure. A result computed with
`compute_view_factors(...; radiating_groups=...)` is deliberately open — the
non-radiating bodies absorb their share but are never assembled — so its rows
sum to less than 1 by construction and a low row sum is not a defect. Use
[`check_reciprocity`](@ref), which is unaffected, to validate such a result.
"""
function check_closure(result::ViewFactorResult; tol::Float64=1e-3)::Bool
    row_sums = vec(sum(result.F_elem, dims=2))
    println("Row sums: min=$(round(minimum(row_sums),digits=6)), " *
            "max=$(round(maximum(row_sums),digits=6))")
    return maximum(row_sums) <= 1.0 + tol
end; export check_closure

end # module Results
