# src/Quadrature.jl
# ---------------------------------------------------------------------------
# Gauss–Legendre quadrature rules on the reference interval [-1, 1] and the
# reference square [-1,1]².
#
# Points and weights are hard-coded up to n=5 for zero-dependency startup;
# a generic n-point rule is provided via the Golub–Welsch algorithm.
# ---------------------------------------------------------------------------

module Quadrature

using LinearAlgebra
export gauss_legendre_1d, gauss_legendre_2d, QuadRule2D

"""
    QuadRule2D

Tensor-product Gauss–Legendre rule on [-1,1]².

Fields
------
- `points`  : `(2, N_pts)` matrix — (ξ, η) coordinates
- `weights` : `N_pts`-vector — quadrature weights (sum = 4)
"""
struct QuadRule2D
    points  :: Matrix{Float64}   # 2 × N
    weights :: Vector{Float64}   # N
end

# ---------------------------------------------------------------------------
# 1-D Gauss–Legendre nodes and weights on [-1, 1]
# ---------------------------------------------------------------------------

# Pre-tabulated rules (points, weights) for n = 1..5
const _GL_TABLE = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}(
    1 => ([0.0],
          [2.0]),

    2 => ([-1/√3,  1/√3],
          [ 1.0,   1.0]),

    3 => ([-√(3/5), 0.0, √(3/5)],
          [5/9,     8/9, 5/9]),

    4 => begin
             a  = √((3 + 2√(6/5))/7)
             b  = √((3 - 2√(6/5))/7)
             wa = (18 - √30)/36
             wb = (18 + √30)/36
             ([-a, -b, b, a], [wa, wb, wb, wa])
         end,

    5 => begin
             a  = √(5 + 2√(10/7))/3
             b  = √(5 - 2√(10/7))/3
             wa = (322 - 13√70)/900
             wb = (322 + 13√70)/900
             wc = 128/225
             ([-a, -b, 0.0, b, a], [wa, wb, wc, wb, wa])
         end,
)

"""
    gauss_legendre_1d(n) -> (points, weights)

Return the `n`-point Gauss–Legendre rule on [-1, 1].
Pre-tabulated for n ≤ 5; uses the Golub–Welsch algorithm otherwise.
"""
function gauss_legendre_1d(n::Int)
    haskey(_GL_TABLE, n) && return _GL_TABLE[n]
    return _golub_welsch(n)
end

"""Golub–Welsch algorithm for arbitrary n."""
function _golub_welsch(n::Int)
    # Build symmetric tridiagonal Jacobi matrix
    β = [i / √(4i^2 - 1.0) for i in 1:n-1]
    J = diagm(0 => zeros(n), 1 => β, -1 => β)
    vals, vecs = eigen(J)
    idx     = sortperm(vals)
    points  = vals[idx]
    weights = 2.0 .* vecs[1, idx].^2
    return points, weights
end

# ---------------------------------------------------------------------------
# 2-D tensor-product rule on [-1,1]²
# ---------------------------------------------------------------------------

"""
    gauss_legendre_2d(n) -> QuadRule2D

Tensor-product `n×n` Gauss–Legendre rule on the reference square [-1,1]².
Total number of quadrature points: n².
"""
function gauss_legendre_2d(n::Int)::QuadRule2D
    pts1d, wts1d = gauss_legendre_1d(n)
    N   = n^2
    pts = Matrix{Float64}(undef, 2, N)
    wts = Vector{Float64}(undef, N)
    k   = 0
    for j in 1:n, i in 1:n
        k += 1
        pts[1, k] = pts1d[i]
        pts[2, k] = pts1d[j]
        wts[k]    = wts1d[i] * wts1d[j]
    end
    return QuadRule2D(pts, wts)
end

end # module Quadrature
