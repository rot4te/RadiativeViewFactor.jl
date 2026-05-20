# src/Geometry.jl
# ---------------------------------------------------------------------------
# Isoparametric geometry for the 8-node serendipity quadrilateral (Quad8).
#
# Reference element: (ξ, η) ∈ [-1,1]²
#
# Node numbering (Gmsh convention, 1-based):
#
#   4---7---3
#   |       |
#   8       6
#   |       |
#   1---5---2
# ---------------------------------------------------------------------------

module Geometry

using LinearAlgebra
using StaticArrays
import ..Quadrature: gauss_legendre_2d

export quad8_shape,
       quad8_physical_point,
       quad8_normal_and_area_element,
       element_area,
       line3_shape,
       line3_physical_point,
       line3_normal_and_length_element

@inline function quad8_shape(ξ::Float64, η::Float64)
    # Shape functions
    N1 = 0.25*(1-ξ)*(1-η)*(-ξ-η-1)
    N2 = 0.25*(1+ξ)*(1-η)*( ξ-η-1)
    N3 = 0.25*(1+ξ)*(1+η)*( ξ+η-1)
    N4 = 0.25*(1-ξ)*(1+η)*(-ξ+η-1)
    N5 = 0.5*(1-ξ^2)*(1-η)
    N6 = 0.5*(1+ξ)*(1-η^2)
    N7 = 0.5*(1-ξ^2)*(1+η)
    N8 = 0.5*(1-ξ)*(1-η^2)
    N  = SVector(N1,N2,N3,N4,N5,N6,N7,N8)

    # ∂N/∂ξ  (derived via product rule)
    dN1dξ = 0.25*(1-η)*(2ξ+η)
    dN2dξ = 0.25*(1-η)*(2ξ-η)
    dN3dξ = 0.25*(1+η)*(2ξ+η)
    dN4dξ = 0.25*(1+η)*(2ξ-η)
    dN5dξ = -ξ*(1-η)
    dN6dξ =  0.5*(1-η^2)
    dN7dξ = -ξ*(1+η)
    dN8dξ = -0.5*(1-η^2)
    dNdξ  = SVector(dN1dξ,dN2dξ,dN3dξ,dN4dξ,dN5dξ,dN6dξ,dN7dξ,dN8dξ)

    # ∂N/∂η  (derived via product rule)
    dN1dη = 0.25*(1-ξ)*(ξ+2η)
    dN2dη = 0.25*(1+ξ)*(-ξ+2η)
    dN3dη = 0.25*(1+ξ)*(ξ+2η)
    dN4dη = 0.25*(1-ξ)*(-ξ+2η)
    dN5dη = -0.5*(1-ξ^2)
    dN6dη = -(1+ξ)*η
    dN7dη =  0.5*(1-ξ^2)
    dN8dη = -(1-ξ)*η
    dNdη  = SVector(dN1dη,dN2dη,dN3dη,dN4dη,dN5dη,dN6dη,dN7dη,dN8dη)

    return N, dNdξ, dNdη
end

"""
    quad8_physical_point(coords, nodes, ξ, η) -> SVector{3,Float64}

Map (ξ,η) to physical space. `nodes` may be a Vector{Int} or SVector{8,Int}.
"""
@inline function quad8_physical_point(coords::Matrix{Float64},
                                       nodes,
                                       ξ::Float64, η::Float64)::SVector{3,Float64}
    N, _, _ = quad8_shape(ξ, η)
    x = @SVector zeros(3)
    for a in 1:8
        xa = SVector{3,Float64}(coords[1,nodes[a]], coords[2,nodes[a]], coords[3,nodes[a]])
        x  = x + N[a]*xa
    end
    return x
end

"""
    quad8_normal_and_area_element(coords, nodes, ξ, η) -> (n̂, dA)

Compute the unit normal and area element at (ξ,η). `nodes` may be a
Vector{Int} or SVector{8,Int}.
"""
@inline function quad8_normal_and_area_element(coords::Matrix{Float64},
                                                nodes,
                                                ξ::Float64, η::Float64)
    _, dNdξ, dNdη = quad8_shape(ξ, η)
    dxdξ = @SVector zeros(3)
    dxdη = @SVector zeros(3)
    for a in 1:8
        xa   = SVector{3,Float64}(coords[1,nodes[a]], coords[2,nodes[a]], coords[3,nodes[a]])
        dxdξ = dxdξ + dNdξ[a]*xa
        dxdη = dxdη + dNdη[a]*xa
    end
    c  = cross(dxdξ, dxdη)
    dA = norm(c)
    return c/dA, dA
end

"""
    element_area(coords, nodes; nquad=4) -> Float64

Numerically integrate the area of one Quad8 element.
"""
function element_area(coords::Matrix{Float64}, nodes; nquad::Int=4)::Float64
    # Inline a minimal GL rule to avoid a circular module dependency
    pts1d = [-√(3/5), 0.0, √(3/5)]
    wts1d = [5/9, 8/9, 5/9]
    if nquad <= 2
        pts1d = [-1/√3, 1/√3]; wts1d = [1.0, 1.0]
    elseif nquad >= 4
        # Use Golub–Welsch via the parent Quadrature module
        rule = gauss_legendre_2d(nquad)
        A = 0.0
        for k in 1:size(rule.points, 2)
            ξ, η = rule.points[1,k], rule.points[2,k]
            _, dA = quad8_normal_and_area_element(coords, nodes, ξ, η)
            A += rule.weights[k] * dA
        end
        return A
    end
    # 3-point rule fallback
    A = 0.0
    for (ξ,wξ) in zip(pts1d,wts1d), (η,wη) in zip(pts1d,wts1d)
        _, dA = quad8_normal_and_area_element(coords, nodes, ξ, η)
        A += wξ * wη * dA
    end
    return A
end


# ---------------------------------------------------------------------------
# Line3 element (3-node 2nd-order line, Gmsh type 8)
#
# Reference element: ξ ∈ [-1, 1]
# Node ordering (Gmsh 1-based): 1 = left endpoint, 2 = right endpoint,
#                                3 = midpoint
#
# Shape functions:
#   N₁ = -½ ξ(1−ξ)   (node 1, left corner)
#   N₂ =  ½ ξ(1+ξ)   (node 2, right corner)
#   N₃ =  (1−ξ²)      (node 3, midpoint)
#
# The in-plane unit normal is computed by rotating the tangent 90°
# counter-clockwise in the xy-plane: n = (−tᵧ, tₓ, 0) / |t|.
# This is correct for curves lying in the z = const plane.  For general
# 3-D planar curves the caller is responsible for ensuring all curves share
# a common normal plane.
# ---------------------------------------------------------------------------

@inline function line3_shape(ξ::Float64)
    N1  = -0.5*ξ*(1.0 - ξ)
    N2  =  0.5*ξ*(1.0 + ξ)
    N3  =  1.0 - ξ^2
    dN1 = -0.5 + ξ
    dN2 =  0.5 + ξ
    dN3 = -2.0*ξ
    return SVector(N1, N2, N3), SVector(dN1, dN2, dN3)
end

"""
    line3_physical_point(coords, nodes, ξ) -> SVector{3,Float64}

Map reference coordinate ξ ∈ [-1,1] to physical space for a Line3 element.
`nodes` is a length-3 index vector [left corner, right corner, midpoint].
"""
@inline function line3_physical_point(coords::Matrix{Float64},
                                       nodes ::Vector{Int},
                                       ξ     ::Float64)::SVector{3,Float64}
    N, _ = line3_shape(ξ)
    x    = @SVector zeros(3)
    for a in 1:3
        xa = SVector{3,Float64}(coords[1,nodes[a]], coords[2,nodes[a]], coords[3,nodes[a]])
        x  = x + N[a]*xa
    end
    return x
end

"""
    line3_normal_and_length_element(coords, nodes, ξ) -> (n̂, dL)

Compute the in-plane unit normal and arc-length element at ξ for a Line3
element lying in the xy-plane (z = const).

Returns
-------
- `n̂`  : SVector{3,Float64} — unit normal, z-component is zero
- `dL` : Float64            — arc-length element |dx/dξ|
"""
@inline function line3_normal_and_length_element(coords::Matrix{Float64},
                                                  nodes ::Vector{Int},
                                                  ξ     ::Float64)
    _, dN = line3_shape(ξ)
    dxdξ  = @SVector zeros(3)
    for a in 1:3
        xa   = SVector{3,Float64}(coords[1,nodes[a]], coords[2,nodes[a]], coords[3,nodes[a]])
        dxdξ = dxdξ + dN[a]*xa
    end
    dL = norm(dxdξ)
    t  = dxdξ / dL                              # unit tangent
    n̂  = SVector(-t[2], t[1], 0.0)              # 90° CCW rotation in xy
    return n̂, dL
end

end # module Geometry
