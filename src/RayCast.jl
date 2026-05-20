# src/RayCast.jl
module RayCast

using StaticArrays
using LinearAlgebra

import ..BVH: BVHTree, intersect_ray_bvh, intersect_seg_bvh

export is_visible

"""
    is_visible(bvh, x_i, x_j; mesh_dim=2) -> Bool

Return `true` if the path from `x_i` to `x_j` is unobstructed.

`mesh_dim=2` (default) — 3-D ray–triangle BVH test.
`mesh_dim=1`            — 2-D ray–segment BVH test (xy components only);
                          the BVH must have been built from a segment soup.
"""
@inline function is_visible(bvh     ::BVHTree,
                             x_i    ::SVector{3,Float64},
                             x_j    ::SVector{3,Float64};
                             mesh_dim::Int = 2)::Bool
    if mesh_dim == 1
        return !intersect_seg_bvh(bvh, x_i, x_j)
    else
        d = x_j - x_i
        L = norm(d)
        L < eps() && return true
        return !intersect_ray_bvh(bvh, x_i, d/L, L)
    end
end

end # module RayCast
