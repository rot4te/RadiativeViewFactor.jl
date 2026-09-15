# src/BVH.jl
# ---------------------------------------------------------------------------
# A simple, self-contained Axis-Aligned Bounding-Volume Hierarchy (BVH) for
# the triangle soup that represents obstruction geometry.
#
# Build once from MeshData.tri_soup; query repeatedly during view-factor
# integration with ray–AABB + ray–triangle tests.
#
# The BVH is a binary tree stored in a flat array (implicit left/right
# children at 2k and 2k+1).  Leaf nodes hold a small list of triangle
# indices; interior nodes hold only the merged AABB.
# ---------------------------------------------------------------------------

module BVH

using StaticArrays
using LinearAlgebra



# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

"""Axis-aligned bounding box."""
struct AABB
    lo :: SVector{3, Float64}
    hi :: SVector{3, Float64}
end

"""A node in the BVH tree (stored in a flat vector)."""
struct BVHNode
    aabb       :: AABB
    left       :: Int          # index into node array; 0 = leaf
    right      :: Int
    tri_start  :: Int          # index into sorted triangle list (leaf only)
    tri_count  :: Int
end

"""
    BVHTree

Flat-array BVH over a triangle soup.

Fields
------
- `nodes`     : flat array of BVHNode
- `tri_idx`   : permutation of triangle indices (sorted during build)
- `tri_soup`  : reference to the (3, 3, N) triangle coordinate array
"""
struct BVHTree
    nodes    :: Vector{BVHNode}
    tri_idx  :: Vector{Int}
    tri_soup :: Array{Float64, 3}   # 3 × 3 × N  (coord, vertex, tri)
end; export BVHTree

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

const LEAF_MAX = 8   # max triangles per leaf node

"""
    build_bvh(tri_soup) -> BVHTree

Build a BVH over the triangle soup.  `tri_soup` must be (3, 3, N):
  - dim 1: xyz coordinate (1, 2, 3)
  - dim 2: vertex index (1, 2, 3)
  - dim 3: triangle index
(the layout produced by `MeshIO._build_group_obs_soups`).
"""
function build_bvh(tri_soup::Array{Float64,3})::BVHTree
    N       = size(tri_soup, 3)
    tri_idx = collect(1:N)
    nodes   = BVHNode[]

    _build_recursive!(nodes, tri_soup, tri_idx, 1, N)

    return BVHTree(nodes, tri_idx, tri_soup)
end; export build_bvh

function _triangle_aabb(tri_soup::Array{Float64,3}, tidx::Int)::AABB
    # Works for both triangle soups (3,3,N) and segment soups (3,2,N):
    # loop over however many vertices/endpoints axis 2 contains.
    nv   = size(tri_soup, 2)
    lox  = tri_soup[1, 1, tidx];  hix = lox
    loy  = tri_soup[2, 1, tidx];  hiy = loy
    loz  = tri_soup[3, 1, tidx];  hiz = loz
    for v in 2:nv
        x = tri_soup[1, v, tidx];  lox = min(lox,x);  hix = max(hix,x)
        y = tri_soup[2, v, tidx];  loy = min(loy,y);  hiy = max(hiy,y)
        z = tri_soup[3, v, tidx];  loz = min(loz,z);  hiz = max(hiz,z)
    end
    return AABB(SVector(lox, loy, loz), SVector(hix, hiy, hiz))
end

function _merge_aabb(a::AABB, b::AABB)::AABB
    AABB(min.(a.lo, b.lo), max.(a.hi, b.hi))
end

function _centroid(tri_soup::Array{Float64,3}, tidx::Int)::SVector{3,Float64}
    nv = size(tri_soup, 2)
    cx = 0.0;  cy = 0.0;  cz = 0.0
    for v in 1:nv
        cx += tri_soup[1, v, tidx]
        cy += tri_soup[2, v, tidx]
        cz += tri_soup[3, v, tidx]
    end
    SVector(cx/nv, cy/nv, cz/nv)
end

"""Recursively build BVH; appends nodes to `nodes` and returns node index."""
function _build_recursive!(nodes::Vector{BVHNode},
                             tri_soup::Array{Float64,3},
                             tri_idx::Vector{Int},
                             start::Int, stop::Int)::Int
    # Compute merged AABB for this range
    box = _triangle_aabb(tri_soup, tri_idx[start])
    for i in start+1:stop
        box = _merge_aabb(box, _triangle_aabb(tri_soup, tri_idx[i]))
    end

    count = stop - start + 1

    # Leaf node
    if count <= LEAF_MAX
        push!(nodes, BVHNode(box, 0, 0, start, count))
        return length(nodes)
    end

    # Choose split axis: longest axis of AABB
    extent = box.hi - box.lo
    axis   = argmax(extent)   # 1=x, 2=y, 3=z

    # Sort triangles by centroid along chosen axis
    mid = (start + stop) ÷ 2
    sort!(view(tri_idx, start:stop);
          by = i -> _centroid(tri_soup, i)[axis])

    # Reserve a slot for this interior node, fill in children later
    push!(nodes, BVHNode(box, 0, 0, 0, 0))
    node_idx = length(nodes)

    left_child  = _build_recursive!(nodes, tri_soup, tri_idx, start, mid)
    right_child = _build_recursive!(nodes, tri_soup, tri_idx, mid+1, stop)

    nodes[node_idx] = BVHNode(box, left_child, right_child, 0, 0)
    return node_idx
end

# ---------------------------------------------------------------------------
# Ray–AABB intersection (slab method)
# ---------------------------------------------------------------------------

"""
Per-axis slab bounds for the ray/AABB test. Handles a ray direction
component of exactly zero explicitly: `inv_d = 1/0 = ±Inf` there, and if the
box also touches the ray's origin on that axis (`lo == o` or `hi == o`,
common for axis-aligned/planar test geometry), the direct formula computes
`0 * Inf = NaN`, which silently makes `_ray_aabb` reject an otherwise-valid
hit (NaN compares false against everything). A ray parallel to an axis
passes that axis's slab whenever the origin lies within `[lo, hi]`,
independent of `t`, so that case is handled as an unconstrained
`(-Inf, Inf)` (or an impossible `(Inf, -Inf)` when the origin is outside).
"""
@inline function _slab_t(lo::Float64, hi::Float64, o::Float64,
                          d::Float64, inv_d::Float64)::Tuple{Float64,Float64}
    if d == 0.0
        return (lo <= o <= hi) ? (-Inf, Inf) : (Inf, -Inf)
    end
    t1 = (lo - o) * inv_d
    t2 = (hi - o) * inv_d
    return t1 < t2 ? (t1, t2) : (t2, t1)
end

"""Return true if ray (origin `o`, direction `d`) hits `box` before t=`tmax`."""
@inline function _ray_aabb(o::SVector{3,Float64}, d::SVector{3,Float64},
                            inv_d::SVector{3,Float64},
                            box::AABB, tmax::Float64)::Bool
    lo1, hi1 = _slab_t(box.lo[1], box.hi[1], o[1], d[1], inv_d[1])
    lo2, hi2 = _slab_t(box.lo[2], box.hi[2], o[2], d[2], inv_d[2])
    lo3, hi3 = _slab_t(box.lo[3], box.hi[3], o[3], d[3], inv_d[3])

    tmin  = max(lo1, lo2, lo3, 0.0)
    tmax2 = min(hi1, hi2, hi3, tmax)

    return tmin <= tmax2
end

# ---------------------------------------------------------------------------
# Ray–triangle (Möller–Trumbore) — see RayCast.jl for the standalone version
# ---------------------------------------------------------------------------

const _EPS = 1e-10

# Barycentric edge tolerance: a ray that pierces exactly on the shared edge
# between two coplanar triangles (e.g. the diagonal split of a quad
# obstruction face) computes u (or v) as a tiny negative number on both
# triangles due to floating-point rounding, so a strict `< 0.0` test rejects
# the hit on *both* sides of the edge — a watertightness crack that silently
# lets rays leak through supposedly opaque obstruction geometry. Accepting a
# small negative slack on the barycentric bounds closes the crack; the bias
# this introduces (a ray passing just outside the triangle by less than this
# margin can register as a hit) is negligible at this scale.
const _BARY_EPS = 1e-9

"""
Return hit distance t > `t_min` if ray hits triangle, else `Inf`.
`v0,v1,v2` are the three vertices of the triangle.
"""
@inline function _ray_triangle(o::SVector{3,Float64},
                                d::SVector{3,Float64},
                                v0::SVector{3,Float64},
                                v1::SVector{3,Float64},
                                v2::SVector{3,Float64},
                                t_min::Float64)::Float64
    e1 = v1 - v0
    e2 = v2 - v0
    h  = cross(d, e2)
    a  = dot(e1, h)
    abs(a) < _EPS && return Inf
    f = 1.0 / a
    s = o - v0
    u = f * dot(s, h)
    (u < -_BARY_EPS || u > 1.0 + _BARY_EPS) && return Inf
    q = cross(s, e1)
    v = f * dot(d, q)
    (v < -_BARY_EPS || u + v > 1.0 + _BARY_EPS) && return Inf
    t = f * dot(e2, q)
    t > t_min ? t : Inf
end

# ---------------------------------------------------------------------------
# BVH traversal
# ---------------------------------------------------------------------------

"""
    intersect_ray_bvh(bvh, origin, direction, t_max) -> Bool

Return `true` if any triangle in `bvh` is hit by the ray
`origin + t * direction` for `t ∈ (0, t_max)`.
"""
function intersect_ray_bvh(bvh      ::BVHTree,
                             origin   ::SVector{3,Float64},
                             direction::SVector{3,Float64},
                             t_max    ::Float64)::Bool
    inv_d = SVector(1.0/direction[1], 1.0/direction[2], 1.0/direction[3])

    stack    = zeros(Int, 64)
    stack[1] = 1
    sp       = 1

    @inbounds while sp > 0
        nidx = stack[sp];  sp -= 1
        node = bvh.nodes[nidx]

        _ray_aabb(origin, direction, inv_d, node.aabb, t_max) || continue

        if node.left == 0   # leaf
            for k in node.tri_start : node.tri_start + node.tri_count - 1
                tidx = bvh.tri_idx[k]
                # tri_soup layout is (xyz, vertex, triangle) -- dim 1 is the
                # coordinate axis, dim 2 is the vertex index (see
                # _build_group_obs_soups in MeshIO.jl and _triangle_aabb /
                # _centroid above, which both read it this way correctly).
                # This was previously transposed here, reading each
                # SVector as (x1,x2,x3) etc. instead of (x,y,z) of one
                # vertex -- i.e. testing three bogus points built from the
                # three vertices' *same-axis* coordinates instead of the
                # three real vertices, which silently broke essentially all
                # 3-D ray/triangle obstruction queries.
                v0 = SVector{3,Float64}(bvh.tri_soup[1,1,tidx],
                                         bvh.tri_soup[2,1,tidx],
                                         bvh.tri_soup[3,1,tidx])
                v1 = SVector{3,Float64}(bvh.tri_soup[1,2,tidx],
                                         bvh.tri_soup[2,1,tidx],
                                         bvh.tri_soup[3,1,tidx])
                v1 = SVector{3,Float64}(bvh.tri_soup[1,2,tidx],
                                         bvh.tri_soup[2,2,tidx],
                                         bvh.tri_soup[3,2,tidx])
                v2 = SVector{3,Float64}(bvh.tri_soup[1,3,tidx],
                                         bvh.tri_soup[2,3,tidx],
                                         bvh.tri_soup[3,2,tidx])
                v2 = SVector{3,Float64}(bvh.tri_soup[1,3,tidx],
                                         bvh.tri_soup[2,3,tidx],
                                         bvh.tri_soup[3,3,tidx])
                t = _ray_triangle(origin, direction, v0, v1, v2, 0.0)
                t < t_max && return true
            end
        else
            sp += 1;  stack[sp] = node.left
            sp += 1;  stack[sp] = node.right
        end
    end
    return false
end; export intersect_ray_bvh


# ---------------------------------------------------------------------------
# 2-D ray–segment intersection and BVH traversal for curve meshes
# ---------------------------------------------------------------------------

"""
    _seg_blocks_ray(xi, xj, v0, v1; tol=1e-8) -> Bool

Return true if the 2-D line segment v0→v1 blocks the ray xi→xj.
Only the x and y components are used (z is ignored).
Uses 2-D line-line intersection via Cramer's rule; both segments are
parameterised and checked for interior intersection (0 < s,t < 1).
"""
@inline function _seg_blocks_ray(xi::SVector{3,Float64}, xj::SVector{3,Float64},
                                  v0::SVector{3,Float64}, v1::SVector{3,Float64};
                                  tol::Float64 = 1e-8)::Bool
    dx = xj[1]-xi[1];  dy = xj[2]-xi[2]   # ray direction (xy only)
    ex = v1[1]-v0[1];  ey = v1[2]-v0[2]   # segment direction
    denom = dx*ey - dy*ex
    abs(denom) < tol && return false        # parallel
    fx = v0[1]-xi[1];  fy = v0[2]-xi[2]
    s  = (fx*ey - fy*ex) / denom           # parameter along ray
    t  = (fx*dy - fy*dx) / denom           # parameter along segment
    return tol < s < 1.0-tol && tol < t < 1.0-tol
end

"""
    intersect_seg_bvh(bvh, x_i, x_j) -> Bool

Return true if the 2-D segment from x_i to x_j is blocked by any segment
in `bvh`. The BVH must have been built from a segment soup (3, 2, N_segs);
only xy-components are used.
"""
function intersect_seg_bvh(bvh::BVHTree,
                             x_i::SVector{3,Float64},
                             x_j::SVector{3,Float64})::Bool
    # Bounding box of the ray for quick node rejection
    ray_xlo = min(x_i[1], x_j[1]);  ray_xhi = max(x_i[1], x_j[1])
    ray_ylo = min(x_i[2], x_j[2]);  ray_yhi = max(x_i[2], x_j[2])

    stack    = zeros(Int, 64)
    stack[1] = 1
    sp       = 1

    @inbounds while sp > 0
        nidx = stack[sp]; sp -= 1
        node = bvh.nodes[nidx]

        # 2-D AABB overlap (ignore z)
        (node.aabb.lo[1] > ray_xhi || node.aabb.hi[1] < ray_xlo ||
         node.aabb.lo[2] > ray_yhi || node.aabb.hi[2] < ray_ylo) && continue

        if node.left == 0  # leaf
            for k in node.tri_start : node.tri_start + node.tri_count - 1
                tidx = bvh.tri_idx[k]
                # Segment soup: (3, 2, N) — axis 1=xyz, axis 2=endpoint, axis 3=seg
                v0 = SVector(bvh.tri_soup[1, 1, tidx],
                              bvh.tri_soup[2, 1, tidx],
                              0.0)
                v1 = SVector(bvh.tri_soup[1, 2, tidx],
                              bvh.tri_soup[2, 2, tidx],
                              0.0)
                _seg_blocks_ray(x_i, x_j, v0, v1) && return true
            end
        else
            sp += 1; stack[sp] = node.left
            sp += 1; stack[sp] = node.right
        end
    end
    return false
end; export intersect_seg_bvh

end # module BVH
