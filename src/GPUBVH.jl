# src/GPUBVH.jl
# ---------------------------------------------------------------------------
# GPU-friendly BVH: flattens the CPU BVHTree into plain typed arrays that can
# be transferred to any KernelAbstractions device, plus the @inline stackless
# traversal function called from inside GPU kernels.
#
# Data layout
# -----------
# nodes_lo   : FloatT (3, M)   — AABB lower bounds per node
# nodes_hi   : FloatT (3, M)   — AABB upper bounds per node
# nodes_meta : Int32  (5, M)   — per node:
#                [1] left       — left child index (0 = leaf)
#                [2] right      — right child index
#                [3] tri_start  — first index into tri_idx (leaf only)
#                [4] tri_count  — number of triangles (leaf only; 0 = interior)
#                [5] miss_link  — next node to visit on AABB miss, or 0 = done
# tri_idx    : Int32  (N,)     — sorted triangle permutation (from BVH build)
# tri_verts  : FloatT (3,3,N)  — triangle vertices [vertex, xyz, tri]
# tri_group  : Int32  (N,)     — physical group tag per triangle (pre-permutation)
#
# Stackless traversal
# -------------------
# The miss_link of each node is the index of the next node to visit after
# completely skipping (or finishing) this node's subtree in a depth-first
# traversal.  Interior nodes are visited by following left_child on a hit and
# miss_link on a miss; leaf nodes test their triangles then follow miss_link.
# No thread-local stack is needed, eliminating MVector register pressure.
#
# Miss link assignment invariant:
#   miss_link(node) = first node visited after node's subtree in DFS order
#                   = right sibling of nearest ancestor for which this node
#                     is in the left subtree; 0 if none exists.
# This is computed in one recursive pass during build_flat_bvh.
# ---------------------------------------------------------------------------

module GPUBVH

using StaticArrays

import ..BVH: BVHTree, build_bvh



# ---------------------------------------------------------------------------
# Flat BVH struct — one type parameter per field for device-array compatibility
# ---------------------------------------------------------------------------

struct FlatBVH{A1, A2, A3, A4, A5, A6}
    nodes_lo   :: A1   # (3, M)    FloatT
    nodes_hi   :: A2   # (3, M)    FloatT
    nodes_meta :: A3   # (5, M)    Int32
    tri_idx    :: A4   # (N,)      Int32
    tri_verts  :: A5   # (3, 3, N) FloatT
    tri_group  :: A6   # (N,)      Int32
end; export FlatBVH

# ---------------------------------------------------------------------------
# Miss-link computation (CPU side, called during flattening)
# ---------------------------------------------------------------------------

"""
    _assign_miss_links!(miss_links, nodes, node_idx, next_node)

Recursively assign miss links to all nodes in the subtree rooted at `node_idx`.
`next_node` is the DFS-order successor of this entire subtree (0 = end of tree).

Call as: `_assign_miss_links!(miss_links, cpu_bvh.nodes, 1, 0)`
"""
function _assign_miss_links!(miss_links::Vector{Int32},
                              nodes,
                              node_idx ::Int,
                              next_node::Int)
    miss_links[node_idx] = Int32(next_node)
    nd = nodes[node_idx]
    if nd.left != 0   # interior node
        # After exhausting the left subtree, visit the right child
        _assign_miss_links!(miss_links, nodes, nd.left,  nd.right)
        # After exhausting the right subtree, go to this node's successor
        _assign_miss_links!(miss_links, nodes, nd.right, next_node)
    end
    # Leaf nodes: miss_link already set above; no children to recurse into
end

# ---------------------------------------------------------------------------
# CPU BVHTree → FlatBVH
# ---------------------------------------------------------------------------

"""
    build_flat_bvh(cpu_bvh, tri_group_cpu, FloatT, ArrayT) -> FlatBVH

Flatten a CPU `BVHTree` to plain typed device arrays.

`tri_group_cpu` is a `Vector{Int32}` of length `size(cpu_bvh.tri_soup, 3)`,
giving the physical group tag of each triangle indexed by its original
(pre-permutation) index.

Miss links are computed here and stored in row 5 of `nodes_meta`.
"""
function build_flat_bvh(cpu_bvh     ::BVHTree,
                         tri_group_cpu::Vector{Int32},
                         ::Type{FloatT}, ArrayT) where FloatT
    M = length(cpu_bvh.nodes)

    lo_cpu   = Array{FloatT}(undef, 3, M)
    hi_cpu   = Array{FloatT}(undef, 3, M)
    meta_cpu = Array{Int32}(undef,  5, M)   # row 5 = miss_link

    # Compute miss links for all nodes in one recursive pass
    miss_links = Vector{Int32}(undef, M)
    _assign_miss_links!(miss_links, cpu_bvh.nodes, 1, 0)

    for (k, nd) in enumerate(cpu_bvh.nodes)
        lo_cpu[1,k]   = FloatT(nd.aabb.lo[1])
        lo_cpu[2,k]   = FloatT(nd.aabb.lo[2])
        lo_cpu[3,k]   = FloatT(nd.aabb.lo[3])
        hi_cpu[1,k]   = FloatT(nd.aabb.hi[1])
        hi_cpu[2,k]   = FloatT(nd.aabb.hi[2])
        hi_cpu[3,k]   = FloatT(nd.aabb.hi[3])
        meta_cpu[1,k] = Int32(nd.left)
        meta_cpu[2,k] = Int32(nd.right)
        meta_cpu[3,k] = Int32(nd.tri_start)
        meta_cpu[4,k] = Int32(nd.tri_count)
        meta_cpu[5,k] = miss_links[k]
    end

    return FlatBVH(
        ArrayT(lo_cpu),
        ArrayT(hi_cpu),
        ArrayT(meta_cpu),
        ArrayT(Int32.(cpu_bvh.tri_idx)),
        ArrayT(FloatT.(cpu_bvh.tri_soup)),
        ArrayT(tri_group_cpu),
    )
end; export build_flat_bvh

"""
    build_flat_bvh_from_mesh(mesh, obstruction_groups, FloatT, ArrayT)
        -> FlatBVH or nothing

Merge triangle soups for the given groups, build a CPU BVH, and flatten it
to device arrays. Each triangle carries its physical group tag so the GPU
traversal can skip triangles belonging to the emitter or receiver group.

Returns `nothing` if no triangle geometry exists for the given groups.
"""
function build_flat_bvh_from_mesh(mesh,
                                   obstruction_groups::Vector{Int},
                                   ::Type{FloatT}, ArrayT) where FloatT
    isempty(obstruction_groups) && return nothing

    soups      = Array{Float64,3}[]
    group_tags = Int32[]

    for g in obstruction_groups
        haskey(mesh.group_tri_soup, g) || continue
        s  = mesh.group_tri_soup[g]
        nt = size(s, 3)
        push!(soups, s)
        append!(group_tags, fill(Int32(g), nt))
    end

    isempty(soups) && return nothing

    total  = sum(size(s, 3) for s in soups)
    merged = Array{Float64,3}(undef, 3, 3, total)
    t = 0
    for s in soups
        nt = size(s, 3)
        merged[:, :, t+1:t+nt] .= s
        t += nt
    end

    return build_flat_bvh(build_bvh(merged), group_tags, FloatT, ArrayT)
end; export build_flat_bvh_from_mesh

# ---------------------------------------------------------------------------
# Stackless GPU BVH traversal
#
# Called from inside @kernel bodies. Uses only scalar registers — no MVector,
# no thread-local stack memory. The miss_link (nodes_meta row 5) encodes the
# DFS successor of each node, so traversal is a simple linear walk:
#
#   AABB hit  + interior → descend into left child
#   AABB hit  + leaf     → test triangles, then follow miss_link
#   AABB miss            → follow miss_link (skip entire subtree)
#
# Arguments
# ---------
#   bvh_lo / bvh_hi / bvh_meta / bvh_tri_idx / bvh_tris / bvh_tri_group
#       — FlatBVH fields (passed individually for GPU kernel compatibility)
#   ox, oy, oz   — ray origin
#   dx, dy, dz   — ray direction (unit vector)
#   t_max        — segment length (maximum hit distance)
#   group_i/j    — emitter/receiver group tags; triangles in these groups
#                  are skipped to replicate the CPU per-pair exclusion logic
#
# Returns true if any non-excluded triangle is hit in (t_eps, t_max).
# ---------------------------------------------------------------------------

const _GPU_T_EPS = 1f-6   # Float32 literal; cast to T inside the function

@inline function gpu_intersect_bvh(bvh_lo, bvh_hi, bvh_meta,
                                    bvh_tri_idx, bvh_tris, bvh_tri_group,
                                    ox::T, oy::T, oz::T,
                                    dx::T, dy::T, dz::T,
                                    t_max::T,
                                    group_i::Int32, group_j::Int32)::Bool where T

    inv_dx = T(1) / dx
    inv_dy = T(1) / dy
    inv_dz = T(1) / dz
    t_eps  = T(_GPU_T_EPS)

    nidx = 1   # start at root

    @inbounds while nidx > 0

        # Ray–AABB slab test
        t1x = (T(bvh_lo[1, nidx]) - ox) * inv_dx
        t2x = (T(bvh_hi[1, nidx]) - ox) * inv_dx
        t1y = (T(bvh_lo[2, nidx]) - oy) * inv_dy
        t2y = (T(bvh_hi[2, nidx]) - oy) * inv_dy
        t1z = (T(bvh_lo[3, nidx]) - oz) * inv_dz
        t2z = (T(bvh_hi[3, nidx]) - oz) * inv_dz

        tentry = max(min(t1x, t2x), min(t1y, t2y), min(t1z, t2z), T(0))
        texit  = min(max(t1x, t2x), max(t1y, t2y), max(t1z, t2z), t_max)

        if tentry <= texit   # AABB hit

            tri_count = Int(bvh_meta[4, nidx])

            if tri_count > 0   # leaf node — test triangles
                tri_start = Int(bvh_meta[3, nidx])
                for k in tri_start : tri_start + tri_count - 1
                    tidx = Int(bvh_tri_idx[k])

                    # Skip emitter/receiver group triangles
                    tg = bvh_tri_group[tidx]
                    (tg == group_i || tg == group_j) && continue

                    # Triangle vertices [vertex 1..3, xyz 1..3, tri]
                    v0x = T(bvh_tris[1, 1, tidx]);  v0y = T(bvh_tris[1, 2, tidx]);  v0z = T(bvh_tris[1, 3, tidx])
                    v1x = T(bvh_tris[2, 1, tidx]);  v1y = T(bvh_tris[2, 2, tidx]);  v1z = T(bvh_tris[2, 3, tidx])
                    v2x = T(bvh_tris[3, 1, tidx]);  v2y = T(bvh_tris[3, 2, tidx]);  v2z = T(bvh_tris[3, 3, tidx])

                    # Möller–Trumbore (fully scalar, no SVector allocations)
                    e1x = v1x-v0x;  e1y = v1y-v0y;  e1z = v1z-v0z
                    e2x = v2x-v0x;  e2y = v2y-v0y;  e2z = v2z-v0z

                    hx = dy*e2z - dz*e2y
                    hy = dz*e2x - dx*e2z
                    hz = dx*e2y - dy*e2x
                    a  = e1x*hx + e1y*hy + e1z*hz
                    abs(a) < T(1e-10) && continue

                    f  = T(1) / a
                    sx = ox-v0x;  sy = oy-v0y;  sz = oz-v0z
                    u  = f * (sx*hx + sy*hy + sz*hz)
                    (u < T(0) || u > T(1)) && continue

                    qx = sy*e1z - sz*e1y
                    qy = sz*e1x - sx*e1z
                    qz = sx*e1y - sy*e1x
                    v  = f * (dx*qx + dy*qy + dz*qz)
                    (v < T(0) || u+v > T(1)) && continue

                    t = f * (e2x*qx + e2y*qy + e2z*qz)
                    t > t_eps && t < t_max && return true
                end
                # Leaf exhausted — follow miss link
                nidx = Int(bvh_meta[5, nidx])

            else   # interior node — descend into left child
                nidx = Int(bvh_meta[1, nidx])
            end

        else   # AABB miss — skip entire subtree
            nidx = Int(bvh_meta[5, nidx])
        end

    end   # while nidx > 0

    return false
end; export gpu_intersect_bvh

end # module GPUBVH
