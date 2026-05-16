# src/GPUBVH.jl
# ---------------------------------------------------------------------------
# GPU-friendly BVH: flattens the CPU BVHTree into plain typed arrays that can
# be transferred to any KernelAbstractions device, plus the @inline traversal
# function called from inside GPU kernels.
#
# Data layout
# -----------
# nodes_lo   : FloatT (3, M)   — AABB lower bounds per node
# nodes_hi   : FloatT (3, M)   — AABB upper bounds per node
# nodes_meta : Int32  (4, M)   — [left, right, tri_start, tri_count] per node
#                                 left == 0 iff the node is a leaf
# tri_idx    : Int32  (N,)     — sorted triangle permutation (from BVH build)
# tri_verts  : FloatT (3,3,N)  — triangle vertices; dim1=vertex, dim2=xyz,
#                                 dim3=triangle (same layout as BVHTree.tri_soup)
# tri_group  : Int32  (N,)     — physical group tag of each triangle, indexed
#                                 by original (pre-permutation) triangle index,
#                                 so bvh_tri_group[tidx] matches bvh_tris[:,:,tidx]
# ---------------------------------------------------------------------------

module GPUBVH

using StaticArrays

import ..BVH: BVHTree, build_bvh

export FlatBVH, build_flat_bvh, build_flat_bvh_from_mesh, gpu_intersect_bvh

# ---------------------------------------------------------------------------
# Flat BVH struct
# ---------------------------------------------------------------------------

struct FlatBVH{FA, IA}
    nodes_lo   :: FA   # (3, M)    FloatT
    nodes_hi   :: FA   # (3, M)    FloatT
    nodes_meta :: IA   # (4, M)    Int32
    tri_idx    :: IA   # (N,)      Int32
    tri_verts  :: FA   # (3, 3, N) FloatT
    tri_group  :: IA   # (N,)      Int32 — physical group tag per triangle
end

# ---------------------------------------------------------------------------
# CPU BVHTree → FlatBVH (transfers to device via ArrayT)
#
# tri_group_cpu must be a Vector{Int32} of length size(cpu_bvh.tri_soup, 3),
# indexed by original triangle index (before BVH permutation).
# ---------------------------------------------------------------------------

function build_flat_bvh(cpu_bvh::BVHTree,
                         tri_group_cpu::Vector{Int32},
                         ::Type{FloatT}, ArrayT) where FloatT
    M = length(cpu_bvh.nodes)

    lo_cpu   = Array{FloatT}(undef, 3, M)
    hi_cpu   = Array{FloatT}(undef, 3, M)
    meta_cpu = Array{Int32}(undef,  4, M)

    for (k, nd) in enumerate(cpu_bvh.nodes)
        lo_cpu[1,k] = FloatT(nd.aabb.lo[1])
        lo_cpu[2,k] = FloatT(nd.aabb.lo[2])
        lo_cpu[3,k] = FloatT(nd.aabb.lo[3])
        hi_cpu[1,k] = FloatT(nd.aabb.hi[1])
        hi_cpu[2,k] = FloatT(nd.aabb.hi[2])
        hi_cpu[3,k] = FloatT(nd.aabb.hi[3])
        meta_cpu[1,k] = Int32(nd.left)
        meta_cpu[2,k] = Int32(nd.right)
        meta_cpu[3,k] = Int32(nd.tri_start)
        meta_cpu[4,k] = Int32(nd.tri_count)
    end

    return FlatBVH(
        ArrayT(lo_cpu),
        ArrayT(hi_cpu),
        ArrayT(meta_cpu),
        ArrayT(Int32.(cpu_bvh.tri_idx)),
        ArrayT(FloatT.(cpu_bvh.tri_soup)),
        ArrayT(tri_group_cpu),
    )
end

"""
    build_flat_bvh_from_mesh(mesh, obstruction_groups, FloatT, ArrayT)
        -> FlatBVH or nothing

Merge the triangle soups for the given groups, build a CPU BVH, and flatten it
to device arrays.  Each triangle carries its physical group tag so that the GPU
traversal can skip triangles belonging to the emitter or receiver element's group,
matching the per-pair exclusion behaviour of the CPU path.

Returns `nothing` if no triangle geometry is available for the given groups.
"""
function build_flat_bvh_from_mesh(mesh, obstruction_groups::Vector{Int},
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
end

# ---------------------------------------------------------------------------
# GPU-safe BVH ray traversal
#
# Designed to be called from inside a @kernel body.  Uses a fixed-size
# MVector stack (on-thread local memory) and only scalar indexing, so it
# compiles cleanly for CUDA and Metal.
#
# Arguments:
#   bvh_lo / bvh_hi / bvh_meta / bvh_tri_idx / bvh_tris — FlatBVH geometry
#   bvh_tri_group  — group tag per triangle (indexed by original triangle index)
#   ox,oy,oz       — ray origin
#   dx,dy,dz       — ray direction (unit vector)
#   t_max          — maximum hit distance (segment length)
#   group_i        — physical group tag of the emitter element
#   group_j        — physical group tag of the receiver element
#
# Triangles whose group matches group_i or group_j are skipped, replicating
# the per-pair exclusion behaviour of the CPU path.
#
# Returns true if any non-excluded triangle is hit between t_eps and t_max.
# ---------------------------------------------------------------------------

@inline function gpu_intersect_bvh(bvh_lo, bvh_hi, bvh_meta,
                                    bvh_tri_idx, bvh_tris, bvh_tri_group,
                                    ox::T, oy::T, oz::T,
                                    dx::T, dy::T, dz::T,
                                    t_max::T,
                                    group_i::Int32, group_j::Int32)::Bool where T
    inv_dx = T(1) / dx
    inv_dy = T(1) / dy
    inv_dz = T(1) / dz
    t_eps  = T(1e-6)     # offset to avoid self-hit at the segment endpoints

    stack    = MVector{64, Int32}(undef)
    stack[1] = Int32(1)
    sp       = 1

    @inbounds while sp > 0
        nidx = Int(stack[sp]); sp -= 1

        # Ray–AABB slab test
        t1x = (T(bvh_lo[1, nidx]) - ox) * inv_dx
        t2x = (T(bvh_hi[1, nidx]) - ox) * inv_dx
        t1y = (T(bvh_lo[2, nidx]) - oy) * inv_dy
        t2y = (T(bvh_hi[2, nidx]) - oy) * inv_dy
        t1z = (T(bvh_lo[3, nidx]) - oz) * inv_dz
        t2z = (T(bvh_hi[3, nidx]) - oz) * inv_dz

        tentry = max(min(t1x, t2x), min(t1y, t2y), min(t1z, t2z), T(0))
        texit  = min(max(t1x, t2x), max(t1y, t2y), max(t1z, t2z), t_max)
        tentry > texit && continue

        tri_count = Int(bvh_meta[4, nidx])

        if tri_count > 0   # leaf node
            tri_start = Int(bvh_meta[3, nidx])
            for k in tri_start : tri_start + tri_count - 1
                tidx = Int(bvh_tri_idx[k])

                # Skip triangles belonging to the emitter or receiver group
                tg = bvh_tri_group[tidx]
                (tg == group_i || tg == group_j) && continue

                # Vertex coordinates (dim1=vertex, dim2=xyz, dim3=tri)
                v0x = T(bvh_tris[1, 1, tidx])
                v0y = T(bvh_tris[1, 2, tidx])
                v0z = T(bvh_tris[1, 3, tidx])
                v1x = T(bvh_tris[2, 1, tidx])
                v1y = T(bvh_tris[2, 2, tidx])
                v1z = T(bvh_tris[2, 3, tidx])
                v2x = T(bvh_tris[3, 1, tidx])
                v2y = T(bvh_tris[3, 2, tidx])
                v2z = T(bvh_tris[3, 3, tidx])

                # Möller–Trumbore (scalar, no SVector allocations)
                e1x = v1x - v0x;  e1y = v1y - v0y;  e1z = v1z - v0z
                e2x = v2x - v0x;  e2y = v2y - v0y;  e2z = v2z - v0z

                hx = dy * e2z - dz * e2y
                hy = dz * e2x - dx * e2z
                hz = dx * e2y - dy * e2x
                a  = e1x * hx + e1y * hy + e1z * hz
                abs(a) < T(1e-10) && continue

                f  = T(1) / a
                sx = ox - v0x;  sy = oy - v0y;  sz = oz - v0z
                u  = f * (sx * hx + sy * hy + sz * hz)
                (u < T(0) || u > T(1)) && continue

                qx = sy * e1z - sz * e1y
                qy = sz * e1x - sx * e1z
                qz = sx * e1y - sy * e1x
                v  = f * (dx * qx + dy * qy + dz * qz)
                (v < T(0) || u + v > T(1)) && continue

                t  = f * (e2x * qx + e2y * qy + e2z * qz)
                t > t_eps && t < t_max && return true
            end

        else   # interior node — push children
            sp += 1;  stack[sp] = bvh_meta[1, nidx]   # left
            sp += 1;  stack[sp] = bvh_meta[2, nidx]   # right
        end
    end

    return false
end

end # module GPUBVH
