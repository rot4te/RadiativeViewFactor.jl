# src/GPUDuffyKernels.jl
# ---------------------------------------------------------------------------
# Device-side Duffy transformation for element pairs that share a vertex or an
# edge (Quad4-Quad4 and Quad8-Quad8 pairs; see DuffyKernel.jl for the
# construction and the derivation of the regions, which are reproduced here
# region for region).
#
# Design
# ------
# The Duffy integrals are needed for only O(N) pairs, so they are not folded
# into the O(N^2) pair kernels. Those run first, giving every pair the plain
# quadrature (or Monte Carlo) value; this module then re-evaluates a *list* of
# pairs on the device and overwrites their entries in the raw matrix. That is
# the same overwrite the CPU path performs (`patch_adjacent_pairs_duffy!`), one
# thread per listed pair:
#
#   * two same-family quads sharing one corner node  -> 4-region vertex integral
#   * two same-family quads sharing two corner nodes -> 6-region edge integral
#   * anything else in the list (merely close pairs, triangles, mixed
#     families)                                       -> plain quadrature via
#                                                        GPUKernels._pair_quadrature
#
# The last case is what the Monte Carlo path needs: its near-pair list holds
# every pair within `factor` element diameters, most of which do not touch.
#
# Kernel-compatibility notes: no heap allocation, no Vector/Dict/Symbol in the
# device code (the CPU version's `for branch in (:plus, :minus)` is an integer
# loop here), and every literal is built from `T = eltype(coords)` so nothing
# is promoted to Float64 (Metal has no Float64).
# ---------------------------------------------------------------------------

module GPUDuffyKernels

using KernelAbstractions
using StaticArrays

import ..GPUBVH:     gpu_intersect_bvh
import ..GPUKernels: _pair_quadrature, _quad4_point_and_jac, _quad8_point_and_jac,
                     _vf_kernel

export launch_duffy_patch!

# ---------------------------------------------------------------------------
# Element evaluation on the unit square
# ---------------------------------------------------------------------------

# Physical point, unit normal and area element at (u,v) in [0,1]^2 for a Quad4
# (family code 2) or Quad8 (code 0). The x4 is the Jacobian of [0,1]^2 -> [-1,1]^2.
@inline function _eval_uv(coords, nodes_quad, fam, idx::Int, u::T, v::T) where T
    ξ = 2u - one(T)
    η = 2v - one(T)
    if fam == 2
        x, n, dA = _quad4_point_and_jac(coords, nodes_quad, idx, ξ, η)
    else
        x, n, dA = _quad8_point_and_jac(coords, nodes_quad, idx, ξ, η)
    end
    return x, n, dA * 4
end

# Local (u,v) -> reference (ξ,η) in [0,1]^2 for the edge running from corner `a`
# to corner `b` (u = 0 at a, u = 1 at b; v = 0 on the edge, v > 0 into the
# element). Quad corner layout: 1=(0,0) 2=(1,0) 3=(1,1) 4=(0,1).
@inline function _edge_local_to_ref(a::Int, b::Int, u::T, v::T) where T
    o = one(T)
    if     a == 1 && b == 2; return u,     v
    elseif a == 2 && b == 1; return o - u, v
    elseif a == 2 && b == 3; return o - v, u
    elseif a == 3 && b == 2; return o - v, o - u
    elseif a == 3 && b == 4; return o - u, o - v
    elseif a == 4 && b == 3; return u,     o - v
    elseif a == 4 && b == 1; return v,     o - u
    else                     return v,     u        # (1, 4)
    end
end

@inline function _eval_edge_uv(coords, nodes_quad, fam, idx::Int,
                                a::Int, b::Int, u::T, v::T) where T
    ξl, ηl = _edge_local_to_ref(a, b, u, v)
    return _eval_uv(coords, nodes_quad, fam, idx, ξl, ηl)
end

# True when the shadow ray xi -> xj is blocked. Mirrors the test in the pair
# kernel (GPUKernels._pair_quadrature).
@inline function _blocked(bvh_lo, bvh_hi, bvh_meta, bvh_tri_idx, bvh_tris, bvh_tri_group,
                           xi::SVector{3,T}, xj::SVector{3,T},
                           gi::Int32, gj::Int32) where T
    rx = xj[1] - xi[1]
    ry = xj[2] - xi[2]
    rz = xj[3] - xi[3]
    rlen = sqrt(rx*rx + ry*ry + rz*rz)
    rlen > T(1e-15) || return false
    inv_r = one(T) / rlen
    return gpu_intersect_bvh(bvh_lo, bvh_hi, bvh_meta, bvh_tri_idx, bvh_tris, bvh_tri_group,
                             xi[1], xi[2], xi[3], rx * inv_r, ry * inv_r, rz * inv_r,
                             rlen, gi, gj)
end

# ---------------------------------------------------------------------------
# COMMON_VERTEX: four "biggest-coordinate" regions, Jacobian rho^3
# ---------------------------------------------------------------------------
@inline function _vertex_integral(coords, nodes_quad, fam,
                                   idx_i::Int, idx_j::Int, ci::Int, cj::Int,
                                   gl_pts, gl_wts, gi::Int32, gj::Int32,
                                   bvh_lo, bvh_hi, bvh_meta,
                                   bvh_tri_idx, bvh_tris, bvh_tri_group, use_bvh::Bool)
    T  = eltype(coords)
    nq = length(gl_wts)
    o  = one(T)

    # Which side (0 or 1) of the unit square the shared corner sits on, per axis.
    u_hi_i = ci == 2 || ci == 3;  v_hi_i = ci == 3 || ci == 4
    u_hi_j = cj == 2 || cj == 3;  v_hi_j = cj == 3 || cj == 4

    Fij = zero(T)
    for region in 1:4
        for iρ in 1:nq, iη1 in 1:nq, iη2 in 1:nq, iη3 in 1:nq
            ρ  = gl_pts[iρ];  wρ  = gl_wts[iρ]
            η1 = gl_pts[iη1]; wη1 = gl_wts[iη1]
            η2 = gl_pts[iη2]; wη2 = gl_wts[iη2]
            η3 = gl_pts[iη3]; wη3 = gl_wts[iη3]

            if region == 1
                du, dv, ds, dt = ρ,      ρ * η1,  ρ * η2,  ρ * η3
            elseif region == 2
                du, dv, ds, dt = ρ * η1, ρ,       ρ * η2,  ρ * η3
            elseif region == 3
                du, dv, ds, dt = ρ * η1, ρ * η2,  ρ,       ρ * η3
            else
                du, dv, ds, dt = ρ * η1, ρ * η2,  ρ * η3,  ρ
            end

            # offsets always move *into* the square from the shared corner
            u = u_hi_i ? o - du : du
            v = v_hi_i ? o - dv : dv
            s = u_hi_j ? o - ds : ds
            t = v_hi_j ? o - dt : dt

            xi, ni, dAi = _eval_uv(coords, nodes_quad, fam, idx_i, u, v)
            xj, nj, dAj = _eval_uv(coords, nodes_quad, fam, idx_j, s, t)

            K = _vf_kernel(xi, ni, xj, nj)
            if K > zero(T)
                if !use_bvh || !_blocked(bvh_lo, bvh_hi, bvh_meta, bvh_tri_idx, bvh_tris,
                                          bvh_tri_group, xi, xj, gi, gj)
                    Fij += wρ * wη1 * wη2 * wη3 * (ρ * ρ * ρ) * K * dAi * dAj
                end
            end
        end
    end
    return Fij
end

# ---------------------------------------------------------------------------
# COMMON_EDGE: a free along-edge coordinate, two (u,s) triangles, and three
# "biggest-coordinate" regions in each; Jacobian (u or s) * rho^2
# ---------------------------------------------------------------------------
@inline function _edge_integral(coords, nodes_quad, fam,
                                 idx_i::Int, idx_j::Int,
                                 a_i::Int, b_i::Int, a_j::Int, b_j::Int,
                                 gl_pts, gl_wts, gi::Int32, gj::Int32,
                                 bvh_lo, bvh_hi, bvh_meta,
                                 bvh_tri_idx, bvh_tris, bvh_tri_group, use_bvh::Bool)
    T  = eltype(coords)
    nq = length(gl_wts)
    o  = one(T)

    Fij = zero(T)
    for branch in 1:2, region in 1:3
        for ia in 1:nq, iρ in 1:nq, iη1 in 1:nq, iη2 in 1:nq
            a  = gl_pts[ia];  wa  = gl_wts[ia]      # the free coordinate (u or s)
            ρ  = gl_pts[iρ];  wρ  = gl_wts[iρ]
            η1 = gl_pts[iη1]; wη1 = gl_wts[iη1]
            η2 = gl_pts[iη2]; wη2 = gl_wts[iη2]

            if region == 1
                w, v, t = ρ,      ρ * η1,  ρ * η2
            elseif region == 2
                w, v, t = ρ * η1, ρ,       ρ * η2
            else
                w, v, t = ρ * η1, ρ * η2,  ρ
            end

            if branch == 1
                u = a;  s = u * (o - w)
            else
                s = a;  u = s * (o - w)
            end

            xi, ni, dAi = _eval_edge_uv(coords, nodes_quad, fam, idx_i, a_i, b_i, u, v)
            xj, nj, dAj = _eval_edge_uv(coords, nodes_quad, fam, idx_j, a_j, b_j, s, t)

            K = _vf_kernel(xi, ni, xj, nj)
            if K > zero(T)
                if !use_bvh || !_blocked(bvh_lo, bvh_hi, bvh_meta, bvh_tri_idx, bvh_tris,
                                          bvh_tri_group, xi, xj, gi, gj)
                    Fij += wa * wρ * wη1 * wη2 * (a * ρ * ρ) * K * dAi * dAj
                end
            end
        end
    end
    return Fij
end

# Corner nodes (rows 1-4 of nodes_quad) two quads have in common, found in the
# same (ci outer, cj inner) order as `DuffyKernel.singularity_type`, so the
# first two matches pair up identically to the CPU classification.
@inline function _shared_corners(nodes_quad, idx_i::Int, idx_j::Int)
    n = 0;  ci1 = 0;  cj1 = 0;  ci2 = 0;  cj2 = 0
    for ci in 1:4, cj in 1:4
        if nodes_quad[ci, idx_i] == nodes_quad[cj, idx_j]
            n += 1
            if n == 1
                ci1 = ci;  cj1 = cj
            elseif n == 2
                ci2 = ci;  cj2 = cj
            end
        end
    end
    return n, ci1, cj1, ci2, cj2
end

# ---------------------------------------------------------------------------
# The patch kernel: one thread per listed pair
# ---------------------------------------------------------------------------
@kernel function _duffy_patch_kernel!(raw_out,
                                       coords, nodes_quad, nodes_tri,
                                       elem_family, elem_node_idx, elem_group,
                                       pair_i, pair_j, offset, npairs,
                                       gl_pts, gl_wts,
                                       quad_pts, quad_wts, tri_pts, tri_wts,
                                       bvh_lo, bvh_hi, bvh_meta,
                                       bvh_tri_idx, bvh_tris, bvh_tri_group,
                                       use_bvh::Bool)
    kg = @index(Global)
    k  = kg + offset

    if k <= npairs
        T  = eltype(coords)
        i  = Int(pair_i[k])
        j  = Int(pair_j[k])
        fi = elem_family[i]
        fj = elem_family[j]
        ni_idx = Int(elem_node_idx[i])
        nj_idx = Int(elem_node_idx[j])
        gi = Int32(elem_group[i])
        gj = Int32(elem_group[j])

        F = zero(T)
        done = false

        if fi == fj && (fi == 0 || fi == 2)
            n, ci1, cj1, ci2, cj2 = _shared_corners(nodes_quad, ni_idx, nj_idx)
            if n == 1
                F = _vertex_integral(coords, nodes_quad, fi, ni_idx, nj_idx, ci1, cj1,
                                     gl_pts, gl_wts, gi, gj,
                                     bvh_lo, bvh_hi, bvh_meta,
                                     bvh_tri_idx, bvh_tris, bvh_tri_group, use_bvh)
                done = true
            elseif n >= 2
                F = _edge_integral(coords, nodes_quad, fi, ni_idx, nj_idx,
                                   ci1, ci2, cj1, cj2,
                                   gl_pts, gl_wts, gi, gj,
                                   bvh_lo, bvh_hi, bvh_meta,
                                   bvh_tri_idx, bvh_tris, bvh_tri_group, use_bvh)
                done = true
            end
        end

        if !done
            F, _, _ = _pair_quadrature(coords, nodes_quad, nodes_tri,
                                        fi, fj, ni_idx, nj_idx, gi, gj,
                                        quad_pts, quad_wts, tri_pts, tri_wts,
                                        bvh_lo, bvh_hi, bvh_meta,
                                        bvh_tri_idx, bvh_tris, bvh_tri_group,
                                        use_bvh, true)
        end

        raw_out[i, j] = F
        raw_out[j, i] = F
    end
end

"""
    launch_duffy_patch!(raw_dev, ga, backend, pair_i, pair_j;
                        flat_bvh=nothing, groupsize=64, max_chunk_seconds=1.0,
                        first_chunk=nothing)

Overwrite `raw_dev[i,j]` and `raw_dev[j,i]` for each listed pair `(pair_i[k],
pair_j[k])` with the value `element_pair_view_factor_duffy` gives on the CPU:
the Duffy vertex or edge integral for two same-family quads sharing a corner
or an edge, plain quadrature (`nquad` from `ga`) for every other pair in the
list. `pair_i`/`pair_j` are device `Int32` arrays of element indices. Pass a
`FlatBVH` as `flat_bvh` to apply obstruction, with the same per-pair exclusion
of the two elements' own groups as the pair kernels.

Each pair is independent and touches only its own two entries of `raw_dev`.
Submitted in chunks whose size is retargeted from the measured rate, so no
single command buffer runs long enough to trip a GPU watchdog (Metal's, in
particular); `first_chunk` overrides the size of the first chunk (for tests) —
a Duffy pair costs `4 nquad^4` (vertex) or `6 nquad^4` (edge)
point-pair evaluations, orders of magnitude more than one pair of the
quadrature kernel.
"""
function launch_duffy_patch!(raw_dev, ga, backend, pair_i, pair_j;
                              flat_bvh          = nothing,
                              groupsize::Int    = 64,
                              max_chunk_seconds::Real = 1.0,
                              first_chunk::Union{Nothing,Int} = nothing)
    npairs = length(pair_i)
    npairs == 0 && return raw_dev
    FloatT = ga.FloatT

    use_bvh = flat_bvh !== nothing
    dummy   = KernelAbstractions.zeros(backend, FloatT, 1, 1)
    bvh_lo      = use_bvh ? flat_bvh.nodes_lo  : dummy
    bvh_hi      = use_bvh ? flat_bvh.nodes_hi  : dummy
    bvh_meta    = use_bvh ? flat_bvh.nodes_meta : KernelAbstractions.zeros(backend, Int32, 1, 1)
    bvh_tri_idx = use_bvh ? flat_bvh.tri_idx    : KernelAbstractions.zeros(backend, Int32, 1)
    bvh_tris    = use_bvh ? flat_bvh.tri_verts  : dummy
    bvh_tri_grp = use_bvh ? flat_bvh.tri_group  : KernelAbstractions.zeros(backend, Int32, 1)

    nq   = length(ga.gl_wts01)
    kern! = _duffy_patch_kernel!(backend, groupsize)

    # ~2e7 point-pair evaluations for the first chunk, then adapt from timings
    # (growth capped at 4x per step so one noisy timing cannot overshoot).
    chunk  = first_chunk === nothing ? max(round(Int, 2.0e7 / (6 * nq^4)), groupsize) :
                                       max(first_chunk, 1)
    offset = 0
    while offset < npairs
        rows = min(chunk, npairs - offset)
        t0 = time()
        kern!(raw_dev,
              ga.coords, ga.nodes_quad, ga.nodes_tri,
              ga.elem_family, ga.elem_node_idx, ga.elem_group,
              pair_i, pair_j, offset, npairs,
              ga.gl_pts01, ga.gl_wts01,
              ga.quad_pts, ga.quad_wts, ga.tri_pts, ga.tri_wts,
              bvh_lo, bvh_hi, bvh_meta, bvh_tri_idx, bvh_tris, bvh_tri_grp,
              use_bvh;
              ndrange=rows)
        KernelAbstractions.synchronize(backend)
        dt = time() - t0
        offset += rows
        rate  = rows / max(dt, 1e-4)
        chunk = max(1, round(Int, min(rate * max_chunk_seconds, chunk * 4)))
    end
    return raw_dev
end

end # module GPUDuffyKernels
