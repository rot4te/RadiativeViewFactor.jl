# src/GPUKernels.jl
# ---------------------------------------------------------------------------
# Backend-agnostic GPU kernels for view factor computation using
# KernelAbstractions.jl.
#
# Design
# ------
# The outer loop over element pairs (i,j) is mapped to a 2-D GPU grid.
# Each thread independently computes the full double quadrature loop for
# its assigned pair — this is embarrassingly parallel with no inter-thread
# communication. Each thread writes area_out[i]; since all threads in row i
# compute the same Ai value, the unconditional write is safe (last-write-wins
# gives the correct result and avoids the need for a separate area kernel).
#
# Obstruction support
# -------------------
# When use_bvh=true the pair kernel casts a shadow ray from each quadrature
# sample xi toward xj before accumulating the integrand.  The BVH data arrive
# as five flat device arrays (see GPUBVH.jl) so that no host-side struct needs
# to live on the GPU.  When use_bvh=false, those arrays are ignored entirely
# and the compiler eliminates the dead branch.
#
# Data layout (all plain arrays, no structs, for GPU compatibility)
# -----------------------------------------------------------------
# coords       : Float64/Float32  (3, N_nodes)
# nodes_quad   : Int32            (8, N_quad_elems)   — Quad8 node indices
# nodes_tri    : Int32            (6, N_tri_elems)    — Tri6  node indices
# elem_family  : Int8             (N_elems,)          — 0=quad, 1=tri
# elem_node_idx: Int32            (N_elems,)          — index into nodes_quad or nodes_tri
# quad_pts     : T               (2, NQ)             — reference quadrature points
# quad_wts     : T               (NQ,)               — quadrature weights
# raw_out      : T               (N, N)              — output: ∬K dAⱼ dAᵢ
# area_out     : T               (N,)                — output: element areas
# ---------------------------------------------------------------------------

module GPUKernels

using KernelAbstractions
using StaticArrays
using LinearAlgebra: cross, dot

import ..GPUBVH: gpu_intersect_bvh, FlatBVH
import ..Quadrature: gauss_legendre_2d

export build_gpu_arrays, launch_vf_kernel!

# ---------------------------------------------------------------------------
# Scalar kernel helpers (type-generic, work in both Float32 and Float64)
# ---------------------------------------------------------------------------

@inline function _quad8_shape(ξ::T, η::T) where T
    N1 = T(0.25)*(1-ξ)*(1-η)*(-ξ-η-1)
    N2 = T(0.25)*(1+ξ)*(1-η)*( ξ-η-1)
    N3 = T(0.25)*(1+ξ)*(1+η)*( ξ+η-1)
    N4 = T(0.25)*(1-ξ)*(1+η)*(-ξ+η-1)
    N5 = T(0.5)*(1-ξ^2)*(1-η)
    N6 = T(0.5)*(1+ξ)*(1-η^2)
    N7 = T(0.5)*(1-ξ^2)*(1+η)
    N8 = T(0.5)*(1-ξ)*(1-η^2)

    dN1dξ = T(0.25)*(1-η)*(2ξ+η)
    dN2dξ = T(0.25)*(1-η)*(2ξ-η)
    dN3dξ = T(0.25)*(1+η)*(2ξ+η)
    dN4dξ = T(0.25)*(1+η)*(2ξ-η)
    dN5dξ = -ξ*(1-η)
    dN6dξ =  T(0.5)*(1-η^2)
    dN7dξ = -ξ*(1+η)
    dN8dξ = -T(0.5)*(1-η^2)

    dN1dη = T(0.25)*(1-ξ)*(ξ+2η)
    dN2dη = T(0.25)*(1+ξ)*(-ξ+2η)
    dN3dη = T(0.25)*(1+ξ)*(ξ+2η)
    dN4dη = T(0.25)*(1-ξ)*(-ξ+2η)
    dN5dη = -T(0.5)*(1-ξ^2)
    dN6dη = -(1+ξ)*η
    dN7dη =  T(0.5)*(1-ξ^2)
    dN8dη = -(1-ξ)*η

    return (SVector{8,T}(N1,N2,N3,N4,N5,N6,N7,N8),
            SVector{8,T}(dN1dξ,dN2dξ,dN3dξ,dN4dξ,dN5dξ,dN6dξ,dN7dξ,dN8dξ),
            SVector{8,T}(dN1dη,dN2dη,dN3dη,dN4dη,dN5dη,dN6dη,dN7dη,dN8dη))
end

@inline function _tri6_shape(ξ::T, η::T) where T
    L1 = 1-ξ-η; L2 = ξ; L3 = η
    N    = SVector{6,T}(L1*(2L1-1), L2*(2L2-1), L3*(2L3-1), 4L1*L2, 4L2*L3, 4L1*L3)
    dNdξ = SVector{6,T}((4L1-1)*(-1), 4L2-1, 0, 4*(L2*(-1)+L1), 4L3, 4L3*(-1))
    dNdη = SVector{6,T}((4L1-1)*(-1), 0, 4L3-1, 4*L2*(-1), 4L2, 4*(L3*(-1)+L1))
    return N, dNdξ, dNdη
end

# Evaluate physical point and (normal, dA) for Quad8.
# Takes the full nodes_quad matrix and the 1-based column index to avoid
# creating a heap-allocated SubArray (view) inside a GPU kernel.
@inline function _quad8_point_and_jac(coords, nodes, col::Int, ξ::T, η::T) where T
    N, dNdξ, dNdη = _quad8_shape(ξ, η)
    x    = @SVector zeros(T, 3)
    dxdξ = @SVector zeros(T, 3)
    dxdη = @SVector zeros(T, 3)
    for a in 1:8
        na = nodes[a, col]
        xa = SVector{3,T}(coords[1,na], coords[2,na], coords[3,na])
        x    = x    +  N[a]*xa
        dxdξ = dxdξ + dNdξ[a]*xa
        dxdη = dxdη + dNdη[a]*xa
    end
    c  = cross(dxdξ, dxdη)
    dA = sqrt(dot(c,c))
    return x, c/dA, dA
end

# Evaluate physical point and (normal, dA) for Tri6.
@inline function _tri6_point_and_jac(coords, nodes, col::Int, ξ::T, η::T) where T
    N, dNdξ, dNdη = _tri6_shape(ξ, η)
    x    = @SVector zeros(T, 3)
    dxdξ = @SVector zeros(T, 3)
    dxdη = @SVector zeros(T, 3)
    for a in 1:6
        na = nodes[a, col]
        xa = SVector{3,T}(coords[1,na], coords[2,na], coords[3,na])
        x    = x    +  N[a]*xa
        dxdξ = dxdξ + dNdξ[a]*xa
        dxdη = dxdη + dNdη[a]*xa
    end
    c  = cross(dxdξ, dxdη)
    dA = sqrt(dot(c,c))
    return x, c/dA, dA
end

@inline function _quad4_shape(ξ::T, η::T) where T
    N    = SVector{4,T}(T(0.25)*(1-ξ)*(1-η), T(0.25)*(1+ξ)*(1-η),
                        T(0.25)*(1+ξ)*(1+η), T(0.25)*(1-ξ)*(1+η))
    dNdξ = SVector{4,T}(-T(0.25)*(1-η),  T(0.25)*(1-η),
                         T(0.25)*(1+η), -T(0.25)*(1+η))
    dNdη = SVector{4,T}(-T(0.25)*(1-ξ), -T(0.25)*(1+ξ),
                         T(0.25)*(1+ξ),  T(0.25)*(1-ξ))
    return N, dNdξ, dNdη
end

# Evaluate physical point and (normal, dA) for Quad4.
@inline function _quad4_point_and_jac(coords, nodes, col::Int, ξ::T, η::T) where T
    N, dNdξ, dNdη = _quad4_shape(ξ, η)
    x    = @SVector zeros(T, 3)
    dxdξ = @SVector zeros(T, 3)
    dxdη = @SVector zeros(T, 3)
    for a in 1:4
        na = nodes[a, col]
        xa = SVector{3,T}(coords[1,na], coords[2,na], coords[3,na])
        x    = x    +  N[a]*xa
        dxdξ = dxdξ + dNdξ[a]*xa
        dxdη = dxdη + dNdη[a]*xa
    end
    c  = cross(dxdξ, dxdη)
    dA = sqrt(dot(c,c))
    return x, c/dA, dA
end

# Evaluate physical point and (normal, dA) for Tri3 (linear triangle).
@inline function _tri3_point_and_jac(coords, nodes, col::Int, ξ::T, η::T) where T
    N    = SVector{3,T}(1-ξ-η, ξ, η)
    dNdξ = SVector{3,T}(-one(T), one(T), zero(T))
    dNdη = SVector{3,T}(-one(T), zero(T), one(T))
    x    = @SVector zeros(T, 3)
    dxdξ = @SVector zeros(T, 3)
    dxdη = @SVector zeros(T, 3)
    for a in 1:3
        na = nodes[a, col]
        xa = SVector{3,T}(coords[1,na], coords[2,na], coords[3,na])
        x    = x    +  N[a]*xa
        dxdξ = dxdξ + dNdξ[a]*xa
        dxdη = dxdη + dNdη[a]*xa
    end
    c  = cross(dxdξ, dxdη)
    dA = sqrt(dot(c,c))
    return x, c/dA, dA
end

# Family codes: 0=Quad8, 1=Tri6, 2=Quad4, 3=Tri3.
# Quad families share the tensor-product GL rule; tri families share the
# Dunavant rule — so the quadrature loop bound depends only on quad-vs-tri.
@inline _is_quad_fam(f) = (f == zero(f)) | (f == oftype(f, 2))

# Single sampler: given a family code and quadrature index, return the
# physical point, normal, area element, and weight, dispatching to the
# correct shape evaluator and reference rule. Inlines to branch-free-per-thread
# code once `f` is known.
@inline function _sample(f, p, coords, nq8, nq4, nt6, nt3, col,
                         quad_pts, quad_wts, tri_pts, tri_wts)
    if _is_quad_fam(f)
        ξ = quad_pts[1,p]; η = quad_pts[2,p]; w = quad_wts[p]
        x, n, dA = f == zero(f) ?
            _quad8_point_and_jac(coords, nq8, col, ξ, η) :
            _quad4_point_and_jac(coords, nq4, col, ξ, η)
        return x, n, dA, w
    else
        ξ = tri_pts[1,p]; η = tri_pts[2,p]; w = tri_wts[p]
        x, n, dA = f == oftype(f, 1) ?
            _tri6_point_and_jac(coords, nt6, col, ξ, η) :
            _tri3_point_and_jac(coords, nt3, col, ξ, η)
        return x, n, dA, w
    end
end

@inline function _vf_kernel(xi::SVector{3,T}, ni::SVector{3,T},
                              xj::SVector{3,T}, nj::SVector{3,T}) where T
    r_vec = xj - xi
    r²    = dot(r_vec, r_vec)
    r²    < T(1e-30) && return zero(T)
    r     = sqrt(r²)
    r̂     = r_vec / r
    cos_i = dot(ni,  r̂)
    cos_j = dot(nj, -r̂)
    (cos_i <= zero(T) || cos_j <= zero(T)) && return zero(T)
    return cos_i * cos_j / (T(π) * r²)
end

# ---------------------------------------------------------------------------
# The main KernelAbstractions kernel
# ---------------------------------------------------------------------------
# One thread per (i, j) pair with i < j (upper triangle).
# Grid is launched as (N, N) and threads where i >= j simply return early.
#
# When use_bvh=true the five bvh_* arrays are used for shadow-ray testing.
# When use_bvh=false those arrays are ignored (pass zero-size dummies).

@kernel function _vf_pair_kernel!(raw_out, area_out,
                                   coords,
                                   nodes_quad, nodes_tri,
                                   nodes_quad4, nodes_tri3,
                                   elem_family, elem_node_idx,
                                   elem_group,
                                   quad_pts, quad_wts,
                                   tri_pts,  tri_wts,
                                   N,
                                   bvh_lo, bvh_hi, bvh_meta,
                                   bvh_tri_idx, bvh_tris, bvh_tri_group,
                                   use_bvh::Bool)
    i, j = @index(Global, NTuple)

    if i <= N && j <= N && i < j

    T   = eltype(coords)
    nq  = length(quad_wts)
    nqt = length(tri_wts)

    gi = Int32(elem_group[i])
    gj = Int32(elem_group[j])

    Fij = zero(T)
    Ai  = zero(T)
    Aj  = zero(T)

    fi     = elem_family[i]
    fj     = elem_family[j]
    ni_idx = Int(elem_node_idx[i])
    nj_idx = Int(elem_node_idx[j])

    npi = _is_quad_fam(fi) ? nq : nqt
    npj = _is_quad_fam(fj) ? nq : nqt

    # ---- compute element areas independently ----
    for p in 1:npi
        _, _, dAi, wi = _sample(fi, p, coords, nodes_quad, nodes_quad4,
                                nodes_tri, nodes_tri3, ni_idx,
                                quad_pts, quad_wts, tri_pts, tri_wts)
        Ai += wi * dAi
    end
    for q in 1:npj
        _, _, dAj, wj = _sample(fj, q, coords, nodes_quad, nodes_quad4,
                                nodes_tri, nodes_tri3, nj_idx,
                                quad_pts, quad_wts, tri_pts, tri_wts)
        Aj += wj * dAj
    end

    # ---- double quadrature loop for view factor integral ----
    for p in 1:npi
        xi, nni, dAi, wi = _sample(fi, p, coords, nodes_quad, nodes_quad4,
                                   nodes_tri, nodes_tri3, ni_idx,
                                   quad_pts, quad_wts, tri_pts, tri_wts)

        inner = zero(T)
        for q in 1:npj
            xj, nnj, dAj, wj = _sample(fj, q, coords, nodes_quad, nodes_quad4,
                                       nodes_tri, nodes_tri3, nj_idx,
                                       quad_pts, quad_wts, tri_pts, tri_wts)

            K = _vf_kernel(xi, nni, xj, nnj)

            if use_bvh && K > zero(T)
                rx = xj[1] - xi[1]
                ry = xj[2] - xi[2]
                rz = xj[3] - xi[3]
                rlen = sqrt(rx*rx + ry*ry + rz*rz)
                if rlen > T(1e-15)
                    inv_r = T(1) / rlen
                    if gpu_intersect_bvh(bvh_lo, bvh_hi, bvh_meta,
                                         bvh_tri_idx, bvh_tris, bvh_tri_group,
                                         xi[1], xi[2], xi[3],
                                         rx * inv_r, ry * inv_r, rz * inv_r,
                                         rlen, gi, gj)
                        K = zero(T)
                    end
                end
            end

            inner += wj * K * dAj
        end

        Fij += wi * inner * dAi
    end

    raw_out[i, j] = Fij
    raw_out[j, i] = Fij
    # Write area_out[i] and area_out[j]: all threads in row i compute the
    # same Ai, and all threads in column j compute the same Aj, so any write
    # gives the correct result. Writing both ensures every element's area is
    # covered — element N never appears as i (no j > N exists) but always
    # appears as j (thread i=N-1, j=N covers it).
    area_out[i] = Ai
    area_out[j] = Aj

    end # if i <= N && j <= N && i < j
end

# ---------------------------------------------------------------------------
# Host-side helpers: flatten MeshData to plain arrays for GPU transfer
# ---------------------------------------------------------------------------

"""
    build_gpu_arrays(mesh, nquad, ArrayT, FloatT) -> NamedTuple

Flatten MeshData into plain typed arrays ready for GPU transfer.
`ArrayT` is the array constructor (e.g. `CuArray`, `MtlArray`, `Array`).
`FloatT` is `Float32` (Metal) or `Float64` (CUDA/CPU).
"""
function build_gpu_arrays(mesh, nquad::Int, ArrayT, FloatT)
    import_mods = Base.loaded_modules   # just for readability below

    elems      = mesh.surface_elems
    coords_cpu = FloatT.(mesh.coords)
    N          = length(elems)

    # Separate elements by family, record per-element family, local index, and group.
    # Family codes: 0=Quad8, 1=Tri6, 2=Quad4, 3=Tri3.
    quad8_node_lists = Vector{Vector{Int32}}()
    tri6_node_lists  = Vector{Vector{Int32}}()
    quad4_node_lists = Vector{Vector{Int32}}()
    tri3_node_lists  = Vector{Vector{Int32}}()
    elem_family     = Vector{Int8}(undef, N)
    elem_node_idx   = Vector{Int32}(undef, N)
    elem_group_cpu  = Vector{Int32}(undef, N)

    for (i, el) in enumerate(elems)
        elem_group_cpu[i] = Int32(el.group)
        if el.family === :quad
            push!(quad8_node_lists, Int32.(el.nodes))
            elem_family[i]   = Int8(0)
            elem_node_idx[i] = Int32(length(quad8_node_lists))
        elseif el.family === :tri
            push!(tri6_node_lists, Int32.(el.nodes))
            elem_family[i]   = Int8(1)
            elem_node_idx[i] = Int32(length(tri6_node_lists))
        elseif el.family === :quad4
            push!(quad4_node_lists, Int32.(el.nodes))
            elem_family[i]   = Int8(2)
            elem_node_idx[i] = Int32(length(quad4_node_lists))
        elseif el.family === :tri3
            push!(tri3_node_lists, Int32.(el.nodes))
            elem_family[i]   = Int8(3)
            elem_node_idx[i] = Int32(length(tri3_node_lists))
        else
            error("GPU backend does not support element family $(el.family) " *
                  "(curve/Line elements are CPU-only).")
        end
    end

    # Pack node lists into per-family matrices (n_nodes × N_family)
    nodes_quad_cpu  = !isempty(quad8_node_lists) ?
        reduce(hcat, quad8_node_lists) : zeros(Int32, 8, 0)
    nodes_tri_cpu   = !isempty(tri6_node_lists)  ?
        reduce(hcat, tri6_node_lists)  : zeros(Int32, 6, 0)
    nodes_quad4_cpu = !isempty(quad4_node_lists) ?
        reduce(hcat, quad4_node_lists) : zeros(Int32, 4, 0)
    nodes_tri3_cpu  = !isempty(tri3_node_lists)  ?
        reduce(hcat, tri3_node_lists)  : zeros(Int32, 3, 0)

    # Quadrature rules
    gl_rule   = gauss_legendre_2d(nquad)
    quad_pts_cpu = FloatT.(gl_rule.points)
    quad_wts_cpu = FloatT.(gl_rule.weights)

    # Dunavant triangle rule (reuse logic from ViewFactorKernel)
    tri_rule = _dunavant_rule(nquad, FloatT)

    # Transfer to device
    return (
        coords        = ArrayT(coords_cpu),
        nodes_quad    = ArrayT(nodes_quad_cpu),
        nodes_tri     = ArrayT(nodes_tri_cpu),
        nodes_quad4   = ArrayT(nodes_quad4_cpu),
        nodes_tri3    = ArrayT(nodes_tri3_cpu),
        elem_family   = ArrayT(elem_family),
        elem_node_idx = ArrayT(elem_node_idx),
        elem_group    = ArrayT(elem_group_cpu),
        quad_pts      = ArrayT(quad_pts_cpu),
        quad_wts      = ArrayT(quad_wts_cpu),
        tri_pts       = ArrayT(tri_rule.points),
        tri_wts       = ArrayT(tri_rule.weights),
        N             = N,
        FloatT        = FloatT,
    )
end

function _dunavant_rule(nquad::Int, ::Type{T}) where T
    if nquad <= 1
        pts = T[1/3; 1/3][:, :]
        return (points=reshape(T[1/3, 1/3], 2, 1), weights=T[0.5])
    elseif nquad == 2
        pts = T[1/6 2/3 1/6; 1/6 1/6 2/3]
        return (points=pts, weights=fill(T(0.5/3), 3))
    elseif nquad == 3
        a1=T(0.101286507323456); b1=1-2a1
        a2=T(0.470142064105115); b2=1-2a2
        pts = T[a1 b1 a1 a2 b2 a2 1/3; a1 a1 b1 a2 a2 b2 1/3]
        w1=T(0.125939180544827/2); w2=T(0.132394440720100/2); w3=T(0.225/2)
        return (points=pts, weights=T[w1,w1,w1,w2,w2,w2,w3])
    else   # 13-point Dunavant degree 7 (weights for reference area 1/2)
        cen=T(1/3)
        a1=T(0.260345966079038); b1=1-2a1
        a2=T(0.065130102902216); b2=1-2a2
        a3=T(0.048690315425316); b3=T(0.312865496004875); c3=1-a3-b3
        # Orbit C: all 6 distinct (ξ,η) permutations of barycentric (a3,b3,c3).
        pts = T[a1 b1 a1 a2 b2 a2 a3 b3 a3 c3 b3 c3 cen;
                a1 a1 b1 a2 a2 b2 b3 a3 c3 a3 c3 b3 cen]
        wc=T(-0.149570044467670/2); wA=T(0.175615257433204/2)
        wB=T(0.053347235608839/2);  wC=T(0.077113760890257/2)
        return (points=pts, weights=T[wA,wA,wA,wB,wB,wB,wC,wC,wC,wC,wC,wC,wc])
    end
end

"""
    launch_vf_kernel!(gpu_arrays, backend; groupsize=16, flat_bvh=nothing)
        -> (raw_out, area_out)

Launch the view factor pair kernel on `backend`.

Returns device arrays `raw_out` (N×N, raw double integrals) and `area_out`
(N, element areas).  Both are filled by the single pair kernel — no separate
area kernel is needed.

Pass a `FlatBVH` as `flat_bvh` to enable obstruction checking; omit or pass
`nothing` for unobstructed computation.
"""
function launch_vf_kernel!(ga, backend;
                            groupsize::Int = 16,
                            flat_bvh       = nothing)
    N      = ga.N
    FloatT = ga.FloatT
    raw_out  = fill!(similar(ga.coords, FloatT, N, N), zero(FloatT))
    area_out = fill!(similar(ga.coords, FloatT, N),    zero(FloatT))

    # Prepare BVH arrays for the pair kernel.
    # When flat_bvh is nothing we pass zero-size dummies and use_bvh=false so
    # the compiler can eliminate the entire shadow-ray branch.
    if flat_bvh !== nothing
        bvh_lo        = flat_bvh.nodes_lo
        bvh_hi        = flat_bvh.nodes_hi
        bvh_meta      = flat_bvh.nodes_meta
        bvh_tri_idx   = flat_bvh.tri_idx
        bvh_tris      = flat_bvh.tri_verts
        bvh_tri_group = flat_bvh.tri_group
        use_bvh       = true
    else
        bvh_lo        = similar(ga.coords, FloatT, 3, 0)
        bvh_hi        = similar(ga.coords, FloatT, 3, 0)
        bvh_meta      = similar(ga.elem_group, Int32, 4, 0)
        bvh_tri_idx   = similar(ga.elem_group, Int32, 0)
        bvh_tris      = similar(ga.coords, FloatT, 3, 3, 0)
        bvh_tri_group = similar(ga.elem_group, Int32, 0)
        use_bvh       = false
    end

    # Pair kernel: 2-D, one thread per (i,j) with i<j
    pair_kern! = _vf_pair_kernel!(backend, (groupsize, groupsize))::Any
    pair_kern!(raw_out, area_out, ga.coords,
               ga.nodes_quad, ga.nodes_tri,
               ga.nodes_quad4, ga.nodes_tri3,
               ga.elem_family, ga.elem_node_idx,
               ga.elem_group,
               ga.quad_pts, ga.quad_wts,
               ga.tri_pts,  ga.tri_wts,
               N,
               bvh_lo, bvh_hi, bvh_meta,
               bvh_tri_idx, bvh_tris, bvh_tri_group,
               use_bvh;
               ndrange=(N, N))

    KernelAbstractions.synchronize(backend)
    return raw_out, area_out
end

end # module GPUKernels
