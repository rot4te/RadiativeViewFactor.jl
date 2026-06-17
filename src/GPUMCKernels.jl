# src/GPUMCKernels.jl
# ---------------------------------------------------------------------------
# GPU Monte Carlo view factor kernel using KernelAbstractions.
#
# Each thread handles one element pair (i,j) and draws n_samples stratified
# random pairs from the two elements, accumulating the MC estimate.
#
# Random number generation on GPU
# --------------------------------
# KernelAbstractions kernels cannot use Julia's AbstractRNG directly (it
# requires heap allocation).  Instead we use a minimal inline xorshift64
# PRNG seeded per-thread from the global seed and thread index.  This gives
# independent pseudo-random streams per thread with no memory overhead.
#
# Stratified sampling
# -------------------
# The n_samples points are divided into s×s strata where s = floor(√n).
# The remaining n - s² points are drawn from the full reference domain.
# This matches the CPU MCKernel stratification exactly.
# ---------------------------------------------------------------------------

module GPUMCKernels

using KernelAbstractions
using StaticArrays
using LinearAlgebra: dot, cross, norm

import ..GPUBVH: gpu_intersect_bvh, FlatBVH
import ..Quadrature: gauss_legendre_2d

export build_gpu_mc_arrays, launch_mc_kernel!

# ---------------------------------------------------------------------------
# Inline xorshift64 PRNG (no allocation, safe inside @kernel)
# ---------------------------------------------------------------------------

@inline function _xorshift64(state::UInt64)::Tuple{Float64, UInt64}
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return (Float64(state >> 11) / Float64(0x001FFFFFFFFFFFFF), state)
end

@inline function _init_rng(global_seed::UInt64, thread_id::Int)::UInt64
    # Mix thread id into seed using splitmix64 to ensure different streams
    z = global_seed + UInt64(thread_id) * 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    return z ⊻ (z >> 31)
end

# ---------------------------------------------------------------------------
# Geometry helpers (same as GPUKernels.jl, duplicated for self-containment)
# ---------------------------------------------------------------------------

@inline function _quad8_shape_gpu(ξ::T, η::T) where T
    N1=T(0.25)*(1-ξ)*(1-η)*(-ξ-η-1); N2=T(0.25)*(1+ξ)*(1-η)*(ξ-η-1)
    N3=T(0.25)*(1+ξ)*(1+η)*(ξ+η-1);  N4=T(0.25)*(1-ξ)*(1+η)*(-ξ+η-1)
    N5=T(0.5)*(1-ξ^2)*(1-η);          N6=T(0.5)*(1+ξ)*(1-η^2)
    N7=T(0.5)*(1-ξ^2)*(1+η);          N8=T(0.5)*(1-ξ)*(1-η^2)
    dN1dξ=T(0.25)*(1-η)*(2ξ+η);       dN2dξ=T(0.25)*(1-η)*(2ξ-η)
    dN3dξ=T(0.25)*(1+η)*(2ξ+η);       dN4dξ=T(0.25)*(1+η)*(2ξ-η)
    dN5dξ=-ξ*(1-η);                    dN6dξ=T(0.5)*(1-η^2)
    dN7dξ=-ξ*(1+η);                    dN8dξ=-T(0.5)*(1-η^2)
    dN1dη=T(0.25)*(1-ξ)*(ξ+2η);       dN2dη=T(0.25)*(1+ξ)*(-ξ+2η)
    dN3dη=T(0.25)*(1+ξ)*(ξ+2η);       dN4dη=T(0.25)*(1-ξ)*(-ξ+2η)
    dN5dη=-T(0.5)*(1-ξ^2);            dN6dη=-(1+ξ)*η
    dN7dη=T(0.5)*(1-ξ^2);             dN8dη=-(1-ξ)*η
    return (SVector{8,T}(N1,N2,N3,N4,N5,N6,N7,N8),
            SVector{8,T}(dN1dξ,dN2dξ,dN3dξ,dN4dξ,dN5dξ,dN6dξ,dN7dξ,dN8dξ),
            SVector{8,T}(dN1dη,dN2dη,dN3dη,dN4dη,dN5dη,dN6dη,dN7dη,dN8dη))
end

@inline function _quad8_eval(coords, nodes_quad, ni_idx::Int, ξ::T, η::T) where T
    N, dNdξ, dNdη = _quad8_shape_gpu(ξ, η)
    x=@SVector zeros(T,3); dxdξ=@SVector zeros(T,3); dxdη=@SVector zeros(T,3)
    for a in 1:8
        na=nodes_quad[a, ni_idx]
        xa=SVector{3,T}(coords[1,na], coords[2,na], coords[3,na])
        x=x+N[a]*xa; dxdξ=dxdξ+dNdξ[a]*xa; dxdη=dxdη+dNdη[a]*xa
    end
    c=cross(dxdξ,dxdη); dA=sqrt(dot(c,c))
    return x, c/dA, dA
end

@inline function _tri6_eval(coords, nodes_tri, ni_idx::Int, ξ::T, η::T) where T
    L1=1-ξ-η; L2=ξ; L3=η
    N=SVector{6,T}(L1*(2L1-1),L2*(2L2-1),L3*(2L3-1),4L1*L2,4L2*L3,4L1*L3)
    dNdξ=SVector{6,T}((4L1-1)*T(-1),4L2-1,T(0),4*(L2*T(-1)+L1),4L3,4L3*T(-1))
    dNdη=SVector{6,T}((4L1-1)*T(-1),T(0),4L3-1,4*L2*T(-1),4L2,4*(L3*T(-1)+L1))
    x=@SVector zeros(T,3); dxdξ=@SVector zeros(T,3); dxdη=@SVector zeros(T,3)
    for a in 1:6
        na=nodes_tri[a, ni_idx]
        xa=SVector{3,T}(coords[1,na],coords[2,na],coords[3,na])
        x=x+N[a]*xa; dxdξ=dxdξ+dNdξ[a]*xa; dxdη=dxdη+dNdη[a]*xa
    end
    c=cross(dxdξ,dxdη); dA=sqrt(dot(c,c))
    return x, c/dA, dA
end

@inline function _quad4_eval(coords, nodes_quad4, ni_idx::Int, ξ::T, η::T) where T
    N    = SVector{4,T}(T(0.25)*(1-ξ)*(1-η), T(0.25)*(1+ξ)*(1-η),
                        T(0.25)*(1+ξ)*(1+η), T(0.25)*(1-ξ)*(1+η))
    dNdξ = SVector{4,T}(-T(0.25)*(1-η),  T(0.25)*(1-η),
                         T(0.25)*(1+η), -T(0.25)*(1+η))
    dNdη = SVector{4,T}(-T(0.25)*(1-ξ), -T(0.25)*(1+ξ),
                         T(0.25)*(1+ξ),  T(0.25)*(1-ξ))
    x=@SVector zeros(T,3); dxdξ=@SVector zeros(T,3); dxdη=@SVector zeros(T,3)
    for a in 1:4
        na=nodes_quad4[a, ni_idx]
        xa=SVector{3,T}(coords[1,na],coords[2,na],coords[3,na])
        x=x+N[a]*xa; dxdξ=dxdξ+dNdξ[a]*xa; dxdη=dxdη+dNdη[a]*xa
    end
    c=cross(dxdξ,dxdη); dA=sqrt(dot(c,c))
    return x, c/dA, dA
end

@inline function _tri3_eval(coords, nodes_tri3, ni_idx::Int, ξ::T, η::T) where T
    N    = SVector{3,T}(1-ξ-η, ξ, η)
    dNdξ = SVector{3,T}(-one(T), one(T), zero(T))
    dNdη = SVector{3,T}(-one(T), zero(T), one(T))
    x=@SVector zeros(T,3); dxdξ=@SVector zeros(T,3); dxdη=@SVector zeros(T,3)
    for a in 1:3
        na=nodes_tri3[a, ni_idx]
        xa=SVector{3,T}(coords[1,na],coords[2,na],coords[3,na])
        x=x+N[a]*xa; dxdξ=dxdξ+dNdξ[a]*xa; dxdη=dxdη+dNdη[a]*xa
    end
    c=cross(dxdξ,dxdη); dA=sqrt(dot(c,c))
    return x, c/dA, dA
end

@inline function _vf_kernel_gpu(xi::SVector{3,T}, ni::SVector{3,T},
                                  xj::SVector{3,T}, nj::SVector{3,T}) where T
    rv=xj-xi; r2=dot(rv,rv)
    r2 < T(1e-30) && return zero(T)
    r=sqrt(r2); rhat=rv/r
    ci=dot(ni,rhat); cj=dot(nj,-rhat)
    (ci<=zero(T) || cj<=zero(T)) && return zero(T)
    return ci*cj/(T(π)*r2)
end

# ---------------------------------------------------------------------------
# MC kernel
# ---------------------------------------------------------------------------

@kernel function _mc_pair_kernel!(raw_out, area_out,
                                   coords,
                                   nodes_quad, nodes_tri,
                                   nodes_quad4, nodes_tri3,
                                   elem_family, elem_node_idx,
                                   n_samples::Int,
                                   global_seed::UInt64,
                                   use_bvh::Bool,
                                   bvh_lo, bvh_hi, bvh_meta,
                                   bvh_tri_idx, bvh_tris, bvh_tri_group,
                                   N::Int)
    i, j = @index(Global, NTuple)

    if i <= N && j <= N && i < j

    T        = eltype(coords)
    fi       = Int(elem_family[i]);  fj = Int(elem_family[j])
    ni_idx   = Int(elem_node_idx[i]); nj_idx = Int(elem_node_idx[j])
    gi       = Int32(0);  gj = Int32(0)   # group tags not needed: BVH exclusion
    # handled via bvh_tri_group in gpu_intersect_bvh

    thread_id = (i-1)*N + j
    rng_state = _init_rng(global_seed, thread_id)

    Ai = zero(T); Aj = zero(T); K_sum = zero(T)

    s = floor(Int, sqrt(n_samples))

    # ---- Stratified samples ----
    sample_k = 0
    for si in 0:s-1
        for sj in 0:s-1
            sample_k += 1

            # Sample on element i
            u1, rng_state = _xorshift64(rng_state)
            u2, rng_state = _xorshift64(rng_state)
            xi, nni, dAi = _sample_on_element(coords, nodes_quad, nodes_tri,
                                               nodes_quad4, nodes_tri3,
                                               fi, ni_idx,
                                               T((si + u1)/s), T((sj + u2)/s))
            Ai += dAi

            # Sample on element j
            u3, rng_state = _xorshift64(rng_state)
            u4, rng_state = _xorshift64(rng_state)
            xj, nnj, dAj = _sample_on_element(coords, nodes_quad, nodes_tri,
                                               nodes_quad4, nodes_tri3,
                                               fj, nj_idx,
                                               T((si + u3)/s), T((sj + u4)/s))
            Aj += dAj

            K = _vf_kernel_gpu(xi, nni, xj, nnj)
            if K > zero(T) && use_bvh
                rx=xj[1]-xi[1]; ry=xj[2]-xi[2]; rz=xj[3]-xi[3]
                rlen=sqrt(rx*rx+ry*ry+rz*rz)
                if rlen > T(1e-15)
                    inv_r=T(1)/rlen
                    if gpu_intersect_bvh(bvh_lo,bvh_hi,bvh_meta,
                                          bvh_tri_idx,bvh_tris,bvh_tri_group,
                                          xi[1],xi[2],xi[3],
                                          rx*inv_r,ry*inv_r,rz*inv_r,
                                          rlen, gi, gj)
                        K = zero(T)
                    end
                end
            end

            K_sum += K * dAi * dAj
        end
    end

    # Remaining samples from full reference domain
    for _ in sample_k+1:n_samples
        u1, rng_state = _xorshift64(rng_state)
        u2, rng_state = _xorshift64(rng_state)
        xi, nni, dAi  = _sample_on_element(coords, nodes_quad, nodes_tri,
                                            nodes_quad4, nodes_tri3,
                                            fi, ni_idx, T(u1), T(u2))
        u3, rng_state = _xorshift64(rng_state)
        u4, rng_state = _xorshift64(rng_state)
        xj, nnj, dAj  = _sample_on_element(coords, nodes_quad, nodes_tri,
                                            nodes_quad4, nodes_tri3,
                                            fj, nj_idx, T(u3), T(u4))
        Ai += dAi; Aj += dAj
        K = _vf_kernel_gpu(xi, nni, xj, nnj)
        if K > zero(T) && use_bvh
            rx=xj[1]-xi[1]; ry=xj[2]-xi[2]; rz=xj[3]-xi[3]
            rlen=sqrt(rx*rx+ry*ry+rz*rz)
            if rlen > T(1e-15)
                inv_r=T(1)/rlen
                if gpu_intersect_bvh(bvh_lo,bvh_hi,bvh_meta,
                                      bvh_tri_idx,bvh_tris,bvh_tri_group,
                                      xi[1],xi[2],xi[3],
                                      rx*inv_r,ry*inv_r,rz*inv_r,
                                      rlen, gi, gj)
                    K = zero(T)
                end
            end
        end
        K_sum += K * dAi * dAj
    end

    # ref_area_i * ref_area_j / n_samples² (absorbed into normalisation).
    # Quad families (0,2) sample [-1,1]² (area 4); tri families (1,3) the
    # reference triangle (area 1/2).
    ref_i   = (fi == 0 || fi == 2) ? T(4) : T(0.5)
    ref_j   = (fj == 0 || fj == 2) ? T(4) : T(0.5)
    raw_val = K_sum * ref_i * ref_j / T(n_samples)

    raw_out[i, j] = raw_val
    raw_out[j, i] = raw_val
    area_out[i]   = Ai * ref_i / T(n_samples)
    area_out[j]   = Aj * ref_j / T(n_samples)

    end # if i <= N && j <= N && i < j
end

# Map a (u1,u2) uniform pair in [0,1]² to a point on element family fi.
# Family codes: 0=Quad8, 1=Tri6, 2=Quad4, 3=Tri3.
@inline function _sample_on_element(coords, nodes_quad, nodes_tri,
                                     nodes_quad4, nodes_tri3,
                                     fi::Int, ni_idx::Int,
                                     u1::T, u2::T) where T
    if fi == 0 || fi == 2   # quad: map [0,1]² → [-1,1]²
        ξ = T(2)*u1 - T(1)
        η = T(2)*u2 - T(1)
        return fi == 0 ? _quad8_eval(coords, nodes_quad,  ni_idx, ξ, η) :
                         _quad4_eval(coords, nodes_quad4, ni_idx, ξ, η)
    else                    # tri: fold [0,1]² into reference triangle
        ξ = u1; η = u2
        if ξ + η > T(1); ξ = T(1)-ξ; η = T(1)-η; end
        return fi == 1 ? _tri6_eval(coords, nodes_tri,  ni_idx, ξ, η) :
                         _tri3_eval(coords, nodes_tri3, ni_idx, ξ, η)
    end
end

# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------

"""
    launch_mc_kernel!(ga, backend, n_samples, seed; groupsize=16, flat_bvh=nothing)
        -> (raw_out, area_out)

Launch the GPU Monte Carlo view factor kernel.
`ga` is the output of `build_gpu_arrays` from GPUKernels.
`seed` is a UInt64 random seed (one per launch; threads derive independent streams).
"""
function launch_mc_kernel!(ga, backend;
                             n_samples ::Int     = 10000,
                             seed      ::UInt64  = rand(UInt64),
                             groupsize ::Int     = 16,
                             flat_bvh           = nothing)
    N      = ga.N
    FloatT = ga.FloatT
    ArrayT = typeof(ga.coords)

    raw_out  = ArrayT(zeros(FloatT, N, N))
    area_out = ArrayT(zeros(FloatT, N))

    use_bvh = flat_bvh !== nothing
    dummy   = ArrayT(zeros(FloatT, 1, 1))   # placeholder when no BVH
    bvh_lo      = use_bvh ? flat_bvh.nodes_lo   : dummy
    bvh_hi      = use_bvh ? flat_bvh.nodes_hi   : dummy
    bvh_meta    = use_bvh ? flat_bvh.nodes_meta  : ArrayT(zeros(Int32,1,1))
    bvh_tri_idx = use_bvh ? flat_bvh.tri_idx     : ArrayT(zeros(Int32,1))
    bvh_tris    = use_bvh ? flat_bvh.tri_verts   : dummy
    bvh_tri_grp = use_bvh ? flat_bvh.tri_group   : ArrayT(zeros(Int32,1))

    kern! = _mc_pair_kernel!(backend, (groupsize, groupsize))
    kern!(raw_out, area_out,
          ga.coords, ga.nodes_quad, ga.nodes_tri,
          ga.nodes_quad4, ga.nodes_tri3,
          ga.elem_family, ga.elem_node_idx,
          n_samples, seed, use_bvh,
          bvh_lo, bvh_hi, bvh_meta, bvh_tri_idx, bvh_tris, bvh_tri_grp,
          N;
          ndrange=(N, N))

    KernelAbstractions.synchronize(backend)
    return raw_out, area_out
end

end # module GPUMCKernels
