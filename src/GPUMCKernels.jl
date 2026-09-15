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
# requires heap allocation).  Each thread is seeded once with splitmix64 from
# the global seed and thread index, then the hot sampling loop draws from a
# fast inline 32-bit xorshift PRNG.  The 32-bit hot path is deliberate: GPUs
# such as Apple Silicon emulate 64-bit integer math in software, so a 64-bit
# per-sample RNG dominates the kernel runtime.  This gives independent
# pseudo-random streams per thread with no memory overhead.
#
# Stratified sampling
# -------------------
# Element i's n_samples points are divided into s×s strata where s = floor(√n)
# (the remaining n - s² points are drawn from the full reference domain);
# element j's points are drawn uniformly, so each sample pair is independent
# and the pair estimate is exactly unbiased (see comment in the kernel body).
# This matches the CPU MCKernel pairing.
#
# Per-thread geometry hoisting
# ----------------------------
# Each thread samples the same two elements n_samples times, so all
# per-element-constant geometry is precomputed once per thread into registers
# (see PreparedElem).  For linear elements this removes the per-sample global
# memory reads and shape-function work entirely — tri3 sampling reduces to an
# affine map with a constant normal (≈9× faster on Apple M-series), quad4 to
# a bilinear map.  Curved elements (quad8/tri6) keep per-sample evaluation.
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

# Per-sample PRNG for the hot loop.  Uses 32-bit xorshift so it runs natively
# on GPUs (e.g. Apple Silicon) that emulate 64-bit integer math in software —
# a 64-bit RNG here dominates the kernel runtime.  Returns a uniform value in
# [0,1) as type `T` (Float32 on Metal, Float64 elsewhere); `state` must stay
# nonzero.  Period 2^32-1 is ample: each thread draws ≪ 2^32 values.
@inline function _xorshift32(state::UInt32, ::Type{T}) where T
    state ⊻= state << 13
    state ⊻= state >> 17
    state ⊻= state << 5
    # top 24 bits → exact Float32 mantissa; harmless extra precision for Float64
    return (T(state >> 8) / T(0x01000000), state)
end

@inline function _init_rng(global_seed::UInt64, thread_id::Int32)::UInt64
    # Mix thread id into seed using splitmix64 to ensure different streams.
    # 64-bit splitmix runs once per thread, so its cost is negligible.
    z = global_seed + UInt64(thread_id % UInt32) * 0x9E3779B97F4A7C15
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

@inline function _quad8_eval(coords, nodes_quad, ni_idx::Int32, ξ::T, η::T) where T
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

@inline function _tri6_eval(coords, nodes_tri, ni_idx::Int32, ξ::T, η::T) where T
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

# ---------------------------------------------------------------------------
# Per-thread element preparation
#
# Each thread samples the same two elements n_samples times, so anything that
# is constant per element is hoisted out of the hot loop into registers:
#
#   tri3  — the map is affine: x = v0 + ξ·e1 + η·e2 with a *constant* normal
#           and area element.  The per-sample cross/sqrt/normalize disappears
#           entirely; a sample costs 6 FMAs.
#   quad4 — the map is bilinear: x = c0 + ξ·c1 + η·c2 + ξη·c3 with
#           dx/dξ = c1 + η·c3 and dx/dη = c2 + ξ·c3.  The four coefficient
#           vectors are precomputed; per sample only the cross/normalize for
#           the (possibly non-planar) quad remains.
#   quad8/tri6 — curved elements keep the full shape-function evaluation and
#           read node coordinates from (cached) global memory as before.
# ---------------------------------------------------------------------------

struct PreparedElem{T}
    fam :: Int32            # 0=Quad8, 1=Tri6, 2=Quad4, 3=Tri3
    idx :: Int32            # column into nodes_quad / nodes_tri (fam 0/1)
    c1  :: SVector{3,T}     # tri3: v0 ; quad4: c0
    c2  :: SVector{3,T}     # tri3: e1 ; quad4: c1 (ξ coefficient)
    c3  :: SVector{3,T}     # tri3: e2 ; quad4: c2 (η coefficient)
    c4  :: SVector{3,T}     # quad4: c3 (ξη coefficient)
    nrm :: SVector{3,T}     # tri3: constant unit normal
    cdA :: T                # tri3: constant area element |e1×e2|
end

@inline function _load_node(coords, na, ::Type{T}) where T
    return SVector{3,T}(coords[1,na], coords[2,na], coords[3,na])
end

@inline function _prepare_elem(coords, nodes_quad, nodes_tri,
                                fam::Int32, idx::Int32, ::Type{T}) where T
    z = zero(SVector{3,T})
    @inbounds if fam == Int32(3)        # tri3: affine geometry, constant normal
        v0 = _load_node(coords, nodes_tri[1, idx], T)
        e1 = _load_node(coords, nodes_tri[2, idx], T) - v0
        e2 = _load_node(coords, nodes_tri[3, idx], T) - v0
        c  = cross(e1, e2); dA = sqrt(dot(c,c))
        return PreparedElem{T}(fam, idx, v0, e1, e2, z, c/dA, dA)
    elseif fam == Int32(2)              # quad4: bilinear coefficients
        v1 = _load_node(coords, nodes_quad[1, idx], T)
        v2 = _load_node(coords, nodes_quad[2, idx], T)
        v3 = _load_node(coords, nodes_quad[3, idx], T)
        v4 = _load_node(coords, nodes_quad[4, idx], T)
        c0 = T(0.25)*( v1+v2+v3+v4)
        c1 = T(0.25)*(-v1+v2+v3-v4)
        c2 = T(0.25)*(-v1-v2+v3+v4)
        c3 = T(0.25)*( v1-v2+v3-v4)
        return PreparedElem{T}(fam, idx, c0, c1, c2, c3, z, zero(T))
    else                                # quad8 / tri6: per-sample evaluation
        return PreparedElem{T}(fam, idx, z, z, z, z, z, zero(T))
    end
end

# Map a (u1,u2) uniform pair in [0,1]² to a point/normal/dA on the element.
@inline function _sample_prepared(pe::PreparedElem{T}, coords,
                                   nodes_quad, nodes_tri,
                                   u1::T, u2::T) where T
    fam = pe.fam
    if fam == Int32(3)                  # tri3: fold into triangle, affine map
        ξ = u1; η = u2
        if ξ + η > T(1); ξ = T(1)-ξ; η = T(1)-η; end
        x = pe.c1 + ξ*pe.c2 + η*pe.c3
        return x, pe.nrm, pe.cdA
    elseif fam == Int32(2)              # quad4: bilinear map
        ξ = T(2)*u1 - T(1)
        η = T(2)*u2 - T(1)
        x    = pe.c1 + ξ*pe.c2 + η*pe.c3 + (ξ*η)*pe.c4
        dxdξ = pe.c2 + η*pe.c4
        dxdη = pe.c3 + ξ*pe.c4
        c = cross(dxdξ, dxdη); dA = sqrt(dot(c,c))
        return x, c/dA, dA
    elseif fam == Int32(1)              # tri6
        ξ = u1; η = u2
        if ξ + η > T(1); ξ = T(1)-ξ; η = T(1)-η; end
        return _tri6_eval(coords, nodes_tri, pe.idx, ξ, η)
    else                                # quad8
        ξ = T(2)*u1 - T(1)
        η = T(2)*u2 - T(1)
        return _quad8_eval(coords, nodes_quad, pe.idx, ξ, η)
    end
end

# View-factor kernel with the visibility test fused in, so the pair distance
# is computed once.  Only 1 sqrt and 2 divides per sample (vs 2 sqrt and
# 5 divides when kernel and ray setup were separate).
@inline function _vf_contribution(xi::SVector{3,T}, ni::SVector{3,T},
                                    xj::SVector{3,T}, nj::SVector{3,T},
                                    use_bvh::Bool,
                                    bvh_lo, bvh_hi, bvh_meta,
                                    bvh_tri_idx, bvh_tris, bvh_tri_group,
                                    gi::Int32, gj::Int32) where T
    rv = xj - xi
    r2 = dot(rv, rv)
    r2 < T(1e-30) && return zero(T)
    inv_r = one(T) / sqrt(r2)
    ci =  dot(ni, rv) * inv_r
    cj = -dot(nj, rv) * inv_r
    (ci <= zero(T) || cj <= zero(T)) && return zero(T)
    K = ci * cj * T(1/π) / r2
    if use_bvh
        if gpu_intersect_bvh(bvh_lo, bvh_hi, bvh_meta,
                              bvh_tri_idx, bvh_tris, bvh_tri_group,
                              xi[1], xi[2], xi[3],
                              rv[1]*inv_r, rv[2]*inv_r, rv[3]*inv_r,
                              r2*inv_r, gi, gj)
            return zero(T)
        end
    end
    return K
end

# ---------------------------------------------------------------------------
# MC kernel
# ---------------------------------------------------------------------------

@kernel function _mc_pair_kernel!(raw_out, area_out,
                                   coords,
                                   nodes_quad, nodes_tri,
                                   elem_family, elem_node_idx,
                                   n_samples::Int,
                                   global_seed::UInt64,
                                   use_bvh::Bool,
                                   bvh_lo, bvh_hi, bvh_meta,
                                   bvh_tri_idx, bvh_tris, bvh_tri_group,
                                   row_offset::Int32,
                                   row_hi::Int32,
                                   N::Int)
    ig, jg = @index(Global, NTuple)
    # Work in 32-bit index space: Apple GPUs emulate 64-bit integer math, so
    # keeping our own index arithmetic in Int32 avoids that tax.  (Array-stride
    # multiplies inside A[i,j] remain 64-bit — that's internal to the device
    # array type.)
    # The launcher submits rows in chunks (row_offset+1 : row_hi) so each
    # command buffer stays short; ig is the row index within the chunk.
    i   = ig % Int32 + row_offset
    j   = jg % Int32
    N32 = N   % Int32

    if i <= row_hi && j <= N32 && i < j

    T        = eltype(coords)
    fi       = Int32(elem_family[i]);  fj = Int32(elem_family[j])
    ni_idx   = elem_node_idx[i] % Int32; nj_idx = elem_node_idx[j] % Int32
    gi       = Int32(0);  gj = Int32(0)   # group tags not needed: BVH exclusion
    # handled via bvh_tri_group in gpu_intersect_bvh

    # Hoist all per-element-constant geometry out of the sample loop.
    pe_i = _prepare_elem(coords, nodes_quad, nodes_tri, fi, ni_idx, T)
    pe_j = _prepare_elem(coords, nodes_quad, nodes_tri, fj, nj_idx, T)

    # Unique per-thread id for RNG seeding (fits Int32 for N ≤ 46340).
    thread_id = (i - one(Int32)) * N32 + j
    # splitmix64 seeding (once per thread), then fold to a nonzero UInt32 that
    # drives the fast 32-bit hot-loop PRNG.
    seed64    = _init_rng(global_seed, thread_id)
    rng_state = UInt32((seed64 ⊻ (seed64 >> 32)) & 0xFFFFFFFF)
    rng_state = ifelse(rng_state == UInt32(0), UInt32(0x9E3779B9), rng_state)

    Ai = zero(T); Aj = zero(T); K_sum = zero(T)

    # unsafe_trunc avoids the checked Float→Int conversion (which boxes/heap
    # -allocates on GPUs); sqrt(n_samples) ≥ 0 so trunc == floor here.
    ns32  = n_samples % Int32
    s     = unsafe_trunc(Int32, sqrt(T(n_samples)))
    inv_s = one(T) / T(s)   # multiply by reciprocal instead of 2 divides/sample

    # Element i is sampled stratified; element j is sampled uniformly over its
    # whole reference domain.  Pairing both elements by the *same* stratum
    # index correlates the two sample positions and biases the pair estimate
    # (only the diagonal stratum blocks of the product domain are ever
    # sampled), and a fixed per-pair stratum shift leaves a conditional error
    # that does not shrink with n_samples.  An independent uniform xⱼ per
    # sample makes each pair term exactly unbiased while keeping the variance
    # reduction from the stratified xᵢ.
    sample_k = Int32(0)
    for si in Int32(0):s-one(Int32)
        for sj in Int32(0):s-one(Int32)
            sample_k += one(Int32)

            # Stratified sample on element i
            u1, rng_state = _xorshift32(rng_state, T)
            u2, rng_state = _xorshift32(rng_state, T)
            xi, nni, dAi = _sample_prepared(pe_i, coords, nodes_quad, nodes_tri,
                                             (T(si) + u1)*inv_s, (T(sj) + u2)*inv_s)
            Ai += dAi

            # Uniform sample on element j
            u3, rng_state = _xorshift32(rng_state, T)
            u4, rng_state = _xorshift32(rng_state, T)
            xj, nnj, dAj = _sample_prepared(pe_j, coords, nodes_quad, nodes_tri,
                                             u3, u4)
            Aj += dAj

            K = _vf_contribution(xi, nni, xj, nnj, use_bvh,
                                  bvh_lo, bvh_hi, bvh_meta,
                                  bvh_tri_idx, bvh_tris, bvh_tri_group, gi, gj)
            K_sum += K * dAi * dAj
        end
    end

    # Remaining samples from full reference domain
    for _ in sample_k+one(Int32):ns32
        u1, rng_state = _xorshift32(rng_state, T)
        u2, rng_state = _xorshift32(rng_state, T)
        xi, nni, dAi  = _sample_prepared(pe_i, coords, nodes_quad, nodes_tri, u1, u2)
        u3, rng_state = _xorshift32(rng_state, T)
        u4, rng_state = _xorshift32(rng_state, T)
        xj, nnj, dAj  = _sample_prepared(pe_j, coords, nodes_quad, nodes_tri, u3, u4)
        Ai += dAi; Aj += dAj
        K = _vf_contribution(xi, nni, xj, nnj, use_bvh,
                              bvh_lo, bvh_hi, bvh_meta,
                              bvh_tri_idx, bvh_tris, bvh_tri_group, gi, gj)
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

    end # if i <= row_hi && j <= N && i < j
end

# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------

"""
    launch_mc_kernel!(ga, backend, n_samples, seed; groupsize=16, flat_bvh=nothing,
                      max_chunk_seconds=1.0, verbose=false)
        -> (raw_out, area_out)

Launch the GPU Monte Carlo view factor kernel.
`ga` is the output of `build_gpu_arrays` from GPUKernels.
`seed` is a UInt64 random seed (one per launch; threads derive independent streams).

The element-pair grid is submitted as a sequence of row-block chunks, each
sized adaptively so a single command buffer runs for at most about
`max_chunk_seconds`.  On Metal the GPU also drives the display, and macOS
kills command buffers that run long ("Impacting Interactivity") — the error
is only logged asynchronously, leaving the output buffers partially written.
Chunking keeps every buffer well under the watchdog limit.  Results are
identical to a single launch: each pair (i,j) seeds its RNG from (i,j) alone.
"""
function launch_mc_kernel!(ga, backend;
                             n_samples ::Int     = 10000,
                             seed      ::UInt64  = rand(UInt64),
                             groupsize ::Int     = 16,
                             flat_bvh           = nothing,
                             max_chunk_seconds::Real = 1.0,
                             verbose   ::Bool    = false)
    N      = ga.N
    FloatT = ga.FloatT

    raw_out  = KernelAbstractions.zeros(backend, FloatT, N, N)
    area_out = KernelAbstractions.zeros(backend, FloatT, N)

    use_bvh = flat_bvh !== nothing
    dummy   = KernelAbstractions.zeros(backend, FloatT, 1, 1)   # placeholder when no BVH
    bvh_lo      = use_bvh ? flat_bvh.nodes_lo   : dummy
    bvh_hi      = use_bvh ? flat_bvh.nodes_hi   : dummy
    bvh_meta    = use_bvh ? flat_bvh.nodes_meta  : KernelAbstractions.zeros(backend, Int32, 1, 1)
    bvh_tri_idx = use_bvh ? flat_bvh.tri_idx     : KernelAbstractions.zeros(backend, Int32, 1)
    bvh_tris    = use_bvh ? flat_bvh.tri_verts   : dummy
    bvh_tri_grp = use_bvh ? flat_bvh.tri_group   : KernelAbstractions.zeros(backend, Int32, 1)

    kern! = _mc_pair_kernel!(backend, (groupsize, groupsize))

    # Row i owns the pairs (i, i+1..N), so its work is proportional to N - i.
    # Each chunk covers rows row0+1 : row0+rows, chosen so its active-pair
    # count stays under pair_budget.  The budget starts conservatively (the
    # per-pair cost is unknown: element family, BVH size and n_samples all
    # matter) and is retargeted from the measured rate after every chunk;
    # growth is capped at 4× per step so one noisy timing can't overshoot
    # into watchdog territory.
    pair_budget = max(5.0e7 / n_samples, 1024.0)
    total_pairs = N * (N - 1) / 2
    pairs_done  = 0.0
    next_report = 0.05
    t_start     = time()
    row0        = 0

    while row0 < N - 1
        rows  = 1
        pairs = N - (row0 + 1)
        while row0 + rows < N - 1 && pairs + (N - (row0 + rows + 1)) <= pair_budget
            rows  += 1
            pairs += N - (row0 + rows)
        end

        t0 = time()
        kern!(raw_out, area_out,
              ga.coords, ga.nodes_quad, ga.nodes_tri,
              ga.elem_family, ga.elem_node_idx,
              n_samples, seed, use_bvh,
              bvh_lo, bvh_hi, bvh_meta, bvh_tri_idx, bvh_tris, bvh_tri_grp,
              Int32(row0), Int32(row0 + rows), N;
              ndrange=(rows, N))
        KernelAbstractions.synchronize(backend)
        dt = time() - t0

        rate        = pairs / max(dt, 1e-4)
        pair_budget = min(rate * max_chunk_seconds, pair_budget * 4)
        row0       += rows
        pairs_done += pairs

        if verbose && pairs_done / total_pairs >= next_report
            println("    MC kernel: ", round(Int, 100 * pairs_done / total_pairs),
                    "% of pairs done, ", round(time() - t_start, digits=1), " s elapsed")
            while next_report <= pairs_done / total_pairs
                next_report += 0.05
            end
        end
    end

    return raw_out, area_out
end

end # module GPUMCKernels
