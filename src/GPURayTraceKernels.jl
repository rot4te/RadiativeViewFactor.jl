# src/GPURayTraceKernels.jl
# ---------------------------------------------------------------------------
# GPU ray-shooting Monte Carlo view factor kernel using KernelAbstractions.
#
# See `RayTraceKernel.jl`'s module docstring for the method (cosine-weighted
# hemisphere sampling + nearest-hit query over the whole radiating scene) and
# why it needs no adjacent-pair Duffy patch. This module is the GPU
# counterpart: one thread per *element* (not per pair, unlike the other GPU
# kernels), each shooting `n_rays` rays serially and tallying which element
# each first hits. `GPUAssembly.jl` does the reciprocity-averaging and
# aggregation afterward, on the CPU, mirroring `Assembly.jl`'s
# `_compute_cpu_raytrace` exactly — that step is O(N²) dense-matrix work
# either way, not worth a kernel of its own.
#
# Element sampling and RNG reuse existing GPU building blocks rather than
# duplicating them: `GPUMCKernels._prepare_elem`/`_sample_prepared` (the same
# per-thread-hoisted geometry used by the pair-area MC kernel) for uniform-
# by-area point+normal sampling, and `GPUMCKernels._xorshift32`/`_init_rng`
# for the inline PRNG (see that module's docstring for why: a 64-bit RNG
# dominates kernel runtime on GPUs that emulate 64-bit integer math in
# software, e.g. Apple Silicon).
# ---------------------------------------------------------------------------

module GPURayTraceKernels

using KernelAbstractions
using StaticArrays
using LinearAlgebra: dot, cross

import ..GPUBVH:      FlatBVH, gpu_nearest_hit_bvh
import ..GPUMCKernels: PreparedElem, _prepare_elem, _sample_prepared,
                       _xorshift32, _init_rng
import ..GPUKernels:   _quad8_point_and_jac, _tri6_point_and_jac,
                       _quad4_point_and_jac, _tri3_point_and_jac

export launch_raytrace_kernel!, launch_area_kernel!

# ---------------------------------------------------------------------------
# Cosine-weighted hemisphere direction sampling (scalar GPU form)
#
# Same branchless orthonormal-basis construction as the CPU kernel
# (RayTraceKernel.cosine_dir; Duff, Burgess, Christensen, Hery, Kensler,
# Liani & Villemin, "Building an Orthonormal Basis, Revisited", JCGT 2017),
# written in scalar components rather than SVector ops to match this file's
# existing GPU style (GPUMCKernels/GPUBVH keep the hot loop free of
# SVector/StaticArrays overhead).
# ---------------------------------------------------------------------------

@inline function _cosine_dir_gpu(nx::T, ny::T, nz::T, rng_state::UInt32) where T
    s = nz >= zero(T) ? one(T) : -one(T)
    a = -one(T) / (s + nz)
    b = nx * ny * a
    tx = one(T) + s*nx*nx*a;  ty = s*b;             tz = -s*nx
    btx = b;                   bty = s + ny*ny*a;   btz = -ny

    u1, rng_state = _xorshift32(rng_state, T)
    u2, rng_state = _xorshift32(rng_state, T)
    r = sqrt(u1); phi = T(2π)*u2
    lx = r*cos(phi); ly = r*sin(phi); lz = sqrt(max(zero(T), one(T)-u1))

    dx = lx*tx + ly*btx + lz*nx
    dy = lx*ty + ly*bty + lz*ny
    dz = lx*tz + ly*btz + lz*nz
    return dx, dy, dz, rng_state
end

# ---------------------------------------------------------------------------
# Element-area kernel — one thread per element, exact quadrature (not
# sampled), reusing the same point/jacobian evaluators the deterministic GPU
# pair kernel uses. Matches the CPU ray-tracing path's choice
# (`precompute_quad(...).Li`) to keep areas exact rather than adding MC
# noise to a quantity that doesn't need it.
# ---------------------------------------------------------------------------

@kernel function _area_kernel!(area_out, coords, nodes_quad, nodes_tri,
                                elem_family, elem_node_idx,
                                quad_pts, quad_wts, tri_pts, tri_wts, N::Int)
    i = @index(Global)
    @inbounds if i <= N
        T  = eltype(coords)
        fi = Int32(elem_family[i])
        idx = Int(elem_node_idx[i])
        Ai = zero(T)
        if fi == Int32(0)          # quad8
            for p in eachindex(quad_wts)
                _, _, dA = _quad8_point_and_jac(coords, nodes_quad, idx, quad_pts[1,p], quad_pts[2,p])
                Ai += quad_wts[p] * dA
            end
        elseif fi == Int32(1)      # tri6
            for p in eachindex(tri_wts)
                _, _, dA = _tri6_point_and_jac(coords, nodes_tri, idx, tri_pts[1,p], tri_pts[2,p])
                Ai += tri_wts[p] * dA
            end
        elseif fi == Int32(2)      # quad4
            for p in eachindex(quad_wts)
                _, _, dA = _quad4_point_and_jac(coords, nodes_quad, idx, quad_pts[1,p], quad_pts[2,p])
                Ai += quad_wts[p] * dA
            end
        else                        # tri3
            for p in eachindex(tri_wts)
                _, _, dA = _tri3_point_and_jac(coords, nodes_tri, idx, tri_pts[1,p], tri_pts[2,p])
                Ai += tri_wts[p] * dA
            end
        end
        area_out[i] = Ai
    end
end

"""
    launch_area_kernel!(ga, backend; groupsize=64) -> area_out

Launch the element-area kernel: one thread per element, exact Gauss/Dunavant
quadrature (reusing `ga.quad_pts`/`quad_wts`/`tri_pts`/`tri_wts`, the same
rule the deterministic pair kernel already carries). Returns a device vector
of length `ga.N`.
"""
function launch_area_kernel!(ga, backend; groupsize::Int = 64)
    N      = ga.N
    FloatT = ga.FloatT
    area_out = KernelAbstractions.zeros(backend, FloatT, N)
    kern! = _area_kernel!(backend, groupsize)
    kern!(area_out, ga.coords, ga.nodes_quad, ga.nodes_tri,
          ga.elem_family, ga.elem_node_idx,
          ga.quad_pts, ga.quad_wts, ga.tri_pts, ga.tri_wts, N; ndrange=N)
    KernelAbstractions.synchronize(backend)
    return area_out
end

# ---------------------------------------------------------------------------
# Ray-shooting kernel
# ---------------------------------------------------------------------------

@kernel function _raytrace_kernel!(hits_out, escaped_out,
                                    coords, nodes_quad, nodes_tri,
                                    elem_family, elem_node_idx,
                                    n_rays::Int,
                                    global_seed::UInt64,
                                    bvh_lo, bvh_hi, bvh_meta,
                                    bvh_tri_idx, bvh_tris, bvh_tri_elem,
                                    row_offset::Int32,
                                    row_hi::Int32,
                                    N::Int)
    ig = @index(Global)
    i  = ig % Int32 + row_offset

    @inbounds if i <= row_hi

    T      = eltype(coords)
    fi     = Int32(elem_family[i])
    idx    = elem_node_idx[i] % Int32
    pe     = _prepare_elem(coords, nodes_quad, nodes_tri, fi, idx, T)
    skip_elem = i

    seed64    = _init_rng(global_seed, i)
    rng_state = UInt32((seed64 ⊻ (seed64 >> 32)) & 0xFFFFFFFF)
    rng_state = ifelse(rng_state == UInt32(0), UInt32(0x9E3779B9), rng_state)

    nr32  = n_rays % Int32
    s     = unsafe_trunc(Int32, sqrt(T(n_rays)))
    inv_s = one(T) / T(s)
    n_escaped = zero(T)

    # Stratified source point on i (same rationale as the pair-area MC
    # kernel's stratified element-i sampling: reduces variance for a fixed
    # ray budget), independent cosine-weighted direction per ray.
    ray_k = Int32(0)
    for si in Int32(0):s-one(Int32)
        for sj in Int32(0):s-one(Int32)
            ray_k += one(Int32)
            u1, rng_state = _xorshift32(rng_state, T)
            u2, rng_state = _xorshift32(rng_state, T)
            x, n, _ = _sample_prepared(pe, coords, nodes_quad, nodes_tri,
                                        (T(si)+u1)*inv_s, (T(sj)+u2)*inv_s)
            dx, dy, dz, rng_state = _cosine_dir_gpu(n[1], n[2], n[3], rng_state)
            hit, elem, front = gpu_nearest_hit_bvh(bvh_lo, bvh_hi, bvh_meta,
                                                    bvh_tri_idx, bvh_tris, bvh_tri_elem,
                                                    x[1], x[2], x[3], dx, dy, dz, skip_elem)
            if !hit
                n_escaped += one(T)
            elseif elem != Int32(0) && front
                hits_out[i, elem] += one(T)
            end
        end
    end
    for _ in ray_k+one(Int32):nr32
        u1, rng_state = _xorshift32(rng_state, T)
        u2, rng_state = _xorshift32(rng_state, T)
        x, n, _ = _sample_prepared(pe, coords, nodes_quad, nodes_tri, u1, u2)
        dx, dy, dz, rng_state = _cosine_dir_gpu(n[1], n[2], n[3], rng_state)
        hit, elem, front = gpu_nearest_hit_bvh(bvh_lo, bvh_hi, bvh_meta,
                                                bvh_tri_idx, bvh_tris, bvh_tri_elem,
                                                x[1], x[2], x[3], dx, dy, dz, skip_elem)
        if !hit
            n_escaped += one(T)
        elseif elem != Int32(0) && front
            hits_out[i, elem] += one(T)
        end
    end

    escaped_out[i] = n_escaped / T(n_rays)

    end # if i <= row_hi
end

# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------

"""
    launch_raytrace_kernel!(ga, flat_scene, backend; n_rays=10000, seed=rand(UInt64),
                            groupsize=64, max_chunk_seconds=1.0, verbose=false)
        -> (hits_out, escaped_out)

Launch the GPU ray-shooting kernel. `ga` is `GPUKernels.build_gpu_arrays`'
output; `flat_scene` is `GPUBVH.build_flat_scene_bvh`'s output (the whole-mesh
scene BVH, tagged by element index — *not* `GPUBVH.build_flat_bvh_from_mesh`'s
group-tagged obstruction-only BVH the other GPU kernels use).

Returns device arrays `hits_out` (N×N, hit *counts* — not yet a fraction or
a raw double-integral value; `GPUAssembly.jl` does that division and the
reciprocity-averaging afterward, on the CPU, mirroring
`Assembly._compute_cpu_raytrace`) and `escaped_out` (N, fraction of each
element's rays that hit nothing — open enclosure).

One thread per *element* (not per pair, unlike the other GPU kernels'
`ndrange=(N,N)`), so the work is `O(N · n_rays)` per launch rather than
`O(N²)`. Submitted in row-block chunks for the same reason
`GPUMCKernels.launch_mc_kernel!` is: on Metal, macOS's GPU watchdog kills
command buffers that run too long, and chunking keeps each one short. Unlike
that kernel's triangular per-pair workload, work here is uniform per
element, so chunking is a plain element-count budget, retargeted from the
measured rate after each chunk.
"""
function launch_raytrace_kernel!(ga, flat_scene::FlatBVH, backend;
                                  n_rays           ::Int     = 10000,
                                  seed             ::UInt64  = rand(UInt64),
                                  groupsize        ::Int     = 64,
                                  max_chunk_seconds::Real    = 1.0,
                                  verbose          ::Bool    = false)
    N      = ga.N
    FloatT = ga.FloatT

    hits_out    = KernelAbstractions.zeros(backend, FloatT, N, N)
    escaped_out = KernelAbstractions.zeros(backend, FloatT, N)

    kern! = _raytrace_kernel!(backend, groupsize)

    elem_budget = max(5.0e6 / max(n_rays, 1), 32.0)
    row0        = 0
    t_start     = time()
    next_report = 0.05

    while row0 < N
        rows = min(N - row0, max(1, round(Int, elem_budget)))

        t0 = time()
        kern!(hits_out, escaped_out,
              ga.coords, ga.nodes_quad, ga.nodes_tri,
              ga.elem_family, ga.elem_node_idx,
              n_rays, seed,
              flat_scene.nodes_lo, flat_scene.nodes_hi, flat_scene.nodes_meta,
              flat_scene.tri_idx, flat_scene.tri_verts, flat_scene.tri_group,
              Int32(row0), Int32(row0 + rows), N;
              ndrange=rows)
        KernelAbstractions.synchronize(backend)
        dt = time() - t0

        rate        = rows / max(dt, 1e-4)
        elem_budget = min(rate * max_chunk_seconds, elem_budget * 4)
        row0       += rows

        if verbose && row0 / N >= next_report
            println("    Ray-trace kernel: ", round(Int, 100 * row0 / N),
                    "% of elements done, ", round(time() - t_start, digits=1), " s elapsed")
            while next_report <= row0 / N
                next_report += 0.05
            end
        end
    end

    return hits_out, escaped_out
end

end # module GPURayTraceKernels
