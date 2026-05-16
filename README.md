# RadiativeViewFactor.jl

[![CI](https://github.com/rot4te/RadiativeViewFactor/actions/workflows/CI.yml/badge.svg)](https://github.com/rot4te/RadiativeViewFactor/actions/workflows/CI.yml)

A Julia package for computing **radiative view factors** between arbitrary surfaces
discretized on 2nd-order surface meshes generated with [Gmsh](https://gmsh.info/).

## Features

- Reads Gmsh `.msh` files via the Gmsh Julia SDK
- Supports **Quad8** (8-node serendipity quadrilateral) and **Tri6** (6-node triangular) surface elements; Quad9 centre node is silently dropped
- Groups surfaces by **Gmsh physical groups**; view factors reported at both element and group level
- Gauss–Legendre quadrature on each element pair (pre-tabulated for n ≤ 5, Golub–Welsch algorithm for n > 5); Dunavant rules for triangular elements
- **Obstruction detection** via Möller–Trumbore ray–triangle intersection on an axis-aligned BVH; works on both CPU and GPU backends
- CPU backend: multi-threaded via `Threads.@threads`; per-pair BVH excludes the emitter and receiver groups from the obstruction geometry
- GPU backends: NVIDIA (CUDA.jl) and Apple Silicon (Metal.jl) via KernelAbstractions.jl; shadow rays are cast inside the GPU kernel using a flat BVH on device memory
- **Reciprocity** and **closure** (row-sum) checks on the assembled matrix

## Project Layout

```
RadiativeViewFactor.jl/
├── src/
│   ├── RadiativeViewFactor.jl   # Package entry-point and public exports
│   ├── MeshIO.jl                # Gmsh mesh loading, physical-group extraction
│   ├── Quadrature.jl            # Gauss–Legendre rules on [-1,1]²
│   ├── Geometry.jl              # Quad8 shape functions, normals, Jacobians, element area
│   ├── BVH.jl                   # Flat-array axis-aligned BVH for triangle soup
│   ├── RayCast.jl               # Segment–BVH visibility test
│   ├── ViewFactorKernel.jl      # Double-area integral kernel; element-pair integrator
│   ├── GPUBVH.jl                # Flat BVH for GPU: CPU→device builder + inline traversal kernel
│   ├── GPUKernels.jl            # KernelAbstractions GPU kernels (Quad8 + Tri6, with obstruction)
│   ├── Assembly.jl              # CPU assembly; group aggregation; reciprocity/closure checks
│   └── GPUAssembly.jl           # GPU assembly path
├── test/
│   └── runtests.jl
└── Project.toml
```

## Quick Start

```julia
using RadiativeViewFactor

# Load a Gmsh mesh (surfaces must be in Physical Surface groups, 2nd-order elements)
mesh = load_mesh("geometry.msh")

# Compute view factors on CPU (multi-threaded)
result = compute_view_factors(mesh; nquad=4)

# result.F_elem  — (N_elem × N_elem) element-level view factor matrix
# result.F_group — (N_group × N_group) physical-group view factor matrix
# result.A_elem, result.A_group — element / group areas

# Post-processing checks
check_reciprocity(result)   # prints max relative error, returns Bool
check_closure(result)       # prints row-sum range, returns Bool
```

### With obstruction detection

```julia
# Tags of physical groups that may occlude rays between other surfaces
result = compute_view_factors(mesh; nquad=4, obstruction_groups=[3, 4])
```

### GPU (CUDA)

```julia
using CUDA
result = compute_view_factors(mesh; nquad=4, backend=CUDABackend())
```

### GPU (Apple Metal)

```julia
using Metal
result = compute_view_factors(mesh; nquad=4, backend=MetalBackend())
```

### GPU with obstruction detection

Obstruction detection works on all backends. Pass `obstruction_groups` exactly as you would for the CPU path:

```julia
using CUDA
result = compute_view_factors(mesh; nquad=4,
                              backend=CUDABackend(),
                              obstruction_groups=[3, 4])
```

The BVH is built on the CPU from the merged triangle soups of the specified groups, then uploaded to the device as flat typed arrays. Each GPU thread traverses the BVH independently using thread-local stack memory. Each triangle carries its physical group tag, so rays between elements of groups i and j automatically skip any obstruction triangle that belongs to group i or j — the same per-pair exclusion behaviour as the CPU path.

## API Reference

### `load_mesh(filename; surface_dim=2, verbose=true) → MeshData`

Load a Gmsh `.msh` file. `surface_dim` selects the element dimension (2 for surfaces).
Returns a `MeshData` with fields `coords`, `surface_elems`, `group_tags`, `group_elems`, and `group_tri_soup`.

### `compute_view_factors(mesh; nquad=4, obstruction_groups=Int[], backend=CPU(), self_vf=false, verbose=true) → ViewFactorResult`

Assemble the full view factor matrix.

| Keyword | Default | Description |
|---|---|---|
| `nquad` | `4` | Gauss points per direction (nquad² quadrature points per element pair) |
| `obstruction_groups` | `Int[]` | Physical group tags that may block rays (CPU and GPU) |
| `backend` | `CPU()` | `CPU()`, `CUDABackend()`, or `MetalBackend()` |
| `self_vf` | `false` | Include diagonal (self) view factors (curved elements; CPU only) |
| `verbose` | `true` | Print progress and row-sum diagnostics |

### `ViewFactorResult` fields

| Field | Type | Description |
|---|---|---|
| `F_elem` | `Matrix{Float64}` | Element-level view factor matrix (N × N) |
| `A_elem` | `Vector{Float64}` | Element areas |
| `F_group` | `Matrix{Float64}` | Group-level view factor matrix (G × G) |
| `A_group` | `Vector{Float64}` | Group areas |
| `group_tags` | `Vector{Int}` | Physical group tags (sorted) |
| `group_names` | `Vector{String}` | Physical group names |

### `aggregate_by_group(result, mesh) → (F_group, A_group, tags, names)`

Re-aggregate element-level results into group-level view factors (useful after modifying `result.F_elem`).

### `check_reciprocity(result; tol=1e-4) → Bool`

Verify the reciprocity relation Aᵢ Fᵢⱼ = Aⱼ Fⱼᵢ. Prints the maximum relative error.

### `check_closure(result; tol=1e-3) → Bool`

Verify that no row sum exceeds 1 + `tol`. Prints the row-sum range.

## Theory

The view factor from surface i to surface j is:

```
F_ij = (1/Aᵢ) ∬_Aᵢ ∬_Aⱼ  cos θᵢ cos θⱼ / (π r²)  H_ij  dAⱼ dAᵢ
```

where θᵢ, θⱼ are the angles between the line of sight and each surface normal,
r is the distance between the differential areas, and H_ij ∈ {0,1} is the
visibility function (0 = obstructed).

The double integral is evaluated numerically using Gauss–Legendre quadrature
on each Quad8 element (mapped from [-1,1]²) or Dunavant quadrature on each
Tri6 element. On CPU the BVH is built once per obstruction-group set and reused across all
element pairs in that set. On GPU the BVH is flattened to plain typed arrays
(AABB bounds, node metadata, triangle vertices, per-triangle group tags) and
uploaded to device memory once before the kernel launch; each thread traverses
it using a fixed-size thread-local stack and skips any triangle whose group tag
matches the emitter or receiver element's group.

## Mesh Requirements

- 2nd-order surface elements: `Quad8` (Gmsh type 16), `Quad9` (type 10, centre node dropped), or `Tri6` (type 9)
- All radiating surfaces must belong to a **Physical Surface** group
- Obstructor surfaces should also be in their own Physical Surface group(s)

In Gmsh, set `Mesh.ElementOrder = 2` (or pass `-order 2` on the command line) before meshing.

## Dependencies

| Package | Role |
|---|---|
| `Gmsh` | Mesh file I/O |
| `KernelAbstractions` | Backend-agnostic GPU kernels |
| `StaticArrays` | Stack-allocated vectors for hot-path geometry |
| `LinearAlgebra`, `SparseArrays` | Standard library |
| `CUDA` *(optional)* | NVIDIA GPU backend |
| `Metal` *(optional)* | Apple Silicon GPU backend |
