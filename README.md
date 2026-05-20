# RadiativeViewFactor.jl

[![CI](https://github.com/rot4te/RadiativeViewFactor/actions/workflows/CI.yml/badge.svg)](https://github.com/rot4te/RadiativeViewFactor/actions/workflows/CI.yml)

A Julia package for computing **radiative view factors** between arbitrary surfaces
or curves discretized on 2nd-order meshes generated with [Gmsh](https://gmsh.info/).

## Features

- Reads Gmsh `.msh` files (v2.2 and v4) via the Gmsh Julia SDK
- **3D surface meshes** (`surface_dim=2`): Quad8, Quad9 (centre node silently dropped), and Tri6 elements
- **2D planar curve meshes** (`surface_dim=1`): Line3 elements; computes view factors per unit depth using the 2D kernel cos θᵢ cos θⱼ / (2r)
- Groups radiating geometry by **Gmsh physical groups**; view factors reported at both element and group level; rows and columns of `F_group` indexed by `result.group_tags` / `result.group_names`
- Gauss–Legendre quadrature on surface element pairs (pre-tabulated for n ≤ 5, Golub–Welsch for n > 5); Dunavant rules for triangular elements; 1-D Gauss–Legendre for Line3 curve elements
- **Obstruction detection** via ray–triangle (3D) or ray–segment (2D) intersection on an axis-aligned BVH, on both CPU and GPU backends
- `obstruction_groups` interface: pass physical group tags of potential occluders; source and destination groups are automatically excluded per pair so a surface never blocks its own rays
- CPU backend: multi-threaded via `Threads.@threads`
- GPU backends: NVIDIA (`CUDABackend`, Float64) and Apple Silicon (`MetalBackend`, Float32) via KernelAbstractions.jl; stackless BVH traversal on device avoids thread-local stack memory pressure
- **Reciprocity** and **closure** (row-sum) checks on the assembled matrix

## Project Layout

```
RadiativeViewFactor.jl/
├── src/
│   ├── RadiativeViewFactor.jl   # Package entry-point and public exports
│   ├── MeshIO.jl                # Gmsh mesh loading; Line3/Quad8/Quad9/Tri6 support
│   ├── Quadrature.jl            # Gauss–Legendre (1-D and 2-D) and Dunavant rules
│   ├── Geometry.jl              # Shape functions, normals, Jacobians for all element types
│   ├── BVH.jl                   # Axis-aligned BVH; triangle and segment soup support
│   ├── RayCast.jl               # CPU visibility test; dispatches on mesh_dim
│   ├── ViewFactorKernel.jl      # 3D and 2D kernels; element-pair double integral
│   ├── Results.jl               # ViewFactorResult, _aggregate, check functions
│   ├── GPUBVH.jl                # Stackless flat BVH for GPU: build + inline traversal
│   ├── GPUKernels.jl            # KernelAbstractions kernels (Quad8 + Tri6, with BVH)
│   ├── Assembly.jl              # CPU assembly; GPU dispatch hook; backend registry
│   └── GPUAssembly.jl           # GPU assembly path; registers GPU hook at load time
├── ext/
│   ├── RadiativeViewFactorCUDAExt.jl    # Registers CUDABackend → CuArray, Float64
│   └── RadiativeViewFactorMetalExt.jl  # Registers MetalBackend → MtlArray, Float32
├── test/
│   └── runtests.jl
└── Project.toml
```

## Quick Start

### 3D surface mesh

```julia
using RadiativeViewFactor

# Load a Gmsh mesh (surfaces must be in Physical Surface groups, 2nd-order elements)
mesh = load_mesh("geometry.msh")           # surface_dim=2 is the default

# Compute view factors on CPU (multi-threaded)
result = compute_view_factors(mesh; nquad=4)

# result.F_group[i,j] is the view factor from group result.group_names[i]
#                      to group result.group_names[j]
println(result.group_names)   # ordered list of group names
println(result.F_group)       # G × G matrix

# Post-processing checks
check_reciprocity(result)   # prints max relative error of Aᵢ Fᵢⱼ = Aⱼ Fⱼᵢ
check_closure(result)       # prints row-sum range (should be ≤ 1 for open geometry)
```

### 2D planar curve mesh (per unit depth)

```julia
# Physical Curve groups required; Line3 elements (Mesh.ElementOrder = 2)
mesh = load_mesh("planar.msh"; surface_dim=1)
result = compute_view_factors(mesh; nquad=6)
# F_group values are view factors per unit depth
```

### Obstruction detection

```julia
# Pass the physical group tags of surfaces that may block rays.
# The source and destination groups are excluded automatically per pair.
result = compute_view_factors(mesh; nquad=4, obstruction_groups=[3, 4])
```

### GPU — NVIDIA CUDA

```julia
using CUDA
result = compute_view_factors(mesh; nquad=4, backend=CUDABackend())

# With obstruction (works on all backends)
result = compute_view_factors(mesh; nquad=4, backend=CUDABackend(),
                              obstruction_groups=[3, 4])
```

### GPU — Apple Metal

```julia
using Metal
# Note: Metal uses Float32 internally; results are promoted to Float64
result = compute_view_factors(mesh; nquad=4, backend=MetalBackend())
```

## API Reference

### `load_mesh(filename; surface_dim=2, verbose=true) → MeshData`

Load a Gmsh `.msh` file and extract all 2nd-order elements belonging to named
physical groups.

| Argument | Default | Description |
|---|---|---|
| `surface_dim` | `2` | Element dimension: `2` for surface meshes, `1` for planar curve meshes |
| `verbose` | `true` | Print element-type summary |

Returns a `MeshData` with fields `coords`, `surface_elems`, `group_tags`,
`group_elems`, `group_tri_soup` (triangle or segment soups per group), and
`mesh_dim`.

### `compute_view_factors(mesh; ...) → ViewFactorResult`

Assemble the full view factor matrix at element and group level.

| Keyword | Default | Description |
|---|---|---|
| `nquad` | `4` | Gauss points per direction for surface elements (nquad² per pair); Gauss points along curve for Line3 elements |
| `obstruction_groups` | `Int[]` | Physical group tags of potential occluders (CPU and GPU) |
| `backend` | `CPU()` | `CPU()`, `CUDABackend()`, or `MetalBackend()` |
| `self_vf` | `false` | Include self view factors (concave elements; CPU only) |
| `verbose` | `true` | Print progress and row-sum diagnostics |

GPU backends do not support `surface_dim=1` (curve meshes); use `CPU()` for 2D problems.

### `ViewFactorResult` fields

| Field | Type | Description |
|---|---|---|
| `F_elem` | `Matrix{Float64}` | Element-level view factor matrix (N_elem × N_elem) |
| `A_elem` | `Vector{Float64}` | Element areas (or arc lengths for curve meshes) |
| `F_group` | `Matrix{Float64}` | Group-level view factor matrix (G × G) |
| `A_group` | `Vector{Float64}` | Group areas or arc lengths |
| `group_tags` | `Vector{Int}` | Physical group tags, sorted; row/column index for `F_group` |
| `group_names` | `Vector{String}` | Physical group names in the same order as `group_tags` |

`F_group[i,j]` is the view factor **from** the group named `group_names[i]`
**to** the group named `group_names[j]`.

To look up a pair by name:

```julia
i = findfirst(==("emitter"),  result.group_names)
j = findfirst(==("receiver"), result.group_names)
println("F(emitter → receiver) = ", result.F_group[i, j])
```

### `aggregate_by_group(result, mesh) → (F_group, A_group, tags, names)`

Re-aggregate element-level results to group level. Useful after modifying
`result.F_elem` directly.

### `check_reciprocity(result; tol=1e-4) → Bool`

Verify Aᵢ Fᵢⱼ ≈ Aⱼ Fⱼᵢ for all element pairs. Prints the maximum relative error.

### `check_closure(result; tol=1e-3) → Bool`

Verify that no row of `F_elem` sums to more than 1 + `tol`. Prints the
row-sum range. Row sums less than 1 are expected for open geometries (radiation
escaping through open boundaries or end caps).

## Theory

### 3D (surface meshes)

The view factor from surface i to surface j:

```
F_ij = (1/Aᵢ) ∬_Aᵢ ∬_Aⱼ  [cos θᵢ cos θⱼ / (π r²)]  H_ij  dAⱼ dAᵢ
```

where θᵢ, θⱼ are the angles between the line of sight and each surface normal,
r is the separation, and H_ij ∈ {0,1} is the visibility function.

Evaluated numerically using Gauss–Legendre on Quad8 elements (mapped from
[-1,1]²) or Dunavant quadrature on Tri6 elements.

### 2D (curve meshes, per unit depth)

```
F_ij = (1/Lᵢ) ∫_Lᵢ ∫_Lⱼ  [cos θᵢ cos θⱼ / (2 r)]  H_ij  dLⱼ dLᵢ
```

Evaluated using 1-D Gauss–Legendre on Line3 elements (mapped from [-1,1]).
The in-plane normal is computed by rotating the element tangent 90°
counter-clockwise; curves must be wound so that normals point toward the
opposing surfaces.

### Obstruction

On CPU: a BVH is built per unique set of active obstruction groups (source and
destination groups excluded) and reused across all element pairs sharing that
set. For 3D meshes the BVH holds a triangle soup and uses Möller–Trumbore
ray–triangle intersection; for 2D meshes it holds a line-segment soup and
uses 2D line–line intersection.

On GPU: the BVH is flattened to typed arrays (AABB bounds, node metadata
including miss-link pointers for stackless traversal, triangle vertices,
per-triangle group tags) and uploaded once before kernel launch. Each GPU
thread traverses the BVH using a stackless while-loop (no thread-local stack
memory), skipping any triangle whose group tag matches the emitter or receiver.

## Mesh Requirements

### Surface meshes (`surface_dim=2`)

- 2nd-order elements: `Quad8` (Gmsh type 16), `Quad9` (type 10, centre node
  dropped), or `Tri6` (type 9)
- All radiating surfaces in a **Physical Surface** group
- Obstruction surfaces in their own **Physical Surface** group(s)
- `Mesh.ElementOrder = 2` (or `-order 2`) before meshing

### Curve meshes (`surface_dim=1`)

- 2nd-order line elements: `Line3` (Gmsh type 8)
- All radiating curves in a **Physical Curve** group
- `Mesh.ElementOrder = 2` before meshing
- Curves wound so that the CCW-rotated tangent normal points toward opposing surfaces

Example Gmsh script for a 2D problem:

```gmsh
Mesh.ElementOrder = 2;
Physical Curve("emitter")     = {1};
Physical Curve("receiver")    = {2};
Physical Curve("obstruction") = {3};
```

## GPU Performance Notes

The GPU path is advantageous for large meshes (N ≳ 300–500 elements) without
obstruction, or N ≳ 500 with obstruction. For small meshes the fixed cost of
kernel compilation and host↔device data transfer dominates.

Metal uses Float32 internally (Apple GPUs do not support Float64 natively);
results are promoted to Float64 before aggregation. For geometries with
near-grazing element pairs or coordinates spanning many orders of magnitude,
Float32 precision (~1e-7 relative) may introduce small errors in individual
element-pair view factors.

The stackless BVH traversal on GPU eliminates the per-thread MVector stack
used in naive implementations, reducing register pressure and avoiding
potential register spilling on both CUDA and Metal backends.

## Dependencies

| Package | Role |
|---|---|
| `Gmsh` | Mesh file I/O |
| `KernelAbstractions` | Backend-agnostic GPU kernels |
| `StaticArrays` | Stack-allocated vectors for hot-path geometry |
| `LinearAlgebra`, `SparseArrays` | Standard library |
| `CUDA` *(optional weak dep)* | NVIDIA GPU backend |
| `Metal` *(optional weak dep)* | Apple Silicon GPU backend |
