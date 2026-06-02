# RadiativeViewFactor.jl

[![CI](https://github.com/rot4te/RadiativeViewFactor/actions/workflows/CI.yml/badge.svg)](https://github.com/rot4te/RadiativeViewFactor/actions/workflows/CI.yml)

A Julia package for computing **radiative view factors** between arbitrary surfaces
or curves discretized on 2nd-order meshes generated with [Gmsh](https://gmsh.info/).

## Features

- Reads Gmsh `.msh` files (v2.2 and v4) via the Gmsh Julia SDK
- **3D surface meshes** (`surface_dim=2`): Quad8, Quad9 (centre node silently dropped), and Tri6 elements
- **2D planar curve meshes** (`surface_dim=1`): Line3 elements; computes view factors per unit depth using the 2D kernel cos θᵢ cos θⱼ / (2r)
- Groups radiating geometry by **Gmsh physical groups**; view factors reported at both element and group level; rows and columns of `F_group` indexed by `result.group_tags` / `result.group_names`
- **Two integration methods** selectable per call:
  - *Gauss–Legendre quadrature*: pre-tabulated for n ≤ 5, Golub–Welsch for n > 5; Dunavant rules for triangular elements; 1-D Gauss–Legendre for Line3 curve elements
  - *Monte Carlo*: stratified area sampling with O(1/N) variance convergence; per-thread independent RNG streams on CPU; xorshift64 per-thread PRNG on GPU
- **Obstruction detection** via ray–triangle (3D) or ray–segment (2D) intersection on an axis-aligned BVH; works on both CPU and GPU backends with both integration methods
- `obstruction_groups` interface: pass physical group tags of potential occluders; source and destination groups are automatically excluded per pair
- Automatic **normal orientation correction** at load time: for curve meshes (`surface_dim=1`), element normals are oriented to point toward the adjacent transfinite surface interior, determined from mesh connectivity
- Optional **normal reversal**: `reverse_normals=true` flips all normals; `reverse_groups=[...]` flips specific physical groups only
- **Mesh visualisation** via an optional Makie extension: load any Makie backend and call `plot_mesh_normals(mesh)` to inspect element geometry and normal directions
- CPU backend: multi-threaded via `Threads.@threads`
- GPU backends: NVIDIA (`CUDABackend`, Float64) and Apple Silicon (`MetalBackend`, Float32) via KernelAbstractions.jl; stackless BVH traversal eliminates per-thread stack memory pressure
- **Reciprocity** and **closure** (row-sum) checks on the assembled matrix

## Project Layout

```
RadiativeViewFactor.jl/
├── src/
│   ├── RadiativeViewFactor.jl   # Package entry-point and public exports
│   ├── MeshIO.jl                # Gmsh mesh loading; element reading; normal orientation
│   ├── Quadrature.jl            # Gauss–Legendre (1-D and 2-D) and Dunavant rules
│   ├── Geometry.jl              # Shape functions, normals, Jacobians for all element types
│   ├── BVH.jl                   # Axis-aligned BVH; triangle and segment soup support
│   ├── RayCast.jl               # CPU visibility test; dispatches on mesh_dim
│   ├── ViewFactorKernel.jl      # 3D and 2D deterministic kernels; element-pair integrator
│   ├── MCKernel.jl              # CPU Monte Carlo integrator with stratified sampling
│   ├── Results.jl               # ViewFactorResult, _aggregate, check functions
│   ├── GPUBVH.jl                # Stackless flat BVH for GPU: build + inline traversal
│   ├── GPUKernels.jl            # KernelAbstractions deterministic kernels (Quad8 + Tri6)
│   ├── GPUMCKernels.jl          # KernelAbstractions Monte Carlo kernel (xorshift64 PRNG)
│   ├── Assembly.jl              # CPU assembly; integration dispatch; GPU hook registry
│   └── GPUAssembly.jl           # GPU assembly path; registers GPU hook at load time
├── ext/
│   ├── RadiativeViewFactorCUDAExt.jl    # Registers CUDABackend → CuArray, Float64
│   ├── RadiativeViewFactorMetalExt.jl  # Registers MetalBackend → MtlArray, Float32
│   └── RadiativeViewFactorMakieExt.jl  # plot_mesh_normals (loaded with any Makie backend)
├── test/
│   └── runtests.jl
└── Project.toml
```

## Quick Start

### 3D surface mesh — deterministic quadrature

```julia
using RadiativeViewFactor

mesh   = load_mesh("geometry.msh")   # surface_dim=2 is the default
result = compute_view_factors(mesh; nquad=4)

# F_group[i,j] = view factor from group_names[i] to group_names[j]
println(result.group_names)
println(result.F_group)

check_reciprocity(result)   # prints max relative error of Aᵢ Fᵢⱼ = Aⱼ Fⱼᵢ
check_closure(result)       # prints row-sum range
```

### 3D surface mesh — Monte Carlo

```julia
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000)

# Reproducible result with a fixed seed
using Random
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               rng=MersenneTwister(42))
```

### 2D planar curve mesh (per unit depth)

```julia
# Physical Curve groups required; Line3 elements (Mesh.ElementOrder = 2)
mesh   = load_mesh("planar.msh"; surface_dim=1)
result = compute_view_factors(mesh; nquad=6)
# F_group values are view factors per unit depth

# Normal orientation is corrected automatically.
# If normals are still wrong for some groups, flip them:
mesh = load_mesh("planar.msh"; surface_dim=1, reverse_normals=true)
mesh = load_mesh("planar.msh"; surface_dim=1, reverse_groups=[2, 5])
```

### Obstruction detection

```julia
# Tags of physical groups that may block rays.
# Source and destination groups are excluded automatically per pair.
result = compute_view_factors(mesh; nquad=4, obstruction_groups=[3, 4])

# Also works with Monte Carlo:
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               obstruction_groups=[3, 4])
```

### GPU — NVIDIA CUDA

```julia
using CUDA

# Deterministic
result = compute_view_factors(mesh; nquad=4, backend=CUDABackend())

# Monte Carlo
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               backend=CUDABackend())

# With obstruction
result = compute_view_factors(mesh; nquad=4, backend=CUDABackend(),
                               obstruction_groups=[3, 4])
```

### GPU — Apple Metal

```julia
using Metal
# Metal uses Float32 internally; results are promoted to Float64
result = compute_view_factors(mesh; nquad=4, backend=MetalBackend())
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               backend=MetalBackend())
```

### Mesh visualisation

```julia
using GLMakie   # or CairoMakie, WGLMakie
using RadiativeViewFactor

mesh = load_mesh("geometry.msh"; surface_dim=1)
fig  = plot_mesh_normals(mesh)

# Options
fig = plot_mesh_normals(mesh;
        normal_scale  = 0.05,          # arrow length in mesh units (auto if omitted)
        show_nodes    = true,          # scatter-plot element nodes
        show_indices  = true,          # annotate elements with index numbers
        group_colors  = Dict(1=>:red, 2=>:blue))  # override colours per group tag

save("normals.png", fig)   # requires CairoMakie for PNG/SVG/PDF output
```

## API Reference

### `load_mesh(filename; ...) → MeshData`

Load a Gmsh `.msh` file and extract all 2nd-order elements in named physical groups.

| Keyword | Default | Description |
|---|---|---|
| `surface_dim` | `2` | `2` for surface meshes; `1` for planar curve meshes |
| `reverse_normals` | `false` | Flip all element normals after loading |
| `reverse_groups` | `Int[]` | Flip normals for specific physical group tags only |
| `verbose` | `true` | Print element-type summary and orientation corrections |

For `surface_dim=1`, element normals are automatically oriented to point toward
the adjacent transfinite surface interior, determined from mesh element
connectivity (works with both `.msh` v2.2 and v4). `reverse_normals` and
`reverse_groups` are applied after this auto-correction.

Returns a `MeshData` with fields `coords`, `surface_elems`, `group_tags`,
`group_elems`, `group_tri_soup`, and `mesh_dim`.

### `compute_view_factors(mesh; ...) → ViewFactorResult`

Assemble the full view factor matrix at element and group level.

| Keyword | Default | Description |
|---|---|---|
| `nquad` | `4` | Gauss points per direction (ignored when `monte_carlo=true`) |
| `obstruction_groups` | `Int[]` | Physical group tags of potential occluders |
| `backend` | `CPU()` | `CPU()`, `CUDABackend()`, or `MetalBackend()` |
| `self_vf` | `false` | Include self view factors (concave elements; CPU only) |
| `monte_carlo` | `false` | Use Monte Carlo integration instead of quadrature |
| `n_samples` | `10000` | MC sample pairs per element pair (ignored when `monte_carlo=false`) |
| `rng` | `Random.default_rng()` | RNG for CPU MC path; ignored on GPU |
| `verbose` | `true` | Print progress and row-sum diagnostics |

Notes:
- GPU backends do not support `surface_dim=1`; use `CPU()` for 2D problems
- Monte Carlo convergence is O(1/√n_samples) for plain sampling, O(1/n_samples) with stratification (default); increase `n_samples` for tighter accuracy
- For reproducible MC results pass an explicit seeded RNG, e.g. `rng=MersenneTwister(42)`

### `ViewFactorResult` fields

| Field | Type | Description |
|---|---|---|
| `F_elem` | `Matrix{Float64}` | Element-level view factor matrix (N_elem × N_elem) |
| `A_elem` | `Vector{Float64}` | Element areas or arc lengths |
| `F_group` | `Matrix{Float64}` | Group-level view factor matrix (G × G) |
| `A_group` | `Vector{Float64}` | Group areas or arc lengths |
| `group_tags` | `Vector{Int}` | Physical group tags, sorted; row/column labels for `F_group` |
| `group_names` | `Vector{String}` | Physical group names in the same order as `group_tags` |

`F_group[i,j]` is the view factor **from** `group_names[i]` **to** `group_names[j]`.

```julia
# Look up a specific pair by name
i = findfirst(==("emitter"),  result.group_names)
j = findfirst(==("receiver"), result.group_names)
println("F(emitter → receiver) = ", result.F_group[i, j])
```

### `aggregate_by_group(result, mesh) → (F_group, A_group, tags, names)`

Re-aggregate element-level results to group level (useful after modifying `result.F_elem`).

### `check_reciprocity(result; tol=1e-4) → Bool`

Verify Aᵢ Fᵢⱼ ≈ Aⱼ Fⱼᵢ for all element pairs. Prints the maximum relative error.

### `check_closure(result; tol=1e-3) → Bool`

Verify no row of `F_elem` sums to more than 1 + `tol`. Prints the row-sum range.
Row sums less than 1 are expected for open geometries.

### `plot_mesh_normals(mesh; ...) → Figure`

Visualise mesh elements with normal arrows. Requires any Makie backend to be
loaded first (`GLMakie`, `CairoMakie`, or `WGLMakie`).

| Keyword | Default | Description |
|---|---|---|
| `normal_scale` | auto | Arrow length in mesh units; auto-estimated from bounding box if omitted |
| `group_colors` | auto | `Dict{Int,Any}` mapping group tag → Makie colour |
| `show_nodes` | `false` | Scatter-plot all element nodes |
| `show_indices` | `false` | Annotate each element with its index number |
| `backend_3d` | auto | Force `Axis3`; defaults to `true` for surface meshes |

## Theory

### 3D (surface meshes)

```
F_ij = (1/Aᵢ) ∬_Aᵢ ∬_Aⱼ  [cos θᵢ cos θⱼ / (π r²)]  H_ij  dAⱼ dAᵢ
```

### 2D (curve meshes, per unit depth)

```
F_ij = (1/Lᵢ) ∫_Lᵢ ∫_Lⱼ  [cos θᵢ cos θⱼ / (2 r)]  H_ij  dLⱼ dLᵢ
```

### Integration methods

**Gauss–Legendre quadrature** evaluates the double integral at fixed reference-space
points. Convergence is spectral for smooth integrands but degrades near singularities
(e.g. nearly-touching or shared-edge element pairs).

**Monte Carlo** draws stratified random sample pairs from each element pair.
The N samples are divided into ⌊√N⌋ × ⌊√N⌋ strata on the reference domain;
one point is drawn uniformly within each stratum. This gives O(1/N) variance
convergence rather than O(1/√N) for plain MC. MC is advantageous for geometries
with many obstructions or near-singular element pairs.

### Obstruction

**CPU**: a BVH is built once per unique set of active obstruction groups and
reused across all pairs sharing that set. Triangle soup for 3D (Möller–Trumbore
intersection); segment soup for 2D (Cramer's rule line–line intersection).

**GPU**: the BVH is flattened to plain device arrays with miss-link pointers for
stackless traversal. Each thread independently traverses the BVH with no
thread-local stack, eliminating register pressure from MVector storage.
Per-triangle group tags allow each thread to skip triangles belonging to the
emitter or receiver group without host-side pre-filtering.

## Mesh Requirements

### Surface meshes (`surface_dim=2`)

- 2nd-order elements: `Quad8` (type 16), `Quad9` (type 10, centre node dropped), or `Tri6` (type 9)
- Radiating surfaces in **Physical Surface** groups; obstructors in separate Physical Surface groups
- `Mesh.ElementOrder = 2` before meshing

### Curve meshes (`surface_dim=1`)

- `Line3` elements (Gmsh type 8); `Mesh.ElementOrder = 2` before meshing
- Radiating curves in **Physical Curve** groups
- Normal orientation is corrected automatically from mesh adjacency; use `reverse_normals` or `reverse_groups` if the auto-correction produces wrong results for specific groups

```gmsh
Mesh.ElementOrder = 2;
Physical Curve("emitter")     = {1};
Physical Curve("receiver")    = {2};
Physical Curve("obstruction") = {3};
```

## Performance Notes

### Quadrature vs Monte Carlo

Quadrature is generally faster and more accurate for smooth geometries with low
obstruction. Monte Carlo is preferable when:
- Many obstructions are present (MC pays the BVH cost only for non-zero kernel samples)
- Near-singular element pairs exist (shared edges, near-touching surfaces)
- A rough estimate is acceptable at low `n_samples` cost

### GPU vs CPU

GPU backends outperform CPU for N ≳ 300–500 elements (deterministic) or
N ≳ 200 elements (Monte Carlo, which has higher arithmetic intensity per thread).
For small meshes, kernel compilation and host↔device transfer dominate.

Metal uses Float32; results are promoted to Float64 before aggregation.
Near-grazing element pairs or geometries spanning many orders of magnitude may
show small Float32 rounding errors in individual element-pair values.

The stackless BVH on GPU uses a fixed number of scalar registers regardless of
tree depth, avoiding register spilling that would occur with a thread-local stack.

## Dependencies

| Package | Role |
|---|---|
| `Gmsh` | Mesh file I/O |
| `KernelAbstractions` | Backend-agnostic GPU kernels |
| `StaticArrays` | Stack-allocated vectors for hot-path geometry |
| `LinearAlgebra`, `SparseArrays`, `Random`, `Statistics` | Standard library |
| `CUDA` *(optional weak dep)* | NVIDIA GPU backend |
| `Metal` *(optional weak dep)* | Apple Silicon GPU backend |
| `Makie` *(optional weak dep)* | Mesh visualisation via `plot_mesh_normals` |
