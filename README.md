# RadiativeViewFactor.jl

[![CI](https://github.com/rot4te/RadiativeViewFactor/actions/workflows/CI.yml/badge.svg)](https://github.com/rot4te/RadiativeViewFactor/actions/workflows/CI.yml)

A Julia package for computing **radiative view factors** between arbitrary surfaces
or curves. Meshes may be **structured or unstructured** and **1st- or 2nd-order**;
any format readable by [Gmsh](https://gmsh.info/) is supported, plus XML VTK
(`.vtu`) via an optional ReadVTK.jl extension.

## Features

- **Structured and unstructured meshes** are both supported. The solver operates
  on a flat list of elements identified purely by node connectivity — there is no
  structured-grid (`i,j`) assumption anywhere in assembly, quadrature, or the BVH.
  Unstructured triangulations (Gmsh's default), structured/transfinite meshes,
  and mixed-element meshes all work.
- **Any Gmsh-readable format**: `.msh` (v2.2 and v4), `.stl`, `.step`/`.stp`,
  Nastran `.bdf`/`.nas`, `.med`, legacy `.vtk`, etc. XML VTK unstructured grids
  (`.vtu`, XML-form `.vtk`) are auto-detected and read through ReadVTK.jl when
  `using ReadVTK` is in scope.
- **1st- and 2nd-order elements**, in any mix within one mesh:
  - **3D surface meshes** (`surface_dim=2`): Tri3, Quad4 (1st order); Tri6, Quad8,
    Quad9 (centre node dropped) (2nd order)
  - **2D planar curve meshes** (`surface_dim=1`): Line2 (1st order), Line3
    (2nd order); computes view factors per unit depth using the 2D kernel
    cos θᵢ cos θⱼ / (2r)
- Groups radiating geometry by **named physical groups**; view factors reported at both element and group level; rows and columns of `F_group` indexed by `result.group_tags` / `result.group_names`. Formats without named groups (e.g. STL, VTK) fall back to a single synthetic `"default"` group, or — for VTK — a per-cell region array
- **Three integration methods** selectable per call:
  - *Gauss–Legendre quadrature* (default): pre-tabulated for n ≤ 5, Golub–Welsch algorithm for n > 5; Dunavant rules for triangular elements; 1-D Gauss–Legendre for Line3 curve elements
  - *Monte Carlo*: stratified area sampling with O(1/N) variance convergence per element pair; per-thread independent RNG streams on CPU; xorshift64 per-thread PRNG on GPU
  - *Duffy transformation* (`use_duffy=true`): Sauter–Schwab singularity regularization for Quad8 element pairs sharing a vertex (8-region decomposition) or edge (5-region decomposition); falls back to standard quadrature for non-adjacent pairs and non-Quad8 families
- **Obstruction detection** via ray–triangle (3D) or ray–segment (2D) intersection on an axis-aligned BVH; works on both CPU and GPU backends and with all three integration methods
- `obstruction_groups` interface: pass physical group tags of potential occluders; source and destination groups are automatically excluded per pair
- Automatic **normal orientation correction** for curve meshes (`surface_dim=1`): element normals are oriented toward the adjacent surface interior, determined from **mesh connectivity** (shared nodes + nearest centroid), not CAD topology — so it works for unstructured curve meshes, not only structured/transfinite ones (and with `.msh` v2.2 and v4). For surface meshes (`surface_dim=2`) normals follow element winding, which Gmsh keeps consistent within each surface; opposing surfaces must be wound to face each other (or use `reverse_normals`)
- Optional **normal reversal**: `reverse_normals=true` flips all element normals at load time
- **Mesh visualisation** via an optional Plots.jl extension: `using Plots` then `plot_mesh_normals(mesh)` to inspect element geometry and normal directions
- CPU backend: multi-threaded via `Threads.@threads`
- GPU backends: NVIDIA (`CUDABackend`, Float64) and Apple Silicon (`MetalBackend`, Float32) via KernelAbstractions.jl; stackless BVH traversal eliminates per-thread stack memory pressure
- **Reciprocity** and **closure** (row-sum) checks on the assembled matrix

## Project Layout

```
RadiativeViewFactor.jl/
├── src/
│   ├── RadiativeViewFactor.jl   # Package entry-point and public exports
│   ├── MeshIO.jl                # Mesh loading (Gmsh + VTK routing); element reading; normal orientation
│   ├── Quadrature.jl            # Gauss–Legendre (1-D and 2-D) and Dunavant rules
│   ├── Geometry.jl              # Shape functions, normals, Jacobians for all element types
│   ├── BVH.jl                   # Axis-aligned BVH; triangle and segment soup support
│   ├── RayCast.jl               # CPU visibility test; dispatches on mesh_dim
│   ├── ViewFactorKernel.jl      # 3D and 2D deterministic kernels; element-pair integrator
│   ├── DuffyKernel.jl           # Sauter–Schwab Duffy transformation for singular pairs
│   ├── MCKernel.jl              # CPU Monte Carlo integrator with stratified sampling
│   ├── Results.jl               # ViewFactorResult, _aggregate, check functions
│   ├── GPUBVH.jl                # Stackless flat BVH for GPU: build + inline traversal
│   ├── GPUKernels.jl            # KernelAbstractions deterministic kernels (Quad4/8 + Tri3/6)
│   ├── GPUMCKernels.jl          # KernelAbstractions Monte Carlo kernel (xorshift64 PRNG)
│   ├── Assembly.jl              # CPU assembly; integration dispatch; GPU hook registry
│   └── GPUAssembly.jl           # GPU assembly path; registers GPU hook at load time
├── ext/
│   ├── RadiativeViewFactorCUDAExt.jl     # Registers CUDABackend → CuArray, Float64
│   ├── RadiativeViewFactorMetalExt.jl    # Registers MetalBackend → MtlArray, Float32
│   ├── RadiativeViewFactorPlotsExt.jl    # plot_mesh_normals (Plots.jl)
│   └── RadiativeViewFactorReadVTKExt.jl  # XML VTK (.vtu) loading via ReadVTK.jl
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

# Look up a specific pair by name
i = findfirst(==("emitter"),  result.group_names)
j = findfirst(==("receiver"), result.group_names)
println("F(emitter → receiver) = ", result.F_group[i, j])

check_reciprocity(result)   # prints max relative error of Aᵢ Fᵢⱼ = Aⱼ Fⱼᵢ
check_closure(result)       # prints row-sum range
```

### With Duffy transformation (near corner/edge singularities)

```julia
# Automatically detects shared vertices and edges between Quad8 elements
# and applies the Sauter–Schwab regularization only where needed.
result = compute_view_factors(mesh; nquad=6, use_duffy=true)
```

### Monte Carlo

```julia
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000)

# Reproducible result with a fixed seed
using Random
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               rng=MersenneTwister(42))
```

### 2D planar curve mesh (per unit depth)

```julia
# Physical Curve groups; Line2 (1st-order) or Line3 (2nd-order) elements
mesh   = load_mesh("planar.msh"; surface_dim=1)
result = compute_view_factors(mesh; nquad=6)
# F_group values are view factors per unit depth

# Normal orientation is corrected automatically toward the adjacent surface
# interior (from mesh connectivity, structured or unstructured). If a group
# comes out facing the wrong way, flip all normals at load time:
mesh = load_mesh("planar.msh"; surface_dim=1, reverse_normals=true)
```

### Other formats (STL, STEP, VTK, …)

```julia
# Any format Gmsh can open is detected from the extension:
mesh = load_mesh("part.stl")        # no named groups → single "default" group
mesh = load_mesh("assembly.step")

# XML VTK unstructured grids (.vtu) need ReadVTK loaded:
using ReadVTK
mesh = load_mesh("grid.vtu")                       # auto-detected
mesh = load_vtu("grid.vtu"; group_field="RegionId") # per-cell region → groups
```

### Obstruction detection

```julia
# Tags of physical groups that may block rays.
# Source and destination groups are excluded automatically per pair.
result = compute_view_factors(mesh; nquad=4, obstruction_groups=[3, 4])

# Also works with Monte Carlo and Duffy:
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               obstruction_groups=[3, 4])
result = compute_view_factors(mesh; nquad=6, use_duffy=true,
                               obstruction_groups=[3, 4])
```

### GPU — NVIDIA CUDA

```julia
using CUDA
result = compute_view_factors(mesh; nquad=4, backend=CUDABackend())
result = compute_view_factors(mesh; monte_carlo=true, n_samples=50000,
                               backend=CUDABackend(), obstruction_groups=[3])
```

### GPU — Apple Metal

```julia
using Metal
# Metal uses Float32 internally; results are promoted to Float64
result = compute_view_factors(mesh; nquad=4, backend=MetalBackend())
```

### Mesh visualisation

```julia
using Plots
using RadiativeViewFactor

mesh = load_mesh("geometry.msh"; surface_dim=1)
fig  = plot_mesh_normals(mesh)

fig = plot_mesh_normals(mesh;
        normal_scale  = 0.05,
        show_nodes    = true,
        show_indices  = true,
        group_colors  = Dict(1=>:red, 2=>:blue))

savefig(fig, "normals.png")
```

> **Full API documentation** is available at the [package documentation site](https://rot4te.github.io/RadiativeViewFactor.jl).

## Theory

### 3D view factor (surface meshes)

$$F_{ij} = \frac{1}{A_i} \iint_{A_i} \iint_{A_j} \frac{\cos\theta_i \cos\theta_j}{\pi r^2} \ H_{ij} \ dA_j \, dA_i$$

### 2D view factor (curve meshes, per unit depth)

$$F_{ij} = \frac{1}{L_i} \int_{L_i} \int_{L_j} \frac{\cos\theta_i \cos\theta_j}{2r} \ H_{ij} \ dL_j \ dL_i$$

The factor of 2 rather than π in the denominator follows from integrating the 2D radiation intensity over the hemisphere, which gives π/2 rather than π.

### Gauss–Legendre quadrature

The double integral is evaluated at fixed tensor-product Gauss points on each element pair. Pre-tabulated rules for n ≤ 5 use classical nodes and weights (Abramowitz & Stegun §25.4); the Golub–Welsch algorithm is used for n > 5. Dunavant rules are used for triangular elements. Convergence is spectral for smooth integrands but degrades near corner singularities.

### Duffy transformation

For Quad8 element pairs sharing a vertex or edge, the `1/r²` singularity in the kernel is integrable but not efficiently resolved by standard quadrature. The Duffy transformation introduces a radial coordinate ρ measuring distance to the singular point; the Jacobian ρ³ of the 4D transformation cancels the singularity, leaving a bounded integrand on which Gauss quadrature converges rapidly. The implementation uses the Sauter–Schwab decomposition:

- **Common vertex**: 8-region decomposition, each integrated with `nquad⁴` points (total `8 × nquad⁴`)
- **Common edge**: 5-region decomposition, each integrated with `nquad⁴` points (total `5 × nquad⁴`)

Pairs with no shared nodes use standard quadrature (`nquad⁴` points). The singularity type is detected automatically from shared global node indices.

Note: the 2D kernel `1/r` for Line3 elements produces a logarithmic divergence (not `1/r²`) at shared endpoints, which is physically real and not regularizable by the Duffy transformation. `use_duffy` has no effect for `surface_dim=1`.

### Monte Carlo integration

The MC estimator for each element pair draws N stratified sample pairs (xᵢ, xⱼ):

$$\iint K \, dA_j \, dA_i \approx \frac{A_i \cdot A_j}{N} \sum_{k=1}^{N} K(x_i^{(k)}, n_i^{(k)}, x_j^{(k)}, n_j^{(k)}) \cdot H_{ij}^{(k)}$$

Samples are drawn on a ⌊√N⌋ × ⌊√N⌋ stratified grid within the reference element, giving O(1/N) variance convergence for smooth integrands rather than O(1/√N) for plain Monte Carlo. Near corner singularities the variance of the MC estimator diverges (infinite variance for the `1/r²` kernel), making `use_duffy` preferable for those geometries.

On GPU, each thread uses an independent xorshift64 pseudo-random number stream seeded by mixing the global seed with the thread index via the splitmix64 hash.

### Obstruction detection

**CPU**: a BVH is built once per unique set of active obstruction groups and reused across all pairs. Triangle soup for 3D (Möller–Trumbore ray–triangle intersection); line-segment soup for 2D (Cramer's rule line–line intersection).

**GPU**: the BVH is flattened to plain device arrays with miss-link pointers for stackless traversal (no per-thread MVector). Per-triangle group tags allow each GPU thread to skip triangles belonging to the emitter or receiver group without host-side pre-filtering.

## Mesh Requirements

Meshes may be **structured or unstructured** — the solver only needs element node
connectivity. Both **1st- and 2nd-order** elements are accepted, and they may be
mixed within a single mesh. Any [Gmsh-readable format](https://gmsh.info/) works;
XML VTK (`.vtu`) works through ReadVTK.jl.

### Surface meshes (`surface_dim=2`)

| Order | Triangle | Quadrilateral |
|---|---|---|
| 1st | `Tri3` (Gmsh type 2) | `Quad4` (type 3) |
| 2nd | `Tri6` (type 9) | `Quad8` (type 16), `Quad9` (type 10, centre node dropped) |

- Radiating surfaces in named groups (**Physical Surface** in Gmsh); obstructors in separate groups. Formats without named groups get a single `"default"` group.
- Element normals follow node winding, which Gmsh keeps consistent within each surface. **Opposing surfaces must be wound to face each other**; use `reverse_normals=true` to flip all normals at load time if a mesh comes in back-to-front.

### Curve meshes (`surface_dim=1`)

| Order | Line |
|---|---|
| 1st | `Line2` (Gmsh type 1) |
| 2nd | `Line3` (type 8) |

- Radiating curves in named groups (**Physical Curve** in Gmsh).
- Normal orientation is corrected automatically toward the adjacent surface interior, computed from mesh connectivity — so it works for unstructured curve meshes, not just structured/transfinite ones. Use `reverse_normals=true` to flip all normals if needed.

```gmsh
// Example Gmsh setup. Mesh.ElementOrder = 1 (default) gives Line2/Tri3/Quad4;
// set it to 2 for Line3/Tri6/Quad8. Structured meshing is optional.
Mesh.ElementOrder = 2;
Physical Curve("emitter")     = {1};
Physical Curve("receiver")    = {2};
Physical Curve("obstruction") = {3};
```

## Performance Notes

### Integration method selection

| Method | Best for | Avoid when |
|---|---|---|
| Quadrature | Smooth geometry, well-separated surfaces | Near corner/edge singularities |
| Duffy | Shared vertices/edges between Quad8 elements | GPU, Monte Carlo, Tri6/Line3 elements |
| Monte Carlo | Many obstructions, near-singular pairs, rough estimates | High-accuracy requirements with few obstructions |

### GPU vs CPU crossover

GPU backends outperform CPU for N ≳ 300–500 elements (quadrature) or N ≳ 200 (Monte Carlo). For small meshes, kernel compilation and host↔device transfer dominate. Metal uses Float32; results are promoted to Float64 before aggregation.

### Duffy transformation cost

The Duffy path evaluates `8 × nquad⁴` (vertex) or `5 × nquad⁴` (edge) kernel evaluations per singular pair, compared to `nquad⁴` for standard quadrature. Since only adjacent element pairs trigger Duffy, the overhead is proportional to the number of shared edges/vertices in the mesh. A lower `nquad` (e.g. 4) with `use_duffy=true` typically outperforms a higher `nquad` (e.g. 16) with standard quadrature for inclined-plate geometries.

## References

The following works informed the numerical methods in this package:

**View factor theory and quadrature:**
- Howell, J. R., Mengüç, M. P., & Siegel, R. (2020). *Thermal Radiation Heat Transfer* (7th ed.). CRC Press. — View factor definitions, reciprocity, crossed-string method, and analytical reference cases.
- Hamilton, D. C., & Morgan, W. R. (1952). *Radiant interchange configuration factors*. NACA Technical Note 2836. — Original tabulation of configuration factor formulae.

**Isoparametric finite elements:**
- Zienkiewicz, O. C., Taylor, R. L., & Zhu, J. Z. (2005). *The Finite Element Method: Its Basis and Fundamentals* (6th ed.). Elsevier. — Quad8 serendipity and Tri6 shape functions, isoparametric mapping, Gauss quadrature.

**Duffy transformation and boundary element singularity treatment:**
- Sauter, S. A., & Schwab, C. (2011). *Boundary Element Methods*. Springer. — Sauter–Schwab common-vertex and common-edge decompositions (§5.3), the definitive reference for the 4D Duffy regularization used in `DuffyKernel.jl`.
- Duffy, M. G. (1982). Quadrature over a pyramid or cube of integrands with a singularity at a vertex. *SIAM Journal on Numerical Analysis*, 19(6), 1260–1262. — Original Duffy transformation paper.

**Gaussian quadrature:**
- Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules. *Mathematics of Computation*, 23(106), 221–230. — Golub–Welsch algorithm used in `Quadrature.jl` for n > 5.
- Dunavant, D. A. (1985). High degree efficient symmetrical Gaussian quadrature rules for the triangle. *International Journal for Numerical Methods in Engineering*, 21(6), 1129–1148. — Dunavant triangle quadrature rules used for Tri6 elements.

**Ray–triangle intersection:**
- Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray/triangle intersection. *Journal of Graphics Tools*, 2(1), 21–28. — Algorithm used in `BVH.jl` for obstruction testing.

**BVH construction and traversal:**
- Wald, I., et al. (2007). *Ray Tracing Gems* — Stackless BVH traversal via miss-link (skip pointer) encoding, used in `GPUBVH.jl`.

**Monte Carlo integration:**
- Pharr, M., Jakob, W., & Humphreys, G. (2023). *Physically Based Rendering: From Theory to Implementation* (4th ed.). MIT Press. — Stratified sampling, variance reduction, and Monte Carlo estimators for light transport integrals.

**GPU random number generation:**
- Marsaglia, G. (2003). Xorshift RNGs. *Journal of Statistical Software*, 8(14). — xorshift64 PRNG used in `GPUMCKernels.jl`.
- Steele, G. L., Lea, D., & Flood, C. H. (2014). Fast splittable pseudorandom number generators. *ACM SIGPLAN Notices*, 49(10). — splitmix64 hash used to derive per-thread seeds.

## Dependencies

| Package | Role |
|---|---|
| `Gmsh` | Mesh file I/O |
| `KernelAbstractions` | Backend-agnostic GPU kernels |
| `StaticArrays` | Stack-allocated vectors for hot-path geometry |
| `LinearAlgebra`, `SparseArrays`, `Random`, `Statistics` | Standard library |
| `CUDA` *(optional weak dep)* | NVIDIA GPU backend |
| `Metal` *(optional weak dep)* | Apple Silicon GPU backend |
| `Plots` *(optional weak dep)* | Mesh visualisation via `plot_mesh_normals` |
| `ReadVTK` *(optional weak dep)* | Loading XML VTK (`.vtu`) unstructured grids |
