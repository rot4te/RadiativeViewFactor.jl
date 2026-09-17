# RadiativeViewFactor.jl

[![CI](https://github.com/rot4te/RadiativeViewFactor/actions/workflows/CI.yml/badge.svg)](https://github.com/rot4te/RadiativeViewFactor/actions/workflows/CI.yml)

A Julia package for computing **radiative view factors** between arbitrary surfaces
or curves. Meshes may be **structured or unstructured** and **1st- or 2nd-order**;
any format readable by [Gmsh](https://gmsh.info/) is supported, plus XML VTK
(`.vtu`) via an optional ReadVTK.jl extension.

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

### Ray-shooting Monte Carlo (fastest for large or obstructed 3D meshes)

```julia
result = compute_view_factors(mesh; raytrace=true, n_rays=10000)               # CPU

using Random
result = compute_view_factors(mesh; raytrace=true, n_rays=10000, rng=Xoshiro(42))

using CUDA   # or Metal
result = compute_view_factors(mesh; raytrace=true, n_rays=10000,
                               backend=CUDABackend())                          # GPU — own kernel
```

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
│   ├── MCKernel.jl              # CPU Monte Carlo integrator with stratified sampling (pair-area)
│   ├── RayTraceKernel.jl        # CPU Monte Carlo integrator via ray-shooting (whole-scene BVH)
│   ├── Results.jl               # ViewFactorResult, _aggregate, check functions
│   ├── GPUBVH.jl                # Stackless flat BVH for GPU: build + inline traversal
│   ├── GPUKernels.jl            # KernelAbstractions deterministic kernels (Quad4/8 + Tri3/6)
│   ├── GPUMCKernels.jl          # KernelAbstractions Monte Carlo kernel (xorshift64 PRNG, pair-area)
│   ├── GPURayTraceKernels.jl    # KernelAbstractions Monte Carlo kernel (ray-shooting, one thread/element)
│   ├── Assembly.jl              # CPU assembly; integration dispatch; GPU hook registry
│   └── GPUAssembly.jl           # GPU assembly path; registers GPU hook at load time
├── ext/
│   ├── RadiativeViewFactorCUDAExt.jl     # Registers CUDABackend → CuArray, Float64
│   ├── RadiativeViewFactorMetalExt.jl    # Registers MetalBackend → MtlArray, Float32
│   ├── RadiativeViewFactorPlotsExt.jl    # plot_mesh_normals (Plots.jl)
│   └── RadiativeViewFactorReadVTKExt.jl  # XML VTK (.vtu) loading via ReadVTK.jl
├── benchmarks/
│   ├── common.jl                # Shared mesh generators and timing helpers
│   ├── quadrature_bench.jl      # Deterministic assembly benchmark (sweeps N)
│   ├── montecarlo_bench.jl      # Monte Carlo assembly benchmark (sweeps n_samples)
│   ├── RESULTS.md               # Before/after numbers for the pre-evaluation optimization
│   └── howell/                  # Validation against Howell's published configuration-factor catalog
│       ├── geom.jl              # Gmsh geometry builders for the catalog cases
│       ├── cases.jl             # Shared case list (geometry + solver options) for run.jl and run_raytrace.jl
│       ├── tables.jl            # Reference values transcribed from the published Howell catalog tables
│       ├── analytic.jl          # Closed-form configuration factors from the Howell catalog
│       ├── run.jl               # Driver: quadrature/Monte Carlo kernels vs. closed-form values
│       ├── run_raytrace.jl      # Driver: ray-shooting Monte Carlo kernel, appends to results.csv
│       ├── results.csv          # Accumulated results from both drivers
│       └── RESULTS.md           # Summary of validation results
├── test/
│   └── runtests.jl
└── Project.toml
```
