# RadiativeViewFactor.jl

[![CI](https://github.com/rot4te/RadiativeViewFactor.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/rot4te/RadiativeViewFactor.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/rot4te/RadiativeViewFactor.jl/branch/main/graph/badge.svg)](https://app.codecov.io/github/rot4te/RadiativeViewFactor.jl)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://rot4te.github.io/RadiativeViewFactor.jl/dev/)

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
# compute on CPU kernel
result = compute_view_factors(mesh; raytrace=true, n_rays=10000)               

# compute using your choice of random seed
using Random
result = compute_view_factors(mesh; raytrace=true, n_rays=10000, rng=Xoshiro(42))

# compute on Nvidia or Apple GPU architecture
using CUDA   # or Metal
result = compute_view_factors(mesh; raytrace=true, n_rays=10000,
                            backend=CUDABackend() #= backend=MetalBackend() =#)
```
