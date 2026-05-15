# ViewFactors.jl

A modular Julia package for computing **radiative view factors** between arbitrary surfaces
discretized on conformal, hexahedral, 2nd-order meshes generated with [Gmsh](https://gmsh.info/).

## Features

- Reads Gmsh `.msh` files via the `Gmsh` Julia SDK
- Supports **2nd-order (serendipity) quadrilateral** surface elements (`Quad8`)
- Groups surfaces by **Gmsh physical groups**
- Computes view factors using **Gaussian quadrature** on each element pair
- **Obstruction detection** via ray–triangle intersection (Möller–Trumbore) on a BVH
- Enforces the **reciprocity** relation and **row-sum** (closure) check
- Outputs a dense or sparse `F[i,j]` matrix between physical groups or individual elements

## Project Layout

```
RadiativeViewFactor.jl/
├── src/
│   ├── RadiativeViewFactor.jl          # Package entry-point, exports
│   ├── MeshIO.jl               # Gmsh mesh loading, physical-group extraction
│   ├── Geometry.jl             # Element geometry: normals, Jacobians, quadrature maps
│   ├── Quadrature.jl           # Gauss–Legendre rules on [-1,1]²
│   ├── BVH.jl                  # Axis-aligned bounding-volume hierarchy for ray casting
│   ├── RayCast.jl              # Möller–Trumbore ray–triangle intersection + obstruction test
│   ├── ViewFactorKernel.jl     # Double-area integral kernel, element-pair VF
│   └── Assembly.jl             # Assemble full F matrix; physical-group aggregation
├── test/
│   └── runtests.jl
└── Project.toml
```

## Quick Start

```julia
using RadiativeViewFactors

# Load mesh
mesh = load_mesh("geometry.msh")

# Compute element-level view factor matrix (with obstruction checking)
F_elem = compute_view_factors(mesh; nquad=4, check_obstruction=true)

# Aggregate to physical groups
F_groups = aggregate_by_group(F_elem, mesh)

println(F_groups)
```

## Dependencies

- `Gmsh` (Julia SDK, wraps the C API)
- `LinearAlgebra`, `StaticArrays`, `SparseArrays` (stdlib / registered)

## Theory

The view factor from surface i to surface j is:

```
F_ij = (1/Aᵢ) ∬_Aᵢ ∬_Aⱼ  [cos θᵢ cos θⱼ / (π r²)] H_ij dAⱼ dAᵢ
```

where θᵢ, θⱼ are angles between the line of sight and each surface normal,
r is the distance between differential areas, and H_ij ∈ {0,1} is the
visibility function (0 = obstructed).

The double surface integral is evaluated numerically using a tensor-product
Gauss–Legendre rule on each pair of `Quad8` elements mapped to [-1,1]².
