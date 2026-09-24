# RadiativeViewFactor.jl

A Julia package for computing **radiative view factors** between arbitrary surfaces
or curves. Meshes may be **structured or unstructured** and **1st- or 2nd-order**;
any [Gmsh](https://gmsh.info/)-readable format is supported, plus XML VTK (`.vtu`)
via an optional ReadVTK.jl extension and Nek5000/NekRS `.re2` meshes through a
built-in reader.

## Overview

RadiativeViewFactor.jl evaluates the double-surface integral that defines the
geometric view factor between pairs of finite surfaces. The geometry is supplied
as a mesh: the solver works from element node connectivity alone, so structured
and unstructured meshes are treated identically, and 1st- and 2nd-order elements
may be mixed freely. Four integration strategies are available:

- **Gauss–Legendre quadrature** — the default; spectral convergence for smooth
  geometries
- **Monte Carlo (pair-area sampling)** — stratified sampling of point pairs on
  each element pair; adjacent and near pairs are patched with the Duffy
  transformation automatically
- **Duffy transformation** — singularity-regularizing change of variables for
  Quad4/Quad8 element pairs sharing a vertex or edge; gives accurate results
  for inclined surfaces with common edges
- **Monte Carlo (ray-shooting)** — cosine-weighted rays shot from each element
  into the whole scene at once; the fastest method for large or obstructed 3D
  meshes, since obstruction and visibility fall out of the same BVH query

All four methods run on CPU and support **obstruction detection** via
BVH-accelerated ray casting. Quadrature, both Monte Carlo variants, and
obstruction also run on GPU (CUDA and Metal), and so does the Duffy
transformation, for 3-D surface meshes.

Beyond the integrators, the package can:

- restrict the radiating surface to a subset of groups while every group still
  shadows (`radiating_groups`), which cuts the cost from `O(N_total²)` to
  `O(N_radiating²)`
- skip element pairs that provably cannot see each other (`facing_cull`,
  on by default)
- check reciprocity and closure of a result, and enforce both on a closed
  enclosure (`enforce_closure`)
- write view factors for a Nek5000/NekRS surface-to-surface radiation case
  (`write_nekrs_view_factors`)

## Installation

```julia
using Pkg
Pkg.add("RadiativeViewFactor")
```

RadiativeViewFactor.jl is registered in the Julia General registry.

Requires Julia 1.10 or later.

For GPU support, install the relevant backend before loading the package:

```julia
Pkg.add("CUDA")   # NVIDIA
Pkg.add("Metal")  # Apple Silicon
```

For mesh visualisation:

```julia
Pkg.add("Plots")   # enables plot_mesh_normals
```

For loading XML VTK (`.vtu`) meshes:

```julia
Pkg.add("ReadVTK")
```

## Quick Example

```julia
using RadiativeViewFactor

mesh   = load_mesh("geometry.msh")
result = compute_view_factors(mesh; nquad=4)

i = findfirst(==("emitter"),  result.group_names)
j = findfirst(==("receiver"), result.group_names)
println("F(emitter → receiver) = ", result.F_group[i, j])

check_reciprocity(result)
check_closure(result)
```

## Contents

```@contents
Pages = [
    "manual/getting_started.md",
    "manual/mesh_requirements.md",
    "manual/integration_methods.md",
    "manual/obstruction.md",
    "manual/gpu.md",
    "manual/visualisation.md",
    "manual/performance.md",
    "theory.md",
    "api.md",
    "references.md",
    "citing.md",
]
Depth = 2
```
