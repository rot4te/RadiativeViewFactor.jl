# API Reference

## Mesh I/O

```@docs
RadiativeViewFactor.load_mesh
RadiativeViewFactor.load_vtu
RadiativeViewFactor.load_re2
RadiativeViewFactor.MeshData
RadiativeViewFactor.SurfaceElement
```

## Mesh manipulation

```@docs
RadiativeViewFactor.reverse_group_normals
RadiativeViewFactor.restrict_to_radiating
RadiativeViewFactor.split_groups_by_tag
```

## View factor computation

```@docs
RadiativeViewFactor.compute_view_factors
RadiativeViewFactor.ViewFactorResult
```

## Post-processing

```@docs
RadiativeViewFactor.aggregate_by_group
RadiativeViewFactor.check_reciprocity
RadiativeViewFactor.check_closure
RadiativeViewFactor.enforce_closure
```

## Nek5000/NekRS export

```@docs
RadiativeViewFactor.write_nekrs_view_factors
```

## Visualisation

```@docs
RadiativeViewFactor.plot_mesh_normals
```

## Element geometry evaluators

Exported helpers that map a reference coordinate to a physical point, unit
normal, and area (or length) element for a single element, and the linear shape
functions behind them.

```@docs
RadiativeViewFactor.quad8_physical_point
RadiativeViewFactor.quad8_normal_and_area_element
RadiativeViewFactor.quad4_shape
RadiativeViewFactor.quad4_physical_point
RadiativeViewFactor.quad4_normal_and_area_element
RadiativeViewFactor.line3_physical_point
RadiativeViewFactor.line3_normal_and_length_element
RadiativeViewFactor.line2_shape
RadiativeViewFactor.line2_physical_point
RadiativeViewFactor.line2_normal_and_length_element
```
