# Citing RadiativeViewFactor.jl

If you use RadiativeViewFactor.jl in your research, please cite it. A
"Cite this repository" button on the
[GitHub repository page](https://github.com/rot4te/RadiativeViewFactor.jl)
generates formatted citations automatically from the `CITATION.cff` file.

## BibTeX

```bibtex
@software{coxe_radiativeviewfactor_2026,
  author  = {Coxe, Alexander M.},
  title   = {{RadiativeViewFactor.jl}},
  url     = {https://github.com/rot4te/RadiativeViewFactor.jl},
  version = {0.5.0},
  year    = {2026}
}
```

## Key references

The numerical methods implemented in this package are described in the
following works. Please also consider citing the relevant references when
using specific features:

**Duffy transformation** (`use_duffy=true`):

```bibtex
@book{sauter_schwab_2011,
  author    = {Sauter, Stefan A. and Schwab, Christoph},
  title     = {Boundary Element Methods},
  publisher = {Springer},
  year      = {2011},
  doi       = {10.1007/978-3-540-68093-2}
}

@article{duffy_1982,
  author  = {Duffy, M. G.},
  title   = {Quadrature over a pyramid or cube of integrands with a
             singularity at a vertex},
  journal = {SIAM Journal on Numerical Analysis},
  volume  = {19},
  number  = {6},
  pages   = {1260--1262},
  year    = {1982},
  doi     = {10.1137/0719090}
}
```

**View factor theory and analytical reference cases**:

```bibtex
@book{howell_2020,
  author    = {Howell, John R. and Meng\"{u}\c{c}, M. Pinar and Siegel, Robert},
  title     = {Thermal Radiation Heat Transfer},
  edition   = {7th},
  publisher = {CRC Press},
  year      = {2020}
}
```

**Ray--triangle intersection** (obstruction detection):

```bibtex
@article{moller_trumbore_1997,
  author  = {M\"{o}ller, Tomas and Trumbore, Ben},
  title   = {Fast, minimum storage ray/triangle intersection},
  journal = {Journal of Graphics Tools},
  volume  = {2},
  number  = {1},
  pages   = {21--28},
  year    = {1997},
  doi     = {10.1080/10867651.1997.10487468}
}
```
