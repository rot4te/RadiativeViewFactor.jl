# References

The following works informed the numerical methods implemented in this package.

## View factor theory

- Howell, J. R., Mengüç, M. P., & Siegel, R. (2020). *Thermal Radiation Heat Transfer* (7th ed.). CRC Press.
  — View factor definitions, reciprocity relation, crossed-string method for 2D geometries, and analytical reference cases used for validation.

- Hamilton, D. C., & Morgan, W. R. (1952). *Radiant interchange configuration factors*. NACA Technical Note 2836.
  — Original tabulation of configuration factor formulae for standard geometries.

## Finite element geometry

- Zienkiewicz, O. C., Taylor, R. L., & Zhu, J. Z. (2005). *The Finite Element Method: Its Basis and Fundamentals* (6th ed.). Elsevier.
  — Quad8 serendipity and Tri6 shape functions, isoparametric mapping, Gauss quadrature on reference elements.

## Singularity treatment (Duffy transformation)

- Duffy, M. G. (1982). Quadrature over a pyramid or cube of integrands with a singularity at a vertex. *SIAM Journal on Numerical Analysis*, 19(6), 1260–1262.
  — Original Duffy transformation. `DuffyKernel.jl` implements an elementary "biggest-coordinate" generalization of this single-simplex transform (4 regions for a common vertex, 6 for a common edge), not the specific Sauter–Schwab region formulas below.

- Sauter, S. A., & Schwab, C. (2011). *Boundary Element Methods*. Springer.
  — Sauter–Schwab common-vertex (§5.3.2) and common-edge (§5.3.3) decompositions; a related but distinct 4D Duffy-type regularization for the same singularity, kept here as background reading rather than as the implemented method.

## Gaussian quadrature

- Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules. *Mathematics of Computation*, 23(106), 221–230.
  — Golub–Welsch tridiagonal eigenvalue algorithm used in `Quadrature.jl` to generate n-point Gauss–Legendre rules for n > 5.

- Dunavant, D. A. (1985). High degree efficient symmetrical Gaussian quadrature rules for the triangle. *International Journal for Numerical Methods in Engineering*, 21(6), 1129–1148.
  — Dunavant triangle quadrature rules used for Tri6 surface elements.

## Ray–triangle intersection

- Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray/triangle intersection. *Journal of Graphics Tools*, 2(1), 21–28.
  — Möller–Trumbore algorithm implemented in `BVH.jl` for obstruction testing in 3D.

## BVH traversal

- Shirley, P., et al. (2019). *Ray Tracing Gems*. Apress.
  — Stackless BVH traversal via miss-link (skip-pointer) encoding, implemented in `GPUBVH.jl` to eliminate per-thread stack memory on GPU.

## Monte Carlo integration

- Pharr, M., Jakob, W., & Humphreys, G. (2023). *Physically Based Rendering: From Theory to Implementation* (4th ed.). MIT Press.
  — Stratified sampling, variance reduction, and Monte Carlo estimators for light transport integrals; basis for the stratified area-sampling scheme in `MCKernel.jl`.

- Cohen, M. F., & Wallace, J. R. (1995). *Radiosity and Realistic Image Synthesis*. Academic Press.
  — Source for the solid-angle identity ``dA_j \cos\theta_j / r^2 = d\Omega`` underlying the cosine-weighted ray-shooting estimator in `RayTraceKernel.jl` and `GPURayTraceKernels.jl`.

## GPU pseudo-random number generation

- Marsaglia, G. (2003). Xorshift RNGs. *Journal of Statistical Software*, 8(14).
  — xorshift64 PRNG used in `GPUMCKernels.jl` for per-thread random streams with no heap allocation.

- Steele, G. L., Lea, D., & Flood, C. H. (2014). Fast splittable pseudorandom number generators. *ACM SIGPLAN Notices*, 49(10), 453–472.
  — splitmix64 mixing function used to derive independent per-thread seeds from the global seed and thread index.
