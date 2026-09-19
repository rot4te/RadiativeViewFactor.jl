# References

The following works informed the numerical methods implemented in this package.

## View factor theory

- Howell, J. R., Mengüç, M. P., Daun, K., & Siegel, R. (2021). *Thermal Radiation Heat Transfer* (7th ed.). CRC Press.
  — View factor definitions, the reciprocity relation, and Hottel's crossed-string method for 2D geometries.

- Howell, J. R. *A Catalog of Radiation Heat Transfer Configuration Factors* (3rd ed.). <https://www.thermalradiation.net>
  — Section C ("Factors from finite areas to finite areas") provides the published reference values used by the validation suite in `benchmarks/howell/`.

- Hamilton, D. C., & Morgan, W. R. (1952). *Radiant-interchange configuration factors*. NACA Technical Note 2836.
  — A study that checks the configuration-factor solutions then in the literature, gives the more complicated ones as families of curves, and adds new cases (rectangles, triangles, finite-length cylinders).

## Finite element geometry

- Zienkiewicz, O. C., Taylor, R. L., & Zhu, J. Z. (2005). *The Finite Element Method: Its Basis and Fundamentals* (6th ed.). Elsevier.
  — Quad8 serendipity and Tri6 shape functions (Chapter 4), isoparametric mapping and Gauss quadrature on reference elements (Chapter 5).

## Singularity treatment (Duffy transformation)

- Duffy, M. G. (1982). Quadrature over a pyramid or cube of integrands with a singularity at a vertex. *SIAM Journal on Numerical Analysis*, 19(6), 1260–1262.
  — The original transformation for integrands with a singularity at a vertex of a square-based pyramid or a cube. `DuffyKernel.jl` implements an elementary "biggest-coordinate" generalization of it (4 regions for a common vertex, 6 for a common edge), not the specific Sauter–Schwab region formulas below.

- Sauter, S. A., & Schwab, C. (2011). *Boundary Element Methods*. Springer.
  — Chapter 5: the Sauter–Schwab regularizing coordinate transformations for the singular panel pairs (identical or common-face, common-edge, common-vertex); a related but distinct 4D Duffy-type regularization for the same singularity, kept here as background reading rather than as the implemented method.

## Gaussian quadrature

- Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules. *Mathematics of Computation*, 23(106), 221–230.
  — The Golub–Welsch algorithm (nodes and weights from the eigenvalues and first eigenvector components of a symmetric tridiagonal matrix) used in `Quadrature.jl` to generate n-point Gauss–Legendre rules for n > 5.

- Dunavant, D. A. (1985). High degree efficient symmetrical Gaussian quadrature rules for the triangle. *International Journal for Numerical Methods in Engineering*, 21(6), 1129–1148.
  — Dunavant triangle quadrature rules used for Tri3 and Tri6 surface elements: the degree 1, 2, 5 and 7 rules, with 1, 3, 7 and 13 points.

## Ray–triangle intersection

- Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray/triangle intersection. *Journal of Graphics Tools*, 2(1), 21–28.
  — Möller–Trumbore algorithm implemented in `BVH.jl` (and `GPUBVH.jl` on the device) for obstruction testing in 3D.

## BVH traversal

- Torres, R., Martín, P. J., & Gavilanes, A. (2009). Ray casting using a roped BVH with CUDA. *Proceedings of the 25th Spring Conference on Computer Graphics (SCCG '09)*, 95–102. ACM.
  — Stackless BVH traversal using a skip pointer (also called a skip connection or escape index): the node to continue with when the ray misses a node's box or the node is a leaf. `GPUBVH.jl` uses this encoding (its `miss_link`) to eliminate per-thread stack memory on GPU.

## Monte Carlo integration

- Pharr, M., Jakob, W., & Humphreys, G. (2023). *Physically Based Rendering: From Theory to Implementation* (4th ed.). MIT Press.
  — Monte Carlo integration and its variance reduction, including stratified sampling (Chapter 2, §2.2.1): background for the stratified area-sampling scheme in `MCKernel.jl`.

- Cohen, M. F., & Wallace, J. R. (1993). *Radiosity and Realistic Image Synthesis*. Academic Press.
  — Form factors and the radiosity method in computer graphics: background for the cosine-weighted ray-shooting estimator in `RayTraceKernel.jl` and `GPURayTraceKernels.jl`.

- Duff, T., Burgess, J., Christensen, P., Hery, C., Kensler, A., Liani, M., & Villemin, R. (2017). Building an orthonormal basis, revisited. *Journal of Computer Graphics Techniques*, 6(1), 1–8.
  — The branchless orthonormal-basis construction (Listing 3, using `copysign`) used to turn a surface normal into the frame for cosine-weighted hemisphere sampling in `RayTraceKernel.jl` and `GPURayTraceKernels.jl`.

## GPU pseudo-random number generation

- Marsaglia, G. (2003). Xorshift RNGs. *Journal of Statistical Software*, 8(14), 1–6.
  — 32-bit xorshift generator used in the sampling hot loop of `GPUMCKernels.jl` (and reused by `GPURayTraceKernels.jl`) for per-thread random streams with no heap allocation.

- Steele, G. L., Lea, D., & Flood, C. H. (2014). Fast splittable pseudorandom number generators. *ACM SIGPLAN Notices*, 49(10), 453–472.
  — The SplitMix 64-bit mixing function (no loops or conditionals, so suitable for GPU code), used to derive independent per-thread seeds from the global seed and thread index.
