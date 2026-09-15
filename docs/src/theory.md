# Theory

## View factor definition

The view factor ``F_{ij}`` from surface ``i`` to surface ``j`` is the fraction
of diffuse radiation leaving surface ``i`` that arrives at surface ``j``:

**3D (surface meshes):**

```math
F_{ij} = \frac{1}{A_i} \iint_{A_i} \iint_{A_j}
    \frac{\cos\theta_i \cos\theta_j}{\pi r^2} \, H_{ij} \, dA_j \, dA_i
```

**2D (curve meshes, per unit depth):**

```math
F_{ij} = \frac{1}{L_i} \int_{L_i} \int_{L_j}
    \frac{\cos\theta_i \cos\theta_j}{2r} \, H_{ij} \, dL_j \, dL_i
```

where:
- ``\theta_i`` is the angle between the outward normal at ``dA_i`` and the
  line of sight ``r_{ij}``
- ``\theta_j`` is the angle between the outward normal at ``dA_j`` and the
  reverse line of sight
- ``r`` is the distance between the two differential elements
- ``H_{ij} \in \{0, 1\}`` is the visibility function (0 = obstructed)

The factor ``2`` rather than ``\pi`` in the 2D kernel follows from integrating
the 2D radiation intensity over the hemisphere, which gives ``\pi/2`` rather
than ``\pi``.

## Reciprocity

The reciprocity relation:

```math
A_i F_{ij} = A_j F_{ji}
```

is a consequence of the symmetry of the kernel and holds exactly in the
continuous case. At the discrete (element) level, reciprocity holds to within
quadrature error; verify with [`check_reciprocity`](@ref).

## Gauss–Legendre quadrature

The isoparametric map ``\mathbf{x}(\xi, \eta)`` transforms the reference
element ``[-1,1]^2`` to physical space. The integral over one Quad8 element
becomes:

```math
\int_{A} f \, dA = \int_{-1}^{1} \int_{-1}^{1}
    f(\mathbf{x}(\xi,\eta)) \left|\frac{\partial\mathbf{x}}{\partial\xi}
    \times \frac{\partial\mathbf{x}}{\partial\eta}\right| d\xi \, d\eta
    \approx \sum_{p=1}^{n^2} w_p \, f(\mathbf{x}(\xi_p,\eta_p)) \, J_p
```

where ``J_p = |\partial_\xi \mathbf{x} \times \partial_\eta \mathbf{x}|`` is
the area Jacobian. A tensor-product ``n \times n`` Gauss–Legendre rule is used.

## Duffy transformation

For element pairs sharing a vertex at ``\mathbf{u}_0 = (u_0, v_0)`` in element
``i``'s unit square and ``\mathbf{s}_0 = (s_0, t_0)`` in element ``j``'s, the
kernel diverges as ``r \to 0``. Near the singularity:

```math
K \sim \frac{1}{r^2}, \quad r \sim \sqrt{(u-u_0)^2+(v-v_0)^2+(s-s_0)^2+(t-t_0)^2}
```

The implementation uses an elementary "biggest-coordinate" Duffy
decomposition (a generalization of Duffy's original single-simplex
transform), not the specific region formulas of Sauter & Schwab's boundary
element method — a related but different decomposition of the same
singularity, and still the reference for the underlying idea.

**Common vertex** (one shared corner node): shift coordinates to
``\tilde u = u-u_0``, etc., so the singularity sits at the origin of
``[0,1]^4``. Split the hypercube into **4 regions** by which of
``(\tilde u, \tilde v, \tilde s, \tilde t)`` has the largest magnitude; in
each region that coordinate is set to a radial variable ``\rho \in [0,1]``
and the other three to ``\rho \eta_k`` (``\eta_k \in [0,1]``), giving a
lower-triangular Jacobian with determinant ``\rho^3``:

```math
K \cdot dA_i \cdot dA_j \cdot |\text{Jac}| \sim \frac{1}{\rho^2} \cdot \rho^2 \cdot \rho^3 = \rho^3 \to 0
\quad \text{as } \rho \to 0
```

The 4 regions exactly tile ``[0,1]^4`` (ties between coordinates have measure
zero), each contributing ``4 \times nquad^4`` total evaluation points.

**Common edge** (two shared corner nodes): in edge-local coordinates the
singularity is the *line* ``u=s,\, v=t=0`` rather than a single point. Split
the ``(u,s)`` square into the two triangles ``u \geq s`` and ``u < s``
(Jacobian ``u`` or ``s`` respectively, a standard ratio parametrization);
within each triangle the remaining three-variable point singularity
``(w, v, t) \to 0`` (where ``w = (u-s)/u`` or ``(s-u)/s``) is resolved the
same "biggest-coordinate" way, now with **3 regions** and Jacobian
``\rho^2``. This gives **6 regions total** (2 triangles × 3 sub-regions),
each contributing ``6 \times nquad^4`` total evaluation points, with combined
Jacobian ``(u\text{ or }s) \cdot \rho^2``.

The transformed integrand is smooth at ``\rho = 0`` in both cases and is
efficiently resolved by standard Gauss–Legendre quadrature.

## Monte Carlo estimator

The unbiased area-sampling estimator for each element pair ``(i,j)`` is:

```math
\iint K \, dA_j \, dA_i
\approx \frac{A_i \cdot A_j}{N} \sum_{k=1}^{N}
K\!\left(x_i^{(k)}, n_i^{(k)}, x_j^{(k)}, n_j^{(k)}\right) \cdot H_{ij}^{(k)}
```

where ``(x_i^{(k)}, x_j^{(k)})`` are i.i.d. uniform samples on ``A_i \times A_j``.

**Stratified sampling** subdivides the reference square into
``\lfloor\sqrt{N}\rfloor \times \lfloor\sqrt{N}\rfloor`` strata and draws one
point per stratum. For smooth integrands, stratified sampling achieves
``O(1/N)`` variance convergence rather than ``O(1/\sqrt{N})`` for plain Monte
Carlo, matching the rate of a 1D Gauss rule.

**Variance near singularities:** for the ``1/r^2`` kernel the variance of the
MC estimator is proportional to ``\iint K^2 \, dA``, which diverges at shared
edges. Infinite variance means no amount of increasing ``N`` gives reliable
convergence — use the Duffy transformation instead for such pairs.

## Ray-shooting Monte Carlo estimator

A second, unrelated Monte Carlo estimator (`raytrace=true`) rewrites the
double-area integral as a single-area integral over the (occlusion-limited)
solid angle ``\Omega_j`` that surface ``j`` subtends from each point
``x \in A_i``:

```math
F_{i \to j} = \frac{1}{A_i} \int_{A_i}
    \left[ \int_{\Omega_j \text{ visible from } x} \frac{\cos\theta_i}{\pi} \, d\Omega \right] dA_i
```

which follows from the solid-angle identity ``dA_j \cos\theta_j / r^2 = d\Omega``.
Drawing a ray direction ``\omega`` from the cosine-weighted hemisphere pdf
``\cos\theta_i / \pi`` gives

```math
\mathbb{E}\big[\mathbb{1}(\text{ray's first hit is } j)\big]
    = \int_{\Omega_j \text{ visible}} \frac{\cos\theta_i}{\pi} \, d\Omega = F_{x \to j}
```

— the hit indicator's expectation is *exactly* the point-to-area form
factor, occlusion included, with no ``\cos\theta_j``, no ``1/r^2``, and no
separate visibility test: a ray hitting the wrong surface first *is* the
visibility test. Averaging over points ``x`` drawn uniformly by area on
``A_i`` gives ``F_{i \to j}`` directly.

Because this estimator never evaluates ``1/r^2``, it has ordinary bounded
(binomial) variance for every element pair, including adjacent ones — no
Duffy patch is applied. Each unordered pair ``\{i,j\}`` gets two independent
estimates (one from each element's own rays); `Assembly.jl` averages them
before dividing back out to ``F``, which enforces reciprocity by
construction rather than as an emergent property of enough samples. See
`src/RayTraceKernel.jl`'s module docstring for the full derivation and a
front/back-face subtlety found while implementing it.
