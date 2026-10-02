# Recon digest: flow/core building blocks for the scalar-IBM design (2026-10-02, Haiku recon; verify anchors)

## Krylov
- **flow `CutcellMG::solveBiCGStab`** (`src/mac_cutcell_mg.hpp:1801`).
  - Pressure BiCGStab; the matvec is a lambda over the ghost-projection overlay (GpOverlay).
  - Preconditioner: a V-cycle (pre/post sweeps). MPI staging through a g = 2 halo.
- **core `MomentumSolver::solveBiCGStab`** (`core/include/peclet/core/solver/csr_bicgstab.hpp:122`).
  - Device, functor-based: operator + pluggable preconditioner z = M⁻¹r (damped Jacobi by
    default; velocity MG / GraphAMG possible).
  - MPI through injected `refresh` (halo) and `dotReduce` callables. The header contains no MPI.
  - Also provides defect correction.
- **core `solver/`** also holds `csr_operator.hpp` (FaceCsrOpT, FvCsrOpT, Jacobi / multicolour GS),
  `face_csr.hpp`, `coloring.hpp`, `graph_amg{,_device}.hpp` and `vector_ops.hpp`.

## MG
- **`CutcellMG`** (`mac_cutcell_mg.hpp:492`, `init(nx, ny, nz, nLevels)`).
  - Per level: ox/oy/oz openness, operator AC/AFX/AFY/AFZ (stored at MReal).
  - Smoother: RB-GS (`cutcellSmoothColorFace`), communication-avoiding under MPI.
  - Coarse operators are REDISCRETIZED from averaged openness (`mg_coarsen_open_avg_k`,
    `buildCutcellOp`).
  - The recon claims it "can embed κ/dt·I + aperture Laplacian + extra diagonal". This is not
    verified: the operator is built from openness only, so a diagonal-shift entry point must be
    checked or added.
- **`VelocityMG`** (`mac_velocity_mg.hpp`).
  - Helmholtz-type: a per-cell diagonal array + diffusion (+ upwind advection).
  - Rediscretized coarse operators.
  - Register: Galerkin is "the robust default" for velocity MG with partial cells; rediscretized
    velocity MG once diverged.

## Geometry
- **core `vof/plic.hpp`.** Every function is `KOKKOS_INLINE_FUNCTION` and container-free:
  - `plicVolume(mx, my, mz, alpha)` (:221): plane → volume fraction (SZ2000);
  - `plicAlpha` (:303): the inverse;
  - `plicSlabVolume` (:339);
  - `plicBoxVolume`.
  - The polygon area/centroid functions sit in flow's VoF drivers (not located precisely).
- **`ccFaceOpenMS`** (`mac_cutcell.hpp:174`).
  - Marching-squares face aperture from 5 trilinear SDF samples (4 corners + centre),
    triangle-fan.
  - Floor 1e-3 (sub-resolution faces snapped closed).
  - Returns the aperture ONLY (no centroid).
- **SDF.**
  - `sdf_` lives on the device over the whole extended block after `set_solid`.
  - Single-rank fill: periodic wrap. MPI fill: GridHalo exchange, then `extendSdfDomainGhosts`
    for non-periodic faces.
  - **Ghost-width conflict between the two recons:** 1 (this one) vs 2 (the earlier one).
    VERIFY. The probe reach is 2.
- **Scalar fields** are registered on a G = 2 block (binding docs: "Register scalar field on G=2
  block").

## Overlay pattern
- **`IbmOverlayT<Space, Real>`** (`cut_cell_ibm.hpp:142`).
  - A SoA list over cut cells: `cell_index` plus per-face factors.
  - Built with a device `parallel_scan` (count + compact).
- **`ibmModifyStencil`** (:394): one thread per cut cell rewrites the cell's 7-point row in place.
- This is the template for a "scalar facet overlay": cell index, facet area/normal/centroid,
  probe cell indices + weights, BC-elimination coefficients.
