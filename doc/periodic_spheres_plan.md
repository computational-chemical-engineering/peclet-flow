# Periodic Spheres Verification Plan

## Findings (Current)
- `scripts/verify_periodic_spheres.py` produces `NaN` for all `K_sim` values at `res=64^3`.
- The IBM geometry passes complete and report reasonable active counts, but the parameter sweep still diverges.

## Hypotheses (Likely Failure Points)
1. **NaN introduction during momentum solve**
   - Potential division by very small diagonal after IBM modifications or zeroed faces.
2. **NaN/Inf in pressure correction**
   - Cut-cell pressure solve may be ill-conditioned for closed/near-closed pores.
3. **Bad fluxes in masked faces**
   - Zeroing solid faces may create discontinuities that destabilize advection or projection.

## Investigation Steps
1. **Locate first NaN**
   - Add device-side NaN checks after: momentum stencil build, IBM stencil modification, pressure solve, projection.
   - Log earliest timestep and field (u/v/w/p/phi) where NaN appears.
2. **Reduce the problem**
   - Run a single `phi` case (e.g., 0.05) with a smaller grid (e.g., 32^3) and fixed timestep.
   - Save intermediate fields every N steps to locate instability origin.
3. **Check diagonal dominance**
   - For IBM-modified stencils, verify diagonal is positive and larger than sum of abs neighbors.
   - Add debug kernel to track min/max of A_C and report if any |A_C| < eps.
4. **Validate cut-cell pressure**
   - Compare divergence norm before/after projection; ensure Poisson residual drops.
   - Confirm `solid_c` scaling and Laplace RHS do not blow up.

## Fix Options (After Diagnosis)
- **Stabilize IBM rows**: clamp diagonal (or add damping) for cells with near-zero `D_rescale`.
- **Adjust projection in solids**: refine `solid_c` or provide a smoother solid `phi` solve (e.g., fixed value / stronger regularization).
- **Timestep control**: enforce a CFL cap in the periodic spheres sweep until the root cause is fixed.
- **Use cell-average IBM polynomials** for low resolutions if needed (align with finite-volume interpretation).

## Acceptance Criteria
- `K_sim` finite for all tested `phi`.
- `K_sim` trends monotonic with `phi` and within 10-20% of literature at `res=64^3`.
