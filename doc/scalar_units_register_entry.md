# Proposed register entries — scalar/energy API in physical units (2026-10-02)

For the orchestrator to place in the umbrella (`suite/docs/decisions/flow.md` and
`suite/docs/DECISIONS.md`); this file is not itself part of the register and can be deleted once
they are placed.

## For `docs/decisions/flow.md` (append)

```markdown
---

### The scalar-transport and phase-change energy API is physical under an armed extent, converted at the boundary
- area: flow
- source: flow `db28d0c` `3e24920` `1c80540` `c978aeb` `5efd9fa` (branch scalar-units); `src/flow_ibm.hpp` UnitScales comment "Scalar transport and the phase-change energy path"
- decided: 2026-10-02
- status: settled (USER DIRECTIVE 2026-10-02: "Do not leave it, but fix it.")
- quote: |
    A transported scalar (temperature, concentration) is never rescaled; every other scalar/energy
    quantity converts by its M-L-T dimension in the internal system (length hRef, time tRef, mass
    rhoRef*hRef^3): diffusivity x tRef/hRef^2, mass flux x tRef/(rhoRef hRef), latent heat
    x tRef^2/hRef^2, conductivity x tRef^3/(rhoRef hRef^4), rho c_p x tRef^2/(rhoRef hRef^2),
    R_int x rhoRef hRef/tRef, divergence source x tRef. Constants are kept verbatim and re-derived
    when a scale is pinned (order-independent); the two FIELD setters (set_mass_flux*,
    set_divergence_source) raise under an extent until set_rho/set_dt have run. Closures on
    rho/mu/force_* take physical parameters (inputs are unscaled fields). Getters
    (vof_interface_area, phase_change_diagnostics, the budget and carry ledger) report physical.
    The operator-flux mdot without set_phase_change_energy RAISES under an extent: its constant-D
    flux carries an implicit rho c_p of one internal unit.
- rejected: documenting the scalar/energy surface as internal (cell) units — "silently wrong" is the failure class U3's four missed setters already showed; converting fields set before the scales are pinned with whatever scale was current (silently wrong if set_rho/set_dt come later); interpreting the constant-D operator mdot's implicit rho c_p as "1 in the caller's units" (unit-system dependent physics)
- why: the plan's contract is one consistent physical system in and out; gates units_scalar_scale_invariance (9.6e-16), units_scalar_sine_decay (exact BE amplitude to 8e-14, order 1.98 on cubic and box cells), units_phase_change_scale_invariance (<= 1.7e-13 incl. 12 full sucking-interface steps); cell units bit-identical (12/12 state hashes)
```

## For `docs/DECISIONS.md` (one line, in the flow section)

```markdown
- **The scalar-transport and phase-change energy API is physical under an armed extent, converted at the boundary (constants re-derived when a scale is pinned; field setters need set_rho/set_dt first; constant-D operator mdot refused)**. **Rejected:** documenting internal (cell) units for scalars; converting fields with unpinned scales; an implicit rho c_p = 1 in the caller's units  <sub>flow</sub>
```
