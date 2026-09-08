# `doc/` — design notes for the code that ships

These describe how `peclet.flow` **works today**: the schemes, why they are what they are, and the
measurements that pin them down. The rule (`../../docs/QUALITY_PLAN.md` §3.H, decision D7) is that a
note stays here only while it describes shipped behaviour or steers work that is still open;
campaign logs, work orders and refuted or superseded design discussion move to
[`history/`](history/README.md), which is indexed and never deleted.

The short working reference is [`../CLAUDE.md`](../CLAUDE.md); suite-wide contracts are in
`../../docs/`.

| note | date | what it describes |
|---|---|---|
| [Robust_Scaled_IBM_Solver.tex](Robust_Scaled_IBM_Solver.tex) | 2026-01 … 04 | The paper draft for the Robust-Scaled cut-cell IBM the solver implements. |
| [ibm_overlay.md](ibm_overlay.md) | 2026-06 | How the momentum cut-cell IBM is structured as a sparse overlay, and what an octree/AMR port has to replace. |
| [flow_multigrid_plan.md](flow_multigrid_plan.md) | 2026-06/07 | Design and benchmarks of the rediscretized geometric pressure multigrid (`CutcellMG`). |
| [velocity_mg_plan.md](velocity_mg_plan.md) | 2026-06/07 | Design of the velocity (Helmholtz) multigrid, its IBM coarse-operator rules, and the Phase-2 `D_rescale` un-scale that is **rejected — do not revive**. |
| [flow_colocated_plan.md](flow_colocated_plan.md) | 2026-06/07 | The collocated variant: the Almgren–Bell–Colella approximate projection and the `GridLayout` policy that makes it share the staggered operators. |
| [collocated_invisible_subspace.md](collocated_invisible_subspace.md) | 2026-08 | The mechanism note for the collocated path: attractor families and the invisible pressure subspace. **Read this first before touching that path.** |
| [collocated_paper_plan.md](collocated_paper_plan.md) | 2026-08 … 09 | Live paper plan and results tracker for that work. |
| [fluid_only_constraint_plan.md](fluid_only_constraint_plan.md) | 2026-08 | The production fluid-only constraint (the `"ghost"` scheme) — design plan, with parts still open. |
| [variable_viscosity_projection.md](variable_viscosity_projection.md) | 2026-07 | The variable-viscosity momentum operator and the rotational-incremental projection it needs. |
| [variable_density_projection.md](variable_density_projection.md) | 2026-07 … 09 | Variable density: momentum, projection scaling, the pressure driver, and (§4) the rank-aware domain-BC repair. |
| [porous_drag_scheme.md](porous_drag_scheme.md) | 2026-07 … 09 | The volume-averaged (porous) CFD-DEM fluid scheme — the gas phase of `peclet.coupling` with `porous=True`. |
| [cutcell_openbc_convergence.md](cutcell_openbc_convergence.md) | 2026-09-01 | The **open defect**: solid geometry intersecting an inflow/outflow face breaks the pressure solve. |
| [anisotropic_metric.md](anisotropic_metric.md) | 2026-09 | The metric in every discrete operator on stretched cells (physical-units Phase 2) — the reference for anisotropic domains. |
| [anisotropic_vof.md](anisotropic_vof.md) | 2026-09 | The same for the geometric two-phase stack (Phase 3). |
| [units_escalation.md](units_escalation.md) | 2026-09 | The escalation channel for the physical-domains work (`../../docs/PHYSICAL_UNITS_PLAN.md` §9.6); kept live while Phase 4 is open. |

[`data/`](data) holds the raw logs and probe scripts the collocated campaign produced.
