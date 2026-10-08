# `doc/` — design notes for the code that ships

These describe how `peclet.flow` **works today**: the schemes, why they are what they are, and the
measurements that pin them down. The rule (`../../docs/QUALITY_PLAN.md` §3.H, decision D7) is that a
note stays here only while it describes shipped behaviour or steers work that is still open;
campaign logs, work orders and refuted or superseded design discussion move to
[`history/`](history/README.md), which is indexed and never deleted.

The short working reference is [`../CLAUDE.md`](../CLAUDE.md), with its measured numbers, rationale
and long API notes in [`CLAUDE_reference.md`](CLAUDE_reference.md) (moved out 2026-10-08);
suite-wide contracts are in `../../docs/`.

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
| [cutcell_openbc_convergence.md](cutcell_openbc_convergence.md) | 2026-09-01 … 09-16 | Solid geometry intersecting an inflow/outflow face: the two defects (SDF ghost wrap, Dirichlet row aperture), the fix, and the sealed-pocket rejection. |
| [anisotropic_metric.md](anisotropic_metric.md) | 2026-09 | The metric in every discrete operator on stretched cells (physical-units Phase 2) — the reference for anisotropic domains. |
| [anisotropic_vof.md](anisotropic_vof.md) | 2026-09 | The same for the geometric two-phase stack (Phase 3). |
| [units_escalation.md](units_escalation.md) | 2026-09 | The escalation channel for the physical-domains work (`../../docs/PHYSICAL_UNITS_PLAN.md` §9.6); kept live while Phase 4 is open. |
| [scalar_ibm_design.md](scalar_ibm_design.md) | 2026-10-02 … 10-04 | **Cut-cell scalar transport** (`add_scalar(..., cutcell=True)`): the contract. It covers the probe-flux walls (Dirichlet / Neumann / Robin), conjugate solids, closures, ScalarMG, advection with small cells, the gates and the open questions (§13), with Amendments A1–A3. |
| [scalar_ibm_STATE.md](scalar_ibm_STATE.md) | 2026-10 | The campaign's one-screen state: where it is, the next action, the open decisions. |
| [scalar_ibm_log.md](scalar_ibm_log.md) | 2026-10 | Append-only log: the 2-D prototype rounds, every work order's numbers, and every ruling (D-WO*). Grep it; do not read it whole. |
| [scalar_ibm_brief.md](scalar_ibm_brief.md), [scalar_ibm_brief_A2.md](scalar_ibm_brief_A2.md) | 2026-10 | The architect briefs behind the design and behind Amendment A2 (steady advection). |
| [scalar_ibm_literature/](scalar_ibm_literature) | 2026-10 | Literature digests: L1 cut-cell/EB, L2 interfaces and conjugate transport, L3 particle-resolved benchmarks, R building blocks. |
| [scalar_ibm_naming_rows.md](scalar_ibm_naming_rows.md) | 2026-10-04 | PROPOSED rows for `../../docs/NAMING.md` (pending Frank's naming decisions). The register entries were placed in `../../docs/decisions/flow.md` and `core.md` (umbrella 4661441). |
| [scalar_ibm_baseline_hashes.txt](scalar_ibm_baseline_hashes.txt) | 2026-10-02 | The 12 legacy state hashes (gate G12) the cut-cell work must keep. |

[`data/`](data) holds the raw logs and probe scripts the collocated campaign produced.
