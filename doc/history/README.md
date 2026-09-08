# History — campaign records, work orders and superseded design notes

Everything here is a **record of what was run and what was found**, kept for the record and moved
out of the live documentation set on 2026-09-08 (`../../../docs/QUALITY_PLAN.md` §3.H.2: *docs
describe the code that exists*). Each file steered a piece of work that has since landed, been
superseded, or been refuted. **Nothing here is maintained**: `file:line` citations, "STATUS" lines,
"DEFAULT since" claims, test counts and "next step" sections are snapshots of their date — read the
source before acting on any of them. Nothing is deleted either.

The authority on how `peclet.flow` behaves today is the code, [`../../CLAUDE.md`](../../CLAUDE.md)
and the reference notes in [`../`](../README.md).

| note | date | what it is |
|---|---|---|
| [claude_md_2026-09-08.md](claude_md_2026-09-08.md) | 2026-01 … 09-08 | `CLAUDE.md` as it stood before the H diet — 174 KiB of rung-by-rung VoF narrative (V0–W3, P0–P3g), the collocated attractor summary, the WO-F/G/H/I/M findings and the retired environment-variable table. |
| [vof_workorders.md](vof_workorders.md) | 2026-08-30/31 | VoF work orders, phase 0 (V-1, S0, S1, V0, V1) — the PLIC toolbox and Weymouth–Yue advection, plus the shared preamble every later VoF work order inherits. Also carries the WO-B/C/H pressure-driver and WO-I drag-ghost findings. |
| [vof_workorders_v2.md](vof_workorders_v2.md) | 2026-08-31 | Rung V2 (two-phase Navier–Stokes, staggered, no surface tension): WO-J plumbing and WO-K momentum consistency, with the three things that construction paid for. |
| [vof_workorders_v34.md](vof_workorders_v34.md) | 2026-08-31 | V3 (curvature), V4 (balanced-force surface tension) and WO-M, the operator-storage precision campaign (`MReal` float vs double, the residual rebound). |
| [vof_workorders_v5.md](vof_workorders_v5.md) | 2026-09-02 … 09-08 | The finishing campaign: V5a cut cells, V-BC open boundaries (WO-R/R2), V5b wetting, V8 collocated, and the gallery examples. |
| [vof_workorders_v6.md](vof_workorders_v6.md) | 2026-09-02 … 09-08 | The remainder of the ladder: V6 dynamic contact angle, V7, V9, the phase-change rungs P0–P3g, and the block container W0–W3 (including the colliding-marker rating). |
| [collocated_first_order_analysis.md](collocated_first_order_analysis.md) | 2026-07-04/05 | Why the collocated solver was first order at curved immersed boundaries — the accumulated design narrative and its dead ends, by its own labelling "useful history, not gospel". |
| [collocated_second_order_literature.md](collocated_second_order_literature.md) | 2026-07-05 | Literature survey of second-order collocated cut-cell schemes that keep an approximate projection, run to choose what to port. |
| [collocated_embed_port_plan.md](collocated_embed_port_plan.md) | 2026-07-05 | The implementation plan for porting Basilisk `embed.h` to the collocated cut-cell solver. |
| [collocated_second_order_open_problem.md](collocated_second_order_open_problem.md) | 2026-07-05 … 08-19 | The narrative-free statement of the open problem as of August — superseded by the attractor campaign. |
| [collocated_ceiling_plan.md](collocated_ceiling_plan.md) | 2026-08-19/20 | The "accuracy ceiling" work plan whose diagnosis section was partly refuted by its own Step 0/1. |
| [collocated_accuracy_ceiling.md](collocated_accuracy_ceiling.md) | 2026-08-20 | The measured record of the (contaminated-protocol) ceiling investigation, kept as the paper's refutation catalogue. |
| [collocated_stall_notes.md](collocated_stall_notes.md) | 2026-08-20/21 | The chronological lab record of the day the plateau was traced to a marginally-stable approximate-projection stall. |
| [colocated_study/](colocated_study/staggered_vs_colocated.md) | 2026-06-22 | Dated staggered-vs-collocated grid-convergence, accuracy and performance study with its figures; its accuracy conclusions were overtaken by the August attractor campaign. |
| [ghost_hardening_plan.md](ghost_hardening_plan.md) | 2026-08-17 | The plan for hardening the ghost-cell projection (full Robust-Scaling + symmetrized deferred correction). |
| [ghost_hardening_findings_A.md](ghost_hardening_findings_A.md) | 2026-08-17/18 | Phase A findings of that plan: hypothesis H1 refuted three ways; the mixed-GPORDER march instability that made `(1, 2)` do-not-use. |
| [advective_cutwall_flux_plan.md](advective_cutwall_flux_plan.md) | 2026-08-31 … 09-08 | The advective cut-wall flux campaign, closed 2026-09-02 when the remaining ten-Cate deficit turned out to be an oversized slab in the scene, not a solver defect. |
| [packing_multires_resolution_report.md](packing_multires_resolution_report.md) | 2026-04-16 | A dated packed-bed multi-resolution target-Re study report (64³/128³/256³ continuation). |
| [packing_parameter_study_workflow.md](packing_parameter_study_workflow.md) | 2026-04 … 09 | The coarse-to-fine continuation workflow for a packed-bed Reynolds sweep. Archived because its driver `scripts/run_packing_multires_re_targets.py` no longer exists and `solver.scale_state(...)` is not in the current bindings — the note says so itself; treat it as methodology, not runnable paths. |

Raw campaign logs and probe scripts stay where they were, under [`../data/`](../data).
