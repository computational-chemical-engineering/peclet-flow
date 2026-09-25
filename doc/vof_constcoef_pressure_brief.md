# Architect brief — E2(a): an opt-in constant-coefficient (Dodd–Ferrante) pressure driver

## 1. Question
Design the opt-in constant-coefficient pressure driver for solid-free boxes, solved by the existing
MG-PCG on a constant operator, so that an Opus implementer can build it without guessing: the
discrete equations on flow's staggered MAC grid, where it sits in `step()`, its state (p^{n-1}),
start-up and restart, MPI, the selection API, its interaction with the variable-density momentum
path, the VoF block CSF and the balanced-force requirement, and the accuracy gates that decide
adoption per case. Deliver it as §12 (addendum) of `doc/vof_step_performance_design.md`.

## 2. Why the architect
A new pressure-velocity coupling scheme in a settled solver: it must keep flow's balanced-force
CSF (the block container's per-marker force summed with the union colour — `doc/vof_overlap_design.md`),
the ABC/approximate-projection conventions, MPI bit-reproducibility (np-independence), and the
dt-divided operator convention — and it deliberately re-opens (scoped) two register rejections.

## 3. Decided (USER DECISION 2026-09-25, register entry "Scoped constant-coefficient
(Dodd–Ferrante) pressure driver", umbrella e390d73)
- Opt-in only; the variable-coefficient projection stays the default. Solid-free boxes (periodic +
  wall BCs, uniform grid; no IBM, no porous). Raise otherwise.
- Scheme (Dodd & Ferrante 2014, Cifani 2019 JCP): lap(p^{n+1})/rho0 = div(u*)/dt +
  div((1/rho0 - 1/rho) grad p_hat), p_hat = 2 p^n - p^{n-1}, rho0 = min rho; the velocity update
  u^{n+1} = u* - dt [ grad p^{n+1}/rho0 + (1/rho - 1/rho0) grad p_hat ].
- Option (a) solver: the existing MG-PCG on the CONSTANT operator (built once, no per-step rebuild,
  no density contrast in coarse operators). Option (b) FFT is a separate later decision — design
  so (b) can slot in behind the same driver interface.
- Order: the main-line performance work orders (WO-0…WO-13 of the design note) come first; E2(a)
  is implemented after them.

## 4. Reference implementation (TBFsolver, read from source)
`poissonEqn/poissonEqn.f90:200-250` (scratchpad copy:
`/tmp/claude-1003/-home-frankp-Codes-suite/ead398a0-dcae-4941-9337-36222cb4b0f1/scratchpad/tbf/src/`):
source term per cell `s = div(u*) * rho0/(alpha dt) + sum_faces (1 - rho0 * 0.5*(1/rho_f + 1/rho_c)) * dp_old_face`
with `computeOldPressDiv` giving the extrapolated old-pressure face gradients (`nl_ = 2` levels
stored, `storeOldField`), rho0 = min(rho_l, rho_g) (`poissonEqn.f90:127`), RK sub-steps
(`alphaRKS`), `makeVelocityDivFree(... rho0, nl)` in `main.f90:201,261`. Note TBFsolver uses the
FACE-AVERAGED 1/rho (arithmetic mean of 1/rho at the two cells), and st (surface tension) enters the
old-pressure divergence (`computeOldPressDiv(..., psi, st, ...)`) — work out whether the CSF force
must enter p_hat's gradient term to keep the balanced-force property, and state it.

## 5. Constraints
- Staggered MAC (collocated: out of scope unless trivial; say so), x-fastest (`src/policy.hpp`),
  device + MPI; np-independence tests must still pass bitwise where they did.
- flow's momentum convention: operator scaled 1/dt ("dt divided"); variable dt between steps (the
  capillary bound changes dt): the extrapolation p_hat = 2p^n - p^{n-1} assumes constant dt — state
  the variable-dt form or the rule.
- The CSF: flow's balanced-force block CSF computes face forces sigma*kappa*grad C that must be
  balanced by the pressure gradient discretely; with the splitting, the pressure gradient is split
  into a constant-coefficient part and an explicit part — design how static equilibrium (a static
  drop: zero velocity) is preserved or how large the residual current is, and gate it.
- Start-up (no p^{n-1}): state the rule (e.g. first step(s) with the variable-coefficient solve).
- Restart / checkpoint: p^{n-1} must be restorable (Python API for the driver scripts, which save
  `get_field("p")` and restore with `set_field("p")`).
- `set_superficial_velocity` (zero net flux) and property models (`set_property_model("rho", ...)`)
  must work with it; the rho closure re-selects the pressure driver ("select the driver last").

## 6. Open — decide
The discrete source and update on flow's grid (face 1/rho: arithmetic of 1/rho, or 1/rho of the
face-interpolated rho as flow's variable-density path uses?) and consistency with flow's existing
variable-density projection so that the SAME density field and CSF enter both; rho0 choice
(min rho vs a safety factor); where the old pressures live; the API name(s) (NAMING.md; developer
vs public tier); the gates and their pass thresholds; whether option (a) needs a special bottom
(constant operator: GraphAMG built once, or the B1 geometric-Krylov bottom).

## 7. Evidence already in hand
- Case and harness: `/home/frankp/Codes/bubble_column_perf/` (ckpt_t43.npz, prof.py, cmp.py,
  run_mpi.py, bench_cpu.sh). Same-node Snellius numbers: TBFsolver 45 ms/step, peclet 187 ms/step
  (projection 125 ms, MG-PCG 13 iterations at rtol 1e-10) on 24 genoa cores.
- A known physics defect is being fixed in parallel (bubbles collect at the LOW wall in peclet in
  both the case and its y-mirror → a code asymmetry, second low-wall defect) — accuracy gates on
  the bubble column must be run after that fix.

## 8. Deliverable
§12 of `doc/vof_step_performance_design.md` in `/home/frankp/Codes/suite/flow-vof-overlap` (branch
vof-perf; commit only that file with a named path; do not push): discrete scheme, placement, state,
start-up/restart, MPI, API, work orders (WO-E2.1…), gates with thresholds, and the expected cost
(ms/step CPU 24 cores and GPU) for the bubble column.

## 9. Out of scope
Option (b) FFT implementation; collocated grids unless trivial; making the driver the default.
