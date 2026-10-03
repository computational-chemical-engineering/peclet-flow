# Architect brief A2: preconditioning steady advection–diffusion in the cut-cell scalar solver

*2026-10-03. The orchestrating session wrote this. The contract is `doc/scalar_ibm_design.md`
(amended by A1 under §5.2). All numbers are in `doc/scalar_ibm_log.md` (WO-3, WO-4 and WO-5
entries).*

## 1. The question

**How should the steady cut-cell scalar solve with advection be preconditioned (or reformulated) so
that it converges robustly up to cell Péclet Pe_h = |u|h/D ≈ 10?** The target is steady Brenner
closure / B-field problems in periodic packed beds at moderate Pe. The answer is an "Amendment A2"
to §5 / §6.7 with work-order changes.

## 2. Why it needs you

- The design's v1 (§6.7) assumed the lumped symmetric surrogate S suffices for steady mode. Its
  rationale was that "G7b and G8 involve no transport across the gradient".
- WO-5 measured that it does not suffice once advection crosses gradients.
- The project's first scientific use (dispersion closure problems, paper A4) needs exactly this
  regime.
- Solver architecture is your remit, and the choice interacts with the A1 coarse construction, the
  MPI parity gates and later the conjugate 2×2 smoother.

## 3. Current state (anchors in the worktree `suite/flow-scalar-ibm`, branch `scalar-ibm`, a12106a)

**Solver.**
- BiCGStab (`src/scalar_krylov.hpp`) on the true operator A.
- A = κ/dt + aperture-FV diffusion bands + the probe-flux facet overlay + advection.
- Advection is implicit first-order upwind (FOU) on all faces in steady mode. In transient mode the
  split is: implicit FOU on faces next to κ < ½ cells, explicit Koren elsewhere (§6.3).

**Preconditioner.**
- One V-cycle of `ScalarMG` (`src/scalar_mg.hpp`) on the SPD surrogate S = diffusion bands + κ/dt +
  lumped wall term, with the coarse wall term averaged at the fine probe distance (A1).
- RB-GS smoother, 7-point rediscretized coarse faces.
- WO-5 added "the lumped outflow restricted onto the coarse MG levels", i.e. the diagonal part of the
  upwind operator. The off-diagonal upwind coupling is NOT in S.

**Measured (WO-5 log).**

| case | iterations |
|---|---|
| transient advecting | 5–6 per step (unchanged from rest) |
| steady advecting, Pe_h = 0 | 13 |
| steady advecting, Pe_h = 0.1 | 40 |
| steady advecting, Pe_h = 0.3 | 89 |
| steady advecting, Pe_h = 1 | 200, not converged |

The A1 contraction guard still passes (0.135), because it sees only S.

**Building blocks.**
- core `solver/` has a functor BiCGStab, multicolour GS, and GraphAMG (smoothed aggregation, device
  apply).
- flow's VelocityMG handles a Helmholtz operator with upwind advection (`src/mac_velocity_mg.hpp`).
- flow's momentum uses implicit FOU + deferred correction.

## 4. Constraints

- Device-first, matrix-free compact stencils, MPI blocks with G = 2 halos.
- MPI parity (G10): bitwise at np = 1; ≤ 1e-10 at np = 2, 4 for rtol 1e-13; iterations ±1.
- Existing paths stay bit-identical (G12).
- Transient mode must not regress (5–6 iterations per step).
- No env knobs. Setters only.

## 5. Already decided (not open)

- The probe-flux discretization, κ storage, and A1's coarse wall term.
- FOU in steady mode for v1 (Q11). A deferred-correction Koren is optional WO-11.
- Krylov on the true operator. The preconditioner approximates.

## 6. Open — decide

The options include, without restricting you:
- (a) Put the FOU advection (off-diagonals included) into the surrogate. That makes it a
  non-symmetric M-matrix: rediscretized upwind coarse operators, with GS ordered or coloured, and
  possibly downstream-ordered or line smoothers for higher Pe_h.
- (b) Pseudo-transient continuation: steady solved by large-dt marching.
- (c) A GraphAMG preconditioner on the assembled FOU operator.
- (d) Defect correction around a stronger solver.
- (e) Restricting the steady advecting mode's Pe_h range.

Say which ships, its expected Pe_h envelope, how it interacts with A1 and with the WO-7 conjugate 2×2
smoother, and which gates change.

## 7. Verification

**Existing gates:** G-iter, G10, G12, and the WO-4/5 iteration rows.

**New gate:** steady advecting rows at Pe_h ∈ {0.1, 1, 10}:
- a periodic box with a sphere and a uniform-ish flow (WO-5's case);
- a periodic simple-cubic sphere array with a Stokes field from the flow solver, as the closure
  problem.

State iteration bounds and growth with resolution. A 2-D prototype extending
`tests/study/scalar_ibm/solver.py` / `mg_coarse_wall.py` is welcome.

## 8. Deliverable

- "Amendment A2" under §5 / §6.7 of `doc/scalar_ibm_design.md`, committed on `scalar-ibm`: a named
  path, with the session trailer lines, not pushed.
- Work-order changes (WO-5b and/or WO-6 adjustments).
- A summary of no more than 20 lines.

## 9. Out of scope

- Higher-order steady advection (WO-11).
- Inflow/outflow flux recovery (Q-E, decided: use the projection's own boundary-face flux).
- The collocated 'ghost' refusal (decided per Q13).
