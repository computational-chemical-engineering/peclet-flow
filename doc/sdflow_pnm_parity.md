# sdflow ↔ pnm_backend parity: status & backlog

Living tracker for converging the canonical **sdflow** solver to match (and eventually replace) the
production **pnm_backend**, using pnm_backend as the **numerical reference** (the mistake during initial
development was *not* doing this — sdflow was built as a from-scratch reimplementation and drifted).

## Established so far (single GPU, sphere-packing Stokes, N=64, grid units)

- **Style / parallelism:** sdflow is the better-structured, MPI-capable code (the whole reason it's
  canonical).
- **Pressure scheme:** sdflow now uses the production **incremental-rotational** correction (ported;
  pressure matches pnm_backend to 0.42% vs 2.36% for classical Chorin).
- **Speed — NOT a real gap:** with the *same* numerics (**simple RB-GS**, no multigrid/PCG), sdflow
  ≈ pnm_backend: **16.1 vs 15.4 ms/step** (~5%). The 1.5–3× "slowdown" reported earlier was entirely
  sdflow's over-engineered defaults (Galerkin multigrid + PCG to rtol=1e-9 *every step* — absurd for a
  steady march). **Caveat about earlier numbers:** the big efficiency gains reported during development
  (reductions ~96×, etc.) were *sdflow-internal* (bug-fix / typedef flips), NEVER vs pnm_backend.
- **Accuracy — a REAL difference, and sdflow looks like the ACCURATE one.** Mean velocity / permeability
  differs (11.6% at N=64), and it is a genuine **discretization** difference, not a solver artifact
  (persists across sdflow's RB-GS vs cut-cell PCG; both ~0.2296). **Grid-convergence study**
  (`scripts/grid_convergence_sdflow_vs_pnm.py`, dimensionless k/N², both verified at *true* steady — `<u>`
  plateaus, no under-convergence):

  | N | sdflow k/N² | pnm k/N² | rel diff |
  |---|---|---|---|
  | 32 | 5.700e-3 | 5.561e-3 | 2.5% |
  | 64 | 5.603e-3 | 5.020e-3 | 11.6% |
  | 128 | 5.586e-3 | 4.294e-3 | 30.1% |

  **sdflow is more grid-stable** (changes −1.7% then −0.2%, settling near ~5.586e-3) than pnm_backend
  (−9.7% N=32→64). **Both solvers are genuinely converged** — verified: pnm at N=64 gives the same k/N²
  for outer-iteration counts 50/200/800 (its production setting is `outer_iterations=800`, RB-GS v=2/p=50;
  my first benchmark using the default `outer_iterations=2` was only ~1.5% under, and the MG variant had
  already landed on the converged value). So the gap is a **real discretisation difference between the
  two cut-cell IBM implementations** (they are separate codes — "same math" was checked bit-exact only vs
  sdflow's *own* serial reference, never vs pnm_backend's kernels), **not** an under-convergence artifact.

  **Corrections to an earlier wrong claim:** (1) it is **NOT** the Brinkman penalization — that lives only
  in pnm_backend's velocity *multigrid*, and these runs used velocity RB-GS, so it was never active.
  (2) The N=128 pnm point (4.294e-3, from velocity RB-GS-80) is **less certain** — RB-GS-80 may under-
  resolve the velocity at N=128, and a full-multigrid re-check was too slow to complete; trust the
  verified N=32/64 trend, not the 30% headline.

  **Which permeability is correct is OPEN.** sdflow being more grid-stable *suggests* it's nearer the
  continuum answer, but a consistently-biased scheme can also be grid-stable to a wrong value, and
  pnm_backend was validated in prior work. Resolving it needs a **ground-truth reference** — N=256 and/or
  a body-fitted (boundary-conforming) solver, or a published permeability for this packing. That, plus
  localising the gap (item 3), is the real next step before claiming either solver is "the accurate one".

## Backlog

1. **[done, but inconclusive] Grid-convergence study** — see the table. sdflow is more grid-stable than
   pnm_backend, and the ~12% gap is real (both converged). But **which is correct is unresolved** — needs
   a ground-truth reference (see item 3). The earlier "sdflow is more accurate / pnm has a Brinkman error"
   was over-claimed and partly wrong (Brinkman wasn't active).
2. **[in progress — user-requested] Make RB-GS the default + do better.** Simple RB-GS matches
   pnm_backend speed (16 vs 15 ms/step) and gives the same sdflow answer as the cut-cell PCG; it's the
   efficient choice for steady marches. Make it the recommended default; the Galerkin-MG/PCG path stays
   available for stiff cases. Then try to beat it (sweep-count tuning, better smoother, cheap accelerator).
3. **[KEY next] Ground-truth the permeability** to resolve item 1: (a) N=256 grid-convergence (does
   sdflow stay ~5.586e-3 and pnm keep drifting, or do they meet?); (b) a body-fitted / published
   reference for this packing; (c) localise the gap — fixed analytic pressure → compare velocity IBM;
   fixed velocity → compare pressure operator. Reference each piece against pnm_backend's actual kernels
   (not sdflow's own serial reimplementation).
4. **[deferred] Crank–Nicolson** — ported then reverted: the simple explicit Laplacian is inconsistent
   with the Robust-Scaled cut-cell IBM (θ=0.5 gave a 4% θ-dependent *steady* error — a bug). Correct CN
   would need the explicit half-step to use the cut-cell operator (research effort). No benefit for
   steady cases anyway (steady is θ-independent). Backward-Euler is the right default.
5. **[deferred / scoped] Non-periodic BCs** (via the cut-cell IBM, not a halo-BC system), physical grid
   spacing (dx≠1), and IBM scheme selection (point-value SCHEME 0 vs cell-average SCHEME 1).
6. **[Phase 3] Retire pnm_backend** once parity (accuracy + speed) is signed off on real cases; tag it
   first as the reference.

## Benchmarks / tools
- `scripts/bench_sdflow_vs_pnm.py` — head-to-head time-to-steady (configurable solver settings).
- `scripts/grid_convergence_sdflow_vs_pnm.py` — the convergence study (item 1).
- `scripts/cross_validate_sdflow_vs_pnm.py` — field-level agreement on channel + single sphere.
