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

  **sdflow is grid-convergent** (changes −1.7% then −0.2%, settling to ~5.586e-3). **pnm_backend is NOT**
  (changes −9.7% then −14.5%, *accelerating* downward — its permeability keeps dropping as the geometry
  resolves). So the gap is **pnm_backend's resolution-dependent error**, not sdflow's — almost certainly
  its **Brinkman penalization** (`add_brinkman_drag_kernel`: a soft drag that smears no-slip over a layer
  and over-damps more as the cut-cell surface resolves), which sdflow's sharp Robust-Scaled cut-cell IBM
  avoids. *Caveat:* "more accurate" is inferred from grid-convergence behaviour; absolute ground truth
  would need N=256 and/or a body-fitted reference, but the convergence evidence is strong.

## Backlog

1. **[DONE] Grid-convergence study** — see the table above. Result: **sdflow is grid-convergent;
   pnm_backend is not** (its permeability drops accelerating with N). So sdflow appears to be the
   *more accurate* solver, and the gap is pnm_backend's resolution-dependent Brinkman-penalization error.
   This flips the earlier worry — sdflow is not "behind on numerics" here; if anything it's ahead.
   Optional follow-up: N=256 and/or a body-fitted/published reference to nail absolute ground truth.
2. **[TODO — user-requested] Revisit RB-GS and do better.** Simple RB-GS already matches pnm_backend
   speed; make it the *default* for sdflow (it's the efficient choice for steady marches), and then see
   if we can beat it — e.g. tuning sweep counts per step, better smoother, or a cheap accelerator that
   doesn't carry the full multigrid cost. (The Galerkin-MG/PCG path stays available for stiff cases.)
3. **[optional] Confirm the gap is the Brinkman drag** — re-run pnm with the Brinkman penalization
   weakened/disabled (if exposable) and check whether its permeability then grid-converges toward
   sdflow's ~5.586e-3. Would turn "almost certainly" into "confirmed".
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
