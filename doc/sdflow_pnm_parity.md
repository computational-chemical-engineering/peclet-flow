# sdflow ↔ pnm_backend parity: status & backlog

Living tracker for converging the canonical **sdflow** solver to match (and eventually replace) the
production **pnm_backend**.

> **STATUS (2026-06-11): accuracy RESOLVED.** Against the Zick & Homsy (1982) SC-sphere Stokes-drag
> ground truth, **both** codes are correct to <0.05% at N=128 with clean grid convergence. sdflow's
> *default* path (`galerkin=False`, RB-GS cut-cell pressure) is as accurate as pnm. The only real defect
> is sdflow's **optional** Galerkin-multigrid pressure path (`galerkin=True`), which is inconsistent and
> must be fixed or removed (backlog item 7) — it was the source of every earlier "sdflow drifts" finding.
> Speed is at parity. Remaining work is the Galerkin-MG fix and Phase-3 retirement of pnm_backend.

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
- **Accuracy — RESOLVED against ground truth (Zick & Homsy 1982). Both codes are correct; sdflow's
  *default* path matches the reference as well as pnm.** The discriminator was an **external** reference,
  not refinement of the two codes against each other: the simple-cubic (SC) array of spheres has a
  semi-analytic Stokes drag factor `K(c)` from Zick & Homsy. pnm_backend was validated against it long
  ago (`output/drag_dimensionless_sc.csv`, <0.1%); `scripts/validate_zick_homsy_sdflow_vs_pnm.py` redoes
  it for **both** codes on the *identical* geometry. `K = F·V_cell/(6π·μ·R·U_sup)`.

  | φ (solid) | Z&H K | sdflow N=128 | pnm N=128 |
  |---|---|---|---|
  | 0.064 | 2.810 | 2.811 | 2.811 |
  | 0.125 | 4.292 | 4.292 | 4.292 |
  | 0.216 | 7.442 | 7.444 | 7.444 |
  | 0.343 | 15.400 | 15.402 | 15.402 |

  Both `<0.05%` at N=128, clean monotone grid convergence (N=32/64/128), agreeing with each other to
  4 digits. **sdflow's cut-cell IBM is correct and grid-convergent** — the long-open "which permeability
  is correct" question is **closed: both are, and they agree with the reference.**

  **The catch — and the explanation for the entire earlier "drift" saga: it was a bug in sdflow's
  OPTIONAL Galerkin-multigrid pressure path (`galerkin=True`), not the physics.** On the same SC sphere:

  | sdflow pressure path | N=64 | N=128 | vs Z&H |
  |---|---|---|---|
  | `galerkin=False` (direct cut-cell RB-GS operator — **the default**) | 4.291 | 4.292 | ✓ |
  | `galerkin=True` (Galerkin-coarsened MG + PCG) | 4.292 | ~4.46 (climbing) | ✗ +4% |

  Decisive test: **seed N=128 from the converged (correct) N=64 solution.** `galerkin=False` *holds*
  K=4.290→4.292 rock-stable over 300 steps (4.292 is its fixed point = Z&H); `galerkin=True` *runs away*
  from 4.292 up to ~4.46 (4.292 is NOT its fixed point). So the RAP/Galerkin coarsening of the cut-cell
  pressure operator is **inconsistent**, converging the per-step pressure solve to a slightly wrong
  operator's solution → a wrong steady that worsens with N. The direct cut-cell RB-GS operator is
  correct.

  **This retroactively invalidates the old `grid_convergence_sdflow_vs_pnm.py` "both drift" table**
  (k/N² 5.700→5.603→5.586→~5.33, and the pnm 5.561→5.020→4.294 column): the sdflow side used
  `galerkin=True` (the buggy path → the apparent down-drift in k/N² = up-drift in K), and the pnm side
  was *mis-configured* (time-marched with default outer-iterations instead of its SIMPLE 800-outer steady
  solve → under-converged). Both halves were broken; the single-sphere Z&H comparison with each code in
  its correct mode supersedes it entirely. (Also: the earlier "NOT Brinkman" correction still stands —
  Brinkman was never active.)

## Backlog

1. **[DONE — RESOLVED] Accuracy / ground-truth.** Settled against Zick & Homsy (1982) SC drag: **both
   codes are correct** to <0.05% at N=128 across φ=0.064–0.343 with clean grid convergence
   (`scripts/validate_zick_homsy_sdflow_vs_pnm.py`). sdflow's *default* (`galerkin=False` RB-GS cut-cell
   pressure) matches the reference as well as pnm. The old "both drift" `grid_convergence_*` table is
   **retired** — its sdflow column used the buggy `galerkin=True` path and its pnm column was
   mis-configured (not the SIMPLE 800-outer steady solve). The real discriminator was an external
   reference, exactly as item 3 anticipated.
2. **[DONE] RB-GS is the default.** Simple RB-GS (`galerkin=False`) matches pnm speed (16 vs 15 ms/step)
   *and* is the **correct** path (item 1 / item 7). It's the recommended default. "Do better" (beat
   RB-GS speed) is still open but lower priority now that correctness is established.
3. **[DONE — see item 1] Ground-truth the permeability.** Zick & Homsy was the reference; both codes
   reproduce it. Body-fitted meshing no longer needed. The "localise the gap" sub-task collapsed into
   item 7: the only real discrepancy was sdflow's Galerkin-MG pressure path.
7. **[NEW — open bug] Fix or remove sdflow's Galerkin-multigrid pressure path (`galerkin=True`).** It is
   **demonstrably inconsistent**: on the SC sphere it converges the per-step pressure solve to a wrong
   operator's solution, giving a steady that drifts ~+4% in K at N=128 (worsening with N), whereas the
   direct cut-cell RB-GS operator (`galerkin=False`) holds the correct Z&H value. Root cause is almost
   certainly the RAP/Galerkin coarsening of the cut-cell pressure operator (coarse operator not a
   consistent restriction of the fine cut-cell operator). Options: (a) fix the coarsening (rediscretise
   the cut-cell operator on each level instead of RAP, or correct the transfer operators), or (b) drop
   the Galerkin path entirely since RB-GS is the correct, fast default. Until fixed, **do not use
   `galerkin=True` / `set_pressure_pcg` for production results** — guard or warn in the API.
4. **[deferred] Crank–Nicolson** — ported then reverted: the simple explicit Laplacian is inconsistent
   with the Robust-Scaled cut-cell IBM (θ=0.5 gave a 4% θ-dependent *steady* error — a bug). Correct CN
   would need the explicit half-step to use the cut-cell operator (research effort). No benefit for
   steady cases anyway (steady is θ-independent). Backward-Euler is the right default.
5. **[deferred / scoped] Non-periodic BCs** (via the cut-cell IBM, not a halo-BC system), physical grid
   spacing (dx≠1), and IBM scheme selection (point-value SCHEME 0 vs cell-average SCHEME 1).
6. **[Phase 3] Retire pnm_backend** once parity (accuracy + speed) is signed off on real cases; tag it
   first as the reference.

## Benchmarks / tools
- **`scripts/validate_zick_homsy_sdflow_vs_pnm.py` — the ground-truth validator** (SC drag K vs Zick &
  Homsy, both codes, identical geometry; coarse→fine seeding). The authoritative accuracy check.
- `output/drag_dimensionless_sc.csv` — pnm_backend's original Z&H grid-convergence sweep (<0.1%).
- `scripts/bench_sdflow_vs_pnm.py` — head-to-head time-to-steady (configurable solver settings).
- `scripts/grid_convergence_sdflow_vs_pnm.py` — **superseded/misleading** (sdflow column used the buggy
  `galerkin=True`, pnm column mis-configured); kept only for history. Use the Z&H validator instead.
- `scripts/cross_validate_sdflow_vs_pnm.py` — field-level agreement on channel + single sphere.
