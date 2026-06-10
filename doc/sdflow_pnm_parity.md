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
  | 256 | **~5.33e-3** | (not run) | — |

  **UPDATE (N=256 overturns the "sdflow is grid-converged" read).** Extending the study to N=256 (16.7M
  cells) shows sdflow **drifts down another −4.6%** (5.586 → ~5.33e-3), *not* a plateau at 5.586. The
  N=256 value is solid — two independent ICs agree on it: a restart seeded from the converged N=128
  solution (upsampled 2×, ×4 since the packing is self-similar so `u_256(2x)≈4·u_128(x)`) relaxes
  *down* from 5.584 → 5.359e-3 (still dropping at step 120), while a cold start rises *up* to 5.197e-3
  (still rising) — both closing on **~5.32–5.34e-3**. (Note: large `dt` to shortcut the long N=256
  transient is *not* viable — `ν·dt≳40` makes the implicit diffusion too stiff for the velocity MG and
  the cut-cell step diverges to NaN; the seed-from-coarse restart is the robust way to reach N=256
  steady cheaply.)

  **So the corrected picture: BOTH cut-cell schemes drift downward with refinement** — sdflow
  −1.7%/−0.3%/−4.6% across 32→64→128→256, pnm faster/further. The flat-looking 64→128 sdflow plateau was
  misleading; sdflow is **not** grid-converged at the resolutions we can afford either. The true
  continuum permeability is **below both**. pnm at N=64 *is* internally converged (same k/N² for
  outer-iteration counts 50/200/800; production uses 800, RB-GS v=2/p=50) — so the gap is a real
  **discretisation** difference between two separate cut-cell IBM codes ("same math" was checked
  bit-exact only vs sdflow's *own* serial reference, never vs pnm_backend's kernels), **not** an
  under-convergence artifact. But neither code is in its asymptotic regime, so grid-convergence of these
  two schemes *alone cannot crown a winner* — an external reference is required.

  **Corrections to an earlier wrong claim:** (1) it is **NOT** the Brinkman penalization — that lives only
  in pnm_backend's velocity *multigrid*, and these runs used velocity RB-GS, so it was never active.
  (2) The N=128 pnm point (4.294e-3, from velocity RB-GS-80) is **less certain** — RB-GS-80 may under-
  resolve the velocity at N=128, and a full-multigrid re-check was too slow to complete; trust the
  verified N=32/64 trend, not the 30% headline.

  **Which permeability is correct is OPEN — and N=256 made it *more* open, not less.** The earlier
  "sdflow is more grid-stable, so probably nearer the continuum" no longer holds: sdflow also drifts at
  N=256. Both schemes converge downward, neither is asymptotic, and grid-refinement on its own is now
  *exhausted* as a discriminator (N=512 = 134M cells is impractical, and both would still be drifting).
  Resolving it now **requires an external reference** — a body-fitted / boundary-conforming solve of this
  exact packing, or a published permeability — plus localising the gap (item 3). Refinement comparisons
  between the two cut-cell codes will not settle it.

## Backlog

1. **[done — refinement exhausted as a discriminator] Grid-convergence study** — see the table, now
   through N=256. **Both** schemes drift downward (sdflow 5.700→5.603→5.586→~5.33e-3; pnm faster); the
   ~12% gap is real (separate codes) but neither solver is asymptotic, so refinement alone cannot say
   which is right. Earlier reads were over-claimed: "sdflow is more accurate" (the 64→128 plateau didn't
   survive to 256) and "pnm has a Brinkman error" (Brinkman wasn't active). Discriminating now needs an
   external reference (item 3) — *not* more refinement of these two codes.
2. **[in progress — user-requested] Make RB-GS the default + do better.** Simple RB-GS matches
   pnm_backend speed (16 vs 15 ms/step) and gives the same sdflow answer as the cut-cell PCG; it's the
   efficient choice for steady marches. Make it the recommended default; the Galerkin-MG/PCG path stays
   available for stiff cases. Then try to beat it (sweep-count tuning, better smoother, cheap accelerator).
3. **[KEY next] Ground-truth the permeability** to resolve item 1: (a) ~~N=256 grid-convergence~~ **DONE
   — sdflow drifts to ~5.33e-3; both codes drift, refinement is exhausted as a discriminator** (see the
   N=256 update above; seed-from-coarse restart is the cheap way to reach N=256 steady, large `dt`
   diverges). (b) **a body-fitted / published reference for this packing — now the only remaining
   discriminator** (a boundary-conforming mesh of the 2×2×2 sphere lattice, or a literature Kozeny-type /
   Stokes-permeability value for SC sphere packs at this solid fraction). (c) localise the gap — fixed
   analytic pressure → compare velocity IBM; fixed velocity → compare pressure operator. Reference each
   piece against pnm_backend's actual kernels (not sdflow's own serial reimplementation).
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
