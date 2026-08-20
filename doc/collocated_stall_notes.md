# The collocated plateau as a marginally-stable approximate-projection stall — working notes

*2026-08-20, mid-investigation. Companion to `collocated_accuracy_ceiling.md` (which still holds
the measured record); these notes assemble the day's measurements + literature into the current
leading mechanism. Verdicts pending: long-march creep, neutral-mode probe, Snellius R=16 sweep.*

## Measurements (this session, all committed or in flight)

1. **dt sweep, R=12 phi=0.60 bed (local RTX 5080, dt-fair stopping, tol 1e-8):**
   staggered k identical to 8 digits between DT=6 and 60 (and +0.004% at DT=600, cap-bound);
   collocated gauge-exact k moves +0.125% (6->60) and +0.180% (6->600), both estimators.
   The col-stag gap swings -0.63% -> -0.45% over the sweep.
2. **S1 probe (N=96, R=6, phi=0.60):** at the march stopping point the ABC loop has NOT closed:
   rms |uf - halfavg(u)| = 4.5e-2 <u> (p99 0.21, max 1.3); alpha-div of the CELL field rms
   3.9e-2 <u> vs 1e-8 for the face field.  Joint-fixed-point argument (rotational update
   P += (rho/dt) phi - mu div(u*) plus A phi = -div(u*) forces (rho/dt + mu A) phi = 0 => phi = 0)
   says the only true fixed point has uf == halfavg(u) and a dt-FREE steady system
   nu L_ibm u + F = G_gp P, D_alpha(avg_half u) = 0.  So the march parks off the fixed point.
3. **Flat-wall displacement sweep (E1/E1b/E2):** all three solvers IDENTICAL on wall-parallel
   force (momentum closure exonerated at flat walls, and theta_face == theta_center there);
   hydrostatic wall-normal force absorbed exactly by staggered, leaked as spurious v by
   collocated: gauge-exact O(h^6)-decaying (5.7e-3 -> 1.2e-6 for N=8..32), plain O(h^1.6).
   CAVEAT: an axis-aligned wall never puts a masked solid cell behind a PARTIALLY open face, so
   S3 and alpha-varying non-adjointness were not engaged.  s=0.50 exact incidence (wall through
   cell centers) is O(1)-wrong for ALL solvers incl. staggered — separate robustness note.
4. **TGV no-solid control:** col-stag field difference order +2.00 exactly (N=16..64).  Core
   collocated scheme clean; the phenomenon is IBM-boundary-tied.

## Literature (Guy & Fogelson, JCP 2005, "Stability of approximate projection methods on
## cell-centered grids" — read in full; PDF cached in session tool-results)

Their setting IS our scheme family: cell-centered approximate projection; pressure updates
PM I (p += phi) vs PM II (p += phi - chi (nu dt/2) D u*, the rotational/Timmermans family = ours);
near-boundary pressure gradient formed by ghost extrapolation of order 0/1/2, or "gradient 2a".

- MAC grid: ALL variants provably stable (eigenvalues < 1) -> matches our dt-flat staggered.
- Cell-centered approximate projection: eigenvalues -> 1 at the highest frequencies; the
  boundary-mode expansion of the step map (their eq. 91) has entries -1 + O(h^2) and
  1 - chi pi^2 h^2 / 4: near-boundary high-frequency modes decay at O(h^2) PER STEP ->
  O(1/h^2) steps to drain them.  REFINEMENT SLOWS THE DRAIN — a "plateau" any fixed-tolerance
  march cannot cross, growing with boundary content (our confinement scaling).
- PM II + "gradient 2" — the 2nd-order one-sided difference (-3p_i + 4p_{i+1} - p_{i+2})/2h,
  VERBATIM our gpCenterGrad cut-cell branch — is the combination that goes UNSTABLE, via
  near-boundary high-frequency (checkerboard-family) oscillations that appear only after
  hundreds of steps and worsen with refinement (their N=128: visible at step 418).  Cf. our
  ghost (1,2) march instability appearing only on >2000-sphere beds.
- Standard n-th order polynomial extrapolation of p into the ghost amplifies the checkerboard
  mode by 2^n/h at the boundary row — the mechanism of the sensitivity.
- Fixes they establish:
  (a) "gradient 2a": extrapolate the GRADIENT, not the pressure:
      (Gp)_1 = (-2p_1 + p_2 + 2p_3 - p_4)/(2h).  2nd-order everywhere AND annihilates the
      checkerboard mode; stable in their practice, full 2nd-order p.
  (b) PM I (chi = 0, drop the rotational term): stable for EVERY gradient tested; velocity
      2nd order; pressure 3/2 order in max norm (acceptable for steady permeability).
- dt enters all eigenvalues through a = 2 nu dt / h^2 -> the measured dt-dependence of where
  the march freezes.

Momentum-interpolation line (Majumdar 1988; Choi 1999; Yu et al. 2002 review): dt- and
relaxation-dependent steady states are the signature disease of inconsistent collocated
coupling; their fixes are RC-specific (remove dt terms from the face interpolation) and do not
transfer directly (our centerToFace carries no dt term), but the diagnosis language matches C2.

## Candidate fixes, ranked (to test AFTER the stall verdicts land)

- F1 **gpCenterGrad "gradient 2a" cut-cell branch** (extrapolate-the-gradient one-sided form).
  One kernel change, C1-C4 compatible (symmetric operator untouched — the gradient is not in
  the pressure matrix), AMR-portable.  Test: bed ladder + dt sweep + long-march m1.
- F2 **PM I ablation** (gate the -mu div(u*) rotational term for the collocated path).
  One-flag experiment; if the plateau collapses, decide production default on p-accuracy needs.
- F3 **Frank's residual re-projection**: every K steps solve div(a grad dp) = div(a r),
  r = momentum residual, a = openness (reuses CutcellMG rails); P += dp.  Drains frozen modes
  regardless of their origin.  BCs: natural alpha=0 Neumann.  Keep as fallback/accelerator.
- F4 Rider-style filter of the non-solenoidal cell-field component (iterated approximate
  projection at steady state).  Diagnostic-grade; production only if F1/F2 fail.

## Predictions to check against the running probes

- Long-march (R=12, 16k steps, dt switches at 8k/12k): k should CREEP toward staggered on a
  ~1e4-step scale (their 1 - pi^2 h^2/4-type factors), m1 decaying slowly; after a dt switch the
  state should keep moving (it is not a fixed point of either map).
- Neutral probe: perturbed/plug-IC marches should land at DIFFERENT k at the plateau scale
  (slow modes repopulated differently), spread >> march noise.
- If instead everything is EXACTLY frozen and IC-independent, the map deviates from the
  Guy-Fogelson structure somewhere and the code must be re-derived against the model.

## Growth-rate measurements (2026-08-20, evening — the instability quantified)

R=12 (N=192) phi=0.60 bed, zero IC, gauge-exact unless noted; growth read from dm1 (m1 = rms
|uf - halfavg(u)|/<u>) between 250-step reports:

| variant           | DT  | growth                         | doubling time (steps) |
|-------------------|-----|--------------------------------|----------------------|
| gauge-exact       | 60  | exponential from ~step 1700    | ~750–780             |
| gauge-exact       | 600 | exponential from ~step 700     | ~82 (rate ∝ dt)      |
| gauge-2a          | 60  | exponential, smaller seed      | ~850 (NOT stabilized)|
| gauge-exact, R=6  | 60  | none through 6000 steps        | — (stable at N=96)   |
| gauge-2a, R=6     | 60  | none through 6000 steps        | —                    |
| PM I (rot off) R=6| 60  | none through 6000 steps        | —                    |
| staggered         | any | none (dt-flat to 8 digits)     | —                    |

Key facts: the unstable mode is nearly MEAN-FREE (at DT=600 m1 grows 1.9e-2 -> 12 while k moves
only 0.25%) — invisible to k-based march criteria; that is how every ladder run sampled a
corrupted field without noticing.  The DT=600 dt-sweep run was not stable, merely short
(600 steps from a tiny seed; its converged=False was the tell).  Growth requires the finer
cut-cell structure of R=12 — R=6 (N=96) shows no growth in 6000 steps for ANY variant, so
N=96 cannot discriminate fixes.

**gauge-2a verdict: REFUTED as a standalone fix** — it delays onset and slows the rate ~0.85x
but the growth remains exponential at N=192.  Either the tight-throat 2-point fallback rows
(their "gradient 1", amplification 2/h) still feed the mode, or the aperture/cut-cell coupling
adds a channel the 1D straight-wall model lacks.  PM I verdict pending (running).

Fallback ladder if PM I also fails: PM III-style update (pressure-free momentum, p recomputed
from phi); Rider-style periodic filtering of the non-solenoidal cell component; Frank's
residual re-projection div(a grad dp) = div(a r).  Combinations (PM I + gauge-2a) also open.

Bookkeeping: set_dt staleness bug found & fixed (stencil diagonal rho/dt now rebuilt on dt
change — the first dt-switch experiment's post-switch explosion was this artifact, not scheme).
