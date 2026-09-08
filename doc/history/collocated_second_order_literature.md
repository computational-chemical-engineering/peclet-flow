# Second-order collocated cut-cell schemes that keep an approximate projection — a literature survey

**Status:** literature survey (2026-07-05) to decide, before the embed rewrite, *which* second-order
embedded-boundary scheme to copy for `SolverColocated` — under the hard constraint of **keeping the
ABC / Bell–Colella–Glaz approximate projection** (no exact-MAC, no Rhie–Chow). Companion to
`collocated_first_order_analysis.md`.

Confidence tags: **[V]** = adversarially verified (3-0/2-0 vote) in the deep-research run;
**[S]** = extracted from the fetched source but verification was cut short by a spend limit (not
refuted — treat as source-grounded, single-source); **[K]** = domain knowledge / inference. No claim
below was refuted.

## The decisive axis: does the scheme keep an approximate projection?

The literature splits cleanly into three camps, and the ABC constraint mostly decides the ranking.

| Scheme (school) | Keeps approx. projection? | Wall reconstruction | Curved-wall *drag* order shown? | Small-cell handling | Graft cost onto our code |
|---|---|---|---|---|---|
| **Basilisk `embed.h`** (Ghigo–Popinet–Wachs) | **Yes [V]** — cell u + face u_f, project u_f, add centered grad back | 2nd-order (bi)linear at the boundary/partial-face **centroid**, along the normal (Johansen–Colella style) [V/S] | Cylinder C_D vs Wu 2009, N=128–8192 [S] | slope-limiter on small cells (not merging) [S] | **Low–med** — same layout, same cs/fs geometry we half-have |
| **Trebotich–Graves 2015 / EBChombo** (Colella) | **Yes [S]** — "second-order approximate projection, not exact/MAC" | Johansen–Colella boundary flux (least-squares/normal gradient) [S/K] | FV EB NS on cut cells [S]; drag order not pinned in this run | flux redistribution (Chombo signature) [K] | **Med** — AMR-native, more machinery than we need |
| **Johansen–Colella 1998** (foundation) | n/a (Poisson only) | the reference 2nd-order EB Dirichlet flux; **sliver-robust so plain point-relaxation MG works** [S] | 2nd-order on exact Poisson solutions [S] | conditioning-safe boundary flux (no merge needed) [S] | this is the *ingredient* both above copy |
| **4th-order EB unsteady Stokes** (SIAM `10.1137/22M1532019`) | **Yes [S]** — "collocated + approximate projection, matches the constraint" | weighted least-squares | 4th-order on curved EB Stokes (L1/L2) [S] | **weighted LSQ, NO merging/redistribution** [S] | **Med–high** — 4th order is likely overkill; the *no-merge LSQ* idea is the takeaway |
| **Gibou–Fedkiw–Cheng–Kang** (Gibou) | Poisson only | symmetric 2nd-order Dirichlet Poisson, "simple discretization" [S] | 2nd-order Poisson [S] | — | the *symmetric* Poisson is elegant but see next row |
| **Gibou-school NS** (JCP `S0021999115001710`) | **No [S]** — "achieves a *stable* projection by adopting a **MAC/staggered** layout" | — | — | — | **counter-evidence**: a Gibou NS scheme *abandoned* collocated for stability |
| **Seo–Mittal 2011** (sharp-interface, Mittal) | **Yes [S]** — collocated fractional-step | ghost-cell momentum; **cut-cell applied ONLY to the pressure-Poisson + velocity-correction** (mass conservation) [S] | sub-2nd-order (~Δx^1.8), and the goal was *moving-boundary pressure oscillations*, not drag order [S] | — | **Low** — cheapest: fixes only the pressure/mass side, keeps our momentum |
| **Mittal et al. 2008 ghost-cell** (Mittal) | Yes (collocated) | 2nd-order velocity via image point; **pressure BC only formally 1st-order (dp/dn=0)** [S] | — | — | this is essentially *what we already are* — and its pressure-BC weakness = our symptom |
| **Directional ghost-cell IBM** (RG 337301778) | Yes | **axis-by-axis** extrapolation (not true-normal) [S] | — | — | also "what we are"; confirms the axis-by-axis vs normal fork |

## Reading of the evidence

1. **Keeping ABC is not the obstacle — the Colella school proves it.** Basilisk `embed.h` **[V]** and
   Trebotich–Graves **[S]** both reach second order on curved embedded walls *while running an
   approximate projection* (project the face field, reconcile the cell velocity with a centered
   pressure gradient). So the constraint you care about is satisfiable; the first-order-ness is our
   implementation, not a law.

2. **The one ingredient we are missing is shared by every 2nd-order winner: the Johansen–Colella
   boundary flux** — a 2nd-order (bi)linear reconstruction of the normal gradient at the
   boundary-fragment **centroid**, used to close **both** the viscous operator **and** the pressure
   Poisson from the **same** `cs`/`fs`/centroid/normal geometry. Our a-priori work already validated
   this gradient for the *wall viscous* flux; what mode 4 lacked was applying the **same geometry to
   the pressure operator** — which is exactly the inconsistency `collocated_first_order_analysis.md`
   diagnosed, and exactly what "one geometry drives every operator" fixes.

3. **The Gibou path is a trap under our constraint.** The symmetric 2nd-order Dirichlet Poisson is
   beautiful, but the one Gibou-school *NS* paper in the survey **adopted a MAC/staggered layout to
   get a stable projection [S]** — i.e. it gave up the collocated approximate projection precisely for
   stability. That is direct counter-evidence to porting the Gibou projection while keeping ABC.

4. **The Mittal/ghost-cell school is our current scheme and its known ceiling.** 2nd-order velocity
   but a **1st-order pressure BC (`dp/dn=0`)** **[S]** — which is our exact symptom. Not a target to
   copy, but a confirmation of the diagnosis.

5. **A genuinely cheaper fork exists — Seo–Mittal's "cut-cell only for mass conservation."** They keep
   ghost-cell momentum and apply the consistent cut-cell treatment **only to the pressure-Poisson +
   velocity-correction** **[S]**. That is far less invasive than a full FV momentum rewrite and speaks
   directly to our finding that *the barrier is the pressure coupling, not the viscous operator*.
   Caveat: their demonstrated order is **sub-second (~Δx^1.8) [S]** and their goal was moving-boundary
   pressure oscillations — so it is a *lead*, not a proven 2nd-order drag recipe.

## Recommendation — the two to copy, and the order to try them

**Primary: Basilisk `embed.h` (Ghigo–Popinet–Wachs).** Best fit by every axis that matters: it is a
*single-grid collocated approximate-projection* solver **[V]** — our exact layout — with demonstrated
curved-wall drag **[S]**, the Johansen–Colella centroid/normal boundary flux driving all operators
from one geometry, and a slope-limiter (not cell-merging) for slivers **[S]**. It is the most direct
"copy this" and the lowest conceptual risk. The port is: build the *single* embedded geometry
(`cs`, `fs`, boundary-fragment area, barycenter `b`, inward normal `n̂` — we already have `fs`) and
route the viscous flux, the divergence constraint, and the pressure-Poisson closure all through the
**same** `dirichlet_gradient`/`embed_flux` reconstruction at `b` along `n̂`.

**Secondary / cross-check: Trebotich–Graves (EBChombo, Colella school).** Same philosophy, independent
implementation, AMR-native. Its value is the **flux-redistribution** small-cell remedy to compare
against Basilisk's slope-limiter, and as a second reference for the pressure-side treatment under an
approximate projection **[S]**. Higher graft cost (more machinery than we need), so a reference rather
than the thing to copy wholesale.

**Cheap experiment worth running first (before the full rewrite): Seo–Mittal's split.** Apply a
consistent cut-cell/one-geometry treatment to the **pressure-Poisson + velocity-correction only**,
leaving the existing (mode-0) momentum untouched. This is far cheaper than mode 4's FV momentum, and
it targets the exact barrier we localized (pressure coupling). If it moves the drag order toward 2,
it is a low-cost win; if it plateaus sub-2nd-order (as their own results did), it confirms the full
`embed` one-geometry rewrite is required. Either outcome is decisive and cheap.

**Not recommended under the ABC constraint:** the Gibou projection path (the surveyed NS variant
needed MAC for stability **[S]**), and any ghost-cell/axis-by-axis scheme (that is our current
first-order state).

## Measured: the Seo–Mittal (pressure-only) split does NOT reach 2nd order here (2026-07-05)

We ran the cheap experiment. Our existing projection ablations are exactly the "keep momentum, make the
projection progressively cut-cell-consistent" ladder: mode 0 already solves the **mass-conserving
cut-cell pressure Poisson** (`cutcell_pressure=True`: `A = −div(o·grad)`, RHS `div(o·u*)` — the
Seo–Mittal GCL ingredient), mode 1 fixes the cell→face map at solid-bordering faces, mode 2 adds its
adjoint pressure force, mode 3 moves the whole constraint to the open-face centroid. Z&H drag error and
local order (N=32→64):

| N | mode 0 | mode 1 | mode 2 | mode 3 |
|---:|---:|---:|---:|---:|
| 32 | +1.004% | +2.296% | +2.668% | +2.566% |
| 48 | +0.685% | +1.550% | +1.715% | +2.025% |
| 64 | +0.598% | +1.277% | +1.445% | +1.630% |
| **order** | **0.75** | **0.85** | **0.89** | **0.65** |

**Every projection-side treatment is first-order and worse than plain mode 0.** The mass-conserving
cut-cell pressure Poisson is *already* in mode 0, so the pressure side was never the ceiling. The
ceiling is the **momentum no-slip reconstruction**: the Robust-Scaled overlay is **axis-by-axis** (six
1-D link intercepts `θ_k`), which is **O(h) at a curved wall in every mode**. A first-order momentum
no-slip caps the drag order regardless of how consistent the projection is — consistent with the survey
(Seo–Mittal's own ~Δx^1.8 was a *moving-boundary spurious-pressure* metric, not steady drag) and with
our a-priori result (the FV viscous operator is 2nd-order *only when fed the true-normal wall
gradient*, but the actual no-slip feeds it the axis-by-axis O(h) one).

**Decision:** the pressure-only shortcut is ruled out. The single lever is the **momentum no-slip /
wall-shear reconstruction** — replace axis-by-axis intercepts with a **true-normal image-point /
centroid gradient** (Basilisk `dirichlet_gradient`), driving *both* the momentum viscous flux and the
pressure closure from the one boundary geometry. Proceed to the Basilisk `embed.h` port (primary
recommendation above); do not spend more effort on projection-side-only variants.

## Provenance / caveats

The deep-research run fetched the sources and verified the load-bearing claims (Basilisk = collocated
approximate projection **[V]**; one-geometry consistency **[V]**; centroid boundary flux **[S, 1-1
contested — cross-checks with Johansen–Colella and my reading]**) before a **monthly spend limit**
killed the remaining ~66 verification agents and the synthesis. The **[S]** rows are therefore
single-source and not adversarially double-checked — verify the specific drag-order numbers against
the papers before betting the implementation on them. Nothing here was refuted. Key sources:
`basilisk.fr/src/navier-stokes/centered.h`, `hal.science/hal-03948786` (Ghigo–Popinet–Wachs),
`projecteuclid …camcos.2015.10.43` (Trebotich–Graves), JCP `S0021999198959654` (Johansen–Colella),
SIAM `10.1137/22M1532019` (4th-order EB Stokes), JHU `Seo_Mittal_2011`, `S0021999115001710`
(Gibou-school NS → MAC).
