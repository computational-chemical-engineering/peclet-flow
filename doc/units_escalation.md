# Physical-domains escalation — questions the implementing session must not answer itself

Protocol: `suite/docs/PHYSICAL_UNITS_PLAN.md` §9.6. One section per question, each with the
measurement that raised it. Nothing here is a decision; every one is for the session that wrote the
design note.

---

## E1 (Phase 2, commit C2) — gate **G1**'s `1e-9` bound is unreachable at the shipped operator precision

**Status: RESOLVED 2026-09-07 (Fable).** Keep **both** configurations in
`units_anisotropic_poiseuille`, making two different statements. The gate exists to prove the
anisotropic operator is pointwise exact on the quadratic, and that statement has to be made where
the float operator storage cannot mask it — so the `1e-9` exactness bound moves to a
float-representable metric: `extent = (16, 10, 16)`, `spacing (1, 0.25, 2)`, `hRef = 0.25` on `y`,
`w = (1/16, 1, 1/64)`, `mu' = 80` (measured **1.388e-15**). The §5.3 configuration
`(16, 12, 16)`, `spacing (1, 0.3, 2)`, stays in the same test as a second stretched case at
**`1e-7`**: it is a real production-shaped configuration (`mu' = 55.5…` unrepresentable) and what it
measures is the WO-M float operator-storage floor (`docs/SCALING_ISSUES.md` #1) — a one-signed
multiplicative factor of ~1e-7 that `-DPECLET_FLOW_MREAL_DOUBLE` removes. It is a **tripwire for
that floor, not an exactness statement**. Option 3 (gate the whole thing in a double build) is
rejected: a gate on a build nobody ships proves nothing about the shipped one. The other three
assertions — `|v| = |w| = 0`, wall rows exactly `0`, and the isotropic control — apply to both
configurations. §8.2 of `doc/anisotropic_metric.md` and the ctest now read that way, and C2 is
pushed. The measurement below is kept as the record of why.

**Original report (kept):** C2 was implemented and every other gate was green; the
`units_anisotropic_poiseuille` ctest was RED **only** on this bound.

| gate | host-openmp | nvidia-cuda |
|---|---|---|
| `tests/kokkos` | **41 / 41** after the resolution above (40/41 as first measured) | **41 / 41** (40/41 as first measured) |
| `tests/kokkos_mpi`, np = 1, 2, 4 | **103 / 103** | **103 / 103** |
| `sdflow_mpi_np1` bit-exact to single-rank | `k_dist = k_ref = 5.84542251e+00`, rel `0.00e+00` | — |
| `tests/regression/sdflow_regression.py` (never `--update`) | — | **PASS**, every recorded number `+0.00 %`, every iteration count identical |
| the five verify scripts, `.npz` vs a build of `12cac0f` | **np.array_equal TRUE**: poiseuille 848/848, periodic_spheres 76/76, channel 31/31, bfs 8/8, lid_cavity 4/4 | — |
| `units_identity` / `units_scale_invariance` / `units_vof_sigma` | unchanged, printed numbers included | unchanged |
| `cutcellmg_aniso` (G4-order) | orders 2.0308 / 2.0052 / 2.0038 / 2.0052, all ≥ 1.95 | identical to the digit |

### What the note asks for

`doc/anisotropic_metric.md` §8.2: `runChannel` on cells `(16, 40, 8)` with `extent = (16, 12, 16)`
so `spacing() == (1.0, 0.3, 2.0)`, walls on the `y` cell centres `3.75` / `8.55` (`H = 4.8`),
`rho = 1, mu = 0.1, F = 0.01, dt = 50`, 300 steps, `velTol 1e-14`, `cutcellPressure = false`, and

> `max_j |u(y_j) − F (y_j − ylo)(yhi − y_j)/(2 mu)| / u_max ≤ 1e-9`

### What it measures (host-openmp, `OMP_NUM_THREADS=4`, this tree)

| run | max rel `|u − parabola|` |
|---|---|
| stretched `(16, 40, 8)` on `(16, 12, 16)` — **the §8.2 configuration** | **3.052e-08** |
| the isotropic control `(16, 16, 16)` at `extent == cells` (same test) | 3.701e-15 |
| the SAME stretched run, rebuilt with `-DPECLET_FLOW_MREAL_DOUBLE` | **8.674e-15** |

`max|v| = max|w| = 0.000e+00` exactly, and the wall/solid `u` rows are exactly `0` — the other three
§8.2 assertions pass, in every row above.

### The cause — found, and it is not the metric

The error PROFILE is a single one-signed **multiplicative** factor: the relative error is exactly
proportional to `u` itself (`7.153e-09 / 0.0675 = 1.06e-07`, …, `3.052e-08 / 0.288 = 1.06e-07` — one
constant across all fifteen fluid rows), i.e. the computed profile is the analytic parabola scaled by
`1 − 1.06e-07`. That is an error in the effective viscous coefficient, not a discretization error and
not a solve residual.

It is the **float momentum-operator storage** (`IbmSolver::FV`, `MReal`; WO-M, `docs/SCALING_ISSUES.md`
#1). The x/z couplings of an x/z-independent profile cancel only if the stored `AC` equals
`idiag + 2bx + 2by + 2bz` to the last bit:

- the **isotropic** control has `hRef = 1`, so `mu' = mu·tRef/hRef² = 5`, `b = (5, 5, 5)` and
  `AC = 1 + 2((5 + 5) + 5) = 31` — every one of them exactly representable in float, so the
  cancellation is EXACT and the gate reads 3.7e-15;
- the **§8.2 stretched** case has `hRef = 0.3`, so `mu' = 0.1·50/0.09 = 55.5555…` and
  `w = (0.09, 1, 0.0225)`; `AC = 1 + 2((5 + 55.5555…) + 1.25) = 124.6111…` — none of them
  representable, and the residue is ~1 float ulp of `AC` (≈ 7.5e-06 absolute) against `2b_y = 111.1`,
  i.e. ~1e-07 relative. That is the measured number.

Two independent confirmations: (i) the double-storage build of the identical source reads 8.674e-15;
(ii) changing ONLY the y extent so that `mu'` and the weights become float-exact restores it —

| `extent` | `spacing()` | `hRef` | `mu'` | `w` | max rel err |
|---|---|---|---|---|---|
| `(16, 12, 16)` | `(1, 0.3, 2)` | 0.3 | 55.5555… | `(0.09, 1, 0.0225)` | 3.052e-08 |
| `(16, 10, 16)` | `(1, 0.25, 2)` | 0.25 | 80 | `(1/16, 1, 1/64)` | **1.388e-15** |
| `(16, 8, 16)`  | `(1, 0.2, 2)`  | 0.2  | 125 | `(0.04, 1, 0.01)` | **2.385e-15** |

(All three are genuinely anisotropic with `hRef = min_a h_a` on the MIDDLE axis, so all three keep
§8.2's stated purpose — "a non-trivial check of `hRef = min`".)

### The question

The `1e-9` bound cannot be met in the DEFAULT (float `MReal`) build at `spacing = (1, 0.3, 2)`, for a
reason that predates and is orthogonal to Phase 2. Which of these is §8.2?

1. **Keep the configuration, relax the bound** to `1e-7` (measured 3.052e-08, with the one-signed
   proportional profile asserted as well so the gate still fails on anything that is not the float
   floor).
2. **Keep `1e-9`, move the extent** to `(16, 10, 16)` — `spacing (1, 0.25, 2)`, `hRef = 0.25`,
   `h' = (4, 1, 8)`, walls at the y cell centres `3.125` / `7.125`, `H = 4.0`,
   `u_max = F H²/(8 mu) = 0.2` — which measures **1.388e-15**, i.e. the isotropic control's own floor.
3. **Keep both, build the gate in double**: a second `test_units` executable compiled with
   `-DPECLET_FLOW_MREAL_DOUBLE` (measured 8.674e-15). Costs a second binary and states the gate about
   a build the solver does not ship.

**Decided:** 1 and 2 together — see the Status paragraph at the top of E1. The ctest now carries
`(16, 10, 16)` at `1e-9` (1.388e-15) AND `(16, 12, 16)` at `1e-7` (3.052e-08), and its comment
records the float-floor evidence (the 1.06e-07 constant ratio and the 8.674e-15 double build).
Option 3 was rejected.

---

## E2 (Phase 2, commit C2 → C3) — MG-PCG does not converge on the stretched hierarchy, so §8.5's ORDER row is measured on a stalled solve

**Status: RESOLVED 2026-09-07 by commit C3, and by the coarsening rule ALONE.**
`CutcellMG`/`VelocityMG` now defer an axis that is already `theta = 2` times coarser than the finest
coarsenable one (`mgChooseRatio`, doc/anisotropic_metric.md §5), and on the stretched ladder MG-PCG
goes **500 / 500 / 500 CAPPED -> 7 / 8 / 8** at N = 16 / 32 / 64, with `r/|b|`
1.3e-11 / 3.0e-11 / 3.1e-11 and the L2 error equal to the exact discrete solution to every printed
digit — i.e. the MG-PCG rows now ARE the FCG control, and the stretched order is the clean
**2.0038** instead of the contaminated 2.0308. Nothing else changed: not the smoother, not the
post-smoothing colour order, not the driver selection, not the bottom solve.

The direct read-out confirms the mechanism. `pr` (`PECLET_FLOW_MG_DEBUG=2`, zero iff the V-cycle is
symmetric w.r.t. the fine operator), median over the FCG iterations at N = 32:

| problem | today's full coarsening | with the aspect rule | the cubic control |
|---|---|---|---|
| all-fluid stretched | **3.555e-01** (max 1.08) | **6.818e-02** | 2.814e-02 |
| Z&H sphere stretched | 5.345e-02 | **4.002e-02** | — |

so the stretched hierarchy's asymmetry falls 5.2x into the neighbourhood of the isotropic periodic
hierarchy's own 0.062 (`flow/CLAUDE.md`, WO-H). The V-cycle rate over cycles 2-8 goes
**0.6920 -> 0.1501** and the Z&H sphere at `phi = 0.216` goes **24 -> 10** iterations at N = 32
(cubic control 9) and **25 -> 10** at N = 64 (cubic 10). `PECLET_FLOW_MG_ASPECT=1e9` reproduces
every "before" number in the table below to the digit, so the ablation IS today's rule. Full record:
`doc/anisotropic_metric.md` §10, the C3 entry.

**The before-numbers, kept as the record:**

`cutcellmg_aniso` (the new ctest), all-fluid periodic box, `(N, 2N, N/2)` with
`setOpenness(…, 1, 4, ¼)`, `levels = 6`, 2/2 sweeps, host-openmp:

| ladder | N = 16 | 32 | 64 | LS order |
|---|---|---|---|---|
| stretched, **MG-PCG** (§8.5's driver) | 500 iters, r/\|b\| 5.2e-06 | 500, 1.9e-04 | 500, 1.5e-04 | **2.0308** |
| cubic control, MG-PCG | 8 iters, 1.0e-11 | 8, 4.2e-11 | 8, 4.9e-11 | 2.0052 |
| stretched, **FCG** (converged control) | 19 iters, 4.5e-11 | 22, 3.9e-11 | 27, 7.5e-11 | **2.0038** |
| cubic control, FCG | 7 iters | 8 | 8 | 2.0052 |

MG-PCG **caps at 500 on every stretched rung** (the residual stalls, non-monotone, around 1e-4…1e-5
relative — an indefinite V-cycle preconditioner, the mechanism `flow/CLAUDE.md` records for high
coefficient contrast). It is NOT the bottom solve (`setAgglomerationMode` 0 / −1 / 1 are identical to
the digit) and NOT the depth (levels 2, 3, 4, 5, 6 all cap). A 2×2 ablation isolates it to the
COMBINATION: the stretched SHAPE with `w = (1,1,1)` converges in 8 iterations and the CUBIC shape with
`w = (1, 4, ¼)` converges in 20; only stretched + stretched weights stalls — i.e. exactly the coarse
level table of §5.2, where today's full-coarsening rule keeps `(1, ½, 2) → (2, 1, 4) → (4, 2, 8)` and
the point RB-GS smoother stops damping.

Consequence for the C2 gate: the L2 error of the MG-PCG rows is contaminated (N = 64 reads 4.7897e-04
against the exact discrete 4.9713e-04, −3.6 %), which is what inflates the order to 2.0308. The gate
passes with margin either way, and the FCG control — the identical operator, converged — reproduces the
exact modal solution of the discrete system to every printed digit and gives the clean **2.0038**. The
ctest runs and gates both ladders and prints the exact discrete value beside each measured one, so a
future C3 run can read off the rate improvement directly. §8.5's rate item (c) asks for exactly this
comparison; these are its "today's rule" numbers.

---

## E3 (Phase 2, commit C4) — §4.4's one sentence about the v3 wall torque admits two readings, and they differ on an anisotropic grid

**Status: OPEN.** Everything else in C4 landed; this ONE sub-path (`hydroForceTorqueReaction` with
`hasMotion_ && cutcellPressure_`, i.e. the v3 transposed-stress wall torque of a MOVING instance)
keeps an explicit `requireIsotropic` refusal naming this entry, rather than have the implementing
session choose. A static scene, and the force+torque of a moving one without cut-cell pressure, are
admitted; every C4 gate is green.

### What the note says

`doc/anisotropic_metric.md` §4.4, second paragraph, in full:

> *Reaction force* `hydroForceTorqueReaction()`: the momentum row of component `a` is a force
> density in the component-`a` normalisation, so the body force is `F_a = - sum_owner R_a * h_a' *
> V'` in `forceTotalToPhys` units (the isotropic `h_a' V' = 1`). **The wall-torque term (v3) takes
> the same `h_a'` on its force factor and the physical lever arm.**

### The two readings

The v3 term (`flow_ibm.hpp`, `peclet::flow::hydro_reaction_torque_transpose`) is an added TRACTION,
`mu (n dA) x Omega`, integrated over one cut cell's wall patch and crossed with the lever arm:

```cpp
const double ax = oxv(i + sx) - oxv(i);   // the aperture wall-area vector, BODY-outward
const double ay = ..., az = ...;          // == -W_a of §4.4's traction paragraph
const double vx = ay * wz - az * wy;      // v = (n dA) x Omega
...
Kokkos::atomic_add(&Td(3 * oi + 0), mu * (r.y * vz - r.z * vy));
```

1. **"the same h_a'" = the factor `F_a` takes, `h_a' V'`, on the FORCE component `a`.** The literal
   reading of the sentence, and the one the session brief restated
   ("and the same hp_a on the v3 wall-torque force factor"). It would give
   `F_x = h_x' V' mu (a_y Omega_z - a_z Omega_y)`.
2. **"the same h_a'" = the per-axis factor §4.4's own AREA VECTOR carries, `V'/h_a'`, on the area
   component.** `n dA` here IS an area vector of exactly the kind the traction paragraph one
   sentence earlier writes as `A_a = W_a V'/h_a'`. It would give
   `F_x = mu ((a_y V'/h_y') Omega_z - (a_z V'/h_z') Omega_y)`.

Reading 2 is what the physics gives: `F_phys = mu_phys A_phys omega_phys`, and putting
`mu_phys = mu' rhoRef hRef^2/tRef`, `A_phys = A' hRef^2`, `omega_phys = Omega'/tRef` gives
`F' = mu' A' Omega'` in `forceTotalToPhys = rhoRef hRef^4/tRef^2` units — so the metric belongs to
the AREA, whose component index is the one the cross product consumes, not to the force component.
The counter-example that separates them: a wall patch whose area vector is purely `x`,
`n dA = (A_x, 0, 0)`, with `Omega = Omega_z e_z`, produces a force purely in `y`. Reading 1 scales
it by `h_y' V'`; reading 2 by `V'/h_x'`. They agree only when `h_y' = 1/h_x'`, i.e. isotropically.

Reading 1 is also not derivable from the reaction paragraph's own argument: that `h_a'` comes from
the component-`a` normalisation of a MOMENTUM ROW, and the v3 term is not a momentum row — it is a
traction the audit adds on top, with no `1/h_a` of surface-over-volume to cancel.

### Why this was not simply implemented as reading 2

Rule 2 of the implementing role: the note must not be second-guessed, and a sentence that names
`h_a'` where the derivation wants `V'/h_a'` is a decision for the session that wrote it — not least
because the same sentence also has to say whether `V'` is in the factor at all. The commit therefore
REFUSES the one path instead of choosing, which no gate exercises (the term is inert unless
`set_instance_motion` has been called AND `cutcell_pressure=True`) and which no isotropic run can
see.

### What C4 did implement, for contrast

* the traction integral `hydroForceTorque()` exactly as §4.4 writes it,
  `dFp_a = p' W_a V'/h_a'` and
  `dFv_a = -mu' sum_b W_b (V'/h_b') [(h_a'/h_b') gu[a][b] + (h_b'/h_a') gu[b][a]]`;
* the reaction force `F_a = -sum R_a h_a' V'`, gated by the periodic-Stokes identity
  `F = F_body * V_fluid` on the stretched grid (`units_anisotropic_sphere`);
* the v4 owner-boundary attribution correction, which §4.4's site list does not name at all: it
  REMOVES a term that is already inside `F_a = -sum R_a h_a' V'`, so it must carry exactly what that
  term carries there — the momentum row's pressure gradient is `w_c (P(i) - P(i-s))` since C1/C2, so
  the combined factor on `pi(i)` is `w_c h_c' V' = V'/h_c'`, the physical area of the face. That one
  is forced, not chosen: any other factor would break the identity the correction exists to keep.

### The question

Which of the two readings is §4.4's, for the v3 wall-torque force factor: `h_a' V'` on the force
component, or `V'/h_b'` on the area component? Once answered, the change is three lines in
`hydro_reaction_torque_transpose` plus dropping the `requireIsotropic` guard beside it, and a gate
would be a rotating-sphere torque against the analytic Stokes couple `8 pi mu R^3 Omega` on a
stretched grid.
