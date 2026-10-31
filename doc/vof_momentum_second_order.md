# Design — a second-order momentum flux for the mass–momentum-consistent VoF transport

Architect design note, 2026-10-29, branch `mom2` (worktree `suite/flow-mom2`, from flow `2732e51`).
Answers `bubble_column_perf/validation/highratio/BRIEF_ARCH_MOMENTUM2.md`. Implementer: an
`opus-implementer` who has seen neither the brief nor the conversation; every formula, index and
threshold it needs is here. It decides nothing this note leaves open — §10 lists those, each with
a default.

A-priori evidence: `tests/study/vof_mom2/apriori1d.py` (numpy, host, ~10 s; committed with this note).

**Status 2026-10-31.** Implemented (WO-1..5) and **`'muscl_phase'` is the default** (USER 2026-10-29:
the default flips only if all of its gates pass; they did — §12). **The diagnosis of §2 is REFUTED**
for the 3D wrong limit by its own falsification gate G-D (2026-10-29): `'muscl_mixed_capped'` at P1
d/h 16 gave Re 39.659 with the cap active (0.91 faces/step), above the pre-registered refutation
threshold 39.50. The emptying-CV ("stayer debit") kick is therefore NOT the cause of `'muscl_mixed'`'s
+3 % wrong limit; **that mechanism is OPEN.** The scheme does not rest on §2: its gates (§9, §12)
judge it directly. §2 is kept as the hypothesis that was tested, with the measured 1D facts.

## 0. Summary

The opt-in MinMod MUSCL (`momentumMuscl`, now `'muscl_mixed'`) converges to a wrong limit at P1
(+3.0 / +3.1 % at d/h 16 / 24). This note attributed it to the extrapolation of the *mixed*
control-volume velocity to the outgoing flux slab: on a control volume (CV) a sweep empties of
liquid, the departing liquid takes a velocity the matter left behind (gas) has to pay for, with
gain `rho_l/rho_g` (§2). **G-D refuted that attribution in 3D** (status above): removing the kick by
the mass-aware cap left the wrong limit in place. Why `'muscl_mixed'` converges to a wrong limit is
OPEN. The kick itself is real in the uniform-gate round-off growth and in the 1D table of §2.4.

The design, `'muscl_phase'`: each phase's mass flux carries that phase's own face velocity,
reconstructed from the one CV velocity with an interface-biased, MC-limited (TVD) slope, and a
local mass-aware cap scales the slopes down where the matter staying behind is too light to pay
for them. The result obeys a per-sweep discrete maximum principle at any density ratio, keeps the
uniform-velocity identity bit-exact, reduces to plain MC MUSCL wherever only one phase is present,
needs no new halo exchange, and carries over unchanged to the collocated grid.

## 1. Problem and scope

**In scope.** The transported face velocity `u^` in `vof::MomentumConsistentAdvector`
(`src/vof/momentum_advect.hpp`), both the uncut (`momentumFluxes`/`momentumUpdate`) and the cut
(`momentumFluxesCut`/`momentumUpdateCut`) paths; the selection API; tests and gates.

**Out of scope** (do not touch): `WyAdvector` (the colour fluxes, planes, clamp, CFL, sweep order);
the frozen dilation term; the recovery floor; surface tension and curvature; face-viscosity averaging;
time integration; the collocated solver itself; AMR; the energy twin `energy_advect.hpp`
(`energyMuscl` — see §6.3; it is NOT changed here); Snellius.

## 2. Diagnosis — why MinMod on the mixed velocity converges to a wrong limit (REFUTED by G-D)

> **REFUTED 2026-10-29 for the 3D wrong limit; mechanism OPEN.** G-D (§9) predicted that the
> mass-aware cap alone (`'muscl_mixed_capped'`) removes the bias: CONFIRMED if Re16 < 38.99,
> REFUTED if Re16 >= 39.50. Measured at P1 d/h 16: Re 39.659 (`'muscl_mixed'` 39.740, `'upwind'`
> 35.327, reference 38.60), cap active on 0.91 faces/step (max 16), `floored` 0. The cap bounds the
> stayer debit of §2.2 by construction (§4.3), and the +3 % stayed, so §2.2 is not its cause and
> §2.3's closing argument ("the logic is closed") does not hold. What §2 still establishes, as
> measurement: uniform-gate round-off grows at 1e4 under `'muscl_mixed'` (ctest `vof_momentum`
> K7, 50 steps: 1.3e-7 host, 6.5e-8 CUDA) and the cap removes that growth (`'muscl_mixed_capped'`
> 7.8e-16 / 8.9e-16, upwind's level); the 1D table of §2.4. No replacement mechanism is claimed here.

### 2.1 The update, split into what stays and what moves

Notation for one sweep in direction `d` of component `e` (code names in brackets). CV `i` has
current shifted colour `C_i` [`cc_`], current velocity `u_i` [`vel_`], frozen flag `k_i` [`flag_`],
fluid fraction `eps_i` [`cvEps`, 1 uncut]. Face `p` [storage index, as `fc(p)`] is the face between
CVs `p` and `p+s_d`; it has the signed geometric Courant number `a_p`, the open-area-weighted one
`at_p` [`af(p)`; `= a_p` uncut] and the signed liquid flux `F_p` [`fc(p)`, same sign as `a_p`,
`|F_p| <= |at_p|`]. Per-phase mass fluxes through face `f`:
`mu_l,f = rho_l |F_f|`, `mu_g,f = rho_g (|at_f| - |F_f|)`, total `mu_f = mu_l,f + mu_g,f`.
The dilation mass is `D_i = rhohat_i (at_+ - at_-)`, `rhohat_i = k_i ? rho_l : rho_g`.

With `O` / `I` the outflow / inflow faces of CV `i`, the conservative update the code evaluates is

    rho_new u_new = R_i w_i + sum_{f in I} mu_f u^_f + D_i u^n_i,
    R_i  = eps_i rho(C_i) - sum_{f in O} mu_f            (mass that stays)
    w_i  = u_i - (1/R_i) sum_{f in O} sum_phase mu_phase,f (u^_phase,f - u_i)     ("stayer velocity")
    rho_new = R_i + sum_{f in I} mu_f + D_i.

The inflow and dilation pieces are convex combinations (weights `<= 1`, independent of the
ratio). The only place a velocity can be *extrapolated* is `w_i`: whatever extra momentum the
outgoing matter takes (`u^_out != u_i`) is debited from the matter that stays.

### 2.2 The defect

Upwind sets `u^_out = u_i`: `w_i = u_i`, so the update is convex at every ratio — the reason the
1e4 gate is flat. MinMod on the mixed velocity sets `u^_out = u_i + 1/2 (1-|a|) sigma` for BOTH
phases. On a CV the sweep empties of liquid (`|F_out| -> C_i`, so `R_i -> rho_g (1-|a|)`):

    w_i - u_i = - rho_l |F_out| (1-|a|) sigma / (2 R_i)  ~  -(rho_l/rho_g) |F_out| sigma / 2 .

The gas left behind is kicked by the ratio times a liquid-scale momentum difference. This is
exactly the "unbounded in the density ratio" coefficient the code comment at
`momentum_advect.hpp:216-231` measured on the uniform gate (round-off growing 3.3e-16 -> 2.2e-10 in
50 steps at 1e4; 3e-16 -> 6.1e-15 already at 1e3, i.e. a per-step gain > 1 at P1's ratio).

### 2.3 Why that is a wrong *limit* and not slow convergence

`sigma ~ h |du/dx|`, `|F_out| <= |a| <= 1/4`. At ratio 1e3, `d/h = 16`, the kick is
`~ 1e3 * 0.1 * 0.4 * (1/16) d|du/dx| ~ 2.5 d|du/dx|` — of the order of the rise velocity itself. It
decays like `h` only once `(rho_l/rho_g) |a| h |du/dx| << U`, i.e. for `d/h` of several hundred:
the same "absurd resolution" (~200 cells/diameter) Rudman and Arrufat et al. report for the
*inconsistent* scheme, and for the same reason — a ratio-amplified mass–momentum inconsistency.
Below that scale the gas-side error is O(U), is clipped and redistributed by the projection, moves
the interface through the mixed CVs' face velocities, and settles into a resolution-independent
bias. Measured: +3.0 % at d/h 16, +3.1 % at d/h 24, moving away.

The logic is closed: upwind converges to the right limit (order 0.99, extrapolation 38.58); MUSCL
differs from it only by the slope terms; of those, the inflow terms have ratio-independent weights
`<= 1`, so the ONLY ratio-dependent contribution is the stayer debit above. A ratio-scale bias can
come from nowhere else. (A cross-interface slope bias, the other candidate, is ratio-independent and
O(h) on an O(h) band, see §4.7, so it vanishes with h.)

### 2.4 A-priori operator test (1D, `apriori1d.py`)

One x-sweep per step of the exact deviation-form update, transporting a transverse velocity
`v0 = sin 2πx + 0.5 cos 6πx` through a sharp liquid slab `[0.3, 0.7)`, `a = 0.2`, periodic, one
period. The 1D PLIC flux of a step is exact, so every error is the momentum reconstruction's.
L1 error of `v`:

| scheme | ratio | N = 64 | N = 128 | N = 512 | order 256→512 |
|---|---|---|---|---|---|
| upwind | 1e3 | 3.12e-1 | 2.14e-1 | 7.41e-2 | 0.82 |
| `muscl_mixed` (today's opt-in) | 1 | 9.20e-2 | 3.63e-2 | 3.52e-3 | 1.77 |
| `muscl_mixed` | 1e3 | 9.40e-2 | 4.04e-2 | 8.52e-3 | **1.00** |
| `muscl_mixed` | 1e4 | 1.03e-1 | 6.65e-2 | 3.67e-2 | **0.17** |
| MC on mixed u, no cap | 1e3 | 4.53e-2 | 4.31e-2 | 1.21e-1 | diverges |
| `muscl_mixed_capped` (§4.3 cap only) | 1e3 = 1e4 | 9.52e-2 | 4.12e-2 | 5.70e-3 | 1.51 |
| **`muscl_phase` (this design)** | 1 | 3.39e-2 | 1.03e-2 | 7.56e-4 | 1.94 |
| **`muscl_phase`** | 1e3 = 1e4 | 3.99e-2 | 1.24e-2 | 9.30e-4 | **1.95** |
| Basilisk rule, unlimited, no cap | 1e3 = 1e4 | 2.94e-2 | 7.56e-3 | 5.81e-4 | 1.84 |

Read: the mixed slope's error grows with the ratio and its order collapses (0.17 at 1e4 — the 1D
face of the wrong limit); the less dissipative MC makes it diverge. The cap alone makes the error
ratio-independent (1e3 and 1e4 agree to 3 digits). `muscl_phase` is ratio-independent, second order
(1.95), within 1.25x of its own single-phase error, and its cap never bites in 1D (it exists for the
multi-D cases of §4.6). Identity: `v == 1` stays exactly 1 for every scheme. The 1D noise test does
NOT discriminate (no scheme grows noise in 1D at uniform `a`; the measured 3D growth needs the
divergent/convergent multi-D sweeps) — robustness is judged by the 3D gate G-M2-2.

What 1D does not show: at ratio 1e3 and N = 64 the mixed scheme's 1D error is barely above its
ratio-1 error. The 3D bias needs the feedback through projection and interface motion. The 3D
falsification test is G-D (§9).

## 3. Constraints and invariants

- **Uniform-velocity identity, bitwise**, every ratio, cut cells included: a uniform `u` must give
  every face value `== u` to the bit, so every `dv == 0.0` and `u_new == u_old`.
- **Ratio-1e4 robustness** at least as good as upwind: no slope term may carry a ratio-dependent gain.
- **One `F`**: the colour and the momentum share the clamped geometric flux; the Weymouth clamp, the
  frozen flag `k`, the dilation coefficient `rhohat (u^n - u_old)(at_+ - at_-)` and the recovery
  floor are unchanged.
- **Conservation telescopes bit-exactly**: each face value is computed ONCE per face, from donor data
  only, and used by both CVs — in-rank and across a rank boundary.
- **Default bit-identical**: `'upwind'` (and the existing `'muscl_mixed'`) keep their current kernels
  and expressions unchanged. The new update form is algebraically equal to the old one but NOT
  bit-equal, so it lives only in the new branch.
- On-device (Kokkos, `MDRange3`, x-fastest), FP64, MPI with the advector's `g = 3` halo, no new
  exchange, no new global synchronization (counters ride the existing reduction).

## 4. The scheme `'muscl_phase'`

### 4.1 Per-phase face velocities from one CV velocity

Keep ONE conserved momentum per CV and component, recovered after every sweep as today. Each face
carries two velocities, `uL_p` for the liquid flux and `uG_p` for the gas flux, both reconstructed
from the donor CV's velocity `u_don` with phase-specific slopes. (Carrying two momenta through the
three sweeps, as Basilisk does, costs three more fields and nine more exchanges per step and buys no
order — rejected, §4.9.)

### 4.2 The interface-biased TVD slope

For face `p`: `s = sign(a_p)`; donor `q = (a_p > 0) ? p : p + s_d`. Differences along `d`:
`Dm = u_q - u_{q-s_d}`, `Dp = u_{q+s_d} - u_q`. Phase holding, on the CURRENT shifted colour
(exchanged at the top of the sweep):

    holdsL(r) := eps_r > 0 && C_r >= 0.5          holdsG(r) := eps_r > 0 && C_r <= 0.5

(uncut: `eps_r = 1`). For phase `K` in {L, G}:

    est_K = !holdsK(q)                                  ? 0
          : holdsK(q-s_d) && holdsK(q+s_d)              ? 0.5 * (Dm + Dp)
          : holdsK(q-s_d)                               ? Dm
          : holdsK(q+s_d)                               ? Dp
          :                                               0
    sgn     = (Dm * Dp > 0) ? (Dm > 0 ? 1 : -1) : 0      // written NaN-safe like vofMinmod
    sigma_K = sgn * min(2|Dm|, 2|Dp|, |est_K|)

Away from the interface this is the MC (monotonized-central) limiter; at the interface it is
Basilisk's interface-biased estimate (central, one-sided away from the other phase, zero in a CV
where the phase is a minority) inside the MC bound. Both phase slopes share the sign `sgn`. A fully
solid neighbour (`eps = 0`) holds neither phase; its stored velocity still enters `Dm`/`Dp` (as in
today's cut MUSCL path).

### 4.3 The mass-aware cap

The donor's other face is `o = (a_p > 0) ? p - s_d : p + s_d`. Recompute its Courant numbers with
EXACTLY the expressions that produce `af` there: `a_o = 0.5*(ud(o - s_e) + ud(o))*dth`,
`at_o = ofac_o * a_o` (cut: `ofac_o = (d == e) ? vofEpsEff(ep(o), kd(o) == kVofSolid)
: 0.5*(od(o - s_e) + od(o))`; uncut: `at_o = a_o`). Face `o` is an outflow face of `q` iff
`at_o * s < 0`.

    muL = rho_l |F_p|,  muG = rho_g (|at_p| - |F_p|),  mu_p = muL + muG
    muO = outflowO ? rho_g |at_o| + (rho_l - rho_g) * min(|at_o|, eps_q C_q) : 0
    R   = eps_q * vofPhaseRho(rho_g, rho_l, C_q) - mu_p - muO + rhohat_q * s * (at_p - at_o)
    m   = min(|Dm|, |Dp|)
    ext = 0.5 * (1 - |a_p|) * |muG * sigma_G + muL * sigma_L|
    beta = outflowO ? 0.5 : 1.0
    theta = (ext > beta * max(R, 0) * m) ? beta * max(R, 0) * m / ext : 1.0

`R` is a lower bound of (stayer mass + dilation mass): exact when `o` is an inflow face (the common
case), conservative by `muO - mu_o >= 0` when `o` is outflow (`muO` uses Weymouth's upper bound
`min(|at_o|, eps_q C_q)` on the other face's liquid flux, so the other face's PLIC volume is never
needed). `rhohat_q` needs the donor's frozen flag, also for donors in the first ghost layer —
§4.8 item 3.

### 4.4 Face values and the update

Flux kernel, per face (store BOTH; `fluxU_` keeps `uL`, new field `fluxUG_` holds `uG`):

    uL_p = u_q + s * (0.5 - 0.5*|a_p|) * (theta * sigma_L)
    uG_p = u_q + s * (0.5 - 0.5*|a_p|) * (theta * sigma_G)

Update kernel (new branch; everything else as today):

    dvLM = uL_{i-s_d} - u_old,  dvLP = uL_i - u_old,  dvGM = uG_{i-s_d} - u_old,  dvGP = uG_i - u_old
    dev  = rho_g * ((aM - fM) * dvGM - (aP - fP) * dvGP) + rho_l * (fM * dvLM - fP * dvLP) + dilCoef

with `aM = af(i-s_d)`, `aP = af(i)`, `fM = fc(i-s_d)`, `fP = fc(i)` (signed; `a - F` has the sign of
`a`) and `dilCoef`, `rn`, the floor and the division exactly as in `momentumUpdate` /
`momentumUpdateCut`. The cap counter `capped` counts faces with `theta < 1`.

`'muscl_mixed_capped'` (diagnosis ablation, §9 G-D): `sigma_L = sigma_G = vofMinmod(Dm, Dp)` then
the cap; `uL == uG`, so it stores into `fluxU_` only and reuses today's update kernel unchanged.

### 4.5 Identity proof

Uniform `u = U`: `Dm = Dp = 0` exactly, so `sgn = 0`, `sigma_K = 0` (or `-0.0`), `ext = 0`,
`theta = 1`, and `uL = uG = U + s*c*0 = U` bit for bit. Every `dv` is `U - U = 0.0`, `dilCoef` is
`rhohat (U - U)(...) = 0`, `dev = 0`, `u_new = U` — whatever `F`, `eps`, the openness, the flag and
the ratio. Next to a solid CV holding another velocity the sign test still returns 0 because the
fluid-side difference is exactly 0.

### 4.6 Boundedness proof (any ratio)

Per sweep, write `rho_new u_new = (R + D) w' + sum_I mu_f u^_f + D (u^n - u_old)` with
`w' = u_old - (1/(R+D)) sum_O sum_K mu_K,f (u^_K,f - u_old)`.
1. *Inflow values* lie in `[u_nb, u_i]`: `|sigma_K| <= 2|D_toward_i|` and `theta <= 1` give
   `|u^ - u_nb| <= (1 - |a|)|u_i - u_nb|`, same sign.
2. *Stayer*: both phase slopes share one sign, so
   `|sum_O sum_K mu_K,f (u^_K,f - u_old)| = sum_O theta_f ext_f <= sum_O beta_f max(R_true,0) m <= (R+D)_true m`,
   hence `w'` moves toward the upstream neighbour by at most `m` — inside
   `[min(u_{i-1},u_i,u_{i+1}), max(...)]`.
3. The remaining term `D (u^n - u_old)/rho_new` is today's upwind dilation term, unchanged.

So the slope terms obey a discrete maximum principle at every ratio, and the scheme is exactly as
robust as upwind: the only non-convex piece left is upwind's own. Why the cap is still needed with
the phase masks: in 3D a CV holding `C >= 1/2` can be emptied within one sweep — a divergent sweep
with `k = 0` loses `|F_-| + |F_+|` up to `2 * 0.25`, a convergent one with `k = 1` drops `C` by the
inflowing gas volume — and its liquid slope would then pay on gas-scale mass. The mask makes these
events rare (1D: zero bites); the cap makes them harmless.

Single-phase inactivity (provable, gated in G-M2-9): with one density, `R >= rho(1 - |a_-|) >= 0.75 rho`
for one outflow face and `R = rho` for two, while MC bounds `ext <= rho |a|(1-|a|) m <= 0.1875 rho m`
at `|a| <= 1/4`; `ext > beta R m` is impossible. In single-phase regions `'muscl_phase'` IS MC MUSCL.

### 4.7 Accuracy

Bulk: MC-limited Fromm/BCG-in-sweep, second order away from extrema. Interface band (CVs whose
d-neighbour holds the other phase): one-sided or zero slopes, i.e. O(h) face errors on a band of width
O(h). The numerical viscosity there is `~|u| h` on a set of measure `O(h)` per unit interface area, so
the dissipation error is O(h^2) — global second order, first order in L-infinity on the band (1D:
L1 order 1.95, L-infinity order about 1.2). This is why Basilisk converges fast with zero minority
slopes. (The claim made here at design time, that the mixed-velocity bias of §2 is the only
obstruction, fell with G-D; see §2.)

### 4.8 Cut cells, ghosts, and the flag

1. Cut path: masses in cell-volume units as written (`eps_q`, `at`, `F` openness-weighted; the
   time-centring `0.5 - 0.5|a|` uses the geometric `|a|`, as today).
2. Reads per face (donor `q` has d-index `[g-1, g+n]`): `C`, `u`, `eps` at `q, q +- s_d`
   (d-index `[g-2, g+n+1]`, exchanged); `ud` at `o` and `o - s_e` (normal-axis index `[g-3, g+n]`,
   transverse `[g-1, g+n-1]`) — inside what `copyFaceVelocity` fills on the solver path (normal axis
   `[0, n+3]`, transverse `[1, n+4]` of the `g = 3` block); `ep`, `od`, `kd` at `o` (cut).
3. **Ghost flags.** `freezeShiftedState` sets `flag_` on inner CVs only. For the capped schemes, right
   after the `s = 0` exchanges of `cc_` (which then hold `C^{e,n}` in the ghosts), run one kernel over
   the ghost shell: `flag(i) = cc(i) > 0.5 ? 1 : 0` — the inner formula on the neighbour's own
   values, so ghost flags equal the owner's bitwise. No exchange. Capped scheme +
   `useCellDilationFlag` throws `std::invalid_argument` (that ablation's ghost flags are undefined).

### 4.9 Alternatives rejected

| alternative | why rejected |
|---|---|
| MinMod/MC slope of the mixed velocity (today's opt-in) | non-convex stayer debit with gain `rho_l/rho_g` (§2); wrong limit at P1; 1D order 0.17 at 1e4, MC diverges |
| Basilisk-faithful: two momenta per component through the sweeps, unlimited central, `C >= 1/2` masks | +3 fields, +9 exchanges/step; unlimited slopes have no maximum principle; the masks alone do not bound 3D divergent/convergent sweeps (§4.6). Its 1D constant is 1.6x better (5.8e-4 vs 9.3e-4) — robustness is a hard constraint, the constant is not |
| Zalesak FCT on `u` with variable mass | provably bounded too, but two passes plus neighbour limiter values (one more exchange or a widened halo) and harder clipping of smooth extrema; the one-pass local cap gives the same guarantee |
| Cap alone on the mixed MinMod slope | ratio-independent, but MinMod-diffusive and zero at most interface CVs (1D 5.7e-3 vs 9.3e-4); kept only as the diagnosis ablation |
| Exact donor mass from the other face's PLIC flux | needs planes one ghost layer further (reconstruction +9 % at n = 64) or an exchange of `fc` (+9/step); the bound in §4.3 is exact whenever the other face is inflow |
| Slope of `M = rho u` | jumps by the ratio at the interface; meaningless |
| Threshold rule "no liquid slope if `rho_new/rho_old < c`" | a tuning knob with no proof |

## 5. Parallel, device, performance

- **MPI.** Every face value is a pure function of exchanged cell data, face velocities and the
  ghost-extended flag, evaluated by the same expression on both sides of a rank boundary, so the
  owner and the neighbour compute the same bits; the sums telescope as today. No new exchange,
  no new message size. Serial == parallel bitwise (G-M2-8).
- **Device.** No new kernel launches in the sweep loop (the slope and cap live in the existing flux
  kernel; the per-phase form in the update kernel); one tiny ghost-shell flag kernel per component per
  step, only for the capped schemes. No atomics. `capped` is a second reduced value of the existing
  clamp-count reduction (array reduction or a small struct reducer), never a separate fence. The
  ghost-flag kernel runs on the advector's execution instance with no fence of its own. Net: the
  same number of global fences and View allocations per step as `'upwind'` (G-M2-10;
  `device_sync_budget` is the instrument).
- **Data.** One new `SField fluxUG_` (`len_` doubles), allocated in `init()`. Extra traffic per face:
  `C` at two neighbours, `ud` at two positions, one flag byte, one store; per CV in the update: two
  loads. Expected cost well under the 1.15x bound of G-M2-6.
- **Determinism.** Face values are independent per face; host OpenMP and CUDA give identical bits for
  identical inputs; only the counters are reductions.

## 6. Where else it applies

### 6.1 Collocated grid

It carries verbatim. Nothing in §4 uses the half-cell shift: on the collocated grid the momentum CV is
the pressure cell, `C := C`, `F := WyAdvector`'s own clamped flux (shared bit for bit, no second PLIC
clipping), `k := WY`'s frozen dilation flag (ghost-extended the same way), and `a :=` the projected
face Courant number the colour already uses. The identity and boundedness proofs hold unchanged. The
other-face bound of §4.3 is needed there too, because WY's flux array also spans only
`[g-1, g+n-1]`. This is the scheme the collocated air–water VoF should inherit.

### 6.2 AMR

The stencil is three CVs and two faces per donor, the same footprint as today's MUSCL. Out of scope.

### 6.3 The energy twin

`energy_advect.hpp`'s `energyMuscl` extrapolates `T` the same way. By §2.2 its stayer debit has gain
`(rho c_p)_l / (rho c_p)_g` (about 3500 for air–water) on a CV a sweep empties. Its header says there
is no such amplification; this note's analysis says there is. Not changed here — §10, F-6.

## 7. API and naming

- C++: `enum class MomentumScheme : int { Upwind, MusclMixed, MusclMixedCapped, MusclPhase }` and
  `MomentumScheme momentumScheme = MomentumScheme::Upwind;` replace `bool momentumMuscl`. Update its two
  readers (`src/flow_ibm_vof.hpp:2356`, `tests/study/case_a/vofmom_probe/vofmom_probe.cpp:105`).
- Solver: `setVofMomentumScheme(const std::string&)` / `vofMomentumScheme()`; the existing
  `setVofMomentumMuscl(bool)` maps to `MusclMixed` / `Upwind`.
- Python, diagnostics tier (opt-in): `sim.diagnostics.set_vof_momentum_scheme(name)` with
  `name in {'upwind', 'muscl_mixed', 'muscl_mixed_capped', 'muscl_phase'}` (an unknown name raises
  `ValueError`), getter `sim.diagnostics.vof_momentum_scheme() -> str`, and
  `vof_momentum_diagnostics()['capped']`. `set_vof_momentum_muscl(on)` stays, as the old spelling of
  `set_vof_momentum_scheme('muscl_mixed' | 'upwind')` (NAMING.md §0: additive since 1.0.0; its
  docstring names the canonical setter; one row in NAMING.md's divergence table).
- The default stayed `'upwind'` until the recorded user decision (§10, U-2); since 2026-10-31 it is
  `'muscl_phase'` (§12).

## 8. Work orders (dependency order; one commit each; named paths only)

**WO-0 (this commit).** Note + `tests/study/vof_mom2/apriori1d.py`.

**WO-1 — advector kernels** (`src/vof/momentum_advect.hpp`). Enum and member (§7); `fluxUG_`;
`capped` in `Diagnostics` and its reduction; the ghost-shell flag kernel after the `s = 0` exchanges
(capped schemes only) and the `useCellDilationFlag` guard; new branches in `momentumFluxes` and
`momentumFluxesCut` (§4.2–4.4, both capped variants); the per-phase branch in `momentumUpdate` and
`momentumUpdateCut` (`MusclPhase` only). `Upwind` and `MusclMixed` code paths byte-for-byte as
today. Rewrite the header's empty "transported velocity in the flux" section and the
`momentumMuscl` comment to point here.
*Accept:* host + CUDA build; existing `tests/kokkos/test_vof_momentum.cpp`,
`tests/kokkos/test_vof_cutcell.cpp` pass with their outputs unchanged; a local check that
`MusclPhase` on the uniform gate gives `max|u - U| == 0.0` at ratios 1, 1e3, 1e4.

**WO-2 — solver and bindings** (`src/flow_ibm_vof.hpp`, `src/flow_bindings.cpp`, solver header as
needed, `../docs/NAMING.md` row — the umbrella file goes to the coordinator, do not commit the
umbrella). *Accept:* a Python smoke test: every name round-trips through the getter, `'bogus'` raises
`ValueError`, `set_vof_momentum_muscl(True)` then the getter returns `'muscl_mixed'`.

**WO-3 — ctests.** (a) Extend the identity tests to loop over all four schemes, ratios
{1, 1e2, 1e3, 1e4}, with and without cut cells (G-M2-1). (b) The 1e4 50-step uniform noise gate
for `muscl_phase` and `muscl_mixed_capped`, double build (G-M2-2). (c) New `test_vof_momentum_slab`:
the §9 G-M2-7 transverse-slab oracle, plus single-phase `capped == 0` (G-M2-9). (d) MPI bit-identity
for `muscl_phase` (G-M2-8): extend the existing VoF-momentum MPI ctest if one exists
(`tests/kokkos_mpi/`), else add one, with `--bind-to none`. *Accept:* all green on host; (a)–(c) on
CUDA.

**WO-4 — studies, local GPU, every run under
`flock /home/frankp/Codes/bubble_column_perf/gpu_diag/mgd/heavy.lock`.** Driver
`validation/highratio/pec/run_hr_scheme.py` = `run_hr.py` + `set_vof_momentum_scheme(<arg>)`
(leave `run_hr.py` untouched). Runs: G-D (`muscl_mixed_capped`, P1 d/h 16, 24); G-M2-3
(`muscl_phase`, P1 d/h 16, 24, 32); G-M2-4 (P2, `muscl_phase` vs `upwind`, validated resolution);
G-M2-6 (ms/step from the d/h 24 runs). Report `capped` per step. Conditional: if G-M2-3 passes on
magnitude but misses on order, ONE extra run of `muscl_phase` at d/h 24 with the WY CFL cap halved
(0.125) to separate time from space error — a measurement, never a retune. Results go into
`highratio/DECISIONS.log` and the table in `validation/HIGHRATIO_VALIDATION.md`.

**WO-5 — default unchanged, device budget.** Full `ctest -LE bench`, host and CUDA; state_hash
digests == main (G-M2-5); the fence/allocation count of G-M2-10.

**WO-6 — records.** flow `CHANGELOG.md`; flow `CLAUDE.md` VoF bullet ("…momentum flux upwind by
default; `'muscl_phase'` is the second-order opt-in; NEVER a slope of the mixed CV velocity without
the cap"); this note's status line with the gate numbers and a row in `doc/README.md`; the register text of
§11 handed to the coordinator for `suite/docs/DECISIONS.md`.

## 9. Verification gates (pre-registered; PASS / MISS / NOT JUDGED; never retune)

| gate | what | criterion |
|---|---|---|
| G-M2-1 | uniform identity, all schemes, ratios 1e0..1e4, ± cut cells | `max|u - U| == 0.0` bitwise |
| G-M2-2 | ratio 1e4 uniform gate, 50 steps, double build, `muscl_phase` (also `muscl_mixed_capped`) | max error <= 1e-14 (prediction: upwind's 3e-16..7e-16 level) |
| G-M2-3 | P1 d/h 16, 24, 32, `muscl_phase`; `e = Re - 38.60` | `|e24|/38.60 <= 2 %` AND the least-squares order of `ln|e|` vs `ln h` over the three >= 1.5, with neither pairwise order < 1.0 and all three `e` of one sign; otherwise MISS, or NOT JUDGED (escalate) if the signs differ or `|e32|` is below 3x the Re averaging standard error |
| G-M2-4 | P2 at its validated resolution | `|Re_phase - Re_upwind|/Re_upwind <= 2 %`; run completes; min dt >= 0.5x upwind's; `floored == 0` |
| G-M2-5 | default unchanged | full `-LE bench` host + CUDA green; state_hash digests == main |
| G-M2-6 | cost, P1 d/h 24 | ms/step <= 1.15x upwind's, same averaging window |
| G-M2-7 (new) | transverse-slab oracle on the device advector: x-faces `U`, y-faces `v(x)`, `a_x = 0.2`, C a y,z-uniform slab `[0.3, 0.7)`, `ny = nz = 4`, periodic, one period, N in {32, 64, 128}, ratios {1, 1e3, 1e4}; with `v = 0.1 v0` the WY Courant check passes, and the field `(U, v(x), 0)` reduces the step exactly to the 1D sweep (the y-sweep sees y-uniform data: equal fluxes, zero dilation, zero `dv`) | L1 within 1 % of `apriori1d.py` run with the same `v`, `a`, N, ratio, wisp eps, at every point; device order 64→128 >= 1.6 at 1e3; `L1(1e4)/L1(1e3)` in [0.99, 1.01]; `capped == 0`; positive control: `muscl_mixed` at 1e4, N = 128 has L1 >= 3x `muscl_phase`'s |
| G-M2-8 (new) | serial vs MPI, `muscl_phase`, moving interface, non-uniform velocity, np 1/2/4 cutting the sweep axes | final velocity and colour `max|diff| == 0.0` |
| G-M2-9 (new) | single phase (C ≡ 1 and C ≡ 0), any velocity, ratio 1e3 | `capped == 0` every step (§4.6) |
| G-M2-10 (new) | device synchronization, one VoF-momentum step on CUDA, `muscl_phase` vs `upwind`, counted with the `device_sync_budget` instrument | global fences and View allocations per step equal |
| G-D (diagnosis) | P1 d/h 16, 24, `muscl_mixed_capped` | CONFIRMED if `Re16 < 38.99` and `|e24| < |e16|`; REFUTED if `Re16 >= 39.50`; PARTIAL otherwise. REFUTED does not fail the scheme; it invalidates §2 and goes to the architect before any default decision |

## 10. Risks and open questions (each with a default)

- **F-1 (fact).** Ghost validity of `ud`, `ep`, `od`, `kd` at the §4.8 indices on the block-container
  and MPI paths (the solver path is checked above). *Default:* G-M2-8 and a block-container identity
  run detect it; if a path does not fill the layer, extend that path's fill by one layer (no change
  for other schemes) and say so in the commit.
- **F-2 (fact).** Whether 3D confirms the diagnosis (G-D). *Default:* proceed with `muscl_phase`
  whatever G-D says; G-D only gates §2 and the default conversation. **Answered: REFUTED**
  (2026-10-29, §2); the wrong-limit mechanism of `'muscl_mixed'` is OPEN.
- **F-3 (fact).** Re averaging noise against a small `e32`. *Default:* NOT JUDGED as in G-M2-3,
  escalate; no extra runs invented by the implementer.
- **F-4 (fact).** Time error may cap the observed order: backward Euler is the settled integrator,
  and at fixed CFL `dt ~ h` (`h^1.5` where the capillary step binds). *Default:* the one conditional
  half-CFL run of WO-4; report and stop. Time integration is out of scope.
- **F-5 (fact).** MC vs unlimited constant at P1. If G-M2-3 misses on magnitude only, the limiter is
  NOT changed by the implementer. *Default:* report MISS; re-consult.
- **F-6 (fact).** The energy twin's predicted `(rho c_p)` gain (§6.3). *Default:* untouched; a
  separate brief, using this construction.
- **F-7 (fact).** Upwind's own dilation term stays non-convex (§4.6 item 3). *Default:* unchanged;
  it is the robustness baseline the brief names.
- **U-1 (user preference).** Public name and tier. *Default:* diagnostics tier,
  `set_vof_momentum_scheme` with the four names of §7.
- **U-2 (user preference).** Flip the default to `'muscl_phase'`. *Default:* stays `'upwind'`;
  recommend the flip if G-M2-1..9 PASS, as a recorded user decision. **Decided** (USER 2026-10-29:
  flip only if all of its gates pass); flipped 2026-10-31 (§12).
- **U-3 (user preference).** Retire `'muscl_mixed'`. *Default:* keep while opt-in (it is G-D's control);
  deprecate under NAMING §0 with the default flip.

## 11. Register entry (text for `suite/docs/DECISIONS.md`)

As landed 2026-10-31 (the design-time text claimed the §2 mechanism; G-D refuted it, so it is gone):

- **The VoF momentum flux defaults to `'muscl_phase'`: per-phase interface-biased MC slopes of the
  one CV velocity plus a mass-aware cap; NEVER a slope of the mixed control-volume velocity without
  the cap.** Second order, a per-sweep discrete maximum principle at any density ratio, the
  uniform-velocity identity bit-exact; P1 (ratio ~1e3) from rest at t60, d/h 16/24/32/48: 37.443 /
  38.001 / 38.289 / 38.523 against 38.60 (pairwise orders 1.62 / 2.28 / 3.44); P2 d/h 48 within
  0.03 % of upwind; 0.70x upwind's cost at P1 d/h 24. **Rejected:** `'upwind'` as the default (first
  order: P1 d/h 16 35.33, d/h 24 36.41); `'muscl_mixed'` (MinMod of the mixed velocity; converges to
  a wrong limit, P1 +3.0/+3.1 % at d/h 16/24 — mechanism OPEN: the cap-only ablation G-D, 39.659 at
  d/h 16, refuted the emptying-CV explanation); MC of the mixed velocity without the cap (1D:
  diverges); Basilisk-faithful two momenta + unlimited slopes (no maximum principle in 3D, +9
  exchanges/step); Zalesak FCT on u (two passes, extra halo); exact donor mass from widened PLIC
  ghosts (+9 % reconstruction or +9 exchanges/step).
  <sub>USER 2026-10-29 (flip iff all gates pass); flow doc/vof_momentum_second_order.md §9, §12;
  bubble_column_perf/validation/highratio/DECISIONS.log</sub>

## 12. Measured results and the default (2026-10-31)

All gates of §9, pre-registered, judged on flow `mom2` (`9e70a30`, the scheme commits before the
rebase onto main `4b59f52`) with `validation/highratio/pec/run_hr_scheme.py`, local RTX 5080.

| gate | verdict | measured |
|---|---|---|
| G-M2-1 | PASS | `max|u - U| == 0.0` bitwise, all four schemes, ratios 1..1e4, slab / tilted / sphere (ctest `vof_momentum`, `vof_cutcell`) |
| G-M2-2 | PASS | ratio 1e4, 50 steps (ctest `vof_momentum` K7, host / CUDA): `'muscl_phase'` 8.9e-16 / 1.0e-15, `'muscl_mixed_capped'` 7.8e-16 / 8.9e-16, upwind 7.8e-16 / 8.9e-16 (`'muscl_mixed'` 1.3e-7 / 6.5e-8, printed) |
| G-M2-3 | PASS | P1 from rest, t60: d/h 16 / 24 / 32 / 48 = 37.443 / 38.001 / 38.289 / 38.523; `e = Re - 38.60` one sign; pairwise orders 1.62 / 2.28 / 3.44; `|e24|` 1.55 %; d/h 48 = Basilisk L8 (38.517) + 0.02 % |
| G-M2-4 | PASS | P2 d/h 48 (t30 -> 40 from the d/h 24 dump): Re 8.0701 vs upwind 8.0674 (0.03 %); `floored` 0, `capped` 0; dt_min 1.6347e-5 s vs upwind's sampled minimum 1.7745e-5 s, an upper bound on its per-step minimum: ratio >= 0.92 |
| G-M2-5 | PASS | default `'upwind'` unchanged: battery green host + CUDA, state_hash digests == main (re-run after the rebase: host 278/278, 805e3e99a93333c7; CUDA 278/278 with three GPU-OOM serial reruns, e7edbb471ea7a742) |
| G-M2-6 | PASS | P1 d/h 24: 56.65 vs 81.41 ms/step (0.70x; shared GPU, noisy) |
| G-M2-7..9 | PASS | ctests `vof_momentum_slab` (oracle within 1 %, order, ratio independence, positive control), `vof_momentum_mpi` (np 1/2/4 bitwise), single phase `capped == 0` |
| G-M2-10 | PASS | `vofmom_sync_budget`: fences and View allocations per step equal to upwind's |
| G-D | REFUTED | P1 d/h 16 `'muscl_mixed_capped'` 39.659 (cap 0.91 faces/step); §2 |

For comparison at P1: `'upwind'` d/h 16 35.327, d/h 24 36.406 (first order); `'muscl_mixed'` d/h 16
39.740, d/h 24 39.79 (wrong limit). The observed P1 order of `'muscl_phase'` (1–1.6 over the free
three-point fits) is attributed in the campaign to the interface viscous-stress error, which is
outside this note (`doc/vof_sharp_viscous_stress.md`, branch `sharpvisc`); the averaging-bracket
spread of the Basilisk reference (L7 arithmetic 38.569 / geometric 39.727 / harmonic 41.174) puts
the P1 reference in a 38.5–39.2 band.

**The default flip** (flow `eeb77d0`, on `mom2land` = `mom2` rebased onto main `4b59f52`):
`vof::MomentumConsistentAdvector::momentumScheme = MomentumScheme::MusclPhase`. No state_hash
digest moves (14/14 lines equal on host and CUDA): no state_hash case calls `enable_vof_momentum`. `'upwind'` stays
selectable by `diagnostics.set_vof_momentum_scheme('upwind')`; `set_vof_momentum_muscl(on)` stays the
former spelling (`'muscl_mixed'` / `'upwind'`).
