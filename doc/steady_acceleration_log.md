# Anderson acceleration of steady marches — measurement log (append-only)

Every number of the `doc/steady_acceleration.md` work orders, with the command that produced it.
Never edit an entry; correct it with a later one. The current position is in
`doc/steady_acceleration_STATE.md`.

---

## 2026-10-02 — WO-1: the Python oracle, first measurements (literal §4.1 constants)

**Build.** Worktree `flow-anderson`, branch `anderson` at `f6b89fe` (= flow main), host tree
`build_omp` (`-DCMAKE_PREFIX_PATH=../extern/install/host-openmp -DCMAKE_BUILD_TYPE=Release`, sibling
`../core`), `OMP_NUM_THREADS=8 OMP_PROC_BIND=false`. Oracle: `tests/study/anderson_oracle.py`
(§3, §4 and §7 of the note in NumPy over `diagnostics.field_view`; state u, v, w, p; c_P from
`unit_scales` = 1/7 on the §11 case; gauged).

**Commands.** (`O="python tests/study/anderson_oracle.py"`, `PYTHONPATH=$PWD/build_omp`;
`--window 0` = the plain march, `march_to_steady(accelerate=False)`)

    $O sphere --scheme {collocated,staggered} --N {16,24} --settings production --window 0 3 5 8 --label s11
    $O sphere --scheme staggered --N 16 --mu 0.05 --beta 0.5 --advection koren --settings production --window 0 3 5 8 --label re10
    $O sphere --scheme staggered --N 16 --mu 0.0158 --dt 1.234e-2 --advection sou --settings production --window 0 3 5 8 --label g7a
    $O sphere --scheme {collocated,staggered} --N 16 --settings tight --window 0 3 5 8 --label g1

**Case identification.**
- §11 = `tests/study/study_avg_velocity_spheres.py`'s case (phi 0.125, rho = mu = F = 1, nu dt/h^2 = 6,
  collocated = AUTO 'ghost'). The oracle's plain march reproduces the study table exactly:
  collocated N = 16 **395 steps, K = 4.2663**, staggered N = 16 **75 steps, K = 4.2120**.
- G7a's advection scheme is not stated in the note; the note's own fact "plain NaN at step ≈ 435"
  identifies it as the default **SOU**: plain march `diverged` (non-finite monitor) at the block
  ending step **440** with SOU, at **710** with Koren.
- Re ≈ 10 as §1.1: Koren, dt = 0.5 h^2 rho/mu = 0.0390625 (D = 0.5), explicit advection (default).

### G1 (tight: PCG(400, 1e-12), velocity residual tolerance 1e-12, rtol 1e-10, max_steps 20000), N = 16

| scheme | m | steps plain / acc | K_plain | K_acc | \|K_acc/K_plain − 1\| | restarts | unstable |
|---|---|---|---|---|---|---|---|
| collocated | 3 | 3520 / 185 | 4.266648173242035 | 4.266648175354302 | 4.95e-10 | 0 | no |
| collocated | 5 | 3520 / 168 | 4.266648173242035 | 4.266648175351929 | 4.95e-10 | 0 | no |
| collocated | 8 | 3520 / 138 | 4.266648173242035 | 4.266648175353933 | 4.95e-10 | 0 | no |
| staggered | 3 | 295 / 86 | 4.211998994555005 | 4.211998994631220 | 1.81e-11 | 0 | no |
| staggered | 5 | 295 / 78 | 4.211998994555005 | 4.211998994630679 | 1.80e-11 | 0 | no |
| staggered | 8 | 295 / 71 | 4.211998994555005 | 4.211998994631779 | 1.82e-11 | 0 | no |

**G1 PASSES** (≤ 1e-8) at every window. No Ritz evaluation happens in certification here (the
residual is below the 1e-10 noise floor by then); max Ritz radius in phase A 0.99618 (collocated,
the checkerboard rate 0.9962 of `collocated_invisible_subspace.md` §11) and ≤ 0.9695 (staggered).

### Production (study settings: PCG(200, 1e-8), 200 velocity sweeps, rtol 1e-4, max_steps 5000)

Columns: steps to `converged=True`; ratio = steps_plain / steps_acc; wall = oracle wall ratio (host
NumPy, indicative only); K vs the case's plain K and (N = 16 §11) vs K_G1; restarts; the first step
at which status became "unstable" and the phase; max Ritz radius in phase A / in certification;
oldest-column drops; phase sequence (a = accelerate, c = certify, p = plain tail; (uns) = status
unstable, (dis) = disabled).

| case | sch | N | m | conv | steps | acc | ratio | wall | K | vs plain | vs K_G1 | rst | unstable | Ritz A | Ritz C | drops | phases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| §11 | coll | 16 | 0 | T | 395 | 0 | – | – | 4.26633370 | – | 7.4e-5 | – | – | – | – | – | |
| §11 | coll | 16 | 3 | T | 125 | 70 | 3.16 | 3.13 | 4.26664818 | 7.4e-5 | 1.4e-9 | 0 | – | 0.996144 | 2.39 | 0 | a61 c30 a9 c25 |
| §11 | coll | 16 | 5 | T | 112 | 57 | **3.53** | 2.95 | 4.26664955 | 7.4e-5 | 3.2e-7 | 0 | **76 / certify** | 0.996177 | 12.6 | 0 | a57 c30(uns) p25 |
| §11 | coll | 16 | 8 | T | 163 | 53 | 2.42 | 2.15 | 4.26664623 | 7.3e-5 | 4.6e-7 | 0 | **73 / certify** | 0.996174 | 5.12 | 8 | a53 c30(uns) p80 |
| §11 | coll | 24 | 0 | T | 90 | 0 | – | – | 4.27913128 | – | – | – | – | – | – | – | |
| §11 | coll | 24 | 3 | T | 65 | 40 | 1.38 | 1.03 | 4.27913487 | 8.4e-7 | – | 0 | – | 0.990390 | 2.05 | 0 | a40 c25 |
| §11 | coll | 24 | 5 | T | **101** | 46 | **0.89** | 0.58 | 4.27913698 | 1.3e-6 | – | 0 | – | 0.999999 | 1.0001 | 5 | a38 c30 a8 c25 |
| §11 | coll | 24 | 8 | T | 63 | 38 | 1.43 | 0.74 | 4.27913477 | 8.2e-7 | – | 0 | **52 / certify** | 0.990983 | 2.21 | 0 | a38 c25(uns) |
| §11 | stag | 16 | 0 | T | 75 | 0 | – | – | 4.21199423 | – | 1.1e-6 | – | – | – | – | – | |
| §11 | stag | 16 | 3 | T | 49 | 24 | 1.53 | 1.42 | 4.21199895 | 1.1e-6 | 1.0e-8 | 0 | – | 0.952289 | 1.50 | 0 | a24 c25 |
| §11 | stag | 16 | 5 | T | **77** | 22 | **0.97** | 0.92 | 4.21199894 | 1.1e-6 | 1.4e-8 | 0 | **41 / certify** | 0.949240 | 3.70 | 0 | a22 c30(uns) p25 |
| §11 | stag | 16 | 8 | T | 47 | 22 | 1.60 | 1.07 | 4.21199868 | 1.1e-6 | 7.4e-8 | 0 | **44 / certify** | 0.953059 | 56.2 | 7 | a22 c25(uns) |
| §11 | stag | 24 | 0 | T | 135 | 0 | – | – | 4.25727877 | – | – | – | – | – | – | – | |
| §11 | stag | 24 | 3 | T | 52 | 27 | 2.60 | 1.49 | 4.25731078 | 7.5e-6 | – | 0 | – | 0.974725 | 0.9957 | 0 | a27 c25 |
| §11 | stag | 24 | 5 | T | 58 | 28 | 2.33 | 1.09 | 4.25731479 | 8.5e-6 | – | 0 | – | 0.976653 | 1.00003 | 2 | a28 c30 |
| §11 | stag | 24 | 8 | T | 114 | 29 | 1.18 | 0.48 | 4.25732745 | 1.1e-5 | – | 0 | – | 1.0000004 | 1.000002 | 34 | a24 c30 a3 c30 a2 c25 |
| Re≈10 | stag | 16 | 0 | T | 345 | 0 | – | – | 4.36428419 | – | – | – | – | – | – | – | |
| Re≈10 | stag | 16 | 3 | T | 166 | 141 | 2.08 | 1.74 | 4.36427354 | 2.4e-6 | – | 0 | – | 0.963304 | 0.966 | 0 | a141 c25 |
| Re≈10 | stag | 16 | 5 | T | 114 | 89 | 3.03 | 1.91 | 4.36424721 | 8.5e-6 | – | 0 | – | 0.965579 | 1.71 | 0 | a89 c25 |
| Re≈10 | stag | 16 | 8 | T | 109 | 84 | 3.17 | 1.57 | 4.36424852 | 8.2e-6 | – | 0 | – | 0.966149 | 2.24 | 0 | a84 c25 |
| G7a | stag | 16 | 0 | **F** | 440 | 0 | – | – | – (diverged) | – | – | – | – | – | – | – | |
| G7a | stag | 16 | 3 | **F** | 322 | 57 | – | – | – (diverged) | – | – | 0 | – | 0.993804 | 0.9959 | 0 | a57 c30 p235(dis) |
| G7a | stag | 16 | 5 | **F** | 255 | 115 | – | – | – (diverged) | – | – | 0 | – | 0.995537 | 0.9925 | 0 | a115 c30 p110(dis) |
| G7a | stag | 16 | 8 | **F** | 223 | 133 | – | – | – (diverged) | – | – | 0 | – | 0.995518 | 0.9928 | 0 | a133 c30 p60(dis) |

(The certification-phase Ritz radius is computed on windows that hold plain-step columns; the
"unstable" status there ends acceleration but not the march, because §7 tests "unstable" only in
phase A — the outer loop then sees `status != "active"` and finishes with the plain instrument.)

**Gates of WO-1:**
- **G1 (tight, N = 16): PASS** — 4.95e-10 collocated, 1.8e-11 staggered, every m.
- **G2 ≥ 3× on collocated N = 16 (m = 5): 395 / 112 = 3.53 — PASS** as measured, but the 112 includes
  a false "unstable" (below) and a certification budget that ran out; the §7 estimate was ≈ 80.
  K vs K_G1 3.2e-7 (≤ 1e-4).
- **G7a: PASS** — `converged=False` for accelerate True (m = 3, 5, 8) and False; no K. The
  accelerated runs end by the certification fallback (budget exhausted with phase A stagnated →
  `disable()` → plain tail → diverged), not by the Ritz guard (max 0.9955 in phase A).
- Restart storms: none — 0 restarts in every run.
- **False "unstable": YES — STOP condition of WO-1** (§9: "a false 'unstable' … stop and report").

**The false "unstable" (5 of 12 production §11 runs: collocated N = 16 m = 5, 8; collocated N = 24
m = 8; staggered N = 16 m = 5, 8; none at m = 3, none on Re≈10 / G7a, none in the tight runs).**
It fires only in phase B, on `acc.step(False)` calls. §4.3 step 7 evaluates the guard on every
active call, and a plain call records history like an accelerated one, so after ≈ m certification
steps the window holds plain-march differences only. Those are nearly collinear (the plain march
moves along its slowest mode): the Jacobi-scaled condition of XX falls from 1e-4–1e-5 in phase A to
**1.6e-11 – 4e-11** at the alarm, which is above `kCondMin` = 1e-12, so the truncated pseudo-inverse
keeps the near-null direction and M = XX⁺·XR is dominated by the inexact-solve noise (production
PCG rtol 1e-8 against residuals ~2e-7). Ritz radii in certification: up to 12.6 (coll N16 m5), 56.2
(stag N16 m8); three consecutive > 1.001 → "unstable".
Cross-checks (scratch `ritz_check.py`, collocated N = 16, m = 5, 57 accelerated + 23 plain calls):
the cached Gram blocks equal a direct W-metric recomputation from the stored vectors to
≤ 4.5e-16 relative; the Gelfand radius equals `max|eig(T)|` (numpy) to ≤ 1e-5; so the value is the
design's, not an oracle defect.

**A second deviation from the note's premises (not a stop condition, recorded for the architect).**
§7 expects the certification to "pass at its first eligible blocks" (20–25 steps). It ran out of
its 30-step budget in 9 of the 24 converged production accelerated runs (phases `c30` above). On
collocated N = 16 m = 5 the monitor after leaving phase A (step 57, u = 0.040087108771) first falls
for 4 steps, then rises with GROWING block increments: block ratios R ≈ 1.8, 1.36, 1.07, 1.01 at
the blocks ending 72, 77, 82, 87, so `0 < R < 1` fails until the fast modes the accelerated iterate
excited have decayed under the slow 0.996 tail. Consequences in the table: steps_acc > steps_plain
on collocated N = 24 m = 5 (101 vs 90) and staggered N = 16 m = 5 (77 vs 75) — the G2 condition
"steps_acc ≤ steps_plain on every case" fails at the default window on two §11 cases.

**Window rule (pre-registered: m = 3 within 10 % of m = 5 in steps on every measured case → default
3).** steps m=3 / m=5: coll N16 125/112 (+11.6 %), coll N24 65/101 (−36 %), stag N16 49/77 (−36 %),
stag N24 52/58 (−10.3 %), Re≈10 166/114 (+46 %), G7a — (not converged, excluded). **Not met**
(under either reading of "within": |s3/s5 − 1| ≤ 0.1, or s3 ≤ 1.1 s5 — coll N16 and Re≈10 fail
both). Default window stays **5**. Provisional: the step counts carry the false-"unstable" and
certification-budget paths above, so the rule should be re-applied after the architect's answer.

**Extra, unplanned:** the first `g7a` invocation omitted `--advection` and so ran STOKES at
mu = 0.0158, dt = 1.234e-2 (nu dt/h^2 = 0.050): plain 2795 steps, K 4.21235538 (8.5e-5 from K_G1);
accelerated m = 3/5/8: 149/115/95 steps (18.8×/24.3×/29.4×), K within 3.2e-7/6.1e-6/3.1e-5 of
K_G1. Not a WO-1 case; kept because it shows the small-D regime.

---

## 2026-10-02 — WO-1: the dense bed (§10 Q2 default), production, early indication for Q1

**Configuration (Q2 default as ruled).** The only phi ≈ 0.6 configuration in
`~/Codes/peclet-study-A1-drag-audit/scripts/` is the phi-scan of `random_arrays.py` (`--phi … 0.6
--dpdx 16`, `--n-spheres` default 64, generator default `grow`, `--seed0` default 0); its lowest
(only) resolution D/dx 16 gives N = `mg_friendly_N(16 · L/D)` = **64**. Arrangement from
`common.random_arrangement(0.6, n_spheres=64, seed=0)` (peclet.dem from the suite venv, separate
process): phi 0.6000000, L = 3.82246 (L/D 3.822), max overlap 6.6e-4 D. Saved to the WO-1 scratch as
`bed_phi0.6_n64_s0.npz` (not committed; regenerated by that call). Solver as A1's
`common.make_solver` 'release' profile at beta 6 (the oracle's `build_solver`: rho = mu = F = 1,
dt = 6 h^2, MG levels log2 N − 1, PCG(200, 1e-8), 200 velocity sweeps); K = A1's
f V / (3 pi mu D U N_p). Host-feasible: 0.08 s/step staggered plain (325 steps in 27 s).

    $O bed --arrangement bed_phi0.6_n64_s0.npz --scheme {staggered,collocated} --N 64 --settings production --window 0 5 3 8 --label bed06

| sch | m | conv | steps | acc | ratio | oracle wall ratio | K | vs plain | rst | unstable | Ritz A | Ritz C | drops | phases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stag | 0 | T | 325 | 0 | – | – | 98.91086496 | – | – | – | – | – | – | |
| stag | 3 | T | 154 | 129 | 2.11 | 0.67 | 98.91573953 | 4.9e-5 | 0 | – | 0.99795 | 0.9980 | 0 | a129 c25 |
| stag | 5 | T | 257 | 202 | **1.26** | 0.33 | 98.91569390 | 4.9e-5 | 0 | – | 0.99979 | 1.00000 | 3 | a202 c30 p25(dis) |
| stag | 8 | T | 334 | 309 | 0.97 | 0.16 | 98.91570065 | 4.9e-5 | 0 | – | 0.99990 | 1.00000 | 14 | a309 c25 |
| coll | 0 | T | 190 | 0 | – | – | 101.41495514 | – | – | – | – | – | – | |
| coll | 3 | T | 193 | 168 | 0.98 | 0.68 | 101.41632817 | 1.4e-5 | 0 | – | 0.99979 | 2.88 | 0 | a168 c25 |
| coll | 5 | T | 178 | 153 | **1.07** | 0.60 | 101.41632830 | 1.4e-5 | 0 | – | 0.99999 | 1.00000 | 0 | a153 c25 |
| coll | 8 | T | 213 | 183 | 0.89 | 0.43 | 101.41632838 | 1.4e-5 | 0 | – | 1.000001 | 1.000004 | 0 | a183 c30 |

**Early indication for Q1 (D11's rule is applied at G2 on the C++ build, not here):** step ratio at
the default window **1.26× staggered, 1.07× collocated** — below the 1.5× wall threshold already in
steps. No false "unstable" on the bed; Ritz A ≤ 1.000001 (G7c bound 1.0005 holds).

**Why the gain is small (diagnostic, scratch `bed_floor.py`, staggered, m = 5, 120 calls).** Phase A
never reaches its target (1 − slow_rate)·rtol = 3e-7: the W-residual falls slowly (8.3e-5 / 1.4e-5 /
4.4e-6 at accelerated step 40 / 80 / 120; min 1.39e-6 in the sweep run, which then left phase A on
the stagnation rule). It is **not an inner-solve floor**: with PCG rtol 1e-12 and velocity residual
tolerance 1e-12 the residual sequence is identical to 3 digits (4.36e-6 at 120 either way). The
plain march's W-residual is still **7.85e-5 at step 300**, yet its monitor certifies at step 325: on
the dense bed the W-residual and the ⟨u_x⟩ instrument decouple by about two decades, so the phase-A
target (tied to the W-residual) demands far more than the certificate (tied to the monitor) needs.
A premise of §7 ("the target … is the residual at which the slowest mode … can carry at most rtol of
remaining change") does not hold on this bed: the slow content of the W-residual is not in ⟨u_x⟩.
