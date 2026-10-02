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


---

## 2026-10-02 — Revision 1 (architect pass on `doc/steady_acceleration_brief2.md`)

**Build and conventions.** Same tree as WO-1: `build_omp` (host-openmp, Release, sibling `../core`),
suite venv, `OMP_PROC_BIND=false`. Thread counts are stated per block. The box is shared, so wall
times are indicative only; step counts and pressure iterations are not affected by load.
`O="python tests/study/anderson_oracle.py"`, `P="python tests/study/anderson_rev1_probes.py"`,
`BED=<the WO-1 arrangement bed_phi0.6_n64_s0.npz>`; the runs used `/tmp/claude-1003/-home-frankp-Codes-suite/0355d123-388a-45e8-b425-3b14d54c9d99/scratchpad/wo1/bed_phi0.6_n64_s0.npz`.
- The probes were first run as scratch scripts; `tests/study/anderson_rev1_probes.py` is their
  committed form. Checked against the scratch numbers: `phase-a`, collocated N = 16, m = 5, V gives
  35 / 36 / 39 / 52 (row below); `certify`, staggered N = 24, m = 8 gives 74 steps; `slowmode` on
  the staggered bed gives an identical per-step W / velocity / pressure residual to step 60.
- The oracle's `--rev 0` reproduces WO-1: staggered N = 16, m = 5: 77 steps, K 4.2119989364,
  "unstable" at 41 in certify.

### R-1. The slow mode of the dense bed (plain march, 1500 steps, 8 threads)

    $P slowmode --case bed --arrangement $BED --scheme staggered  --N 64 --steps 1500 --save 300,600,1000,1500
    $P slowmode --case bed --arrangement $BED --scheme collocated --N 64 --steps 1500 --save 300,600,1000,1500

**Staggered:**
- W / velocity / pressure residual: step 100 3.49e-4 / 1.56e-5 / 3.48e-4; 200 1.35e-4 / 3.31e-6;
  300 7.85e-5 / 1.33e-6; 325 7.07e-5 / 1.12e-6; 600 3.19e-5 / 3.10e-7; 1000 1.64e-5 / 1.20e-7;
  1500 9.43e-6 / 5.98e-8.
- Per-step W rate: 0.9906 (100–200), 0.9946 (200–300), 0.9962 (300–400), 0.9974 (400–600),
  0.9981 (600–800), 0.9985 (800–1000), 0.9988 (1000–1200), 0.9990 (1200–1500).
- Plain ⟨u_x⟩ error against the accelerated reference 9.36075772e-4: 5.6e-5 at 300, 1.7e-5 at 600,
  7.3e-6 at 1000, 3.7e-6 at 1500. The plain march reaches 1e-4 at step 217, 3e-5 at 431, 1e-5 at 828.
- Aperture graph: 1,086 fluid components at aperture > 0, a main component of 103,767 of 104,935
  fluid cells; 1,094 / 1,118 / 1,142 / 1,193 components at > 0.01 / 0.05 / 0.1 / 0.2.
- Off-main share of the (fluid-mean-free) pressure residual energy at r300 / r600 / r1000 / r1500:
  0.584 / 0.644 / 0.626 / 0.592 at > 0, and 0.687 / 0.766 / 0.828 / 0.848 at > 0.2. The
  per-component-constant share is 0.565 / 0.614 / 0.601 / 0.579 at > 0.
- Pressure share of W²: 0.99971 / 0.99991 / 0.99995 / 0.99996.
- 90 % of the pressure energy sits in 105 / 66 / 50 / 41 cells; 90 % of the velocity energy in
  7,897 / 1,091 / 119 / 37 entries.
- cos between successive late residuals (W): 0.869 (300, 600), 0.928 (600, 1000), 0.956 (1000, 1500).

**Collocated:**
- The ghost scheme prints "205 fluid components; decoupled 216 pocket cells".
- W / velocity residual: 300 5.00e-5 / 7.65e-7; 1500 2.80e-6 / 6.62e-9.
- Off-main share at > 0.2: 0.743 / 0.891 / 0.984 / 0.999. Pressure share of W² ≥ 0.9998.
- Plain ⟨u_x⟩ error against 9.12994916e-4: 1.4e-5 at its certification (190).

### R-2. Velocity vs W metric, unconditional phase A (`acc.step(True)` every call)

**What was run:** the WO-1 oracle (`16393b9`) with c_P set to 0 for "V" (pressure out of the
metric, roles otherwise unchanged) and revision 0's guard. Production settings; 6 threads for the
bed and 4 for the spheres.
- Reference monitor: the bed values above; collocated N = 16, 4.008712493775e-2 (the m = 5 W run's
  last value, = K_G1 to 1e-9); the other cases use their own m = 5 W run's last value.
- Equivalent today: `$P phase-a --case ... --window m --metric V|W --steps 200|250|300 --ref <ref>`.
- "Steps run" < the requested count means the run stopped on revision 0's guard at the noise floor;
  see R-3.

| case | m | metric | monitor ≤ 3e-5 from | ≤ 1e-5 from | velocity res ≤ 3e-7 at | W res ≤ 3e-7 at | monitor err at the velocity handover | steps run |
|---|---|---|---|---|---|---|---|---|
| s11N16_collocated | 3 | V | 37 | 38 | 40 | 51 | 5.6e-07 | 300 |
| s11N16_collocated | 3 | W | 33 | 45 | 41 | 61 | 1.2e-05 | 300 |
| s11N16_collocated | 5 | V | 35 | 36 | 39 | 52 | 1.4e-06 | 300 |
| s11N16_collocated | 5 | W | 31 | 43 | 48 | 57 | 3.2e-06 | 300 |
| s11N16_collocated | 8 | V | 34 | 36 | 38 | 49 | 3.8e-06 | 300 |
| s11N16_collocated | 8 | W | 30 | 34 | 39 | 53 | 4.0e-06 | 300 |
| s11N16_staggered | 3 | V | 12 | 13 | 22 | 25 | 3.7e-07 | 200 |
| s11N16_staggered | 3 | W | 12 | 15 | 23 | 24 | 2.2e-07 | 200 |
| s11N16_staggered | 5 | V | 11 | 15 | 21 | 23 | 4.6e-07 | 200 |
| s11N16_staggered | 5 | W | 13 | 14 | 22 | 22 | 2.0e-08 | 200 |
| s11N16_staggered | 8 | V | 10 | 15 | 21 | 23 | 3.4e-07 | 200 |
| s11N16_staggered | 8 | W | 12 | 13 | 21 | 22 | 6.5e-07 | 200 |
| s11N24_collocated | 3 | V | 11 | 12 | 24 | 43 | 1.3e-06 | 300 |
| s11N24_collocated | 3 | W | 14 | 15 | 25 | 40 | 7.4e-07 | 300 |
| s11N24_collocated | 5 | V | 9 | 14 | 23 | 42 | 2.1e-06 | 300 |
| s11N24_collocated | 5 | W | 13 | 15 | 24 | 38 | 1.8e-06 | 300 |
| s11N24_collocated | 8 | V | 9 | 12 | 23 | 40 | 1.6e-06 | 300 |
| s11N24_collocated | 8 | W | 9 | 12 | 24 | 38 | 7.7e-07 | 300 |
| s11N24_staggered | 3 | V | 18 | 27 | 28 | 29 | 6.8e-06 | 44 |
| s11N24_staggered | 3 | W | 21 | 25 | 27 | 27 | 5.5e-06 | 200 |
| s11N24_staggered | 5 | V | 18 | 23 | 25 | 26 | 6.1e-06 | 200 |
| s11N24_staggered | 5 | W | 20 | 24 | 26 | 28 | 3.7e-06 | 130 |
| s11N24_staggered | 8 | V | 18 | 22 | 24 | 25 | 5.0e-06 | 200 |
| s11N24_staggered | 8 | W | 20 | 22 | 23 | 24 | 6.3e-06 | 98 |
| re10_staggered | 3 | V | 118 | 133 | 136 | 136 | 7.7e-06 | 300 |
| re10_staggered | 3 | W | 121 | 137 | 140 | 141 | 7.7e-06 | 300 |
| re10_staggered | 5 | V | 92 | 102 | 106 | 106 | 7.0e-06 | 162 |
| re10_staggered | 5 | W | 73 | 85 | 89 | 89 | 7.2e-06 | 300 |
| re10_staggered | 8 | V | 64 | 74 | 78 | 78 | 6.1e-06 | 300 |
| re10_staggered | 8 | W | 70 | 80 | 84 | 84 | 6.5e-06 | 300 |
| bed_staggered | 3 | V | 49 | 67 | 86 | — | 4.4e-06 | 250 |
| bed_staggered | 3 | W | 31 | 39 | 138 | — | 1.0e-07 | 250 |
| bed_staggered | 5 | V | 51 | 76 | 68 | — | 1.4e-05 | 250 |
| bed_staggered | 5 | W | 37 | 52 | 104 | — | 1.1e-06 | 250 |
| bed_staggered | 8 | V | 53 | 66 | 65 | — | 1.1e-05 | 196 |
| bed_staggered | 8 | W | 43 | 55 | 98 | — | 1.5e-06 | 250 |
| bed_collocated | 3 | V | 29 | 32 | 74 | — | 1.8e-07 | 250 |
| bed_collocated | 3 | W | 33 | 38 | 109 | — | 1.4e-08 | 250 |
| bed_collocated | 5 | V | 29 | 36 | 58 | — | 9.0e-07 | 126 |
| bed_collocated | 5 | W | 23 | 33 | 74 | — | 6.9e-08 | 250 |
| bed_collocated | 8 | V | 27 | 34 | 57 | — | 9.9e-07 | 250 |
| bed_collocated | 8 | W | 22 | 32 | 72 | — | 1.3e-07 | 250 |

### R-3. Ritz readings in the R-2 runs (revision-0 guard: every call, floor 1e-10)

| run | first reading > 1.001 (step, residual) | max reading at residual ≥ 1e-6 | ≥ 1e-7 | ≥ 1e-8 |
|---|---|---|---|---|
| bed_collocated_m3_V | none | 0.981756 | 0.986085 | 0.997166 |
| bed_collocated_m3_W | none | 1.000002 | 1.000002 | 1.000002 |
| bed_collocated_m5_V | 125, 5.4e-09 | 0.987012 | 0.994453 | 0.997809 |
| bed_collocated_m5_W | none | 1.000002 | 1.000002 | 1.000002 |
| bed_collocated_m8_V | none | 0.986785 | 0.994750 | 0.997828 |
| bed_collocated_m8_W | none | 1.000002 | 1.000002 | 1.000002 |
| bed_staggered_m3_V | none | 0.990389 | 0.994510 | 0.998976 |
| bed_staggered_m3_W | none | 0.999336 | 0.999336 | 0.999336 |
| bed_staggered_m5_V | 212, 2.9e-09 | 0.992792 | 0.998293 | 0.999030 |
| bed_staggered_m5_W | none | 0.999820 | 0.999840 | 0.999840 |
| bed_staggered_m8_V | 190, 2.9e-09 | 0.993225 | 0.998453 | 0.999149 |
| bed_staggered_m8_W | none | 0.999854 | 0.999872 | 0.999872 |
| re10_staggered_m3_V | 193, 3.8e-09 | 0.960855 | 0.960855 | 0.969065 |
| re10_staggered_m3_W | none | 0.963304 | 0.963304 | 0.967268 |
| re10_staggered_m5_V | 155, 4.2e-09 | 0.965457 | 0.965457 | 0.965457 |
| re10_staggered_m5_W | none | 0.965579 | 0.965579 | 0.965579 |
| re10_staggered_m8_V | none | 0.965896 | 0.965896 | 0.965908 |
| re10_staggered_m8_W | none | 0.966149 | 0.966149 | 0.967344 |
| s11N16_collocated_m3_V | 80, 3.4e-10 | 0.996703 | 0.996703 | 0.996703 |
| s11N16_collocated_m3_W | none | 0.996083 | 0.996083 | 0.996097 |
| s11N16_collocated_m5_V | none | 0.997303 | 0.997303 | 0.997303 |
| s11N16_collocated_m5_W | none | 0.996177 | 0.996177 | 0.996177 |
| s11N16_collocated_m8_V | none | 0.996803 | 0.997464 | 0.997464 |
| s11N16_collocated_m8_W | none | 0.996174 | 0.996180 | 0.996180 |
| s11N16_staggered_m3_V | 40, 7.7e-10 | 0.921029 | 0.961255 | 0.961255 |
| s11N16_staggered_m3_W | none | 0.944773 | 0.969268 | 0.969268 |
| s11N16_staggered_m5_V | none | 0.932592 | 0.957736 | 0.957736 |
| s11N16_staggered_m5_W | none | 0.947290 | 0.949240 | 0.949240 |
| s11N16_staggered_m8_V | none | 0.926234 | 0.973866 | 0.973866 |
| s11N16_staggered_m8_W | none | 0.948164 | 0.953059 | 0.955073 |
| s11N24_collocated_m3_V | 122, 1.3e-10 | 0.946350 | 0.972654 | 0.990638 |
| s11N24_collocated_m3_W | none | 0.978305 | 0.996859 | 0.996859 |
| s11N24_collocated_m5_V | none | 0.944483 | 0.972823 | 0.995069 |
| s11N24_collocated_m5_W | none | 0.980489 | 0.996978 | 0.996990 |
| s11N24_collocated_m8_V | 86, 1.4e-10 | 0.946137 | 0.974699 | 0.994257 |
| s11N24_collocated_m8_W | none | 0.980678 | 0.996994 | 0.997037 |
| s11N24_staggered_m3_V | 22, 1.4e-06 | 1.006176 | 1.006176 | 1.019399 |
| s11N24_staggered_m3_W | 94, 1.7e-10 | 0.974725 | 0.976524 | 0.998116 |
| s11N24_staggered_m5_V | 43, 1.9e-08 | 0.977679 | 0.977679 | 1.002273 |
| s11N24_staggered_m5_W | 41, 4.5e-08 | 0.975780 | 0.977758 | 1.001064 |
| s11N24_staggered_m8_V | 42, 1.7e-08 | 0.973223 | 0.976202 | 1.003985 |
| s11N24_staggered_m8_W | 33, 5.5e-08 | 0.976436 | 0.976436 | 1.001490 |

Every reading > 1.001 lies at residual ≤ 5.5e-8, apart from one isolated 1.006176 at 1.4e-6
(staggered N = 24, m = 3, V, step 22). The production velocity residual stalls at ≈ 2e-8 on
staggered N = 24 (m = 3, steps 39–44). Hence the revision-1 Ritz floor max(1e-10, 1000·τ).

### R-4. Certification traces (the driver with the monitor read after every step, 4 threads)

**What was run:** the intermediate revision-1 oracle (metric V, Ritz on mixed windows above 100·τ,
restart floor 100·τ) at revision 0's budget of 6 blocks. Monitor samples at the instrument's local
steps 4, 9, …; d relative to |m|.

    $P certify --scheme collocated --N 16 --window 3|5|8 --budget 6
    $P certify --scheme staggered  --N 24 --window 8 --budget 6;  --scheme staggered --N 16 --window 3 --budget 6
    $P certify --scheme collocated --N 24 --window 3 --budget 6

| case | trace (d/\|m\|, R per block; P = pass) |
|---|---|
| coll N16 m3 (65) | +7.9e-8; +5.3e-8 R 0.677 P; +3.6e-8 R 0.679 P; +2.1e-8 R 0.583 P |
| coll N16 m5 (95) | +4.8e-8; +2.4e-8 R 0.493 P; +6.2e-9 R 0.263 P; −3.3e-9 R −0.535; −7.5e-9 R 2.253 → budget; a1; +2.4e-8; R 0.982 P; 0.988 P; 0.988 P |
| coll N16 m8 (99) | −1.0e-7; R 1.180; R 1.054; R 0.992 P; R 0.962 P → budget; a1; R 1.000; 0.988 P; 0.985 P; 0.984 P |
| stag N24 m8 (182; the R-5 run of the same configuration took 214 at 3 threads — the resume loop is round-off sensitive) | −7.8e-8; R 0.897 P; 1.030; 1.039; 1.020 → budget; a1; −1.2e-8 growing, R 1.233, 1.210, 1.157, 1.112 → budget; a2; R 31.8, 1.89, 1.36, 1.18 → budget; a1; R 2.97, 1.44, 1.21, 1.12 → budget; a14; then \|d\| = 5.6e-12, 6.7e-13, 4.0e-12 of \|m\| → pass on the roundoff path (≤ 1e-11) |
| stag N16 m3 (83) | −3.5e-8; R 0.675 P; 1.091; 1.041; 0.949 P → budget; a1; R 1.201; 0.867 P; 0.820 P; 0.809 P |
| coll N24 m3 (119) | +1.8e-8; R −1.637; 1.707; 1.079; 1.046 → budget; a1; R 1.175, 1.098, 1.051, 1.022 → budget; a9; R 0.802 P ×3 |

With the final 12-block budget the same staggered N = 24, m = 8 run passes at block 10 (R 1.030,
1.039, 1.020, 1.003, then 0.990 P, 0.982 P, 0.976 P; 74 steps).

### R-5. Budget ablation (intermediate revision-1 oracle as in R-4, budgets 6 / 9 / 12, 3 threads)

    $O sphere --scheme {collocated,staggered} --N {16,24} --settings production --window 3 5 8 --budget B
    $O sphere --scheme staggered --N 16 --mu 0.05 --beta 0.5 --advection koren --settings production --window 3 5 8 --budget B
    $O sphere --scheme staggered --N 16 --mu 0.0158 --dt 1.234e-2 --advection sou --settings production --window 3 5 8 --budget B

(Steps; phases a = accelerate, c = certify, p = plain tail. Adjacent certify runs, i.e. a resume
with 0 accelerated steps, merge into one "c".)

| case | m | budget 6 | budget 9 | budget 12 |
|---|---|---|---|---|
| g7a stag N16 | 3 | diverged @312 | diverged @312 | diverged @312 |
| g7a stag N16 | 5 | diverged @270 | diverged @270 | diverged @270 |
| g7a stag N16 | 8 | unstable @18 | unstable @18 | unstable @18 |
| re10 stag N16 | 3 | 161 (a136 c25) | 161 (a136 c25) | 161 (a136 c25) |
| re10 stag N16 | 5 | 131 (a106 c25) | 131 (a106 c25) | 131 (a106 c25) |
| re10 stag N16 | 8 | 103 (a78 c25) | 103 (a78 c25) | 103 (a78 c25) |
| sphere coll N16 | 3 | 65 (a40 c25) | 65 (a40 c25) | 65 (a40 c25) |
| sphere coll N16 | 5 | 95 (a39 c30 a1 c25) | 110 (a39 c45 a1 c25) | 89 (a39 c50) |
| sphere coll N16 | 8 | 99 (a38 c30 a1 c30) | 73 (a38 c35) | 73 (a38 c35) |
| sphere coll N24 | 3 | 119 (a24 c30 a1 c30 a9 c25) | 69 (a24 c45) | 69 (a24 c45) |
| sphere coll N24 | 5 | 48 (a23 c25) | 48 (a23 c25) | 48 (a23 c25) |
| sphere coll N24 | 8 | 79 (a23 c30 a1 c25) | 58 (a23 c35) | 58 (a23 c35) |
| sphere stag N16 | 3 | 83 (a22 c30 a1 c30) | 62 (a22 c40) | 62 (a22 c40) |
| sphere stag N16 | 5 | 51 (a21 c30) | 51 (a21 c30) | 51 (a21 c30) |
| sphere stag N16 | 8 | 51 (a21 c30) | 51 (a21 c30) | 51 (a21 c30) |
| sphere stag N24 | 3 | 53 (a28 c25) | 53 (a28 c25) | 53 (a28 c25) |
| sphere stag N24 | 5 | 50 (a25 c25) | 50 (a25 c25) | 50 (a25 c25) |
| sphere stag N24 | 8 | 214 (a24 c30 a1 c30 a2 c30 a1 c30 a12 c30 a4 c20) | 150 (a24 c45 a1 c80) | 74 (a24 c50) |

### R-6. The revision-1 matrix (final oracle: `--rev 1` default — budget 12 with the slow exit, Ritz floor 1000·τ, restart floor 1e-10)

    $O sphere --scheme {collocated,staggered} --N {16,24} --settings production --window 0 3 5 8 --label s11      # 3 threads
    $O sphere --scheme staggered --N 16 --mu 0.05 --beta 0.5 --advection koren --settings production --window 0 3 5 8 --label re10
    $O sphere --scheme staggered --N 16 --mu 0.0158 --dt 1.234e-2 --advection sou --settings production --window 3 5 8 --label g7a
    $O sphere --scheme {collocated,staggered} --N 16 --settings tight --window 3 5 8 --label g1
    $O bed --arrangement $BED --scheme {staggered,collocated} --N 64 --settings production --window W --label bed06   # 4 threads, one W per process

| case | settings | m | converged | reason | steps | accelerated | K | pressure iterations | phases |
|---|---|---|---|---|---|---|---|---|---|
| bed06 coll N64 | production | 0 | True | certified | 190 | 0 | 101.4149551388 | 6319 |  |
| bed06 coll N64 | production | 3 | True | certified | 163 | 78 | 101.4163271346 | 4922 | a74 c60 a4 c25 |
| bed06 coll N64 | production | 5 | True | certified | 83 | 58 | 101.4162483839 | 2588 | a58 c25 |
| bed06 coll N64 | production | 8 | True | certified | 82 | 57 | 101.4162467119 | 2560 | a57 c25 |
| bed06 stag N64 | production | 0 | True | certified | 325 | 0 | 98.9108649594 | 3733 |  |
| bed06 stag N64 | production | 3 | True | certified | 111 | 86 | 98.9152562526 | 1272 | a86 c25 |
| bed06 stag N64 | production | 5 | True | certified | 93 | 68 | 98.9143585121 | 1067 | a68 c25 |
| bed06 stag N64 | production | 8 | True | certified | 90 | 65 | 98.9146455374 | 1034 | a65 c25 |
| g1 coll N16 | tight | 3 | True | certified | 149 | 129 | 4.2666481754 | 1604 | a129 c20 |
| g1 coll N16 | tight | 5 | True | certified | 141 | 121 | 4.2666481754 | 1521 | a121 c20 |
| g1 coll N16 | tight | 8 | True | certified | 118 | 98 | 4.2666481753 | 1260 | a98 c20 |
| g1 stag N16 | tight | 3 | True | certified | 85 | 65 | 4.2119989946 | 815 | a65 c20 |
| g1 stag N16 | tight | 5 | True | certified | 79 | 59 | 4.2119989946 | 774 | a59 c20 |
| g1 stag N16 | tight | 8 | True | certified | 72 | 52 | 4.2119989946 | 687 | a52 c20 |
| g7a stag N16 | production | 3 | False | diverged | 312 | 57 | — | 2297 | a57 c60 p195(dis) |
| g7a stag N16 | production | 5 | False | diverged | 270 | 90 | — | 2002 | a90 c60 p120(dis) |
| g7a stag N16 | production | 8 | False | unstable | 18 | 18 | 15.0636873198 | 124 | a17 a1(uns) |
| re10 stag N16 | production | 0 | True | certified | 345 | 0 | 4.3642841907 | 2415 |  |
| re10 stag N16 | production | 3 | True | certified | 161 | 136 | 4.3642744865 | 1122 | a136 c25 |
| re10 stag N16 | production | 5 | True | certified | 131 | 106 | 4.3642731991 | 913 | a106 c25 |
| re10 stag N16 | production | 8 | True | certified | 103 | 78 | 4.3642492530 | 717 | a78 c25 |
| s11 coll N16 | production | 0 | True | certified | 395 | 0 | 4.2663336995 | 3148 |  |
| s11 coll N16 | production | 3 | True | certified | 65 | 40 | 4.2666491472 | 513 | a40 c25 |
| s11 coll N16 | production | 5 | True | certified | 89 | 39 | 4.2666539753 | 720 | a39 c50 |
| s11 coll N16 | production | 8 | True | certified | 73 | 38 | 4.2666353529 | 533 | a38 c35 |
| s11 coll N24 | production | 0 | True | certified | 90 | 0 | 4.2791312754 | 714 |  |
| s11 coll N24 | production | 3 | True | certified | 69 | 24 | 4.2791322211 | 499 | a24 c45 |
| s11 coll N24 | production | 5 | True | certified | 48 | 23 | 4.2791312179 | 352 | a23 c25 |
| s11 coll N24 | production | 8 | True | certified | 58 | 23 | 4.2791318817 | 422 | a23 c35 |
| s11 stag N16 | production | 0 | True | certified | 75 | 0 | 4.2119942333 | 472 |  |
| s11 stag N16 | production | 3 | True | certified | 62 | 22 | 4.2119986525 | 382 | a22 c40 |
| s11 stag N16 | production | 5 | True | certified | 51 | 21 | 4.2119983296 | 316 | a21 c30 |
| s11 stag N16 | production | 8 | True | certified | 51 | 21 | 4.2119983354 | 315 | a21 c30 |
| s11 stag N24 | production | 0 | True | certified | 135 | 0 | 4.2572787678 | 859 |  |
| s11 stag N24 | production | 3 | True | certified | 53 | 28 | 4.2573076958 | 358 | a28 c25 |
| s11 stag N24 | production | 5 | True | certified | 50 | 25 | 4.2573087322 | 335 | a25 c25 |
| s11 stag N24 | production | 8 | True | certified | 74 | 24 | 4.2573097578 | 506 | a24 c50 |

**Fixed point.**
- G1 tight, against the WO-1 plain K (collocated 4.266648173242035, staggered 4.211998994555005):
  - collocated m = 3 / 5 / 8: 4.266648175365904 / 4.266648175401971 / 4.266648175297917, i.e.
    4.98e-10 / 5.06e-10 / 4.82e-10;
  - staggered: 4.211998994633273 / 4.211998994631651 / 4.211998994630338, i.e. 1.86e-11 /
    1.82e-11 / 1.79e-11.
- Production §11 N = 16 against K_G1: collocated 2.3e-7 / 1.4e-6 / 3.0e-6; staggered 8.1e-8 /
  1.6e-7 / 1.6e-7.
- Production, accelerated vs plain K: bed staggered 3.5e-5 – 4.4e-5; bed collocated 1.3e-5 –
  1.4e-5; the others ≤ 7.5e-5. All ≤ 2·rtol, as §2.3 bounds it.

**Ratios at m = 5** (steps / pressure iterations):

| case | steps ratio | pressure-iteration ratio |
|---|---|---|
| §11 coll N16 | 4.44 | 4.37 |
| §11 coll N24 | 1.88 | 2.03 |
| §11 stag N16 | 1.47 | 1.49 |
| §11 stag N24 | 2.70 | 2.56 |
| Re ≈ 10 | 2.63 | 2.65 |
| bed stag | 3.49 | 3.50 |
| bed coll | 2.29 | 2.44 |

Pressure iterations per step, plain / accelerated (m = 5): bed staggered 11.49 / 11.47; bed
collocated 33.26 / 31.18; Re ≈ 10 7.00 / 6.97; §11 collocated N = 16 7.97 / 8.09.

**Guard census.**
- 0 restarts in every run.
- "unstable" only on G7a at m = 8: readings 1.124 at step 12 (residual 9.4e-3), 1.090, 1.077,
  1.059, so "unstable" at step 18. The plain march diverges there, so this is a true positive.
- Every other eligible reading ≤ 0.9974 over all runs. The R-5 runs with the 100·τ floor had one
  reading > 1.0005: 1.00618 at residual 1.4e-6 on staggered N = 24, m = 3, isolated. The 1000·τ
  floor excludes it, and no step count changed: R-6 equals R-5's budget-12 column run for run.

**Window rule (pre-registered, re-applied):** m = 3 within 10 % of m = 5 on every case? No:
collocated N16 65 vs 89, collocated N24 69 vs 48, bed collocated 163 vs 83. The window stays 5.

### R-7. Host map time, plain vs m = 5, on the dense bed (8 threads, one process at a time, box shared with other users)

    for rep in 1 2: $O bed --arrangement $BED --scheme staggered --N 64 --settings production --window 0, then 5
    $O bed --arrangement $BED --scheme collocated --N 64 --settings production --window 0, then 5

| run | scheme | m | steps | map seconds | seconds/step | pressure iterations |
|---|---|---|---|---|---|---|
| time1 | staggered | 0 | 325 | 33.0 | 0.1014 | 3733 |
| time1 | staggered | 5 | 93 | 10.4 | 0.1120 | 1067 |
| time2 | staggered | 0 | 325 | 32.6 | 0.1004 | 3733 |
| time2 | staggered | 5 | 93 | 12.0 | 0.1291 | 1067 |
| time1 | collocated | 0 | 190 | 64.4 | 0.3387 | 6319 |
| time1 | collocated | 5 | 83 | 28.9 | 0.3480 | 2588 |

- Map-time ratio (plain / accelerated, solver.step() only): staggered 3.17× and 2.72×; collocated
  2.23×.
- Per-step map time at mixed iterates: +10 % / +28 % staggered, +3 % collocated, at equal pressure
  iterations. Not attributable on a loaded box: the velocity-solve sweeps are not exposed, and the
  oracle's NumPy history traffic between steps leaves the cache cold. Open as Q12; G2 decides.

---

## 2026-10-02 — WO-2: `Solver::marchState()` and the refusals (rev 1)

**Build.** Worktree `flow-anderson`, branch `anderson` (from `8a956bb`); core headers from
`../core-anderson` at `9ff3bd2` (configure: `[peclet] peclet-core headers from
PECLET_SIBLING_PECLET_CORE -> /home/frankp/Codes/suite/core-anderson/include`). Host tree
`build_omp`: `-DCMAKE_PREFIX_PATH=../extern/install/host-openmp -DCMAKE_BUILD_TYPE=Release
-DPECLET_FLOW_BUILD_TESTS=ON -DPECLET_FLOW_MPI=ON -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
-DMPIEXEC_PREFLAGS="--bind-to;none" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache`. CUDA tree `build_cuda`
(`nvidia-cuda` prefix, tests ON, MPI OFF).

**What landed.** `Solver::MarchState` + `marchState()` (§5.1 rev 1: fields u, v, w Velocity; P
Carried; collocated with `advect_ && ufAdvect_` + uf, vf, wf Carried; `innerTolerance` =
`velocityResidualTolerance()` if > 0 else `useChebyshev_ ? chebRtol_ : pcgRtol_`; signature (dt,
rho, mu, F) internal; MPI comm + distributed). The twelve §5.3 refusals, each with its own message
and the hint "pass accelerate=False". ctest `march_state` (`tests/kokkos/test_march_state.cpp`).

**Implementation choices (not numerics; recorded so they can be reverted).**
- DECISION: refusal order — phase change is tested BEFORE VoF, because `enable_phase_change` calls
  `enableVof()` itself, so in the note's list order refusal 3's message could never be reached (the
  test of "each refusal has its own message" would be impossible). Drag precedes cell forces as
  listed (it registers force_*). Alternative: the literal list order. Reversible by swapping two
  `if`s in `flow_ibm_diagnostics.hpp`.
- "Variable rho or mu" = `varRho_ || varProps_ ||` a closure whose target is "rho"/"mu" `||` a
  registered "rho"/"mu" field. The collocated test is `ghostProjection_ && faceInterp_ == 0` (the
  'ghost' scheme; AUTO resolves at `set_solid`).

**G0(a) — state hashes, before (= `8a956bb` against core-anderson) and after.**

    OMP_NUM_THREADS=1 PYTHONPATH=build_omp python tests/regression/state_hash.py
    OMP_NUM_THREADS=1 PYTHONPATH=build_omp mpirun --bind-to none -np 2 python tests/regression/state_hash.py mpi

All 13 lines identical (12 serial entry paths + `mpi_np2`; e.g. `staggered_bed bdc54811…b86e`,
`colocated_ghost 14f3287f…93b8`, `mpi_np2 9fd78958…75a4`). Also an `.npz` of u, v, w, p after 60
plain steps of §11 N = 16 (staggered + collocated) on both builds: `np.array_equal` True on all 8
arrays.

**ctest `march_state`:** all checks pass on host-openmp and on CUDA (RTX 5080): the three §3.1 rows
(each field aliases the solver's buffer, padded e = n + 4), `innerTolerance` 1e-8 (default = PCG
rtol) / 1e-9 (explicit) / 3e-7 (tolerance 0 under Chebyshev), the twelve refusals, a static scene
allowed.

**G0(c), first attempt (WO-2 tree):** `OMP_NUM_THREADS=8 ctest --test-dir build_omp -LE bench -j4`
(189 = 188 + `march_state`): 87 of 189 run, **87 passed, 0 failed**, then stopped by hand after 2 h:
the box was at load 107–119 on 48 cores (other sessions' jobs), and with 8 OpenMP threads per MPI
rank the distributed tests crawled (`velocitymg_bc_mpi_np2` 2712 s, `sdflow_mpi_np2` > 47 min).
The battery is re-run once on the WO-4 tree with the MPI tests at 2 threads per rank (entry below).

---

## 2026-10-02 — WO-4: adapter, bindings, Python driver, packaging (rev 1)

**What landed.** `src/anderson_accelerator.hpp` (`AndersonAccelerator<Grid>`: core's
`AndersonCore` around `Solver::step()`, §4.3; explicitly instantiated beside the solver in
`src/flow_solver_{staggered,colocated}.cpp`); the private classes `_AndersonAccelerator` /
`_AndersonAcceleratorColocated` and `s.diagnostics.anderson_accelerator(window=5, mixing=1.0)`
(Releasable, keep_alive on the solver); `packaging/flow_steady.py` → `peclet/flow/steady.py`
(`march_to_steady`, `MarchResult`, the §7 rev-1 driver: velocity-residual target, budget
2·(num_passes + 3), early "slow" exit; `accelerate=True` by default as the note states, pending
the D11 rule at G2); the import in `flow_init.py`; CMake `configure_file` + `install`; ctest
`march_to_steady`; the study `study_avg_velocity_spheres.py` flow column switched to
`march_to_steady(accelerate=False)` (Q11). `tests/kokkos_mpi/test_anderson_mpi.cpp` (G4c) and
`tests/study/steady_acceleration_gates.py` (the WO-5 instrument) are in the same commit.

**Implementation choices (recorded so they can be reverted).**
- DECISION: the adapter refuses to step (std::logic_error) when the solver's state buffers were
  reallocated since construction (a redistribute / rebalance), instead of mixing into buffers
  the solver no longer reads. Alternative: re-bind silently (would need a fresh history anyway).
  Reversible: delete the pointer check in `AndersonAccelerator::step`.
- DECISION: developer-tier property `acc.seconds` — cumulative accelerator wall time excluding
  `solver.step()`, device-fenced at both ends of each part (`Kokkos::fence()`, no effect on
  numerics) — the G8 instrument. Name not in the note's list; reversible by renaming.
- DECISION: G4c's test runs tight inner solves (PCG 1e-12, velocity residual tolerance 1e-12) at
  nu dt/h^2 = 0.5, so the momentum solve is RB-GS on every block size (the AUTO V-cycle needs ≥ 16
  cells per axis per rank) and converges; the remaining np-dependence is reduction order.
- The 12 refusals carry the hint "pass accelerate=False"; the allocation failure is re-thrown as
  "AndersonAccelerator: <core message>; pass accelerate=False".

**Gate: C++ vs oracle** (host `build_omp4`, 8 threads, §11 collocated N = 16, production, m = 5,
30 unconditional `step(True)` on two identical solvers):

    python tests/study/steady_acceleration_gates.py oracle sphere --scheme collocated --N 16 --window 5 --steps 30 --trace

Max relative difference of the per-step relative velocity residual over the first 30 steps
**2.54e-9** (step 29; ≤ 1e-6 required). Steps 1–4 agree to ≤ 2.3e-15.
Beyond the gate: `march_to_steady` on the C++ build reproduces the revision-1 oracle table
(log R-6) step for step — host: §11 collocated N = 16 m = 5 89 steps (39 accelerated) K
4.2666539753, staggered 51 (21) K 4.2119983296; CUDA (WO-5 runs): every compared case equal in
steps (§11 coll N16 395/65/89/73, N24 90/69/48/58; stag N16 75/62/51/51, N24 135/53/50/74; bed
stag 325/111/93/90, bed coll 190/163/83/82; G7a 440 diverged / 312 / 270 / unstable at 18).

**Gate: G0(b)** — `march_to_steady(accelerate=False)` vs the study's `march()`, §11 production:

| scheme | N | study steps | driver steps | <u_x> identical (==) |
|---|---|---|---|---|
| staggered | 16 | 75 | 75 | yes (0.040607286951873406) |
| collocated | 16 | 395 | 395 | yes (0.04009007980112628) |
| staggered | 24 | 135 | 135 | yes (0.04017534857334543) |
| collocated | 24 | 90 | 90 | yes (0.03997018260520385) |

**Gate: ctest `march_to_steady`** (`build_omp`, 8 threads): PASS in 35.9 s — certify (i) "pass"
after 9 blocks, (ii) "slow" at step 15, (iii) "budget" at step 60; G0(b) N = 16 both schemes;
small G1 N = 12 tight: staggered plain 165 / accelerated 58 steps, |K_acc/K_plain − 1| = 2.76e-12,
collocated 2035 / 106, 2.22e-10; G7a: accelerate=False diverged at 440, True diverged at 270.

**G0(a) after WO-4:** state hashes (12 serial + mpi_np2) identical to the baseline.
**G4c — ctest `anderson_mpi`** (np = 1, 2, 4; 2 threads): max|u_np − u_1|/max|u_1| = 0 / 1.3e-15 /
2.6e-15; γ, window and status bitwise equal on all ranks after every one of the 40 steps (0
mismatches); residual 1.323022e-09 at np = 1, 2, 4 and single-rank.

**G0(c) on the WO-4 tree** (`build_omp`, 193 ctests with `-LE bench` = the 188 existing +
`march_state`, `march_to_steady`, `anderson_mpi_np{1,2,4}`):

    OMP_NUM_THREADS=8 ctest --test-dir build_omp -LE bench -E '_np[0-9]+$' -j3    # 63 tests
    OMP_NUM_THREADS=2 ctest --test-dir build_omp -LE bench -R '_np[0-9]+$' -j4    # 130 MPI tests

**63/63 and 130/130 passed (193/193)**, 766 s + < 10 min. (2 OpenMP threads per MPI rank: with 8,
on this shared box, the first attempt's MPI tests crawled — see the WO-2 entry.)

---

## 2026-10-02 — WO-5: the gate campaign (C++ build; STOPPED on G1, G3 bed, G7c, G8)

**Builds.** host `build_omp` (WO-4 tree, host-openmp, MPI), CUDA `build_cuda` (RTX 5080, shared
with another user's job at ~50–75 % utilisation), host `build_omp4` (MPI module, G4) and
`build_omp_serial` (non-MPI module, G4's serial side), all against core-anderson `9ff3bd2`. Host
box: 48 cores at load 40–120 from other sessions throughout — wall times are indicative; step
counts and iterations are not affected. 8 OpenMP threads per process (2 per rank under MPI).
**Instrument:** `tests/study/steady_acceleration_gates.py`; the queue that ran it:

    Q=<scratch>/wo5/queue.sh   # wraps: G="python tests/study/steady_acceleration_gates.py"
    $G run sphere --scheme {collocated,staggered} --N {14,16,18,20,24} --settings {production,tight} --window W
    $G run sphere --scheme staggered --N 32 --phi {0.343,0.45} --settings {production,tight} --window W
    $G run bed --arrangement $BED --scheme {staggered,collocated} --N 64 --settings {production,tight} --window W
    $G g3 sphere --scheme S --N 16 --window 5 --steps 400 --betas 6 60 600 1e4
    $G g3 bed --arrangement $BED --scheme S --N 64 --window 5 --steps 400 --betas 60 600 1e4
    $G g5 sphere --scheme S --N 16 --settings tight --window 5 --at 25
    $G run sphere --scheme S --N 16 --mu 0.05 --dt 3.906e-2 --advection koren --settings tight --window W
    $G run bed --arrangement $ARR --scheme staggered --N 80 --beta 2.0 --force F --advection sou --implicit-advection --settings tight --window W
    $G run sphere --scheme staggered --N 16 --mu 0.0158 --dt 1.234e-2 --advection sou --settings production --window W
    $G g8 sphere --scheme S --N 64 --window 5 --steps 50 --warmup 150
    mpirun --bind-to none -np P $G mpi sphere --scheme S --N 32 --settings tight --window 5 [--hash --steps 60 [--serial]]

`BED` = the WO-1 arrangement (A1 `random_arrangement(0.6, 64, seed 0)`), `ARR` = A1
`random_arrangement(0.3, n_spheres=64, seed=307)` (phi 0.3000, L 4.8160, N = `grid_for(16, L/D)` =
80), F = A1 `re_scan.force_for_re(Re, 0.3, L, 64)` = 694.9558 (Re ≈ 10), 13400.17 (Re ≈ 100).
Raw records: `<scratch>/wo5/{cuda,omp}/*.jsonl`.

DECISION (Q3 reading): the A1 finite-Re configuration is `re_scan.py`'s own documented one
(`scan --phi 0.3 --dpdx 16 --seed 307`, 64 spheres) at its steady pseudo-step `beta_steady = 2`
with A1 `make_solver`'s implicit advection (SOU), plus the tight settings. No A1 rescan result
exists to pin another phi. Alternative: none named by A1. Reversible: rerun G6 with another `ARR`.

### G2 — production, window 5 (bars) and 3 / 8 (reported); steps identical on host and CUDA

| case | plain | m = 3 / 5 / 8 | steps ratio (m 5) | wall ratio host / CUDA | pressure-iteration ratio | \|K5/K_G1 − 1\| | p-its/step plain vs accelerate phase |
|---|---|---|---|---|---|---|---|
| §11 coll N14 | 445 | 73 / 81 / 60 | 5.49 | 5.19 / 5.96 | 5.49 | 1.0e-5 | 6.94 / 6.88 |
| **§11 coll N16** | 395 | 65 / **89** / 73 | **4.44** | **4.93 / 4.57** | 4.37 | 1.4e-6 | 7.97 / 7.67 |
| §11 coll N18 | 335 | 62 / 65 / 135 | 5.15 | 5.39 / 5.00 | 5.22 | 4.5e-7 | 8.91 / 8.69 |
| §11 coll N20 | 70 | **83** / 70 / 68 | 1.00 | 1.46 / 1.24 | 1.03 | 2.1e-6 | 7.76 / 7.27 |
| §11 coll N24 | 90 | 69 / 48 / 58 | 1.88 | 2.75 / 2.98 | 2.03 | 1.3e-6 | 7.93 / 7.57 |
| §11 stag N14 | 90 | 43 / 41 / 40 | 2.20 | 2.33 / 2.89 | 1.99 | 2.0e-8 | 6.08 / 6.25 |
| §11 stag N16 | 75 | 62 / 51 / 51 | 1.47 | 1.41 / 0.89 | 1.49 | 1.6e-7 | 6.29 / 6.48 |
| §11 stag N18 | 175 | 72 / 96 / 57 | 1.82 | 1.95 / 2.22 | 1.82 | 1.7e-6 | 6.00 / 6.00 |
| §11 stag N20 | 165 | 51 / 48 / 47 | 3.44 | 3.08 / 3.05 | 3.17 | 1.5e-6 | 6.25 / 6.52 |
| §11 stag N24 | 135 | 53 / 50 / 74 | 2.70 | 2.68 / 2.74 | 2.56 | 4.4e-6 | 6.36 / 6.44 |
| Z&H 0.343 stag N32 | 105 | **163** / 70 / 59 | 1.50 | 1.41 / 1.57 | 1.50 | 2.0e-5 | 7.04 / 7.07 |
| Z&H 0.45 stag N32 | 115 | 59 / 54 / 54 | 2.13 | 2.03 / 2.31 | 2.16 | 7.0e-6 | 6.43 / 6.62 |
| **bed stag** | 325 | 111 / **93** / 90 | **3.49** | **3.43 / 3.71** | 3.50 | 1.4e-5 | 11.49 / 11.65 |
| **bed coll** | 190 | 163 / **83** / 82 | **2.29** | **2.44 / 2.59** | 2.44 | 7.9e-7 | 33.26 / 32.12 |

- Bars at window 5: §11 coll N16 steps 4.44 ≥ 3.0 and wall 4.93 (host) / 4.57 (CUDA) ≥ 2.7
  **pass**; steps_acc ≤ steps_plain on every case **pass** (coll N20 70 = 70); |K5/K_G1 − 1| ≤
  1e-4 every case **pass** (max 2.0e-5; K_G1 = the CUDA tight plain K). At m = 3 two cases take
  more steps than the plain march (coll N20 83 vs 70, Z&H 0.343 163 vs 105) — reported, not a bar.
  The CUDA wall ratio < 1 on stag N16 (2.74 s vs 2.43 s) is the shared GPU on a 16³ case.
- Every step count equals the revision-1 oracle where both exist (log R-6).
- **D11 / Q1 (Q13: the staggered bed decides).** Wall ratio, plain / m = 5, including the
  accelerator: **host 3.38, 3.45, 3.38** (28.87/8.54, 28.05/8.14, 28.17/8.33 s), **CUDA 3.87,
  3.55, 3.76, 3.71** (21.49/5.55, 28.66/8.07, 27.07/7.19, 29.06/7.84 s). Collocated bed beside it:
  host 2.53, 2.48, 2.49; CUDA 2.38, 2.64, 2.73, 2.59. **≥ 1.5 → `accelerate=True` is the default**
  of `march_to_steady` (already the code's default; nothing to change).
- Q12 (inner work at mixed iterates): pressure iterations per step in the accelerate phase vs the
  plain march: bed stag 11.65 vs 11.49 (+1.4 %), bed coll 32.12 vs 33.26 (−3.4 %); host time per
  step, bed stag: 89.6 ms accelerated (incl. the accelerator) vs 86.7 ms plain (+3 %).

### G1 — tight, window 5 (CUDA; host in a later entry)

| case | plain steps | m = 5 steps | \|K_acc/K_plain − 1\| | verdict |
|---|---|---|---|---|
| §11 coll N14 | 2105 | 64 | — (converged=False: "unstable" at 64) | **FAIL** |
| §11 coll N16 | 3520 | 141 (m 3: 149, m 8: 118) | 5.06e-10 | pass |
| §11 coll N18 | 2935 | 131 | 4.00e-10 | pass |
| §11 coll N20 | 3715 | 185 | 6.42e-10 | pass |
| §11 coll N24 | 2470 | 153 | 6.67e-10 | pass |
| §11 stag N14 | 220 | 60 | 7.0e-12 | pass |
| §11 stag N16 | 295 | 79 (m 3: 85, m 8: 72) | 1.82e-11 | pass |
| §11 stag N18 | 1165 | 139 | 1.42e-10 | pass |
| §11 stag N20 | 6550 | 51 | — (converged=False: "unstable" at 51) | **FAIL** |
| §11 stag N24 | 3465 | 207 | 6.90e-10 | pass |
| Z&H 0.343 N32 | 2530 | 174 | 4.09e-10 | pass |
| Z&H 0.45 N32 | 5645 | 396 | 1.44e-9 | pass |
| bed stag | 19510 | 536 (a376 c40 p120, disabled) | **1.44e-8** | **FAIL** |
| bed coll | 1835 | 321 | 1.75e-9 | pass |

**The two false "unstable" (the Ritz guard at the tight floor).** The oracle (`--rev 1`)
reproduces coll N14 exactly: "unstable" at step 64, readings 1.00412 / 1.00531 / 1.00527. Every
eligible reading > 1.0005 in the tight runs, (step, residual, radius):
- coll N14: (19, 5.47e-6, 1.0021) (20, 5.41e-6, 1.0037) (22, 5.42e-6, 1.0047) (31, 4.78e-6,
  1.0007) (33, 4.78e-6, 1.0062) (62, 6.65e-9, 1.0041) (63, 6.16e-9, 1.0053) (64, 5.85e-9, 1.0053)
- stag N20: (39, 5.35e-9, 1.0010) (42, 5.26e-9, 1.0015) (44, 5.25e-9, 1.0013) (47, 5.23e-9,
  1.0026) (49, 5.23e-9, 1.0011) (50, 5.18e-9, 1.0011) (51, 5.17e-9, 1.0014) → "unstable" at 51
- coll N12 (ctest case, m 5): (16, 1.49e-5, 1.0120); stag N18: (22, 1.73e-6, 1.0111) (23, 1.71e-6,
  1.0047); stag N24: (58, 1.78e-8, 1.0072); Z&H 0.343 max 1.0030, Z&H 0.45 max 1.0018.
- Reading: at tight settings the Ritz floor is 1000·τ = 1e-9, but the map's velocity residual
  stalls at **5–7e-9** (stag N20: 5.35e-9 → 5.17e-9 over steps 39–51), so readings at 5× the
  floor are noise, as rev 1 found for production at ≈ 2τ. Separately, isolated readings of 1.002–
  1.012 occur at residual 1e-6 – 1.5e-5 (coll N12/N14, stag N18) — far above any floor, on a
  stagnating stretch of phase A (coll N14: residual ≈ 5e-6 for steps 19–37). None of these maps
  is unstable (the plain marches converge). Production: 96 accelerated runs, 946 eligible
  readings, max 0.9825, none > 1.0005.

**The bed: the plain march, not the accelerated one, is off.** A plain tight march of the
staggered bed run to 60 000 steps (CUDA; K every 2000 steps):

    step 20000 K=98.9156981196 | 30000 98.9156990401 | 40000 98.9156992810 | 50000 98.9156993670 | 60000 98.9156994068

per-2000-step changes 4.3e-7 → 9.5e-8 → 2.9e-8 → 6.1e-9, ratio creeping 0.71 → 0.89 (per step
0.99983 → 0.99994; the power-law tail of log R-1). K∞ ≈ 98.9156994–98.9156995 by geometric
extrapolation. The accelerated K **98.9156994503** is within ~5e-10 of it; the plain march
certified at step 19 510 with **98.9156980268**, 1.4e-8 low: the instrument assumes slow_rate
0.997 and this tail is ~0.9999, so at rtol 1e-10 the plain certificate is premature. G1's
reference, not the acceleration, misses 1e-8 here.

### G3 — 400 unconditional accelerated steps, tight (host and CUDA agree)

- §11 N16 collocated and staggered at nu dt/h² = 6, 60, 600, 1e4: status active, 0 restarts,
  running-min residual 1e-16 – 3.5e-16, final ≤ 1.1× min; K spread over the four dt 2.2e-14
  (coll) / 4.2e-16 (stag): **pass**.
- bed collocated at nu dt/h² = 60, 600, 1e4: active, 0 restarts, min residual 1.0e-12 / 1.1e-11 /
  9.3e-12, K spread 3.6e-9: **pass**.
- bed staggered: active, 0 restarts, final = min residual (still falling), but min residual
  6.3e-10 / **2.0e-9 / 1.0e-9** (CUDA; host 7.1e-10 / 1.7e-9 / 1.3e-9) and K spread **3.8e-7**
  (host 4.1e-7): **FAIL** on "running min ≤ 1e-9" at 600 and 1e4 and on the 1e-8 K agreement.
  Not instability — 400 steps do not converge this bed's slow tail to 1e-9 (cf. G1: 376
  accelerated steps reach the 3e-17 target only with plain help).

### G4 / G4c — MPI (host, kokkos_mpi build, `--bind-to none`, 2 threads per rank)

| scheme N32 tight m5 | np 1 | np 2 | np 4 |
|---|---|---|---|
| staggered: steps / K | 128 / 4.27720623198430 | 128 / 4.27720623198496 (1.5e-13) | 128 / 4.27720623198932 (1.2e-12) |
| collocated: steps / K | 221 / 4.28443927102403 | **247** (+11.8 %) / 4.28443927101270 (2.6e-12) | 221 / 4.28443927102430 (6.3e-14) |

|K_np/K_1 − 1| ≤ 1e-9 **pass**; step counts within ±10 % except collocated np 2 (+11.8 %,
reported). np = 1 (MPI build, `init_mpi`) vs the serial build after 60 accelerated steps: state
hashes **identical** (staggered `42a80399…0865`, collocated `05fc177e…29c5`). G4c (ctest
`anderson_mpi`): see the WO-4 entry — pass.

### G5 — restart at step 25 (tight, m 5)

| scheme | uninterrupted | get_field/set_field: total, \|ΔK\|/K | set_state: total, \|ΔK\|/K |
|---|---|---|---|
| collocated N16 (CUDA / host) | 141 | 148 / 150 (+7 / +9), 1.0e-11 | 173 / 173 (+32), 1.7e-11 |
| staggered N16 (CUDA / host) | 79 | 75 / 75 (−4), 2.9e-13 | 99 / 99 (+20), 3.7e-14 |

K ≤ 1e-8 **pass**; extra steps ≤ 15 with get_field/set_field **pass**; set_state reported.

### G6 — finite Re, tight

| case | plain | m 5 | \|K_acc/K_plain − 1\| | verdict |
|---|---|---|---|---|
| §11 stag N16, Re ≈ 10 (host = CUDA) | 685 | 283 | 4.6e-11 | pass (2.4×) |
| §11 coll N16, Re ≈ 10 (n_s = 7), CUDA / host | 2885 | 1043 / 1030 (disabled → plain tail) | < 1e-10 | pass (2.8×) |
| A1 array phi 0.3 N80, Re ≈ 10 (CUDA) | 8015 | 449 | 3.3e-9 | pass (17.9×) |
| A1 array phi 0.3 N80, Re ≈ 100 (CUDA) | 2775 | 2630 (a300 c15 p2315, disabled) | 4e-12 | pass, 1.06× |

### G7 — (a) staggered N16 mu 0.0158 dt 1.234e-2 SOU, production (host = CUDA, steps identical)

plain diverged at 440; m 3 diverged 312 (a57 c60 p195); m 5 diverged 270 (a90 c60 p120);
m 8 "unstable" at 18 (readings 1.124 …): `converged=False` everywhere **pass**. (b) core U4: pass
(core-anderson). (c) false-alarm census: production **pass** (above); tight **FAIL** (above).

### G8 — accelerator overhead and memory (64³ §11 sphere, m 5, mean over 50 steps, after 150 plain
warm-up steps so both means are taken late in the march; two repetitions)

| backend | scheme | plain step | step under acceleration | accelerator | % of plain step | bar |
|---|---|---|---|---|---|---|
| CUDA | staggered | 16.02 / 14.56 ms | 11.47 / 10.41 ms | 0.846 / 0.773 ms | **5.28 / 5.31 %** | ≤ 5 %: **FAIL** |
| CUDA | collocated | 40.43 / 40.66 ms | 15.94 / 16.03 ms | 0.795 / 0.806 ms | 1.97 / 1.98 % | pass |

`memory_bytes` = 130 803 712 = (2·5+3)·4·8·68³ exactly, both schemes, both backends: **pass**.
The 0.8 ms is latency, not bandwidth (the §6.2 traffic, ~120 MB, is ~0.13 ms at 960 GB/s): one
step makes ~26 host-synchronising Kokkos calls (3 + 15 per-column reductions for 3 velocity
fields, 4 COUNT reductions, 4 device-to-device deep_copy). The note's named first optimisation
(fuse pass 2's per-column reductions in pairs) is a core change — not made here. (Host G8 with
the late-march protocol: later entry; the first host run with a 10-step warm-up read 3.8 % / 1.1 %.)

---

## 2026-10-02 — WO-6: register entries (DRAFTS for the caller; not committed to `../docs/`)

Rev 1 changes two entries already in `../docs/decisions/flow.md` and adds three; each is a NEW
recorded decision naming what changed. The guard-scope entry (3) is **not ready to settle**: its
own gate (G7c) failed at tight settings in WO-5 — keep it as "pending the architect" or drop it.

    ### The steady-acceleration metric is the velocity alone; P and the collocated face field are carried (rev 1)
    - area: flow
    - source: flow doc/steady_acceleration.md "Revision 1" R1, D3 (rev 1), §1.2, §3.2; WO-5 §1.3
    - decided: 2026-10-02
    - status: settled (architect rev 1; implemented flow 2ba357c, core-anderson 9ff3bd2)
    - supersedes: the metric clause of "Steady marches are accelerated by type-II Anderson on the
      full march state…" ("weighted c_P = h/(mu + rho h^2/dt), gauge removed")
    - quote: |
        The state stays (u, P [, u_f]); P and u_f are mixed, stored and differenced but not
        measured. The residual is the relative velocity residual ||g_u - x_u|| / ||g_u||.
    - rejected: the c_P-weighted, gauge-centred pressure term (rev 0); per-pocket gauge removal;
      the unweighted Euclidean norm of (u, P); a metric on the face gradient of P
    - why: on the dense bed 99.97 % of the W-residual is pressure in sealed / near-contact pockets
      that moves no velocity (rev-0 bed gain 1.26x in steps); a pressure error is measured by the
      velocity it drives. Measured on the C++ build: dense bed 3.49x (staggered) / 2.29x
      (collocated) in steps, 4.44x on the collocated sphere

    ### Steady-state certification after acceleration: velocity-residual handover, budget 2(num_passes + 3) blocks, early "slow" exit (rev 1)
    - area: flow
    - source: flow doc/steady_acceleration.md "Revision 1" R2, R4, D6 (rev 1), §7
    - decided: 2026-10-02
    - status: settled (architect rev 1; implemented flow 2ba357c)
    - supersedes: in "Steady state is certified by the unchanged stop instrument on PLAIN steps…",
      the residual (now the velocity residual) and the budget (was num_passes + 3, no early exit)
    - quote: |
        Phase A hands over at (1 - slow_rate) rtol on the relative velocity residual; phase B
        certifies with the unchanged instrument on plain steps within 2 (num_passes + 3) blocks;
        a block failing on the remainder bound with R in [slow_rate^check_every, 1) resumes
        acceleration at once; budget or slow exit: target x 0.1, resume with the history.
    - rejected: the 6-block budget (staggered N24 m8: 214 steps vs 135 plain; 74 at 12 blocks); no
      budget; handing over on the monitor's own changes along the accelerated sequence
    - why: after an Anderson iterate the monitor's block changes start near zero and change sign or
      grow for up to five blocks before the slow tail emerges

    ### The Anderson Ritz guard runs only on mixed calls over all-mixed windows above max(1e-10, 1000 tau) (rev 1) — PENDING
    - area: flow, core
    - source: flow doc/steady_acceleration.md "Revision 1" R3, D5 (rev 1), §4.1, §4.3 step 7
    - decided: 2026-10-02 (architect rev 1)
    - status: PENDING — gate G7c FAILED in WO-5 at tight settings (tau = 1e-12): false "unstable"
      on §11 collocated N14 and staggered N20 (readings 1.001-1.005 at residual 5-7e-9 = 5x the
      floor), isolated readings up to 1.012 at residual 1e-6 - 1.5e-5; production 0 of 946 > 1.0005
    - quote: |
        Evaluated only on a mixed call whose window columns were all formed at mixed calls, while
        max(1e-10, 1000 tau) <= residual <= 1e-2; an ineligible call resets the count.
    - rejected: evaluating on plain calls (radii 12.6-56 from near-collinear plain differences, false
      "unstable" in 5 of 12 rev-0 runs); a stricter pseudo-inverse truncation
    - why: below ~1000 tau the window differences are inner-solve noise

    ### `march_to_steady(accelerate=)` defaults to True — the pre-registered D11 rule, measured
    - area: flow
    - source: flow doc/steady_acceleration.md D11, §1.3; log WO-5 G2
    - decided: 2026-10-02 (by the user's pre-registered rule, Q1 / Q13)
    - status: settled
    - quote: |
        Staggered dense bed (phi 0.6, N 64), production, window 5, wall time incl. the accelerator:
        plain / accelerated = 3.38-3.45x on host-openmp, 3.55-3.87x on CUDA (>= 1.5) -> True.
        Collocated bed beside it: 2.48-2.53x / 2.38-2.73x.
    - rejected: False (opt-in acceleration)
    - why: the rule's threshold is met by more than 2x on both backends

    ### (core.md) AndersonCore carries no pressure metric: roles Velocity / Carried, `innerTolerance` (rev 1, WO-3b)
    - area: core
    - source: flow doc/steady_acceleration.md "Revision 1", §9 WO-3b, Q14; core-anderson 42fca87, 9ff3bd2
    - decided: 2026-10-02
    - status: settled (Q14 default yes; implemented on core branch anderson, not yet tagged)
    - supersedes: the descriptor list "(state views + roles + mask + metric weight + gauge flag + comm)"
      in "Anderson acceleration (AndersonCore) lives in core…"
    - quote: |
        AndersonState = padded state views + roles (Velocity = 0, Carried = 2) + extent + ghost +
        innerTolerance + AndersonComm. The Pressure role, sdf, cP, gauged and pass 1 are deleted.
    - rejected: keeping the unused pressure metric until after the tag (a breaking change then)
    - why: no consumer uses it after rev 1; removes one collective per step and the sdf dependency

---

## 2026-10-02 — WO-5 addendum: the host runs that finished after the WO-5 commit

**G1 tight, host-openmp (8 threads)** — the same verdicts as CUDA:

| case | plain | m = 5 | \|K_acc/K_plain − 1\| | note |
|---|---|---|---|---|
| §11 coll N14 | 2105 | 64 | — | "unstable" at 64 (max reading 1.00622) — **FAIL** |
| §11 coll N16 / N18 / N20 / N24 | 3520 / 2935 / 3715 / 2470 | 141 / 131 / 186 / 151 | 4.9e-10 / 4.0e-10 / 6.4e-10 / 6.7e-10 | pass |
| §11 stag N14 / N16 / N18 | 220 / 295 / 1165 | 60 / 79 / 130 | 7.0e-12 / 1.8e-11 / 1.3e-10 | pass (N18 reading 1.0111) |
| §11 stag N20 | 6550 | 51 | — | "unstable" at 51 (max reading 1.00264) — **FAIL** |
| §11 stag N24 | 3465 | 270 (CUDA 207) | 7.1e-10 | pass (reading 1.0079) |
| Z&H 0.343 / 0.45 N32 | 2530 / 5645 | 190 / 333 (CUDA 174 / 396) | 4.2e-10 / 1.3e-9 | pass (readings 1.0030 / 1.0010) |
| bed stag | 19510 | 536 (a376 c40 p120) | **1.44e-8** | **FAIL** (the plain certificate; WO-5) |
| bed coll | 1835 | 321 | 1.75e-9 | pass |

Accelerated step counts at tight settings differ between backends where phase A runs at the
noise floor (stag N24 270 vs 207, Z&H 190/333 vs 174/396); production counts are identical.

**G6, host:** the A1 array at Re ≈ 10: 8015 / 449 steps, K identical to CUDA to 10 digits
(12.1870408213 / 12.1870408616); wall 3326.6 s / 159.4 s = 20.9×. Re ≈ 100: 2775 / 2630
(a300 c15 p2315, disabled), K 27.4359507844 / …7845; wall 1216 s / 1130 s = 1.08×.

**G8, host, late-march protocol** (`--warmup 150 --steps 50`, two repetitions):

| scheme | plain step | step under acceleration | accelerator | % of plain step | bar ≤ 8 % |
|---|---|---|---|---|---|
| staggered | 74.0 / 79.2 ms | 57.9 / 59.1 ms | 4.10 / 4.23 ms | 5.53 / 5.34 % | pass |
| collocated | 374.1 / 381.3 ms | 80.1 / 79.8 ms | 3.79 / 4.12 ms | 1.01 / 1.08 % | pass |

`memory_bytes` = formula, both. The step under acceleration is cheaper than the plain step timed
at steps 151–200 (collocated 4.7×): the inner solves cost less near the fixed point, so "% of the
plain step" depends on where in the march the plain step is timed (relative to the step under
acceleration: 7.1–7.2 % staggered, 4.7–5.2 % collocated).
