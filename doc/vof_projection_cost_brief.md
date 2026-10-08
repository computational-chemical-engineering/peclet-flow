# Architect brief — D (+ C, E, F, H): the cost of the variable-density projection and momentum solve on the CPU

Date 2026-10-08. USER DIRECTIVE 2026-09-25: peclet on par with or better than SOTA on every case it
handles, no GPU↔host transfers in the step; yardstick TBFsolver on the bubble column (46 ms/step on
the same genoa 24 cores; peclet 93.1 ms at 1×24, flow 38e80e6). USER DECISIONS 2026-10-08: E2(b)
FFT pressure solver DROPPED (register); A(b) host bottom factor done (bitwise, −~15 ms expected, being
confirmed on genoa); "design D and G, etc." — G (curvature) is a separate parallel brief
(`doc/vof_curvature_cost_brief.md`).

## 1. The question
Design the next package that cuts the projection (55.3 ms/step at 1×24 genoa, of which the bottom
factor 16.9 is being removed by A(b)) and the momentum solve (10.6 ms) — centrally **D, an FP32
V-cycle preconditioner inside the FP64 PCG (design §10 "E1"/§14 "U-2")** — plus the smaller items C,
E, F, H below. For each: estimated saving (host 1×24 genoa and GPU RTX 5080), numerics class
(bitwise / recorded change / accuracy-relevant), risk, gates. Deliver a ranked design note with work
orders.

## 2. Why the architect
D re-opens the precision question that cost the project a silent-invalidity failure (float operator
storage broke A·1 = 0 at high contrast; SCALING_ISSUES #1; double is now default). The register
permits float *preconditioners below an exact double Krylov* but nobody has designed or gated one
for the pressure V-cycle. E (a non-zero initial guess) collides with a registered divergence. Both
need first-principles analysis, not tuning.

## 3. Current state (recon of flow main f19ac4b; anchors are `src/…:line`, verify)
- Types: `MReal` = double by default (`CMakeLists.txt:93`, `mac_cutcell_mg.hpp:74-81`); operator
  arrays per level `FPV AC, AFX, AFY, AFZ` (Level struct `:874-895`); ALL vectors (x, rhs, res,
  openness, Krylov) are `CCField = View<double*>` (`mac_cutcell.hpp:22-25`). No float vector path
  exists. Guard ctest `no_float_operator_casts` (regex `(float)`, `static_cast<float>`,
  `View<float*`; `// PRECISION-EXEMPT:` lines skipped) — it does NOT see `BottomDirect<float>` or
  `template <class FR = float>`: the FP32 bottom factor (production since 2026-10-03, preconditions an
  FP64 FCG at the bottom) is outside the guard.
- Single-rank resident MG-PCG `solvePCGResident` (`mac_cutcell_mg.hpp:1919-2052`): r = b − A x,
  removeMean(r), r0 = max|r|; z = M r (one symmetric V-cycle `precondVcycle` :2525, which ZEROES z
  every call :2538), p = z; loop { Ap = A p (exact double operator, `matvecOverlap`); if
  meanRemovalAll_ removeMean(Ap); dot p·Ap ("mgdot"); update x, r ("mgpcg_update"); removeMean(r)
  ("mgmeanr"/"mgmeans"); max|r| ("mgmax"); ONE 4-double device→host packet read (:2000); stop if
  rn < rtol·r0 (:2017); z = M r; dot r·z; p = z + βp ("mgaypx") }. Krylov scalars live on device.
  Stop is relative to the INITIAL residual of the supplied x.
- V-cycle `vcycleImpl` (:2563-2670): RB-GS smoother `cutcellSmoothFaceCell` (`mac_pressure.hpp:137`:
  phi(i) = (b − Σ AF·phi_nb)/AC, operator reads cast to double), (pre, post, bottom) = (2, 2, 12) from
  `flow_ibm_project.hpp:1507`; residual (wrap variant on single rank); `restrictAvgZeroX`;
  `prolongAdd`; removeMean at L == 0 exit. Coarse operators are REDISCRETIZED from averaged openness
  (`coarsenOpenAvg`, `buildCutcellOpFace`), not Galerkin. Bottom: agglomerated → `'direct'` (FP32
  block-tridiagonal factor + FP64 FCG) on single rank, GraphAMG distributed.
- Variable density REBUILDS every level's operator every step (`projectBuildCoefficients`
  `flow_ibm_project.hpp:1280-1365` → `buildRhoCoeff` → `mg_.setOpenness` → per-level
  `buildCutcellOpFace` + `facStale_ = true`).
- Mean-removal scope: ALREADY `'fine'` by default (`meanRemovalAll_ = false` :4122; register
  "Fine-scope MG-PCG promoted to default", 2026-08-09). The 44 mean-removal launches/step at 1×24
  are the per-iteration residual projection and the L0 V-cycle exit — so the orchestrator's earlier
  item "B" (switch to fine) is void; only fusing them (H-5 iii) remains.
- Initial guess: ZERO every step (`pwarm_ = false`, `projectSolve` :1371-1372). No extrapolation
  anywhere. `set_pressure_warmstart(True)` exists but is registered as DIVERGENT on the steady Stokes
  march (192³ bed, k → −1.7e120 by step 400; register flow.md:2714) — design §5 D4/Q11 hypothesis:
  the r0-relative stop under-solves whenever the warm guess is worse than zero; root cause never found.
- Momentum (staggered VoF): velocity MG `vmg_.solve` (`mac_velocity_mg.hpp:734-822`), stop
  max|r| ≤ resTol·max(max|b|, max|Au|) with max-norm reductions `vmg_maxabs`/`vmg_maxabsdiff` and a
  separate residual kernel each V-cycle; tolerance follows the pressure rtol (register 2026-09-02).
- Possible defects found by recon (fact, unverified — check and report, do not fix in the design):
  (i) the resident MG-PCG ignores `stopRef_` (the balanced-force "relative to full rhs" stop) that
  the host PCG, FCG and Chebyshev honour (:1754, :2098, :3863 vs :2017) — matters for collocated
  var-ρ (BFP default ON there), not the staggered bubble column; (ii) defaults disagree: member
  `pcgRtol_ = 1e-10`, `pcgMaxit_ = 500` vs the binding `set_pressure_pcg(max_iter=200, rtol=1e-8)`.

Measured (genoa 1×24 contiguous, `'direct'`, rtol 1e-8, 30-step kernel timer, S-1 job 27770799;
10.19 PCG iterations/step; step 93.7): projection 55.3 (mg_bottom_factor 16.9, cc_smooth 12.4 / 240
launches, mgmeanr+mgmeans 5.6 / 44, mgdot 3.5 / 20, prolong 2.7 / 30, cc_residual 1.5, cc_apply_exact
1.4, restrict 0.9, mgpcg_update 0.8, mg_bottom_direct 1.9 / 10, operator rebuild: rho_coeff 0.5,
cc_build_op 0.5, rhs_var 1.8, correct_var 0.45); momentum 10.6 (vmg_resid 3.6, ibm_build_diff_var
3.5, ibm_rbgs 2.9, bc_vel 2.3 / 47, vmg_maxabs + maxabsdiff 1.1); host deep copies 1.8 / 11 and
zero-memsets 1.2 / 62 (sites: `precondVcycle` z = 0 per iteration; `block_exchange.hpp:253` per VoF
block; Krylov/VMG staging copies `:1929/:1937/:1949/:2048/:2050`, `mac_velocity_mg.hpp:741-742`).
Spread placement (12 CCDs) is 10 % faster than contiguous → the step is memory-bandwidth bound.
Raw: `~/Codes/bubble_column_perf/s1/summary_kprof.txt`, `summary_main.txt`.

## 4. Constraints and invariants
- Register "defect-correction rule" (flow.md:2251, settled): the Krylov matvec and residual are the
  EXACT double operator in flux form (A·1 = 0 bitwise); V-cycles/smoothers/AMG below are
  preconditioners and MAY stay float. Register "precision policy rule" (flow.md:1798):
  identity-bearing quantities are stored in the precision the identity is asserted. Register 3579
  (device bottom): A·1 = 0, the mean projection and the stopping residual stay FP64; an UNREFINED
  FP32 bottom inside the FP64 V-cycle was rejected (§13.3 row e: error ≈ ε32·κ unverified, grows
  silently with contrast). Operator storage stays double by default (flow.md:2759).
- Success criterion for any precision change (flow.md:2161): match the full-double attainable
  residual floor, iteration parity, never "reaches rtol 1e-8" alone. Gate battery named by design
  §10: `tests/study/vardensity_solver_probe.py` (ratio 10³ walled, packing_ring 10⁴).
- High coefficient contrast makes the V-cycle preconditioner indefinite above ~10³ (CG drivers cap;
  Chebyshev healthy). The bubble column's density ratio: check the case
  (`peclet-examples/benchmarks/bubble-column/scripts/run_peclet.py`).
- Device and host paths: one Kokkos source; host-only branches by `kHostMemory` / HostSpace; device
  bits unchanged by host work unless recorded. `-ffp-contract=off` on host. No env knobs that change
  numbers; setters on `s.diagnostics` for ablations. MPI np-parity (np = 1 bitwise gate for every
  distributed default).
- Physical case rtol is 1e-8 (USER, register 2026-10-03) — NOT open.

## 5. Already decided, not open
ABC/MAC projection; PCG (Krylov) not RB-GS for cut-cell; rotational incremental pressure update;
fine-scope mean removal default; momentum tol follows pressure rtol; no FFT/constant-coefficient
driver (E2(a) parked, E2(b) dropped); case rtol 1e-8; host `'direct'` bottom; A(b) done.
**Open (yours):** D in full — what is stored in FP32 (operators per level? vectors inside the
V-cycle? both?), on which levels, arithmetic precision, PCG vs FCG with a non-exactly-symmetric
preconditioner, how A·1 = 0 / the null space is protected inside an FP32 V-cycle, expected bytes and
time saved per kernel on host and GPU, failure modes and the gate that would catch each; and whether
D is the default or an opt-in. Plus:
- **C** = §14 H-5 Krylov reduction fusion (designed already in §14.3; confirm or amend; it is bitwise).
- **E** = a better initial guess (warm start / extrapolation p̂ = 2φⁿ − φⁿ⁻¹ or similar, with a stop
  that cannot under-solve, e.g. relative to |b| or to the zero-guess residual): first resolve or
  bound the registered warm-start divergence (Q11) from first principles; estimated saving ≈ 3.5 ms
  per iteration saved.
- **F** = momentum-solve fusion (max-norm inside the residual kernel, bc_vel launch merging) — bitwise.
- **H** = operator-rebuild and copy/zero elimination (the z = 0 per preconditioner call that the
  V-cycle overwrites anyway, staging copies, the three var-ρ builds) — bitwise.
- Whether the remaining launch overhead (937 launches/step) warrants F4 after all.

## 6. Already tried / rejected (with evidence)
- Float operator storage as the default: silently breaks A·1 = 0 at high contrast (dense beds
  invalid) → double default (SCALING_ISSUES #1, flow.md:2759). The double-DIAGONAL fallback: 65×
  worse on divergence, retired.
- Defect-correction outer loop with a float hierarchy (flow.md:597): superseded by the rule above.
- Unrefined FP32 bottom inside the FP64 V-cycle: rejected (§13.3 e); the shipped FP32 factor sits
  INSIDE an FP64 FCG.
- `set_pressure_warmstart(True)` as default: diverges on the steady Stokes march (flow.md:2714).
- Dodd–Ferrante constant-coefficient splitting: fails static balance ~5000× (parked); FFT dropped.
- Chebyshev smoother / GraphAMG GPU bottom lost to fine-scope MG-PCG (flow.md:912).
- §14 H-0…H-4 landed (f19ac4b); model said 69 ms, measured 93.1 — the miss is mostly the bottom
  factor (now A(b)) and H-5 not yet in.

## 7. Verification
Bubble-column checkpoint `~/Codes/bubble_column_perf/ckpt_t43.npz` with `tests/study/vof_perf/
prof.py` / `run_mpi.py --dump`; G-BIT scripts in `~/Codes/bubble_column_perf/` (WARNING: their
`rc $?` after `$(basename …)` always prints 0 — gate on real exit status and rebuilt-module
timestamps); state_hash; battery 231 `-LE bench`; `vardensity_solver_probe.py`; static-drop /
Hysing study gates (`tests/study/vof_surface_tension.py`); bubble-column statistics vs the published
rtol-1e-8 run; genoa timing via the S-1 scripts `~/Codes/bubble_column_perf/s1/`; RTX 5080 timing.

## 8. Deliverable
`flow/doc/vof_projection_cost_design.md`, committed and pushed to flow main (USER DIRECTIVE: no PR;
shared checkout — `git pull --rebase`, commit only your file with a pathspec). Sections: measured
breakdown and byte model; D design (with the null-space/A·1 = 0 argument); E analysis (incl. Q11);
C/F/H confirmations; ranked table (saving host/GPU, numerics class, risk); work orders for an
`opus-implementer` with gates; register entries to add; open questions with defaults; and an honest
end-state estimate vs TBFsolver's 46 ms combining A(b), this package and G's estimate if you can.

## 9. Out of scope
Curvature (G brief); distributed H-6/H-7 (already designed in §14); the case tolerance; FFT /
constant-coefficient drivers; implementing anything.
