# Architect brief — the variable-density multi-marker VoF step must match or beat TBFsolver on CPU and exploit the GPU

## 1. The question
What changes to peclet.flow's variable-density step (pressure projection first, then the block
VoF container and the host kernel-launch structure) bring the bubble-column case to **≤ TBFsolver's
cost per step on the same CPU cores, and several times faster on one GPU** — without changing the
numerics of the variable-coefficient projection the register has settled? Deliver a ranked design
with expected savings per item, what each costs to build, and the gates that prove it inert.

## 2. Why the architect
USER DIRECTIVE (2026-09-25, standing suite directive): peclet must be on par with or better than
SOTA codes on every case it can handle; GPU↔host transfers inside the step are not acceptable.
The measured gap is structural (solver architecture, MG launch structure, container design), not
a local bug — the local defects are already fixed (below).

## 3. Measured state (all numbers from real runs, 2026-09-25)
Case: `peclet-examples/benchmarks/bubble-column` — 128×96×64 cubic cells (D/h = 16), 16 VoF block
markers (Weymouth–Yue per marker on moving boxes, union colour max_k C_k, per-marker CSF summed),
ρ_g/ρ_l = μ_g/μ_l = 0.02, walls in y, periodic x/z, zero net flux (`set_superficial_velocity`),
MG-PCG pressure at rtol 1e-10 (13–15 iterations), dt 1.59e-3 (0.25 × capillary limit).
TBFsolver (Cifani et al. 2018) on the same case: dt ≈ 1.53e-3 on average, so cost per step ≈ cost
per simulated time.

**Snellius genoa, same node (tcn538, 2× EPYC 9654), same 24 cores, flow main 7fdec0d, ms/step:**

| layout | TBFsolver | peclet |
|---|---|---|
| 8 MPI × 3 OMP | 45–46 | 199–203 |
| 1 × 24 OMP | — | 186–189 |
| 64 × 3 (192 cores) | 25–33 | 76–81 |

peclet 1×24 per stage: step 187, **projection 125.2 (67 %)**, momentum solve 23.4, curvature 14.4,
block advect 13.2, predictor 3.5, debris 1.9, csf 0.7. The VoF stages (~30 ms) alone ≈ 2/3 of
TBFsolver's whole step. Workstation host profile (1×8 threads, 344.6 ms/step): top kernels per step
`cc_smooth` (RB-GS) 51.0 ms / 312 launches, `mg_pfill3` (periodic ghost fill) 35.2 ms / 416 launches
(~85 µs each, launch-bound on coarse levels), `ibm_rbgs` 25.3, `cc_residual` 22.6, `prolong` 21.7,
mean removal (`mgmeans`+`mgmeanr`) 19.6 / 56 launches, `vmg_resid` 15.1, `pv_batch` 13.8 / 1. A
Kokkos MDRange 7-point stencil (x-fastest, default tile (nx,2,2)) costs 1.32 ms vs a hand flat x
loop 0.56 ms at 1 thread (2.4×); the host Kokkos prefix is built without `-march`.

**GPU (RTX 5080, quiet), ms/step:** MG-PCG 54.1 (Chebyshev 104.1): projection 27.8 of which the
**host GraphAMG bottom solve ≈ 15.5** (default depth 4 leaves a 16×12×8 bottom; `auto` bottom
agglomerates it and rebuilds a host PCG-AMG every step under variable ρ; 44 D→H + 44 H→D copies +
7 operator mirrors per step — the only bulk transfers left); curvature 13.0 (the tier-3 PV
paraboloid fallback, one launch now, ~10 ms, latency-bound: one thread fits a 5³ paraboloid per
cell and ~35 % of interfacial cells fall back); block advect 8.0 (~5 ms per-block serialization:
per-block WY compaction scans returning counts to host, ~8 host-result statistics reductions per
block); momentum 3.9. 474 scalar reductions/step still return to the host (Krylov dots, block
stats). Chebyshev: 30 of its 44 V-cycles/step are spectral re-estimation (variable ρ invalidates
the bounds each step).

TBFsolver's step (read from its source, `poissonEqn/poissonEqn.f90:200-250`, `main.f90:194-261`):
constant-coefficient pressure with ρ0 = min(ρ_l, ρ_g) and the old pressure extrapolated (nl_ = 2
levels; Dodd & Ferrante 2014, Cifani 2019 JCP "Analysis of a constant-coefficient pressure equation
method…"), solved directly by FFT(x,z) + tridiagonal(y) (FAST_MODE), ~7 ms/solve on the
workstation. Per-step RK sub-stages (alphaRKS) — check whether it solves more than one Poisson
problem per step.

## 4. Constraints and invariants
- Staggered MAC, x-fastest storage and iteration (`src/policy.hpp`), Kokkos (CUDA/HIP/OpenMP),
  every method on device and MPI-distributable; host paths only as oracles.
- The collocated projection is ABC, NEVER Rhie–Chow (suite register).
- Must remain general: IBM cut cells / SDF solids, porous, arbitrary factor-of-two structure,
  MPI telescoping — a fast path for a special geometry is allowed only as a selected driver with
  the general path intact.
- A structural change is proved bitwise at the old configuration; a numerics change is a
  recorded decision with gates.

## 5. Already decided — not open
- **Dodd–Ferrante constant-coefficient splitting is REJECTED twice in the register**
  (`docs/decisions/flow.md`: "Part III bubbly-flow container reuses V0–V4 kernels; Dodd–Ferrante
  FFT not needed" — "varRho MG projection is stronger"; "S-ladder plan; Dodd–Ferrante splitting
  rejected for the pressure driver" — "error ~ σκ = the dominant field at pore scale"). You may
  analyse a scoped fast path (e.g. solid-free periodic/walled boxes only, as an opt-in driver)
  with its accuracy cost quantified, but recommending it reverses a settled decision and must be
  framed as such for the user.
- MG-PCG is the terminal fallback; Chebyshev is the variable-density default because it has no
  global dot products per iteration (multi-rank scaling). The case uses MG-PCG on one rank.
- The S-ladder (S2 "Chebyshev-bound amortization at capillary dt") is prior planned work.
- The block VoF container's design (one WY field per marker on a moving box, union MAX colour,
  per-marker CSF SUM) is settled (`doc/vof_overlap_design.md`); batching its execution is open.

## 6. Genuinely open — decide these
1. The pressure path: how to get the variable-coefficient projection from ~125 ms (24 cores) /
   ~28 ms (GPU) to ≤ ~20 ms / ≤ ~8 ms: device-resident coarse solve (replace the host GraphAMG
   bottom), deeper/cheaper hierarchy for grids with a factor 3, fewer launches (fused ghost fill +
   smoother, level batching), Chebyshev bound reuse, iteration count vs rtol (1e-10 may be stricter
   than the VoF needs — what tolerance does the balanced-force CSF actually require?).
2. The host launch structure: MDRange tile-loop overhead (2.4×) and per-kernel OpenMP region cost —
   fused loops, flat x loops, team policies; what the design rule should be.
3. The block container: batching all markers in one launch per stage (compaction, statistics on
   device, curvature fallback warp-per-cell — note `pvFitAdd` lives in core, tagged first).
4. The expected end state per stage, CPU 24 cores and GPU, vs TBFsolver.

## 7. Already tried, with evidence
- Fixed (bitwise, landed flow 7fdec0d): 16 serial PV-fallback launches → 1 (curvature 90 → 13 ms
  GPU); on-device zero-flux constraint (host round trip removed); persistent Chebyshev vectors (8
  cudaMalloc/step → 0); MG periodic fill 3 kernels → 1; 45 MG deep_copies without fences; block
  debris sums one launch; 14 fences dropped; MDRange x-fastest everywhere (host 7.5× per stencil,
  1×8 step 471 → 399 ms).
- Depth 6 instead of 4 removes the host bottom but raises MG-PCG iterations 13 → 21.
- Chebyshev vs MG-PCG on one GPU: 104 vs 54 ms/step (re-estimation dominates).

## 8. Verification
`ctest -LE bench` (175), `tests/state_hash.py` (12 cases + np2, bitwise for structural changes),
the bubble-column 50-step bitwise dump from `scratchpad/perf/ckpt_t43.npz`, MPI np-independence
tests, and the Snellius same-node comparison script `/projects/0/prjs1022/peclet/bubble-cpu/
bubble_cpu.slurm` (reruns in 15 min). Hysing rising bubble and the static-drop parasitic-current
gates for any numerics change.

## 9. Deliverable
`flow/doc/vof_step_performance_design.md`: ranked items with expected ms saved (CPU 24 cores and
GPU), build cost, risk, inert-proof gates, and work orders an Opus implementer can execute; a
separate section framing any register reversal for the user's decision.

## 10. Out of scope
The physics difference between the codes (a separate investigation: peclet's bubbles collect at
one wall, TBFsolver's stay homogeneous); AMR; multi-node scaling beyond one node.
