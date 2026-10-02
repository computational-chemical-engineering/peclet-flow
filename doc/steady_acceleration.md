# Anderson acceleration of steady marches — design note

*2026-10-02, architect pass on `doc/steady_acceleration_brief.md`. Status: **proposed**, not
implemented. The decisions in §1 go to `../docs/decisions/flow.md` once approved. An implementer
should not need to choose anything this note leaves open. Where a choice depends on a measurement,
§10 states the default to use and the experiment that would change it.*

---

## 1. Decision summary (register style)

Each entry gives the decision, the alternative rejected and the reason. Evidence is in §1.1.

**D1. Placement: the data path is C++/Kokkos inside flow; the control path is a small pure-Python
driver.** The data path covers the state, the history, the reductions and the safeguards, in
`AndersonCore` + `AndersonAccelerator<Grid>`. The control path covers the phases and the §3.2 stop
instrument, in `peclet.flow.march_to_steady`.
- *Rejected:* a pure-Python driver over `diagnostics.field_view`. Its device path would be
  CuPy-only, so it would not run on the HIP or OpenMP backends that the Kokkos source supports. It
  cannot see the collocated face field `uf_` (that field is not in the registry). It cannot enforce
  refusals from the solver's own configuration flags. Every layout fact would be hard-coded twice.
- *Rejected:* a C++ march loop that includes the stop instrument. The instrument would then exist
  twice (C++ and the study scripts), and per-step Python hooks (logging, checkpoints) would be
  lost.
- *Reason:* everything that touches fields stays on the device and runs on every backend and under
  MPI. Everything that is policy stays readable and user-adjustable.

**D2. The state vector is exactly what `step()` reads across steps.** It is the full padded buffers
of the velocity (`C[0..2].u`) and the accumulated pressure `P_`. For `SolverColocated` with
projected-face advection on, it also includes the face field `uf_/vf_/wf_`.
- *Rejected:* a velocity-only state. Measured, it stalls at 1.9e-4 error, because the slow
  checkerboard lives in P (§1.1).
- *Rejected:* adding derived fields (u*, φ). They are recomputed every step, so they are not state.
- *Reason:* Anderson accelerates a map x ↦ g(x). If g reads anything outside x, the map Anderson
  sees depends on history, and its secant model is wrong.

**D3. The metric is velocity at unit weight plus pressure weighted by c_P.** The pressure term is
c_P² times the fluid-centred, gauge-centred pressure. The face field is mixed but not measured.
- *Rejected:* the unweighted Euclidean norm, which depends on the unit system.
- *Rejected:* the Chorin scale Δt/(ρh), which grows without bound as Δt → ∞ and lets the pressure
  swamp the metric.
- *Rejected:* a metric on the face gradient of P. It would need ghost-consistent history and would
  depend on the partition.
- *Reason:* c_P = h/(μ + ρh²/Δt) is the velocity response of the momentum operator to a pressure
  jump across one face. It interpolates between Chorin's scale (D → 0) and the Stokes scale
  (D → ∞), so u and P are commensurate at every Δt (§3).

**D4. Algorithm: type-II Anderson.** Window m = 5 (compile-time cap 8) and mixing β = 1. The least
squares is solved by normal equations through a Jacobi-scaled eigendecomposition. The oldest column
is dropped while the scaled Gram condition exceeds 1e12, i.e. κ(ΔR) > 1e6.
- *Rejected:* QR or TSQR. It costs m sequential reductions or a TSQR tree, and at κ ≤ 1e6 it buys
  nothing because γ only steers the path, not the fixed point.
- *Rejected:* m = 2, which was measured insufficient on the collocated case (§1.1).
- *Rejected:* β < 1 as the default. It slows the fast modes, and the plain step is already the
  scheme's own stable preconditioner.

**D5. Safeguards.**
- Mixing engages after two consecutive residual decreases.
- History restarts when the residual exceeds 4× the minimum since the last restart (applied only
  above the noise floor 1e-10).
- Acceleration is disabled after 5 restarts.
- A non-finite residual or a failed pressure solve at a mixed iterate restores the last plain-map
  output and disables acceleration.
- An instability guard watches the spectral radius of the window's Ritz matrix. If it exceeds
  1 + 1e-3 on 3 consecutive evaluations while the residual is in [1e-10, 1e-2], the status becomes
  "unstable" and the march returns not-converged.
- *Rejected:* a tight bound on the mixing coefficients Σ|α|. Extrapolating the 0.996/step
  checkerboard needs Σ|α| ≈ 2/(1−λ) ≈ 500; a tight bound forbids the acceleration being sought.
- *Rejected:* no instability guard. Measured: Anderson keeps marching where the plain march at the
  same Δt diverges. Anderson, like GMRES, can converge to fixed points that the plain march cannot
  reach (§1.1, §2.4).

**D6. Stop interplay: two phases.**
- Accelerate until the relative residual reaches (1 − slow_rate)·rtol.
- Then certify with the **unchanged §3.2 instrument on consecutive plain steps**, within a budget of
  num_passes + 3 blocks.
- If the budget is exhausted without a pass, tighten the target ×0.1 and resume acceleration.
- If the plain steps grow, disable acceleration and finish as the plain march would.
- *Rejected:* applying the §3.2 rule to the accelerated sequence. Anderson iterates are not plain
  steps. A stagnating Anderson sequence has small ⟨u_x⟩ changes far from the fixed point, which
  means false stops.
- *Rejected:* a residual-norm-only stop. Its error bound would need a non-normal bound on
  (I − J)⁻¹ that nobody has verified, and it abandons the instrument the user chose.

**D7. MPI: bit-exact rank-count independence is not required.**
- np = 1 must be bit-identical to the serial build. np > 1 must agree to tolerance.
- Each step does one `MPI_Allreduce` of the sum packet (≤ 5m+1 doubles). A gauged pressure adds a
  3-double pre-pass, and the inner state check adds a 1-double count.
- Rank 0 broadcasts the decision packet. **Identical γ on every rank is a correctness requirement:**
  ghosts are mixed locally, so they stay consistent only if all ranks use the same coefficients.
- *Rejected:* reproducible (binned or integer) summation for rank-count bit-exactness. It needs a
  custom exact accumulator, and the existing Krylov path is already only tolerance-exact at np > 1
  (`tests/kokkos_mpi/test_sdflow_mpi.cpp:125`: "np=1 bit-exact; np>1 the MG-PCG reduction-order
  floor").

**D8. Memory: double precision, full padded buffers, 2m+3 state vectors** (history 2m, plus X, R_prev
and G_prev).
- *Rejected:* float history. Halving memory is not needed below about 7.5 M cells per 16 GB card,
  and the suite's precision policy argues against it (open question Q8).
- *Rejected:* storing inner entries only. That loses state held in ghost indices (the high-side
  outflow face) and needs extra ghost-refresh logic.
- *Reason:* a linear combination of padded buffers is exactly the linear combination of the states.

**D9. Checkpoint: the Anderson history is not checkpointed.** A restart rebuilds it in m + 3 steps.
The checkpoint is `get_field`/`set_field` of `u, v, w, p`, not `set_state`, which restores velocity
only and loses P.
- *Rejected:* checkpointing the history. It makes a checkpoint about 14× larger (2m + 3 = 13 extra
  state vectors on top of the one state) and adds nothing to correctness.

**D10. Scope.**
- Supported: staggered `Solver`, and `SolverColocated` with the fluid-only `"ghost"` scheme. Stokes
  or finite Re, periodic or domain BCs.
- Refused, with a named error: gauge-exact, plain and embed collocated schemes; VoF; phase change;
  transported scalars; porous continuity; variable ρ/μ closures; `drag_beta`; cell forces (CFD-DEM);
  moving scene instances; superficial velocity; pressure warm start; the balanced-force projection.
- *Rejected:* allowing gauge-exact. Its fixed points form an attractor family
  (`collocated_invisible_subspace.md` §4), and Anderson may select a different member than the
  march would, with a spread of 5e-4 in k.

**D11. API.**
- Public: `peclet.flow.march_to_steady(solver, monitor, rtol=1e-4, …)`, returning a `MarchResult`.
- Developer tier: `s.diagnostics.anderson_accelerator(window=5, mixing=1.0)`.
- `Solver.step()` is untouched, so the unaccelerated path stays bit-identical.
- `accelerate=True` is the default of the new function only.
- *Rejected:* an `enable_*` switch that changes what `step()` does, which would change an existing
  entry point.
- *Rejected:* an environment variable (QUALITY_PLAN D3).

**D12. Finite Re: same design, with smaller gains.**
- Anderson stays appropriate while the plain march contracts. With explicit advection, Δt is
  CFL-bound and the slow spectrum is a continuum near 1, not a few outliers, so the gain falls to
  about 2.5× (measured).
- Unsteady or march-unstable regimes are handled by the D5 guard plus the certification-growth
  fallback (§7).

### 1.1 Evidence: a throwaway prototype (2026-10-02, measured)

**What was run:**
- A NumPy prototype of §4 without the safeguards, run on host build
  `../flow-vof-b1/build_omp` (OpenMP, 6–8 threads, an existing VoF-branch build; the single-phase
  step is assumed to equal main's). Script: scratchpad `aa_proto.py`, which is ephemeral; WO-1
  rewrites it.
- Case: §11 of `collocated_invisible_subspace.md`. One sphere per periodic unit cell, φ = 0.125,
  ρ = μ = F = 1, νΔt/h² = 6, N = 16, pressure PCG rtol 1e-10, 200 velocity sweeps.
- State: (u, v, w, p) through `diagnostics.field_view` padded buffers. Metric as in §3, with
  c_P = 1/(1+D) = 1/7.
- "Steps to tol" is the first step after which |⟨u_x⟩/U∞ − 1| stays below tol. U∞ comes from a
  long plain march.

**Steps to tolerance:**

| case | plain 1e-4 / 1e-6 / 1e-8 | Anderson m=5 | m=3 | m=8 | m=2 |
|---|---|---|---|---|---|
| collocated ghost, Stokes | 315 / 1522 / 2652 | **27 / 55 / 77** | 30 / 58 / 96 | 28 / 43 / 78 | 56 / >120 / — |
| staggered, Stokes | 22 / 77 / 165 | **11 / 20 / 32** | 12 / 20 / 35 | — | — |
| staggered, Re ≈ 10 (μ = 0.05, Koren, dt = 3.906e-2, D = 0.5) | 262 / 392 / 524 | 105 / 181 / >200 | — | — | — |
| collocated, Stokes, **velocity-only state** | 315 / … | stalls: error 1.9e-4 at 150 steps | | | |

**Fixed point:**
- Staggered: after 120 Anderson steps, the error against the 2000-step plain reference is 3.3e-16.
- Collocated: 3.6e-9 against a 3000-step plain reference. That reference is itself about 2e-9
  short: its last-step change is 9.2e-12 at a rate of 0.996, so about 2.3e-9 remains.

**Ritz estimate ρ(I+M) on the window (§4.6):**
- Collocated: converges to **0.99616**, which matches the measured checkerboard tail rate 0.9962 at
  N = 16. The window captures the slow mode.
- Staggered: ≤ 0.954, except one spike of 1.09 at relative residual < 5e-12. That is round-off,
  hence the noise floor in D5.
- Re ≈ 10: ≤ 0.966.

**A march-unstable case** (staggered, μ = 0.0158, dt = 1.234e-2, which is CFL 0.5 on the Stokes
velocity estimate):
- The **plain march diverges** to NaN at step about 435; max|u| was 3.57 at step 400.
- Anderson is still bounded at step 900 (⟨u⟩ = 2.018, residual 1.45e-4, slowly converging). Its Ritz
  radius stayed at ≈ 0.992, with one non-consecutive spike of 1.013.
- So Anderson carries the march where the plain map at that Δt cannot go. §2.4 and §7 handle this
  case.

**Cost per step** (prototype, collocated N = 16, 6 threads): plain 17.2 ms/step over its first 315
steps, with 9.2 pressure iterations per step. The prototype runs 25.7 ms/step, but that includes
host copies and Python-loop Gram products, which the C++ design does not have (§6.4). Even so, wall
time to 1e-4 error is 5.4 s plain against 0.69 s accelerated.

---

## 2. Fixed-point argument

### 2.1 What the accelerator can and cannot change

Let g be the step map the solver actually computes, including its inexact inner solves (momentum
residual stop, MG-PCG or BiCGStab to rtol). Its state is x = (u, P [, u_f]) as in D2.

The accelerator never modifies g. It only chooses the input to the next evaluation:

    x_{k+1} = Σ_i α_i [ β g(x_i) + (1−β) x_i ],      Σ_i α_i = 1.

Discrete operators, coefficients, defaults and inner tolerances are untouched. `step()` is called
unchanged.

### 2.2 The reported state is a plain-march state, certified by the plain-march instrument

The driver reports the state after a **run of consecutive plain steps** x_{n+1} = g(x_n). The run
starts from the accelerated iterate and passes the §3.2 instrument with its constants unchanged. So
the accelerated driver reports exactly what the unaccelerated driver reports: a state of the plain
march that passed the same acceptance test. The only difference is where that plain run starts.

By design, the accelerator applies its mix lazily, at the start of the *next* call (§4.2). Between
calls the solver therefore always holds a genuine map output, with every derived quantity
consistent: `uStar_` for the reaction force, the face field, and the `last_*` diagnostics.

### 2.3 Uniqueness: both marches go to the same point

For the supported configurations (D10), g has a unique fixed point modulo the pressure constant.

- **Exact solves.** By Prop. 1 of `collocated_invisible_subspace.md`, P-stationarity gives
  ((ρ/Δt)I + μA_p)φ = 0, hence φ = 0, D_αΠu = 0, and the Δt-free steady system −μLu = F − GP. For
  the staggered scheme and the collocated ghost scheme that system has a unique solution modulo
  gauge (C2; the ghost scheme drops the solid rows that create §4's family; staggered has
  ker G = constants).
- **Inexact solves.** Let the pressure solve return φ with |A_p φ + DΠu*| ≤ τ|DΠu*| for relative
  tolerance τ. Write A_pφ = −DΠu* + e with |e| ≤ τ|DΠu*|. P-stationarity gives
  (ρ/Δt)φ = μDΠu* = −μ(A_pφ − e), i.e. ((ρ/Δt)I + μA_p)φ = μe.
  Then |DΠu*| ≤ |A_pφ| + |e| ≤ C τ |DΠu*| with C = O(1 + ‖μA_p((ρ/Δt)I + μA_p)⁻¹‖) ≤ 2, so
  DΠu* = 0 and φ = 0 whenever τ < 1/2.
- **The momentum solve.** It is warm-started from u^n and always runs at least one sweep. At an
  exact steady state its initial residual is zero, and a stationary sweep leaves an exact solution
  unchanged (`collocated_invisible_subspace.md` §5).

So the inexact map's fixed point is the exact one. Every certified state (§2.2) lies within the
instrument's remainder bound of it, whichever path led there.

**Consequence.** |K_acc − K_plain| ≤ |K_acc − K*| + |K* − K_plain| ≤ 2·rtol·|K| at production
settings, and → 0 at a tight stop. Gate G1 tests the tight form at ≤ 1e-8.

**Gauge.** In a periodic (or wall-only) box, g(x + c·1_P) = g(x) + c·1_P. The constant P mode is
neutral: it affects neither u nor K nor the certificate. It is removed from the metric (§3) and
otherwise left alone.

### 2.4 What the argument does not cover (stability), and how the design handles it

Type-II Anderson on a linear map is GMRES-like on (I − J)x = c (Walker & Ni). It converges whenever
I − J is nonsingular, **including when ρ(J) > 1**. So Anderson can reach a fixed point that the
plain march at the same Δt never reaches. Two kinds exist:

- **Numerical:** explicit advection beyond its CFL limit. Measured in §1.1. The fixed point is the
  Δt-independent discrete steady state, so a plain march at a stable Δt would reach the same point.
- **Physical:** an unstable steady branch of an unsteady flow. The plain march never converges.

Three defences, in the order they act:

1. **The Ritz guard (D5).** Anderson can suppress a growing mode only if that mode is in its
   window, because the mode's growth dominates the secant differences. The Galerkin projection of
   J − I onto the window then has spectral radius ρ(I+M) > 1. Three consecutive detections end the
   march as not-converged ("unstable").
2. **Certification growth (§7).** If the plain steps grow from the accelerated state (residual
   ×10, or non-finite), acceleration is disabled and the plain march continues on its own terms.
   The outcome is then the unaccelerated outcome from that state.
3. **The residual risk is shared with the plain march.** An unstable mode that the transient never
   excites (it sits at round-off, for example by symmetry) is invisible to Anderson and to the
   plain march alike. The §3.2 instrument can stop the plain march on such a state too, so the
   answers agree.

**Out of scope, refused (D10):**
- Configurations whose fixed points are not unique, because Anderson would select among them
  differently from the march: gauge-exact, plain and embed collocated schemes.
- Configurations whose map is time-dependent and has no fixed point: moving geometry, VoF, phase
  change, CFD-DEM.

### 2.5 Mixing preserves the affine constraints

With β = 1, x_{k+1} lies in the affine hull of the g outputs, because Σα = 1. Every linear
constraint that all outputs satisfy, homogeneous or affine, is therefore satisfied by the mix:

- the discrete divergence of the staggered faces or of the collocated u_f (to solver tolerance);
- the solid masks;
- Dirichlet ghost values (inflow, walls).

With β < 1 the hull also contains earlier inputs; for those, the constraints hold to the extent that
the inputs satisfied them. Correctness does not need this, since the plain march starts from any
state. It explains why mixed inputs behave like plain-march inputs, for example why the projection's
right-hand side does not jump.

---

## 3. State vector and scaling

### 3.1 State (per grid; full padded buffers, `n_pad = e.x·e.y·e.z`, G = 2)

| configuration | fields (role) | n_s |
|---|---|---|
| `Solver` (staggered), any advection | `C[0].u, C[1].u, C[2].u` (Velocity, on faces), `P_` (Pressure) | 4 |
| `SolverColocated`, ghost scheme, advection off **or** `set_uf_advection(False)` | `C[0..2].u` (Velocity, cell), `P_` (Pressure) | 4 |
| `SolverColocated`, ghost scheme, advection on with projected-face advection (`advect_ && ufAdvect_`, the default) | the above + `uf_, vf_, wf_` (Carried) | 7 |

How to read the table:
- **"Carried"** fields are mixed, stored and differenced like the others, but excluded from the
  metric. They are a deterministic function of the previous step's (u*, φ), so their residual adds
  no independent information.
- Decide the collocated row from the *configuration* (`advect_ && ufAdvect_`), not from
  `faceFieldValid_`.
- Collocated Stokes: `uf_` is output-only, since every projection overwrites it and nothing reads it
  before then. It is not state.

### 3.2 Metric (inner product)

For two state vectors a and b on one rank, with inner entries I (G ≤ x < e.x−G, and likewise for y
and z):

    ⟨a,b⟩_W = Σ_{f ∈ Velocity} Σ_{i ∈ I} a_f(i) b_f(i)
            + c_P² Σ_{i ∈ I, sdf(i) > 0} (a_P(i) − ā_P)(b_P(i) − b̄_P)

    ā_P = (1/N_F) Σ_{i ∈ I, sdf(i) > 0} a_P(i)   if the pressure is gauged, else 0

Definitions:
- "Gauged" means the pressure has a constant nullspace, i.e. there is no Dirichlet-pressure
  (outflow) domain face.
- N_F is the **global** count of inner fluid-centred cells.
- Sums are global, over ranks; inner regions partition the global grid, so each entry counts once.
- `sdf` is the solver's cell-centred SDF (`sdf_`); only its sign is used.
- Velocity entries count at every inner position, including masked solid ones, whose residual is 0.
- Pressure counts only at fluid-centred cells. In the ghost scheme, solid-centred P is not a
  constraint DOF. In the staggered scheme, solid-centred cut-cell P errors show up in r_u.

**Pressure weight**, evaluated in the solver's internal unit system (where the registry fields
live; h_int = 1, ρ, μ and Δt being the values that build A_u):

    c_P = h / (μ + ρ h² / Δt)  =  1 / (μ_int + ρ_int / Δt_int)

**Why this weight.** A pressure jump δp across one face drives a face velocity change
δu = (δp/h) / (ρ/Δt + μ/h²) = c_P·δp, which is the grid-scale response of A_u. As D = νΔt/h² → 0
this tends to Chorin's Δt/(ρh); as D → ∞ it tends to the Stokes h/μ. With the first `set_rho` and
`set_dt` pinning ρ_int = Δt_int = 1, c_P = 1/(1 + D). That is 1/7 on the §11 case, the value
measured in §1.1. Because the checkerboard is nearly invisible to the velocity, this weighting is
what makes it visible to the least squares.

**Velocity scale and relative residual:**

    U² = Σ_{f ∈ Velocity} Σ_{i ∈ I} g_f(i)²          (g = the step output just computed)
    residual = sqrt(⟨r,r⟩_W / U²)                    (r = g(x) − x; = 0 if r = 0 and U = 0; +inf if U = 0 ≠ r)

**The units trap.** Everything here is internal: the fields and c_P alike. No physical conversion
happens anywhere in the accelerator, and none is needed, because mixing is linear and the metric
is scale-consistent.

---

## 4. Algorithm

### 4.1 Fixed constants (C++ `constexpr`, not user-settable)

| name | value | meaning |
|---|---|---|
| `kMaxWindow` | 8 | compile-time cap on m |
| `kEngageDecreases` | 2 | consecutive residual decreases before mixing engages |
| `kRestartGrowth` | 4.0 | restart if ρ_k > 4·ρ_min (since the last restart), at a mixed iterate |
| `kMaxRestarts` | 5 | disable acceleration at the 5th restart |
| `kNoiseFloor` | 1e-10 | below this relative residual, no restart test and no Ritz test |
| `kCondMin` | 1e-12 | minimum eigenvalue ratio of the Jacobi-scaled Gram (κ(ΔR) ≤ 1e6) |
| `kRitzDelta` | 1e-3 | instability if ρ(I+M) > 1 + 1e-3 |
| `kRitzHi` | 1e-2 | Ritz test only while residual ≤ 1e-2 (outside the strongly nonlinear initial transient) |
| `kRitzConsecutive` | 3 | consecutive detections that declare "unstable" |
| `kGelfandSquarings` | 20 | ρ(T) ≈ ‖T^(2^20)‖^(2^-20) |

User-settable: `window` m ∈ [1, 8] (default 5), and `mixing` β ∈ (0, 1] (default 1.0).

### 4.2 Data held by `AndersonCore`

- **Device:** `X`, `Rprev`, `Gprev`, `dR[0..m−1]` and `dG[0..m−1]`. Each is a set of n_s padded
  `Kokkos::View<double*>`, one per state field. That makes 2m + 3 state vectors; the history slots
  are used circularly.
- **Host:** `mk` (columns in use); slot indices ordered oldest→newest; the m×m blocks `RR`, `GG`
  and `GR`, where GR(i,j) = ⟨Δg_i, Δr_j⟩_W; per-column fluid P-means `mR[j]` and `mG[j]` (gauged
  only); `havePrev`, `pending`, `gamma[m]`, `engaged`, `decCount`, `rhoPrev`, `rhoMin`,
  `numRestarts`, `numResets`, `ritzCount`, `status` ∈ {active, disabled, unstable}, `reason`,
  `residual`, and `ritzRadius` (last value, NaN if not computed).
- **Signature**, for change detection: internal Δt, ρ, μ and the body force (3 components) at the
  last call.

`reset()` clears the history and the counters: mk = 0; pending = engaged = false;
decCount = ritzCount = 0; rhoMin = rhoPrev = +inf. It keeps `havePrev`.

### 4.3 One call: `step(accelerate)` (the adapter wraps the core around `solver.step()`)

    step(accelerate):
      # --- 0. change detection -------------------------------------------------------------
      if signature(solver) != stored: reset(); havePrev = false; numResets++; store signature
      if havePrev:
          nChanged = Σ_{inner entries, all fields} [buf ≠ Gprev]           # device count, then MPI_SUM
          if nChanged > 0: reset(); havePrev = false; numResets++          # state written externally
      # --- 1. lazy mix (prepare) -------------------------------------------------------------
      mixed = accelerate and pending and status == active
      if mixed:                                    # buf holds g_{k-1} (== Gprev, just checked)
          for every padded entry e of every field (kernel MIX):
              c = Σ_j gamma[j]·dG_j(e);   d = Σ_j gamma[j]·dR_j(e)
              x = buf(e) − c − (1−β)·(Rprev(e) − d)
              buf(e) = x;  X(e) = x
      else:
          X = buf                                  # copy (kernel COPY)
      pending = false
      # --- 2. the map --------------------------------------------------------------------------
      try:   solver.step()
      except e:
          if mixed: buf = Gprev; status = disabled; reason = "step failed at an accelerated iterate: " + e
                    reset(); notice once (rank 0); return            # do NOT rethrow
          else:     rethrow                                            # the plain march's own failure
      # --- 3. provisional new column (does not touch Rprev / Gprev) ----------------------------
      if havePrev:
          s = (mk < m) ? next free slot : oldest slot
          kernel DIFF over padded entries: r = buf − X;  dR_s = r − Rprev;  dG_s = buf − Gprev;  X = r
      else:
          kernel R: X = buf − X                     # X now holds r_k
      # --- 4. reductions (§6.1) ---------------------------------------------------------------
      if gauged: pass 1: fluid P-sums of X, dR_s, dG_s  -> Allreduce(3) -> means mX, mR_s, mG_s
      pass 2: ⟨X,X⟩_W, U², and for every window column j (plus s):
              RR(s,j), GG(s,j), GR(s,j), GR(j,s), b_j = ⟨dR_j, X⟩_W   -> Allreduce
      # --- 5. host decision (all ranks compute; rank 0's packet is broadcast and wins) ---------
      rho = sqrt(⟨X,X⟩_W);  residual = rho / sqrt(U²)
      if not finite(rho) or (mixed and solver.pressureSolveFailed()):
          if mixed: buf = Gprev;  reason = "non-finite residual / failed pressure solve at an accelerated iterate"
          else:     reason = "non-finite residual on a plain step"
          status = disabled; reset(); havePrev = false; return
      if havePrev: accept column s: copy the new RR/GG/GR row and column, mR[s], mG[s]; mk = min(mk+1, m)
      if mixed and residual ≥ kNoiseFloor and rho > kRestartGrowth·rhoMin:
          reset(); numRestarts++; rhoMin = rho
          if numRestarts ≥ kMaxRestarts: status = disabled; reason = "too many restarts"
      decCount = (rho < rhoPrev) ? decCount+1 : 0;  if decCount ≥ kEngageDecreases: engaged = true
      rhoPrev = rho;  rhoMin = min(rhoMin, rho)
      # --- 6. commit on the device --------------------------------------------------------------
      swap(Rprev, X)                                # Rprev := r_k (handle swap; X becomes scratch)
      Gprev = buf                                   # copy (kernel COPY)
      havePrev = true
      # --- 7. next coefficients ------------------------------------------------------------------
      if status == active and engaged and mk ≥ 1:
          while mk > 1 and condScaled(RR) < kCondMin: drop the oldest column (mk--)
          gamma = solveTruncated(RR, b)             # §4.4
          pending = true
          if mk ≥ 2 and kNoiseFloor ≤ residual ≤ kRitzHi:
              ritzRadius = gelfand(I + pinvTruncated(XX)·XR)   # §4.6
              ritzCount = (ritzRadius > 1 + kRitzDelta) ? ritzCount+1 : 0
              if ritzCount ≥ kRitzConsecutive:
                  status = unstable; reason = "the plain map is locally unstable at this dt (Ritz radius …)"
                  pending = false; reset()
      broadcast (rank 0 → all): status, reason code, mk and slot order, gamma[0..mk-1], pending, residual,
                                ritzRadius, numRestarts

**Notes that are not left to the implementer:**
- The **first call** has `havePrev = false`. It evaluates one plain step, records `Rprev`/`Gprev`,
  and forms no column. The residual of that first evaluation is defined (x = the initial state)
  and counts for engagement. With β = 1, the initial state never enters a mix (§2.5).
- **A plain call** (`accelerate = False`) records history exactly like an accelerated one. A plain
  step is an Anderson step with γ = 0. The pending mix is discarded, not deferred.
- **Restart** keeps `Rprev`/`Gprev` (step 6 still commits). The next call therefore forms a column
  at once, and mixing resumes after re-engagement (two decreases).
- **After `status ≠ active`,** `step()` keeps working as a plain step with no history maintenance.
  It skips the reductions too; residual stays at its last value. The driver uses `solver.step()`
  directly then anyway.
- **The state check (step 0) compares inner entries only.** Getters and ghost exchanges
  (`exchange_field`, `max_open_divergence`, the force getters) may refresh ghosts between calls;
  ghosts refreshed from unchanged inner values are not a state change. The mix acts on the full
  padded buffer as found.

### 4.4 The least-squares solve (`solveTruncated`, host, deterministic)

The goal is min_γ ‖r_k − ΔR γ‖_W, i.e. RR γ = b, with mk ≤ 8.

1. D = diag(RR)^{1/2}. If any D_j = 0, drop that column first.
2. A = D⁻¹ RR D⁻¹ (unit diagonal); b̃ = D⁻¹ b.
3. Eigendecompose A by cyclic Jacobi rotations, in fixed sweep order (p < q, row-major). Stop when
   the off-diagonal Frobenius norm is ≤ 1e-15·‖A‖_F, or after 50 sweeps.
4. `condScaled` = λ_min/λ_max. The caller drops the oldest column while it is < `kCondMin`.
5. y = Σ_i v_i (v_iᵀ b̃)/λ_i; γ = D⁻¹ y.

### 4.5 Coefficient identity used by MIX

x_{k+1} = g_k − ΔG γ − (1−β)(r_k − ΔR γ), where ΔR = [Δr_j] and ΔG = [Δg_j] over the window, in
slot order. This equals Σα_i[βg_i + (1−β)x_i] with Σα = 1, and α never needs to be formed. For
diagnostics, Σ|α| is computable from γ (α_oldest = γ_oldest, …, α_newest = 1 − γ_newest after
differencing). Log it; do not test it (D5).

### 4.6 The Ritz guard (host)

From the cached blocks:
- XX = GG − GR − GRᵀ + RR, i.e. ⟨Δx_i, Δx_j⟩_W with Δx = Δg − Δr.
- XR = GR − RR, i.e. ⟨Δx_i, Δr_j⟩_W.

Then:
1. M = XX⁺·XR, where XX⁺ is the truncated pseudo-inverse from §4.4 with eigenvalues below
   `kCondMin`·λ_max discarded and no column dropping.
2. T = I + M.
3. ρ(T) by Gelfand: A ← T/‖T‖_max, L ← log‖T‖_max. Then 20 times: A ← A·A; s ← ‖A‖_max;
   A ← A/s; L ← 2L + log s. Result: ρ = exp(L/2^20), with ρ = 0 if A vanishes.

The bias is ≤ ln(C)/2^20 ≈ 1e-5 for C ≤ 1e6, well below `kRitzDelta`.

---

## 5. Placement and API

### 5.1 Files

| file | content |
|---|---|
| `src/anderson.hpp` | `peclet::flow::AndersonCore` — grid-agnostic: takes a `MarchState` descriptor (views + roles + metric data + comm), implements §4 (kernels MIX/COPY/DIFF/R/COUNT, the reductions, host LS, guards, MPI packets). Header-only, `inline` members; Kokkos device code in `.hpp` as everywhere in flow. No `Solver` dependency — unit-testable with synthetic maps. |
| `src/anderson_accelerator.hpp` | `template <class Grid> class AndersonAccelerator` — holds `Solver<Grid>&` + an `AndersonCore`; `step(bool accelerate)` = §4.3 around `solver.step()`; signature read; status accessors. Explicitly instantiated in `src/flow_solver_staggered.cpp` / `src/flow_solver_colocated.cpp` with a matching `extern template` at the end of the header (the G.8 pattern; link via `peclet_flow_solver`). |
| `src/flow_ibm.hpp` | declare `struct MarchState` and `MarchState marchState();` (public C++, **not bound**). |
| `src/flow_ibm_diagnostics.hpp` | define `marchState()`: builds the field list of §3.1, roles, `sdf_` view, `e_`, `G`, `cP`, `gauged = !hasOutflow_`, the signature values, and (MPI) `comm_`/`distributed_`; throws `std::runtime_error` naming the first refusal of §5.3. |
| `src/flow_bindings.cpp` | bind the two adapters as private classes `_AndersonAccelerator` / `_AndersonAcceleratorColocated`; add the factory `diagnostics.anderson_accelerator(window=5, mixing=1.0)` (returns a new accelerator; `nb::keep_alive` on the solver). |
| `packaging/flow_steady.py` → installed as `peclet/flow/steady.py` | `march_to_steady` and `MarchResult` (pure Python, §7); `flow_init.py` adds `from .steady import march_to_steady, MarchResult`; `CMakeLists.txt` gets the `configure_file` (build tree) and `install` lines mirroring those for `flow_init.py` (lines 194, 353). |

`MarchState` fields: `std::vector<CCField> fields`; `std::vector<Role> roles` (Velocity / Pressure
/ Carried); `CCConst sdf`; `C3 e`; `int G`; `double cP`; `bool gauged`; `std::array<double,6>
signature` (dt, rho, mu, Fx, Fy, Fz, all internal); under `PECLET_FLOW_MPI`, `MPI_Comm comm` and
`bool distributed`.

### 5.2 Python surface

**Public tier** (module level; NAMING: verbs for actions, `num_*` counts, `rtol` like
`set_pressure_pcg`):

```python
peclet.flow.march_to_steady(solver, monitor, rtol=1e-4, max_steps=5000, accelerate=True,
                            window=5, check_every=5, num_passes=3, slow_rate=0.997,
                            roundoff=1e-11, callback=None) -> MarchResult
```

- `monitor()` returns the scalar the §3.2 instrument watches, for example the mean of the velocity
  along the force. **Under MPI it must return the same value on every rank** (a global reduction);
  every stop decision is a pure function of it and of `acc.residual`, which is rank-consistent by
  construction.
- `callback(steps, phase)` is optional and called after every step, with phase ∈ {"accelerate",
  "certify", "plain"}. Use it for logging and checkpoints.
- `MarchResult` is a frozen dataclass: `converged: bool`, `steps: int`, `accelerated_steps: int`,
  `reason: str` ∈ {"certified", "max_steps", "unstable", "diverged"}, `num_restarts: int`,
  `monitor: float` (the last value).
- `accelerate=True` on an unsupported configuration **raises** with the refusal reason and the hint
  "pass accelerate=False". An allocation that does not fit raises with the byte count (§6.3).
- No parameter is a cell-unit quantity. `rtol`, `slow_rate` and `roundoff` are dimensionless;
  `max_steps`, `check_every`, `num_passes` and `window` are iteration counts.

**Developer tier** (`s.diagnostics`):

```python
acc = s.diagnostics.anderson_accelerator(window=5, mixing=1.0)
acc.step(accelerate=True)   # one map evaluation (+ the lazy mix when accelerate and engaged)
acc.residual                # property: relative W-residual of the last evaluation (inf before the first)
acc.status                  # "active" | "disabled" | "unstable";   acc.reason: str
acc.num_restarts, acc.num_resets, acc.num_columns, acc.ritz_radius, acc.window, acc.mixing,
acc.memory_bytes            # properties
acc.reset(); acc.disable()  # methods
```

**Opt-in vs default.** `Solver.step()` is unchanged, and a hand-written `for … s.step()` loop is
bit-identical to today. Acceleration is the default *of the new function* `march_to_steady`, which
ships only after the gates (§8) pass. `accelerate=False` reproduces the study's `march()` bit for
bit (G0).

### 5.3 Refusals (`marchState()` throws)

The check runs at construction. It runs again at every `step` call: this is a re-evaluation of the
host flags only, so a feature enabled after construction is refused at the next call.

Each refusal throws with its own message:
1. `SolverColocated` whose active scheme is not the fluid-only ghost projection (gauge-exact, plain,
   embed, `set_face_interp` overrides). Message: "the fixed point is not unique / the march is
   unstable for this scheme".
2. `enable_vof`, `enable_vof_momentum` or VoF blocks.
3. `enable_phase_change`.
4. Any transported scalar (`add_scalar`).
5. Porous continuity.
6. Variable ρ or μ (any property closure or `rho`/`mu` field).
7. A `drag_beta` field.
8. Cell force fields (`force_*`).
9. Moving scene instances.
10. `set_superficial_velocity`.
11. Pressure warm start on (register: "set_pressure_warmstart(True) diverges on the steady Stokes
    march").
12. The balanced-force projection active.

Allowed:
- domain BCs of every type (inflow, wall, slip, outflow, periodic);
- static scenes;
- outer Picard iterations;
- any pressure driver except with warm start;
- velocity MG on or off;
- the wall-banded blend and rotational filter on staggered, since they are part of the map.

---

## 6. MPI and GPU design

### 6.1 Reductions, collectives and determinism

**Kernels:**
- Elementwise kernels (COUNT, MIX, COPY, DIFF, R) are a 1-D `Kokkos::RangePolicy` over the padded
  linear index, i.e. storage order, which satisfies the `iteration_order` rule.
- Reductions over inner entries use `MDRange3<CCExec>` from `policy.hpp`, x fastest.

**Pass 2 packet layout:**
- One fused reduction per window column j, each a 5-double `Kokkos::Sum` on a small struct:
  RR(s,j), GG(s,j), GR(s,j), GR(j,s), b_j.
- One reduction for ⟨X,X⟩_W and U².
- Each reduction reads at most 5 vectors (dR_s, dG_s, dR_j, dG_j, X), so its value type stays small
  enough for GPU shared memory. A single ~45-double reducer is rejected for that reason.
- The partial results are packed into one host array of 5·mk + 2 doubles, followed by **one**
  `MPI_Allreduce(SUM)` on the solver's communicator.

**Collectives per accelerated step** (all latency-bound, ≤ 45 doubles):
1. COUNT → `MPI_Allreduce(SUM)`, 1 double.
2. Pass 1 (gauged only) → `MPI_Allreduce(SUM)`, 3 doubles.
3. Pass 2 → `MPI_Allreduce(SUM)`, 5·mk + 2 doubles.
4. The decision packet → `MPI_Bcast` from rank 0, ≤ 16 doubles plus ints.

A step already performs dozens of collectives (MG-PCG two per iteration, the momentum stop, and so
on), so these four add < 0.1 % at 1536 ranks. On a distributed run with ~30 µs collectives that is
about 120 µs against a step of ~0.8 s.

**Determinism contract:**

| comparison | guarantee |
|---|---|
| np = 1 (MPI build) vs the serial build | **bit-identical** (Allreduce and Bcast on one rank are identities) |
| same np, same backend, same build | **bit-identical run to run** (deterministic Kokkos reductions; the suite's `state_hash` gates rely on the same) |
| np = 2, 4, … vs np = 1 | **tolerance**: iterates differ at reduction-order round-off amplified by κ(ΔR) ≤ 1e6; the converged K agrees to the fixed-point tolerance (G4) |
| across ranks within one run | **identical γ, status, mk and slot order** (rank 0 broadcast) — a correctness requirement: each rank mixes its own ghosts, and they stay equal to the neighbour's mixed inner values only if all ranks use the same coefficients |

### 6.2 Memory traffic per accelerated step (double, n_s = 4, m = 5; units: state vectors)

| kernel | vectors read + written |
|---|---|
| COUNT (buf vs Gprev, inner) | 2 |
| MIX (buf, Rprev, 2m history → buf, X) | 2m + 4 = 14 |
| DIFF (buf, X, Rprev, Gprev → dR_s, dG_s, X) | 7 |
| pass 1 (gauged; P part only) | ≈ 0.75 |
| pass 2 (m+1 reductions × ≤ 5 reads) | ≈ 30 |
| commit (Gprev ← buf) | 2 |
| **total** | **≈ 56 vectors ≈ 1.8 KB per cell** |

A plain step moves the equivalent of ≈ 48 KB per cell (4.7–6.2e-8 s/cell-step on an RTX 5080 at
~960 GB/s), so the overhead is ≈ 4 % of a step. Gate G8 bounds it at 5 % on CUDA. If the gate
fails, the first optimisation is to fuse pass 2's per-column reductions in pairs; numerics are
unchanged.

### 6.3 Memory (the binding constraint)

    bytes = (2m + 3) · n_s · 8 · n_pad          (n_pad = Π (n_axis + 4) for a block of n_axis inner cells)

| configuration | m = 3 | **m = 5 (default)** | m = 8 |
|---|---|---|---|
| n_s = 4 (staggered; collocated Stokes), per inner cell, unpadded | 288 B | **416 B** | 608 B |
| … with the G = 2 padding of a 212³ block (×1.058) | 305 B | **440 B** | 643 B |
| … of a 32³ MPI block (×1.42) | 409 B | 591 B | 863 B |
| n_s = 7 (collocated + projected-face advection), unpadded | 504 B | 728 B | 1064 B |

**On the 16 GB card** (the brief's 9.5 M-cell limit is 1.65 KB/cell, i.e. 15.7 GB usable):
- n_s = 4, m = 5: the largest grid is 15.7e9 / (1650 + 440) ≈ **7.5 M cells (≈ 196³)**.
- m = 3: ≈ 8.0 M.
- m = 2 (7 vectors, 237 B): ≈ 8.3 M.

**At 9.5 M cells no window ≥ 2 fits.** The constructor's allocation raises with the message:

> "AndersonAccelerator: window m needs B bytes (formula); window 2 needs B2; run on more ranks or pass accelerate=False"

The run then either uses two GPUs (the accelerator is distributed) or the plain march. On an 80 GB
H100 the history is not binding.

---

## 7. Interaction with the stop criterion

`march_to_steady` keeps the §3.2 instrument **verbatim** as the acceptance test. The instrument is
`certify()` below: d = the change of `monitor()` over a block of `check_every` steps; a pass needs
0 < R < 1 and |d|/(1 − max(R, slow_rate^check_every)) < rtol·|m|, or |d| ≤ roundoff·|m|;
`num_passes` consecutive passes. It is applied only to runs of **consecutive plain steps**.

    march_to_steady(solver, monitor, rtol, max_steps, accelerate, window, ...):
        steps = 0
        if not accelerate:
            return certify(solver.step, budget=None)                 # == study march(), bit for bit
        acc = solver.diagnostics.anderson_accelerator(window)        # raises on refusal / allocation
        target = (1 - slow_rate) * rtol
        while steps < max_steps:
            # ---- phase A: accelerate -----------------------------------------------------------
            best, since, stagnated = inf, 0, False
            while steps < max_steps and acc.status == "active" and acc.residual > target:
                acc.step(True); steps += 1; callback(steps, "accelerate")
                if acc.status == "unstable": return MarchResult(False, steps, "unstable", ...)
                if acc.residual <= 0.5 * best: best, since = acc.residual, 0
                else: since += 1
                if since >= 10 * window: stagnated = True; break
            if acc.status != "active":
                return certify(solver.step, budget=None)             # plain march from here
            # ---- phase B: certify on consecutive plain steps ----------------------------------
            out = certify(lambda: acc.step(False), budget=num_passes + 3,
                          growth_ref=acc.residual)                   # see below
            if out == "pass": return MarchResult(True, steps, "certified", ...)
            if out == "max":  return MarchResult(False, steps, "max_steps", ...)
            if out == "growth" or stagnated:
                acc.disable(); return certify(solver.step, budget=None)
            target *= 0.1                                            # out == "budget": resume A

The details below are fixed:
- **`certify(step_fn, budget, growth_ref=None)`** runs the §3.2 loop with a *fresh* block state
  (`prev = dprev = None`, `passes = 0`). Its step counter starts at 0, so `monitor()` is sampled at
  local steps 4, 9, 14, … exactly as the study does.
  - It returns "pass" at a stop.
  - It returns "budget" after `budget` blocks without a stop (never when `budget is None`).
  - It returns "max" at `max_steps`.
  - With a `growth_ref`, it returns "growth" as soon as `acc.residual > 10·growth_ref`, or as soon
    as the residual or the monitor is non-finite.
  - A non-finite `monitor()` in a `budget=None` run returns `MarchResult(False, …, "diverged")`.
    This is the one behavioural addition to the plain path; it acts only on runs that are already
    diverging, and G0 tests converging cases.
- **The target** is (1 − slow_rate)·rtol, i.e. 3e-7 at the defaults. It is the residual at which the
  slowest mode the instrument assumes (rate `slow_rate`) can carry at most rtol of remaining change,
  since its error is at most residual/(1 − slow_rate). Anderson deflates that mode, so in practice
  the certification passes at its first eligible blocks.
- **Why certify with a budget and resume** rather than let the plain tail run on: if the target was
  not tight enough, the plain tail decays at the slow rate (0.996 per step on the collocated case),
  while resuming Anderson with the history intact is ~10× faster. Plain steps through
  `acc.step(False)` keep the history warm (§4.3).
- **Why a growth exit returns to the plain march** rather than failing: growth means the plain map
  at this Δt departs from the accelerated state. The honest outcome is whatever the unaccelerated
  march does from there: converge elsewhere, oscillate until `max_steps`, or diverge.
- **Expected cost** on the §11 collocated case, N = 16, estimated from the §1.1 residual trace: the
  residual reaches 3e-7 at about step 57, then the certification takes 20–25 steps, giving ≈ 80
  steps against 395 for the plain march to stop (brief §3.2). That is **≈ 5×**.
  - Staggered: ≈ 22 + 20–25 ≈ 45 against 75, **≈ 1.7×**. The instrument's fixed cost dominates.
  - Re ≈ 10: ≈ 150 + 20–25 against ≳ 270, **≈ 1.6×**.
  - G2 measures all three.

---

## 8. Verification gates

**Common settings:**
- "Tight" means: `set_pressure_pcg(True, 400, 1e-12)` (the ghost scheme's BiCGStab takes the same
  rtol); `set_velocity_residual_tolerance(1e-12)`; `march_to_steady(rtol=1e-10, max_steps=20000)`.
- "Production" means: the study settings (`set_pressure_pcg(True, 200, 1e-8)`, 200 velocity
  sweeps); `rtol=1e-4`; window 5.
- K is the Zick–Homsy-normalised drag of `tests/study/study_avg_velocity_spheres.py`
  (`drag_K(⟨u_x⟩)`), or the case's own K for beds.
- Host means `host-openmp` with `OMP_NUM_THREADS=8 OMP_PROC_BIND=false`. CUDA means the RTX 5080
  `nvidia-cuda` prefix.
- **A twice-failed gate stops the work order.** Numerics and constants are never tuned to pass; this
  is the register's escalation rule.

| gate | what, measured how | configurations | pass threshold |
|---|---|---|---|
| **G0** inertness | (a) `tests/regression/state_hash.py` before/after; (b) `march_to_steady(accelerate=False)` vs the study's `march()`; (c) full ctest battery | (a) all entry paths; (b) §11 φ=0.125 N = 16, 24, collocated + staggered, production; (c) host tree, `-LE bench` | (a) identical hashes; (b) **identical step count and bit-identical ⟨u_x⟩**; (c) 188 existing + new tests all pass |
| **G1** fixed point | K accelerated vs K plain, both `converged=True`, tight | §11 N = 14, 16, 18, 20, 24 (collocated ghost + staggered); Z&H SC array φ = 0.343 and 0.45 at N = 32 (staggered); dense random bed φ ≈ 0.6 (§10 Q2), staggered + collocated ghost | **\|K_acc/K_plain − 1\| ≤ 1e-8** every case |
| **G2** speed-up | steps to `converged=True` and wall time incl. accelerator overhead, production | all G1 cases, host and CUDA | §11 collocated N = 16: **steps ratio ≥ 3.0 and wall ratio ≥ 2.7**; every case: steps_acc ≤ steps_plain **and** \|K_acc/K_G1 − 1\| ≤ 1e-4; report all ratios (expected ≈ 5× collocated, ≈ 1.7× staggered) |
| **G3** no new instability | 400 unconditional `acc.step(True)` (no stop) | §11 N = 16 collocated + staggered at νΔt/h² ∈ {6, 60, 600, 1e4}; dense bed at Δt = 60 and 600 (cell units, as in `collocated_invisible_subspace.md` §3) and at νΔt/h² = 1e4 | status stays "active"; ≤ 1 restart per 100 steps; running-min residual reaches ≤ 1e-9; residual at step 400 ≤ 10 × its running min; the four-Δt K agree to 1e-8 (C2 under acceleration) |
| **G4** MPI | tight `march_to_steady`, kokkos_mpi tree, `mpirun --bind-to none` | §11 N = 32 collocated ghost + staggered, np = 1, 2, 4 | \|K_np/K_1 − 1\| ≤ 1e-9; np=1 (MPI build) vs serial build: identical state hash after 60 accelerated steps; step counts within ±10 % across np (report) |
| **G4c** MPI ctest | `test_anderson_mpi` (C++, `tests/kokkos_mpi`) | staggered N = 16 sphere, 40 `acc.step(True)`, np = 1, 2, 4; plus U7 | max\|u_np − u_1\| ≤ 1e-8·max\|u_1\|; γ bitwise equal on all ranks of a run (assert after the Bcast) |
| **G5** restart | interrupt at step 25: `get_field` u,v,w,p → fresh solver, same setup → `set_field` → `march_to_steady` | §11 N = 16 collocated + staggered, tight | K vs uninterrupted accelerated K ≤ 1e-8; total steps ≤ uninterrupted + 15 (report); variant with `set_state` (velocity only): K ≤ 1e-8, extra steps reported |
| **G6** finite Re | tight `march_to_steady`, accelerated vs plain | staggered §11 sphere N = 16 at μ = 0.05, dt = 3.906e-2 (Re ≈ 10, measured stable); collocated ghost same case (state n_s = 7); a random array at Re ≈ 10 and ≈ 100 (§10 Q3) | \|K_acc/K_plain − 1\| ≤ 1e-8; steps_acc ≤ steps_plain; report ratios (prototype: 2.5× to 1e-4 error at Re ≈ 10) |
| **G7** unstable / unsteady | (a) both drivers on a case whose plain march diverges; (b) synthetic unstable maps; (c) false-alarm census | (a) staggered §11 sphere N = 16, μ = 0.0158, dt = 1.234e-2 (plain NaN at step ≈ 435, measured); (b) core unit test U4; (c) every G1–G6 run | (a) `converged=False` for accelerate True **and** False, no K reported; (b) status "unstable" within 3m steps of engagement; (c) status never "unstable" and logged `ritz_radius` ≤ 1.0005 throughout the active range |
| **G8** performance + memory | accelerator time (inside the adapter, excluding `solver.step()`), mean over 50 steps; `memory_bytes` | 64³ §11-type case, staggered + collocated, CUDA and host | overhead ≤ 5 % (CUDA), ≤ 8 % (host) of the mean plain step; `memory_bytes` = formula §6.3 exactly |

**Core unit tests** (`tests/kokkos/test_anderson_core.cpp`, synthetic maps on Views, host and CUDA):
- **U1 linear contraction:** x ← Jx + c, J diagonal with 10⁴ entries in [0, 0.996]. Window 5
  reaches residual ≤ 1e-12 in ≤ 80 steps; the plain march needs > 3000 (assert both).
- **U2 affine hull:** every g output satisfies Σx_i = S. Mixed states satisfy it to 1e-14·|S|.
- **U3 restore:** a map that returns NaN at the 3rd mixed iterate. Afterwards buf == Gprev
  bitwise, status "disabled", and the next `step(false)` works.
- **U4 instability:** J with eigenvalues {0…0.9, 1.02} and a 2×2 rotation block of modulus 1.01.
  Status becomes "unstable". A stable twin (max 0.996) never does, and its ritz_radius converges
  within 1e-3 of 0.996.
- **U5 conditioning:** a map whose differences are rank-1. Columns drop, γ stays finite, no NaN.
- **U6 change detection:** writing an inner entry between calls resets the history (`num_resets`
  increments, no mix applied). Writing only ghosts does not.
- **U7 (MPI):** U1 distributed over np = 1, 2, 4. γ is equal on all ranks, and iterates are within
  1e-13 of np = 1 after 20 steps.

---

## 9. Work orders (dependency order; each ends with its gate)

All work happens in a worktree `../flow-anderson` on branch `anderson`. Commit named paths only, and
run the blocking clang-format 18.1.8 check on `src/` and `tests/`. Record numbers in
`doc/steady_acceleration_log.md`, which is append-only.

**WO-1 — Python oracle and parameter confirmation (host only; no production code).**
- Write `tests/study/anderson_oracle.py`. It implements §3 and §4 exactly (constants of §4.1,
  `solveTruncated`, the Ritz guard) plus the §7 driver, in NumPy over `diagnostics.field_view` on a
  host build of main.
- Run it on: §11 N = 16 and 24 (collocated ghost and staggered); the Re ≈ 10 case; the G7a case; and
  the dense bed if host-feasible. Sweep m ∈ {3, 5, 8}.
- **Pre-registered rule:** if m = 3 is within 10 % of m = 5 in steps to `converged=True` on *every*
  measured case, the default `window` becomes 3, recorded in the log. Nothing else may change here.
- If any §4.1 constant causes a restart storm (> 1 per 100 steps), a false "unstable", or a missed
  G7a, **stop and report**. Do not tune.
- *Gate:* G1 (tight) on §11 N = 16 collocated and staggered via the oracle; G2's ≥ 3× on collocated
  N = 16; G7a via the oracle.

**WO-2 — `Solver::marchState()` and refusals.**
- Declaration in `flow_ibm.hpp`, definition in `flow_ibm_diagnostics.hpp` (§5.1, §5.3).
- `tests/kokkos/test_march_state.cpp` checks the field list and roles for the three §3.1 rows, and
  that each §5.3 refusal throws with its own message.
- *Gate:* G0(a) and G0(c).

**WO-3 — `AndersonCore` (`src/anderson.hpp`).**
- Kernels, reductions and packets (§6.1), host LS/Jacobi/Gelfand (§4.4–4.6), and the state machine
  (§4.3, with the core taking a step callback or split into `prepare`/`complete` around the
  caller's step; the split form is preferred).
- *Gate:* U1–U6 on host and CUDA; U7 at np = 1, 2, 4 in `tests/kokkos_mpi`.

**WO-4 — Adapter, bindings, Python driver, packaging.**
- `src/anderson_accelerator.hpp`, the explicit instantiations, the bindings (§5.1–5.2),
  `packaging/flow_steady.py` plus the `flow_init.py` import and the CMake lines.
- `tests/python/test_march_to_steady.py`, registered as ctest `march_to_steady`. It contains G0(b)
  at N = 16, a small G1 (§11 N = 12, collocated and staggered, tight, ≤ 1e-8), and G7a.
- *Gate:*
  - G0(b).
  - C++ vs oracle on host, §11 N = 16 collocated: the per-step `residual` sequences agree to 1e-6
    relative over the first 30 steps (reduction order differs, so bit equality is not expected).
  - The new ctest passes.

**WO-5 — Gate campaign.**
- Write `tests/study/steady_acceleration_gates.py` (an instrument, not a ctest).
- Run G1–G8 on host and CUDA, and G4 with the kokkos_mpi tree. Log every number.
- *Gate:* all of §8.

**WO-6 — Documentation.**
- Add a CLAUDE.md subsection "Steady marches" covering usage, scope, refusals, the memory formula,
  and "checkpoint with get_field/set_field".
- Add a "Measured" table to this note, replacing the §1.1 estimates.
- Draft the D1–D12 entries for `../docs/decisions/flow.md`. The caller commits them once approved.
- *Gate:* the docs build (`docs.yml`) is clean.

---

## 10. Risks and open questions

Each item is labelled **[fact]** (knowable by a measurement) or **[preference]** (the user's call),
and has a default that work proceeds with.

| # | question | kind | default (proceed with this) | settles it |
|---|---|---|---|---|
| Q1 | Should `march_to_steady` default to `accelerate=True`? | preference | **True** (the function is new; nothing existing changes) | user; trivially reversible (one default) |
| Q2 | Which dense random bed (φ ≈ 0.6) and resolution is the G1/G3 reference? | fact | the smallest dense-bed configuration in `~/Codes/peclet-study-A1-drag-audit/scripts/`, at its lowest resolution | A1 owner names the file |
| Q3 | Which random array and Δt rule for G6 at Re ≈ 10, 100? | fact | the A1 finite-Re configuration at its lowest resolution with A1's own Δt rule; if none exists, the §11 sphere at Re ≈ 10 only (measured stable), Re ≈ 100 reported as "not run" | A1 owner |
| Q4 | Do the §4.1 safeguard constants cause false restarts/"unstable" on production beds? | fact | constants as stated; WO-1 and G7c measure; a failure stops the WO (no tuning) | WO-1/WO-5 numbers |
| Q5 | Should A1 use an interim driver on the released wheel 1.2.0? | preference | **no**; A1 adopts on the flow release carrying this (the oracle is host-only and copies) | user (A1 budget vs effort) |
| Q6 | Largest per-GPU grid in A1 production? Runs > 7.5 M cells on a 16 GB card do not fit at m = 5 | fact | such runs pass `accelerate=False` or go to two GPUs; the constructor's error names the bytes | A1 owner |
| Q7 | Should the collocated face field be registered (`uf`, `vf`, `wf` in the field registry) so a collocated-advection checkpoint round-trips exactly? It would also make `redistribute` carry it, fixing CLAUDE.md's open item but changing a rebalanced run's numerics | preference (changes an existing path) | **not done here**; the collocated-advection restart re-seeds u_f (a transient, same fixed point) | user, as its own recorded decision |
| Q8 | Is float history storage acceptable (memory ÷ 2 for the history)? | fact + preference | **double** | an oracle run with float32 history: if steps-to-stop are unchanged on all WO-1 cases, the user decides |
| Q9 | Promote `AndersonCore` to `core` for `peclet.amr`? | preference | flow-local until amr adopts it (amr's map has the same (u, P) incremental structure, and its lmax = 0 march reproduces flow's step counts exactly — §11); then promote, tagging `core` first | user, when amr work starts |
| Q10 | Public names `march_to_steady`, `MarchResult`, `diagnostics.anderson_accelerator` and the keyword names of §5.2 | preference | as stated (checked against NAMING §1: verbs, `num_*`, `rtol`, no cell units) | user |
| Q11 | Is the §3.2 instrument library API from now on (`slow_rate=0.997`, `roundoff=1e-11`, `check_every=5`, `num_passes=3` as library defaults)? | preference | yes, the study's constants; the study script switches to `march_to_steady(accelerate=False)` in WO-4 (G0(b) proves equivalence) | user |

**Risks stated plainly:**
- **Staggered gains are modest (≈ 1.7×).** The instrument's fixed certification cost (20–25 plain
  steps) dominates a 75-step march. A shorter certificate would mean changing the instrument; that
  is not proposed.
- **Finite-Re gains are modest (≈ 1.6–2.5×).** CFL-bound Δt gives a continuum of slow modes;
  windows ≤ 8 cannot deflate a continuum.
- **Anderson can reach states the plain march at that Δt cannot** (§2.4, measured). The Ritz guard
  and the certification-growth fallback cover the cases where the plain march would visibly diverge
  or Anderson actively suppresses growth. A weakly unstable steady state can still be certified, but
  then the unaccelerated instrument would certify it too.
- **Inner-solve cost at mixed iterates** may differ from plain iterates. G2's wall-time ratio
  measures it; the prototype could not, since its host-copy overhead dominated at N = 16.
- **Monitor consistency under MPI** is the caller's responsibility (documented). A monitor that
  differs across ranks can deadlock the driver, as it would the study's `march()`.

**Transfer to other codes.** To AMR: yes. The core is View-based and grid-agnostic; an amr adapter
supplies leaf u and P and its own refusals. To VoF: no, it is time-accurate. Not designed here.
