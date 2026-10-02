# Architect brief: Anderson acceleration of steady marches in peclet.flow

*Written 2026-10-02 for an architect design pass that Frank Peters commissioned. This file is
self-contained: the session that reads it has none of the conversation that produced it.*

## 1. The question

Design an **Anderson (type-II, windowed) acceleration of the steady-state march** in `peclet.flow`.
It must cut the number of steps to a converged steady state severalfold, and it must provably leave
the converged answer unchanged. Decide:

- where the acceleration lives (a Python driver over `Solver.step()`, or C++ inside flow);
- what the state vector is;
- which safeguards it needs;
- how it behaves under MPI and on the GPU;
- what gates prove it correct.

The deliverable is a design note (§8) that an `opus-implementer` can execute without making
decisions of its own.

## 2. Why it needs the architect

A wrong acceleration does not fail loudly. It either converges to a different fixed point, or it
destabilizes a scheme whose stability margins are subtle. `doc/collocated_invisible_subspace.md`
§3 shows a "harmless" wall-banded rotational blend that is stable or unstable depending on geometry
and aperture order. The payoff is large: every steady campaign in the paper programme marches to a
steady state. Examples are the A1 drag dataset (§4.4), permeabilities and Zick–Homsy benchmarks.

## 3. Current state

### 3.1 The step (the fixed-point map)

From `doc/collocated_invisible_subspace.md` §1 (collocated; the staggered `Solver` has the same
structure on MAC faces). Steady Stokes, body force F:

```
(momentum)  A_u u* = (ρ/Δt) uⁿ + F − G Pⁿ            A_u = (ρ/Δt)I − μL (Robust-Scaled cut-cell Laplacian)
(Poisson)   A_p φ  = −D_α Π u*                        A_p = D_α G_f, MG-PCG
(faces)     u_f^{n+1} = Π u* − G_f φ                  (D_α u_f^{n+1} = 0)
(cells)     u^{n+1}  = M (u* − G φ)                   M = solid mask
(pressure)  P^{n+1}  = Pⁿ + (ρ/Δt) φ − μ D_α Π u*     incremental-rotational update
```

- The **production collocated default is the fluid-only "ghost" scheme**
  (`set_collocated_scheme("ghost")`, AUTO). It has a unique Δt-independent fixed point ("C2",
  `collocated_invisible_subspace.md` §2, §6).
- The **staggered `Solver`** is the production path for most permeability and drag work.
- At finite Re, advection is explicit (SOU/Koren), so the map is nonlinear.

### 3.2 How steady marches are driven today

A driver loops over `solver.step()` and stops on a criterion. The current best criterion is from
`tests/study/study_avg_velocity_spheres.py` (2026-10-01):

```python
def march(step, umean):
    """Every CHECK_EVERY steps, d = change of <u_x> over the block and R = d / (previous d).
    A block PASSES when 0 < R < 1 and the geometric remainder |d| / (1 - max(R, RHO_SLOW**CHECK_EVERY))
    is below TOL_K |<u_x>| -- or |d| is at round-off. N_PASS consecutive passes stop the march."""
```

With it, the worst measured |K_stop/K_∞ − 1| is 7.4e-5 against a 1e-4 target. The older
"per-step Δ⟨u⟩ < tol" stop **halted falsely** at 2.4e-4 where two modes of opposite sign crossed.

Steady campaigns in the A1 drag study (`~/Codes/peclet-study-A1-drag-audit/scripts/`) run on the
released wheel `peclet-flow-cu13 1.2.0`. They drive the march from Python through zero-copy
views (nanobind; device views on CUDA).

### 3.3 The slow modes that set the step count (measured)

- **Collocated (ghost and gauge-exact): the (π,0,0) pressure checkerboard**
  (`collocated_invisible_subspace.md` §11).
  - Mechanism: the bulk central cell gradient annihilates p ∝ (−1)ⁱ. The mode couples to velocity
    only through one-sided wall rows.
  - Rate ≈ 0.996 per step: 0.9926 / 0.9962 / 0.9955 / 0.9971 / 0.9968 at N = 14/16/18/20/24, and
    0.9985 at N = 32. It is **independent of Δt** (0.9957–0.9964 over νΔt/h² = 1.5…10⁴), while the
    physical modes go from 0.68 per step to nothing.
  - 80–84 % of the late pressure-increment energy is the single Fourier mode (π,0,0).
  - It is excited only by symmetric geometry: a sphere centred on a grid vertex with N even. A
    shift of h/2 cuts it by ~10³.
  - It does **not** change the converged answer (Δt limits agree to 8e-9).
  - Collocated took 490 steps to stop at N = 16, against 70 for staggered.
- **Staggered: a slow near-wall pressure mode**, 0.95–0.99 per step.
- **Variability:** dense beds take 150–250 steps per Stokes solve (A1 measurement, RTX 5080), and
  finite-Re points are budgeted at ~400 steps.

### 3.4 Cost context

- Measured: 4.7–6.2e-8 s per cell-step on an RTX 5080.
- Memory: 1.65–1.9 KB per cell; the 5080 (16 GB) holds at most ~9.5 M cells.
- The A1 budget is ≈ 343 local GPU-h + ≈ 290 H100-h, and is dominated by march steps
  (`~/Codes/peclet-study-A1-drag-audit/work/RERUN_PLAN.md`).
- Anderson with window m stores ~2m extra copies of the state. That is the binding constraint on
  the GPU.

## 4. Constraints and invariants

1. **The fixed point must be unchanged.** The accelerated iteration must converge to the steady
   state of the unaccelerated scheme to solver tolerance. That means C2, Δt-independence, and no
   change to discrete equations, operators or defaults.
2. **No change to the numerics of `step()`.** The acceleration wraps the map; it does not alter
   it. The unaccelerated path stays bit-identical (suite rule: never change numerics while
   changing structure).
3. **Device-first and MPI-distributable.**
   - The state lives on the device. Anderson's small least-squares problem (m ≤ ~10) may run on
     the host after global reductions.
   - **Reductions:** flow has multi-rank bit-exactness gates (`tests/kokkos_mpi`, np = 1, 2, 4).
     State whether the accelerated path must be rank-count bit-exact, or only rank-count
     independent to tolerance, and design the reductions accordingly.
4. **Memory.** Specify the window m and the storage per cell. It must fit dense-bed production
   runs; say what happens at the 9.5 M-cell limit of a 16 GB card.
5. **Restart-safe.** A run stopped and restarted from `get_state`/`set_state` (verify what that
   round-trips; it may not include everything the map depends on) must converge to the same
   answer. State whether the Anderson history is part of the checkpoint or discarded on restart.
6. **Physical-units API** (suite directive): every new setting takes physical inputs. No
   cell-unit API surface and no numerics-changing environment variables (QUALITY_PLAN D-rules;
   `docs/NAMING.md` for any public name).
7. **Stability.** Anderson can diverge on non-normal maps. The step map is non-normal (the
   cut-cell momentum operator is nonsymmetric; ‖S−Sᵀ‖/‖S‖ = 1.4e-2 measured in the M1 model). It
   needs safeguards: restarts, a residual-decrease test, a damping β, and conditioning control of
   the least-squares problem.

## 5. Already decided, and what is open

**Decided:**
- Frank chose **Anderson acceleration**. A known-subspace (checkerboard) deflation and a pressure
  filter are *not* to be designed here.
- Filters are rejected because they can move the fixed point (the steady P contains checkerboard
  content set by the wall rows) and because of the blend experience.
- The stop criterion of §3.2 stays as the acceptance instrument, possibly re-expressed on the
  accelerated residual.

**Open (decide them):**
1. Where it lives: a Python driver (CuPy/NumPy views; fast to ship, works on the released wheel)
   or C++/Kokkos in flow (no copies; MPI-native).
2. The state vector x and the residual r = g(x) − x. Which fields are included: cell u, face u_f,
   accumulated P, and anything else the map reads? What weighting or scaling makes velocity and
   pressure commensurate in the least-squares problem?
3. Window m, mixing β, the restart policy, and the safeguards. Also: should acceleration start
   only after the fast transient (e.g. once successive R are steady)?
4. Finite Re: is Anderson still appropriate with explicit advection? Where are its limits, and do
   unsteady regimes need to be detected and excluded?
5. Collocated and staggered: one design for both solvers?
6. How it interacts with the stop criterion of §3.2, which assumes geometric contraction of
   ⟨u_x⟩.
7. MPI: the reductions, determinism, and the communication cost per accelerated step.

## 6. Already tried and rejected

- **The wall-banded rotational blend w0**, as a stabilizer. It is not a cure: it fails with today's
  order-2 apertures (M1 study `~/Codes/peclet-study-M1-collocated-attractors/transplant/README.md`).
  This is evidence that "small" modifications of the update can bite.
- **The per-step Δ⟨u⟩ stop**: it halts falsely (§3.2).
- **Chebyshev** acceleration of the *pressure* solve gave no gain at 1536 ranks. That was a
  different loop (the linear solve, not the outer march), so don't conflate the two.
- Nobody has tried Anderson, Krylov or extrapolation on the outer march in this codebase.

## 7. How the answer will be verified

The note must specify gates. At least:

1. **Fixed point unchanged.** On the §11 case (one sphere per periodic cell, φ = 0.125,
   νΔt/h² = 6, N = 14…24, collocated ghost and staggered), accelerated K equals unaccelerated K to
   ≤ 1e-8 relative at a tight stop. The same holds on a dense random bed (φ ≈ 0.6) and on the
   Zick–Homsy SC array.
2. **Speed-up.** Steps to the §3.2 stop, and wall time including the Anderson overhead, against
   unaccelerated. Target ≥ 3× on the §11 collocated case, with a report for staggered and the
   dense bed.
3. **No new instability.** Long marches at large Δt (60…1e4 in the cases' units) with acceleration
   on: no divergence, and the residual decreases monotonically on average.
4. **MPI.** np = 1, 2, 4 give the same K to tolerance (or bit-exact, per the §4 decision).
5. **Restart.** A stop and restart mid-march converges to the same K.
6. **Finite Re.** Re = 10 and 100 on a random array: the same answer as unaccelerated, with the
   step reduction reported.

## 8. Deliverable

The design note goes in `doc/steady_acceleration.md` in this repo (flow), with these sections:

1. Decision summary.
2. Fixed-point argument (why the converged answer is unchanged).
3. State vector and scaling.
4. Algorithm, with pseudocode (window, mixing, safeguards, restarts).
5. Placement and API (physical-units names, defaults; opt-in vs default).
6. MPI/GPU design (reductions, memory).
7. Interaction with the stop criterion.
8. Gates (from §7, made concrete).
9. Work orders for an `opus-implementer`, each with its gate.
10. Risks and open questions.

Commit it on a branch, not main. Record the design decisions in `docs/decisions/flow.md` of the
umbrella (`/home/frankp/Codes/suite`) when the note is approved. Write no production code.

## 9. Out of scope

- Known-subspace (checkerboard) deflation.
- Pressure or velocity filters.
- Changes to the discrete scheme, its defaults, or the linear-solver internals (MG-PCG settings).
- Unsteady (time-accurate) simulations, except for detecting them in order to switch acceleration
  off.
- AMR (`peclet.amr`) and VoF. Mention whether the design transfers, but do not design for them.
