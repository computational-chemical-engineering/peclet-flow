# The collocated advecting velocity: the projected face field, not the cell→face average

**Decision, 2026-09-21.** On the collocated grid, momentum advection now uses the **projected,
discretely divergence-free MAC face field `uf_/vf_/wf_`** that the approximate (ABC) projection
produced at the end of the previous step. The un-projected cell→face average ½(u_i+u_j) — the
phase-2 form — survives as the developer-tier ablation `diagnostics.set_uf_advection(False)`.

This closes open decision **P1** of `../../amr/docs/amr_flow_uniform_parity.md` §7, whose default
was *"not taken"*. The staggered grid is untouched, bit-for-bit: there the stored velocity **is**
the face velocity, and after the projection it is already the divergence-free one.

---

## 1. Why

Three independent statements, all pre-existing, all pointing the same way:

1. **flow's own design note prescribes it.** `doc/flow_colocated_plan.md` §1 step 3: *"Correct the
   faces: `u_f = u_f* − grad_f(phi)` … The face field is now discretely divergence-free (the
   approximate-projection guarantee). These `u_f` become the **advecting** velocities for the next
   step's advection."* Step 3 shipped; the swap did not. `src/colocated_advection.hpp` said so in
   its own header — *"this header's `adv_vel` is where that swap happens"* — for three months.

2. **It is the Almgren–Bell–Colella prescription**, and the reason the coupling is called an
   *approximate* projection in the first place: the field the projection just made solenoidal IS
   the conservative advective flux. Advecting with a field that is not discretely divergence-free
   on the stencil `divergOpen` measures makes the conservative flux form non-conservative by
   exactly that residual.

3. **The FOU operator's row-sum identity needs it.** `../docs/decisions/flow.md:977` already
   recorded that `fou_operator`'s conservative row sum equals `rho_f · div_h(u^k)` and *"holds
   only for uniform/div-free advecting field"*. With the projected field it holds exactly, so the
   implicit-FOU operator has zero row sum and the explicit `(SOU − FOU)` deferred correction
   cancels at steady state instead of leaving a residual.

And one empirical statement: it was the **last uniform-grid difference** between `peclet.flow` and
`peclet.amr` — see §3.

**It is not Rhie–Chow, and this is not a step toward it.** The suite-wide prohibition
(`../docs/DECISIONS.md`, held independently by `flow`, `amr` and `voro`) stands: the residual
*cell* divergence is intrinsic to cell-centred velocity placement and no advecting-velocity choice
removes it. What changes here is only which field the *advection* reads; the pressure coupling,
the flux reconstruction (Koren/SOU/FOU), the openness weighting and the cut-cell closures are all
untouched.

---

## 2. What the code does

`cadv::adv_vel` (`src/colocated_advection.hpp`) gained a `bool uf`:

| `uf` | advecting velocity at the `+fd` face of cell `(x,y,z)` |
|---|---|
| `true` (production) | `U(x+1,y,z)` — the face field read **verbatim**, in flow's low-face convention (`U(i)` is the velocity at the −x face of cell `i`, so the cell's `+x` face is its neighbour's low face) |
| `false` (ablation) | `0.5*(U(x,y,z) + U(x+1,y,z))` — the cell→face average |

The flag is threaded through `cadv::advect / advect_sou / advect_fou / fou_operator` and the
`GridLayout` policies (`src/grid_layout.hpp`; `Staggered` accepts and ignores it so the call sites
stay grid-agnostic). One predicate decides it:

```cpp
// src/flow_ibm_core.hpp
bool Solver<Grid>::ufAdvVelocity() const {
  return Grid::collocated && ufAdvect_ && faceFieldValid_;
}
```

and **every** consumer of the advecting velocity reads that one predicate, because the implicit
FOU operator and the explicit deferred correction must agree or the cancellation in §1.3 is lost:

- `buildRhs`, `buildRhsForced`, `buildRhsVar`, `buildRhsColoFF` — the explicit `advect_sou` /
  `advect` / `advect_fou` terms (`src/flow_ibm_project.hpp`);
- `buildAdvStencil`, `buildAdvStencilVar` — the implicit-FOU operator (`src/flow_ibm_core.hpp`);
- `vmg_.restrictAdvVelocities` — the velocity-MG coarse FOU preconditioner
  (`src/flow_ibm_project.hpp`).

The **advected** field stays the cell field (`advVelView(c)`) throughout: what moves is momentum at
cell centres; only the flux velocity comes from the faces.

### Timing: lagged one step, frozen through the Picard loop

`step()` runs the predictor before the projection, so `buildRhs` reads the face field of the
**previous** step's projection — which is what the design note says ("*the next step's
advection*") and what `peclet.amr` does (`amr/include/peclet/amr/flow.hpp:1192`). It is therefore
also frozen across Picard iterations, unlike the advected field.

### The open boundary is the one face that must NOT be the projected value

At an outflow, `bcCorrectOutflow` writes onto exactly one face the correction that closes the
discrete mass balance (the operator's Dirichlet-`p` row owns that face). Its value is therefore set
by **global continuity**, not by the fluid next to it. Feeding it into the upwind advective flux of
the last cell — and, on the domain-BC path, into the implicit-FOU operator's diagonal there —
closes a positive feedback loop. Measured, on the developing channel (H=16, L=112, Re=100,
dt=0.5): the outflow column gains mass from step 2 (mean `u` 1.009 against an inflow of 0.998),
`max|u|` reaches 4.5e+02 by step 50, and the pressure PCG loses its preconditioner at step 69.
`verify_colocated_bfs` failed the same way.

So the advecting field is the projected face field **with each open domain face replaced by its
zero-gradient extrapolation** — `buildOpenFaceField()` / `openFaceView(c)`, a lazily allocated
triple that exists only when there *is* an outflow. That value is local, and it is what momentum
advection at an open boundary has always used (it is what the cell→face average of the
outflow-filled cell ghosts came to). With it, the channel converges to the developed profile
(`u_max/U_mean` → 1.486, divergence 7.7e-05, stable).

The **colour and scalar** transport keep reading the raw `uf_`: their outlet flux *should* be the
mass-conserving one — that is what `bcCorrectOutflow` is for, from their point of view. Hence a
separate field rather than an in-place fix-up. The divergence diagnostic reads the same open-face
view (it already wanted exactly this transformation, and used to perform it destructively on
`uf_`).

`amr` never met this: its parity cases are periodic or walled, so it has no open-boundary
configuration and the ABC prescription's one exception never came up there.

### Where it falls back, and why that is not a silent scheme switch

`faceFieldValid_` is false only before any face field exists at all. `set_state` / `set_velocity`
both seed it from the cell field with the same `centerToFace` map `project()` uses
(`seedFaceFieldFromCells`), so in practice the fallback covers exactly one situation: a solver
that has had no velocity set and no step taken, where the field is zero either way. The ablation
`set_uf_advection(False)` is the only way to get the average deliberately.

### Two staleness holes this change forced us to close

The face field is **block scratch, not a registry field**, so nothing migrates or rebuilds it
automatically. Both of these were already wrong for the colour and scalar transport (which ride
`uf_` too, and whose divergence guard waves a zero field through as "perfectly solenoidal"); they
became wrong for momentum as well, so they are fixed here:

- **`redistribute()`** (`src/flow_ibm_mpi.hpp`) — `allocateBlock` hands out fresh **zero** buffers
  on the new partition and the old partition's face field is simply gone. Now re-seeded from the
  migrated cell velocity at the end of the rebalance.
- **`setSolid` / `setSolidDevice`** (`src/flow_ibm_geometry.hpp`) — `buildVelocityOverlays(resetU)`
  zeroes the cell velocity but left `uf_` holding the previous geometry's projected field. Now
  zeroed with it, and `faceFieldValid_` cleared.

A third, related repair: **`max_open_divergence()` no longer mutates `uf_`.** It used to
re-impose the zero-gradient outflow face *in place* to keep the diagnostic comparable with the
staggered one — harmless when nothing read `uf_` between steps, but now that reading a diagnostic
would have changed the next step's advecting velocity at the outflow plane, it reads the
`openFaceView` copy built above instead.

And a coverage repair: the ctest **`colocated_open_boundary`**
(`tests/python/test_colocated_open_boundary.py`). Nothing registered covered the collocated open
boundary — every other collocated ctest is Stokes (no advecting velocity at all) or
periodic/walled — so the outflow failure above broke the channel AND the BFS while all 161 ctests
still passed. A defect no gate can see is a gate bug as much as a code bug. The new test is the
cheap version of the same signal (one coarse channel, 400 steps, ~1 min: finiteness,
mass in == mass out, u_max/U_mean → 1.5, face divergence); the two full-fidelity scripts stay
hand-run, because ~20 minutes each is past what CI can carry.

---

## 3. Measured

All numbers host-openmp, 6-8 OpenMP threads, `build_uf` vs `build_base` (the same worktree at
`6adbb4b`). Reproduce with `tests/study/uf_advection_ab.py` (see its header for the exact
commands). The last digit of a round-off-level figure moves with the thread reduction order; the
byte gate in §3.5 is the reproducible-to-the-bit one, at one thread.

### 3.1 The structural claim: is the advecting field divergence-free?

max |div_h(advecting field)| after 40 Taylor–Green steps, periodic, all fluid:

| N | projected `uf` | ½(u_i+u_j) | ratio |
|---|---:|---:|---:|
| 32 | 4.62e-15 | 7.49e-06 | 1.6e+09 |
| 64 | 2.30e-16 | 1.71e-07 | 7.4e+08 |

The old advecting field carried the intrinsic approximate-projection **cell** divergence; the new
one is solver-tolerance zero. This is the point of the change, and it is not a tuning number.

### 3.2 Accuracy: 2-D Taylor–Green against the exact solution

ν=0.05, dt=0.5, 40 steps, L2 relative to the initial field norm:

| N | projected `uf` | order | ½(u_i+u_j) | order |
|---|---:|---:|---:|---:|
| 32 | 4.7453e-03 | — | 4.7858e-03 | — |
| 64 | 3.4317e-04 | 3.79 | 3.4858e-04 | 3.78 |
| 128 | 2.2834e-05 | 3.91 | 2.3263e-05 | 3.91 |

**Indistinguishable — ~1 % better, at the same observed order, at every resolution.** This matches
what `amr` measured independently (`amr_flow_uniform_parity.md` §3a: 0.4 % apart, same order). The
case for the change is consistency, not accuracy; §3.1 and §3.3 are the reasons, and this table is
the evidence that taking it costs nothing where it does not help.

The registered gate `verify_colocated_taylor_green` (100 steps, N=32/64) passes both ways, with
the error slightly lower: 7.876e-04 vs 7.916e-04 at N=64.

### 3.3 Momentum conservation, and the production regime

Galilean-shifted vortex (mean u = 0.5, periodic, no body force, dt=0.1, 120 steps): the continuum
answer is an exactly constant mean momentum.

| N | projected `uf` | ½(u_i+u_j) |
|---|---:|---:|
| 32 | 4.22e-15 | 4.22e-15 |
| 64 | 4.11e-15 | 4.11e-15 |

Both conserve to round-off — again matching `amr`'s independent measurement. (Aside, not a gate:
at dt=0.5 the explicit periodic case is past its stability limit, and there the average diverged
to NaN at both resolutions while the projected field survived at N=64. Real, but a stability
observation from outside the valid range, not a claim.)

The cut-cell bed at large dt — the regime the dt² scaling of the difference points at, and flow's
production regime — 32³ eight-sphere periodic bed, µ=0.02, f=1e-3, implicit advection, dt=20,
200 steps:

| solver | ⟨u_x⟩ over fluid | max_open_divergence |
|---|---:|---:|
| staggered (reference) | 2.185937e-01 | 2.6e-16 |
| collocated, projected `uf` | 2.115886e-01 | 1.8e-04 |
| collocated, ½(u_i+u_j) | 2.115907e-01 | 1.8e-04 |

**The two choices differ by 1e-5 % here** — the regime that was flagged as the one where this
could bite does not move. (The 3.2 % collocated-vs-staggered gap and the 1.8e-04 cut-cell
divergence residual are both pre-existing and identical on both sides of the A/B: they are the
collocated cut-cell accuracy problem, not this change.)

### 3.4 The last `amr` ↔ `flow` uniform-grid difference

`amr/tests/study/flow_parity`, 20 steps of full Navier–Stokes, N=32, gauge-exact, implicit
advection, relative L2 over the shared fluid cells:

| case | velocity | pressure |
|---|---:|---:|
| both engines on their default (`amr` `uf`, `flow` **now** `uf`) | **1.90e-11** | **4.41e-11** |
| `amr` ablated to the average, `flow` on `uf` | 2.49e-04 | 3.95e-04 |

The gap the parity study measured (2.5e-04) was exactly this one function, and it is closed. Note
the consequence for `amr`'s gate: its `tg_advect_matched.json` case ablates `amr` to *flow's old*
choice, so it now measures the difference rather than the agreement — that case wants updating on
the `amr` side (either drop the `"uf_advection": false` line, or ablate both engines).

### 3.5 Byte gate

`tests/regression/state_hash.py`, one thread. The ten pre-existing cases are **bit-identical**
before and after: the staggered bed, all four collocated schemes (`ghost` / `gauge-exact` /
`plain` / `embed` — all Stokes, `set_advection(False)`), the staggered channel, the VoF droplet,
the Boussinesq scalar, the porous continuity and the moving scene.

Two cases were **added**, because nothing in the gate exercised collocated advection at all — the
gap that let this defect live for three months:

- `colocated_advect` — periodic, explicit Koren advection;
- `colocated_advect_bc` — inflow/outflow/wall, so `implicitAdv()` holds and the implicit FOU
  operator plus the explicit deferred correction both run.

Both change, as intended. And with `set_uf_advection(False)` the new binary reproduces the old
one's hashes on both, **bit for bit** — the ablation is exact, so the plumbing adds nothing of its
own:

```
colocated_advect      e7e376f79469…  (build_base == build_uf + set_uf_advection(False))
colocated_advect_bc   93bde5cd6f16…  (build_base == build_uf + set_uf_advection(False))
```

---

## 4. Scope and known limits

- **Staggered: nothing changes**, and the byte gate proves it rather than asserting it.
- **Collocated Stokes: nothing changes.** With `set_advection(False)` there is no advecting
  velocity, so every permeability / drag baseline (`tests/regression/perf_baseline_colocated*.json`,
  the Zick–Homsy and bed studies) is untouched.
- **Moving geometry does not interact.** `advWallInputs()` is `!Grid::collocated && …`, so the A0
  wall-velocity advection inputs are staggered-only and never meet the face field. If the
  wall-aware advection inputs are ever extended to the collocated grid, note that `centerToFace`
  builds the face field from the **masked** cell field: the wall velocity would have to enter the
  face field, not the cell field, for advection to see it.
- **Velocity-MG coarse operator, one inconsistency, preconditioner-only.** `restrictAdvVelocities`
  now restricts the face field, but the coarse FOU is built by `sadv::fou_operator_aniso`
  (`src/mac_velocity_mg.hpp`), whose `adv_vel` always averages two stored values and is not
  parameterised by `uf`. So on coarse levels the restricted face values get averaged again. Level 0
  is `buildAdvStencil`'s operator and is exact; the mismatch lives only in the multigrid
  preconditioner, and only in the (explicitly opted-in) collocated + velocity-MG + implicit-FOU
  configuration. Recorded here rather than fixed: closing it means teaching the coarse builder the
  face convention, which is a separate change with its own gate.
- **Cost.** One extra `deep_copy` of three cell-sized fields per step, and only when the case has
  an outflow (`buildOpenFaceField`). Nothing else: `adv_vel` with `uf == true` does *fewer*
  arithmetic operations than the average it replaces. If the copy ever matters, only the
  open-boundary plane actually differs from `uf_`, so a save/restore of that plane would do —
  correctness first, and the copy is provably right.
- **Exactly conservative in the bulk, residual at cut faces.** The projection zeroes the
  openness-**weighted** divergence `sum(o_f u_f^+ − o_f u_f^-)` (`divergOpen`), while `cadv`'s flux
  form is unweighted. So the advective flux sum telescopes to exactly zero where every face
  openness is 1 — the all-fluid bulk, which is what §3.1 measures — and carries a residual in the
  cut band. That residual is what §1.3's row-sum identity is subject to, and it is the same trade
  `amr` recorded in 2026-07-24 ("div-free bulk, residual-small band", `../docs/decisions/amr.md`).
  It is strictly better than the average, which carries the O(h²) cell divergence *everywhere*,
  but "divergence-free advecting field" should be read as "in the bulk".
- **Not measured: large-dt steady driving on a dense cut-cell bed at finite Re.** The
  uf-vs-average difference scales as dt² (`amr_flow_uniform_parity.md` §3), so it is small at a
  CFL-limited dt and largest exactly there. `tests/study/uf_advection_ab.py bed` is the probe.

## 5. Reversing this

`set_uf_advection(False)` restores the old numerics bit-for-bit — that is what §3.4 measures — so
any baseline that turns out to depend on the old choice can be reproduced without reverting code.
Changing the **default** back is a new recorded decision, per `../docs/DECISIONS.md`.
