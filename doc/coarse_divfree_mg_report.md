# Coarse divergence-free multigrid investigation

## Summary

The main issue was **not** the central differencing itself, and it was also **not** fixed by simply doing more coarse pressure or velocity iterations. The decisive bug was in the experimental coarse pressure multigrid path: the V-cycle had stopped behaving like a standard defect-correction multigrid solve.

The fix that materially improved robustness was to restore a proper pressure V-cycle based on:

1. pre-smoothing on the current level,
2. residual computation on that level,
3. restriction of the residual to the coarse right-hand side,
4. coarse correction solve with zero initial correction,
5. prolongation of that correction,
6. post-smoothing.

With that change in place, packed-bed startup no longer showed the previously severe practical timestep restriction. On the tested continuation runs, `dt = 1.0` remained finite and the momentum residual decayed strongly on both `128^3` and `256^3`.

## What was checked

### 1. Convection / IBM suspicion

The earlier suspicion was that central differencing near the IBM surface might create large convective terms because the IBM was not driving the near-wall velocity low enough. That was investigated, including earlier fixes restoring central differencing near the particle surface and repairing missed IBM classification near the periodic seam.

Those changes were important for consistency, but they did **not** explain the remaining strong timestep sensitivity in the packed-bed startup case.

### 2. Coarse-level pressure solve

The coarse pressure hierarchy was then audited. The critical finding was that the experimental `pressure_v_cycle()` no longer solved a fixed coarse correction problem. Instead, it had drifted into a non-standard procedure that:

- restricted coarse state instead of the fine-level defect,
- mutated coarse `u/v/w/p` state inside the V-cycle,
- recomputed coarse right-hand sides inside the V-cycle.

That means the PDE being solved changed inside the multigrid cycle itself. In practice this explained why “more coarse work” often made the solve worse instead of better.

## Implemented solution

### Pressure multigrid

The coarse pressure multigrid was corrected so that:

- the **finest level** keeps the cut-cell / IBM-aware operator,
- the **coarser levels** use the plain periodic 7-point Laplacian,
- the V-cycle is a standard residual-restriction defect-correction cycle,
- mean removal is preserved after smoothing/correction.

This matches the intended interpretation from the robust scaled IBM formulation: the finest level uses the cut-cell decoupling outside the particle, while coarse levels should solve the continued field with the bare periodic Laplacian.

### Coarse divergence-free projection prototype

The branch also keeps the coarse-state projection machinery that restricts coarse `u/v/w/p`, recomputes the coarse divergence source, and applies coarse pressure correction outside the V-cycle. After the pressure fix, this path became reasonable again.

At the moment, however, the clear benefit comes from the **pressure-side correction**. The velocity multigrid / coarse momentum path is still only mildly beneficial and should still be treated as experimental.

## Results

### `128^3` packed bed

After restoring proper defect correction:

- `dt = 0.25`: momentum residual dropped from about `1.76e+01` (earlier broken path) to about `1.22e+01`,
- `dt = 1.0`: continuation remained finite and the momentum residual decreased over 4 steps:
  - no velocity-MG: `1.79e+01 -> 8.63e+00 -> 4.77e+00 -> 2.71e+00`
  - with velocity-MG prototype: `1.75e+01 -> 8.54e+00 -> 4.69e+00 -> 2.65e+00`

### `256^3` packed bed

For the `256^3` case, continuation with `dt = 1.0` also remained finite, with momentum residual decreasing over 3 startup steps:

- `9.32e+01 -> 4.56e+01 -> 2.17e+01`

## Interpretation

The previously observed “CFL-like” restriction was, at least in large part, a multigrid implementation artifact rather than a fundamental limit of the implicit convection / projection formulation.

The evidence for that is:

- increasing coarse pressure/velocity work on the broken V-cycle made convergence worse,
- restoring standard defect-correction immediately improved large-`dt` behavior,
- `dt = 1.0` continuation became viable on both `128^3` and `256^3`.

## Tests updated

The Python regression scripts were updated to match the current pybind API and now cover:

- fluid-fraction geometry sanity,
- auto-`dt` smoke coverage,
- single-sphere large-`dt` continuation,
- IBM smoke behavior,
- implicit Poiseuille verification.

## Remaining caveats

1. The pressure multigrid fix is the main validated improvement.
2. The coarse velocity multigrid path is still experimental and does not yet clearly reduce outer iterations.
3. Running multiple `256^3` solvers in the same Python process can still exhaust GPU memory; that is a process-memory issue, not the corrected multigrid algorithm itself.
