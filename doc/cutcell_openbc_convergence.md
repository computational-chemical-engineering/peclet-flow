# Cut-cell IBM + open domain boundaries: the pressure solve stalls or diverges

**Status: OPEN, needs work.** Found 2026-09-01 while building the `foxberry-scaling` benchmark
(peclet-examples), which reproduces FoxBerry's packed-bed scaling case: a 5000-sphere bed in a
unit box with an inlet, an outlet and four no-slip walls.

## The defect

Cut-cell IBM (`set_solid`) **together with** inflow/outflow domain BCs (`set_domain_bc` types 2/3)
makes the cut-cell pressure multigrid unusable: the solve caps at its iteration limit with a
divergence residual six orders too large, and at shallow MG depth it diverges outright.

Each ingredient on its own is healthy. Measured on the committed FoxBerry bed at 100³, μ=1, dt=1,
`MGLEVELS=3`, MG-PCG rtol 1e-8, cap 200, `BOTTOM=auto` (the benchmark's `BCMODE` ablation knob):

| configuration | pressure iters/step | final `max｜div(open·u)｜` |
|---|---|---|
| all-fluid (`set_pressure_geometry`) + inlet/outlet/4 walls | 27.5 | 7.4e-09 |
| packed bed + fully periodic | 15.0 | 2.9e-10 |
| packed bed + 6 no-slip walls | 17.0 | 2.0e-09 |
| **packed bed + inlet/outlet/4 walls** | **200 = CAP** | **2.5e-03** |

Depth / driver dependence on the failing row (cap raised to 300):

| variation | outcome |
|---|---|
| `BOTTOM=smoother` | still caps at 300 — **not** the agglomerated bottom |
| `MGLEVELS=2` | **diverges**: `<u>`=NaN, `max｜div｜`=inf, driver reports 0 iterations |
| `MGLEVELS=1` (pure RB-GS) | **diverges**: `max｜div｜`=1.8e+268, 0 iterations |
| `PRESSURE=cheby` | **diverges** to NaN |
| `PRESSURE=fcg` | caps, like PCG |

The shallow hierarchy blows up; the deeper one stalls. No outer Krylov driver survives, so this is
the preconditioner/operator, not the accelerator — the same signature class as WO-H, where an
inconsistent coarse-level ghost made the V-cycle an asymmetric preconditioner and MG-PCG ran to
its cap on every 3-D wall-bounded grid.

## Why it was never caught

**Nothing in the test matrix combines the two.** No `tests/kokkos_mpi` test calls `setDomainBc`
and `setSolid` together (`test_multiphysics_mpi` sets a *scalar* BC on a periodic flow; the
velocity BC tests are all-fluid). None of the `scripts/verify_*_sdflow.py` domain-BC scripts —
lid cavity, channel, BFS — carries an immersed solid; the BFS step is deliberately realized as an
inlet profile rather than a solid. The porous/IBM studies are all periodic. So the product of the
two features has never been exercised, single-rank or multi-rank.

## Where to look

`CLAUDE.md` (§ Domain boundary conditions) documents the two-role openness split at open
boundaries: the **operator** openness α is 0 at walls/inflow (Neumann) and open at outflow
(Dirichlet p=0, ghost held at 0, mean-removal off), while the **flux** openness β stays open at
inflow and outflow so their flux is counted. `CutcellMG` re-imposes the boundary face openness on
every coarse level, with `applyNeumannGhost` (added by WO-H) and `applyOutflowGhost` filling the
non-periodic ghosts before prolongation.

The walls-only row passing while the inflow/outflow row fails points at the **Dirichlet (outflow)
half** of that split meeting the cut-cell aperture rediscretization. At an open face the coarse
operator has to reconcile a rediscretized cut-cell aperture with a Dirichlet ghost, and because
mean removal is switched off there (the operator is non-singular), an inconsistent coarse operator
is fatal rather than merely inaccurate — which matches the observed NaN at depth ≤ 2.

## Next steps

1. **Bisect the BC set** — outflow-only (+5 periodic), inflow-only, then both — to confirm the
   outflow face is the trigger.
2. **Move the bed off the open faces.** It currently starts 4 cells from the inlet (FoxBerry's
   placement rule). If the stall tracks *cut cells near an open boundary*, that localizes it to
   the α/β reconciliation at those faces.
3. **Read the symmetry out directly**: `PECLET_FLOW_MG_DEBUG=2` prints `pr`, zero iff M is
   symmetric w.r.t. the fine operator (0.008–0.086 healthy, 0.42–0.52 was the WO-H stall). Compare
   all four rows of the first table.
4. **`PECLET_FLOW_MG_BCGHOST=0`** — the WO-H ablation — to check whether the new Neumann ghost is
   itself interacting badly with cut cells at an open face.
5. **Gate it**: whatever the fix, the regression test is the first one that combines `setDomainBc`
   with `setSolid`, in `tests/kokkos_mpi` (np 1/2/4), plus a single-rank `tests/kokkos` case.

## Reproducing

```bash
cd <peclet-examples>/benchmarks/foxberry-scaling
source ~/Codes/suite/.venv/bin/activate
for m in foxberry walls periodic; do
  PYTHONPATH=~/Codes/suite/flow/build_mpi \
    PACK=results/packing_foxberry_n5000_phi0.45_s0.npz CASE=packed BCMODE=$m \
    GN=100 NSTEPS=2 WARMUP=1 MGLEVELS=3 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
    python foxberry_bench.py 2>&1 | grep -E "^\[(perf|sanity)"
done
```

Full benchmark context, the FoxBerry mapping and the deliberate deviations:
`<peclet-examples>/benchmarks/foxberry-scaling/README.md`.
