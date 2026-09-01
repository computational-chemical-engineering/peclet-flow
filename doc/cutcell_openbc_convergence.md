# Solid geometry intersecting an OPEN domain face breaks the pressure solve

**Status: OPEN, narrower than first reported.** Found 2026-09-01 while building the
`foxberry-scaling` benchmark (peclet-examples).

> **Correction (same day).** This was first written up as "cut-cell IBM + inflow/outflow BCs does
> not solve", which was **wrong**, and wrong in a way worth remembering: the bed used to find it had
> spheres *clipped by the inlet and outlet planes*, an artifact of how that bed was built, not a
> property of the configuration under test. With a bed whose spheres are whole and clear of the open
> faces — which is what the case being reproduced actually specifies — the identical configuration
> converges fine. The real defect is the narrower one below. The lesson: when an A and a B differ,
> check that they differ *only* in the thing you think they do.

## The defect

When immersed solid **intersects an open (inflow type 2 / outflow type 3) domain face**, the
cut-cell pressure solve stalls: MG-PCG runs to its iteration cap and the divergence residual sits
three to four orders too high. With the same bed, same grid, same BCs and the solid merely pulled
clear of those faces, it converges normally.

Measured A/B — identical everything (128³, μ=1, dt=0.78, `MGLEVELS=4`, MG-PCG rtol 1e-8, cap 300,
5000 spheres, φ=0.45), only the bed differs:

| bed | pressure iters | capped | final `max｜div(open·u)｜` |
|---|---|---|---|
| whole spheres inside [0.01, 0.99] (wall-grown) | **32.7** (max 37) | none | 9.6e-05 → **1.95e-06** over 42 steps |
| spheres **clipped by the inlet/outlet planes** | 260.8 (max 300) | **5 of 6 steps** | 4.0e-03 |

The healthy row keeps converging: over 42 steps its divergence falls to 1.95e-06 and its iteration
count settles at 29.2, and `<u>` tracks the inlet velocity to 2 % (1.018e-3 against 1.0e-3).

The earlier, coarser evidence still stands on its own terms and localizes which BC is implicated —
with the clipped bed, the same geometry is healthy under periodic BCs (15 iterations) and under six
no-slip walls (17), and an all-fluid domain is healthy under the open BCs (27). So it is the
open-boundary treatment meeting solid *at that boundary*, not either one alone. At `MGLEVELS` ≤ 2
the clipped case diverges outright (NaN, 1e+268, 0 iterations); `BOTTOM=smoother` also caps, so the
agglomerated bottom is exonerated; FCG caps and Chebyshev NaNs.

## Why it was never caught

No test covers solid together with domain BCs at all: no `tests/kokkos_mpi` test calls
`setDomainBc` with `setSolid`, and none of the `verify_*_sdflow.py` domain-BC scripts carries an
immersed solid (the BFS step is deliberately an inlet profile rather than a solid). The narrower
case — solid *cutting* an open face — is a subset of that gap.

## Where to look

`CLAUDE.md` (§ Domain boundary conditions) describes the two-role openness split at open
boundaries: the **operator** openness α is 0 at walls/inflow (Neumann) and open at outflow
(Dirichlet p=0, ghost held at 0, mean-removal off), while the **flux** openness β stays open at
inflow and outflow so their flux is counted. The defect is presumably in how a *cut* cell on such a
face reconciles those two roles — a face that is simultaneously partially solid (aperture < 1) and
an open boundary. Because mean removal is off on the outflow path (the operator is non-singular
there), an inconsistent row there is fatal rather than merely inaccurate, which matches the NaN at
shallow depth. `CutcellMG` re-imposes boundary face openness on every coarse level, so the coarse
levels are equally suspect.

## Next steps

1. **Minimal reproducer**: a single sphere straddling an outflow face on an otherwise empty domain,
   at a few grid sizes. Far cheaper to reason about than a 5000-sphere bed, and it should show the
   stall immediately if the mechanism above is right.
2. **Bisect the BC set** on that reproducer — outflow only, inflow only, both.
3. **Read the symmetry out**: `PECLET_FLOW_MG_DEBUG=2` prints `pr`, zero iff M is symmetric w.r.t.
   the fine operator (0.008–0.086 healthy, 0.42–0.52 was the WO-H stall).
4. **Decide the contract.** It may be legitimate to *reject* solid intersecting an open face with a
   clear error rather than fix it — but that has to be a decision, not the current silent stall, and
   a partially blocked inlet is a physically reasonable thing for a user to want.
5. **Gate it**: whatever the outcome, the regression test is the first one combining `setDomainBc`
   with `setSolid`, in `tests/kokkos` and `tests/kokkos_mpi` (np 1/2/4).

## Reproducing

```bash
cd <peclet-examples>/benchmarks/foxberry-scaling
source ~/Codes/suite/.venv/bin/activate
for bed in walls n; do            # walls = clear of the open faces; n = clipped by them
  case $bed in
    walls) P=results/packing_foxberry_walls_n5000_phi0.45_s0.npz ;;
    n)     P=results/packing_foxberry_n5000_phi0.45_s0.npz ;;
  esac
  PYTHONPATH=~/Codes/suite/flow/build_mpi PACK=$P CASE=packed BCMODE=foxberry \
    GN=128 NSTEPS=6 WARMUP=2 MGLEVELS=4 PMAXIT=300 \
    OMP_NUM_THREADS=8 OMP_PROC_BIND=false python foxberry_bench.py 2>&1 |
    grep -E "^\[(sdf|perf|sanity)"
done
```

Benchmark context and the deliberate deviations:
`<peclet-examples>/benchmarks/foxberry-scaling/README.md`. Prioritized alongside the campaign's
other findings in `<suite>/docs/SCALING_ISSUES.md`.
