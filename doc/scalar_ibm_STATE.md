# Scalar transport with immersed solids — campaign STATE

*Rewritten in place, never appended. History goes in `scalar_ibm_log.md`.*

## Objective (Frank, 2026-10-02)

Design scalar transport in the presence of solids **properly**, as a peclet suite feature, not a
paper. It must cover:

- Dirichlet;
- Neumann, homogeneous and prescribed flux;
- conjugate transport (solid-side diffusion, partition coefficient, contact resistance);
- Robin.

It must generalize later to AMR (`amr/`) and to VoF / level-set interfaces. The method is chosen
from the literature and adapted to peclet:

- GPU (Kokkos);
- MPI blocks;
- matrix-free compact stencils, RB-GS / geometric MG / PCG;
- SDF at cell centres plus face apertures;
- divergence-free MAC face flux.

Then a plan, and a step-by-step implementation.

## Where we are

- **Branches.** Work happens on branch `scalar-ibm` in worktree `suite/flow-scalar-ibm`, off
  `origin/main` 39680a4. The prototypes live on `peclet-papers` branch `a4-exploration` (worktree
  `~/Codes/peclet-papers-a4`), commit e0d8cbf.
- **First study done.** See `work/A4/proto/RESULTS.md` in that worktree.
- **Running now:** the literature search (L1 cut-cell/EB, L2 interface/conjugate/VoF species,
  L3 particle-resolved and benchmarks) and a recon of flow's phase-change scalars and amr's cut
  cells. Digests go to the session scratchpad under `lit/`. Copy them into `doc/` when they land.

## First-study gates (2-D disc; relative error at 32 / 128 cells per diameter)

| condition | scheme | error at 32 / 128 | order |
|---|---|---|---|
| Neumann | aperture FV, κ storage | 1.0e-3 / 6e-5 | 2 |
| Neumann | flow today (unit storage) | 11 % / 2.8 % | ~1 |
| Dirichlet | linear ghost (symmetric) | 9e-4 / 6e-5 | 2 |
| Dirichlet | flow's per-cell mask | 16 % / 4 % | 0.7–1 |
| Dirichlet | cut-cell FV, centroid wall flux | 2.9 % / 0.7 % | 1 |
| Robin | FV series resistance | — / 0.3–0.75 % | 1 |

No compact scheme so far is second order for Robin at all Biot numbers.

## Next action

1. When the literature and recon land: prototype the candidates in 2-D. These are the symmetric
   Robin FV (Papac/Gibou), face-centroid FV (Johansen–Colella / AMReX), the "aperture + link"
   hybrid, and the conjugate variants (BCCM, two-sided cut cell, VIM).
2. Write the architect brief → design note → work orders.

## Open decisions (defaults)

None yet need Frank. Pushing waits for him; nothing goes to main.
