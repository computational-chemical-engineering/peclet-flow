# Solid geometry intersecting an OPEN domain face

**Status: FIXED 2026-09-16** (`tests/kokkos/test_openbc_solid.cpp`,
`tests/kokkos_mpi/test_openbc_solid_mpi.cpp` — the first tests anywhere to call `set_domain_bc`
together with `set_solid`). Found 2026-09-01 while building the `foxberry-scaling` benchmark
(peclet-examples); prioritized as issue 3 in `<suite>/docs/SCALING_ISSUES.md`.

> **Correction (2026-09-01).** This was first written up as "cut-cell IBM + inflow/outflow BCs does
> not solve", which was **wrong**, and wrong in a way worth remembering: the bed used to find it had
> spheres *clipped by the inlet and outlet planes*, an artifact of how that bed was built, not a
> property of the configuration under test. With a bed whose spheres are whole and clear of the open
> faces the identical configuration converges fine. The real defect was the narrower one below. The
> lesson: when an A and a B differ, check that they differ *only* in the thing you think they do.

## What it was — THREE defects, and none of them can be fixed alone

Each one masked the next, and fixing one without the others makes the answer worse rather than
better: (1) hid (2) by making the aperture read ~1 so the operator's literal 1.0 agreed with it,
(2) hid (3) by overwriting the plane the halo had corrupted, and repairing (3)'s openness without
its velocity breaks a conservation identity that had been closing at 1e-14.

### 1. The SDF ghost band outside a non-periodic domain face was filled by PERIODIC WRAP

`setSolidUploadSdf` fills the whole extended block by a periodic gather, and only free-slip faces
were repaired afterwards (`mirrorSdfSlipFaces`, added when a type-4 face turned the far wall into a
phantom solid on the symmetry plane). Every other non-periodic face kept the wrap — so the geometry
a wall / inlet / outlet face saw was **teleported from the opposite side of the domain**.

That decides the **boundary-face aperture**, because `ccFaceOpen` samples the SDF *at* the face,
i.e. halfway between the last inner cell and that ghost. A solid cell sitting against the inlet was
handed the far side's fluid and its inflow face came out as **fully open**:

```
[dbg] argmax div = 1.0000e+00 at cell (0,7,4)   sdf = -0.7293      <- the cell centre is SOLID
      ox- 1.0000  ox+ 0.0000  oy- 0.0000  oy+ 0.0000  oz- 0.0000  oz+ 0.0000
      u-  1.0000  (the prescribed inlet velocity)
```

The *flux* openness β on that face is 1, so the prescribed inflow is counted into the cell; the
*operator* openness α is 0 there (an inflow is a Neumann face) and 0 on the other five faces (solid).
The row is `0·p = U`: **inconsistent**. MG-PCG cannot converge — it ran to its iteration cap with
`max|div|` stuck at exactly the inflow velocity.

**Fix**: `extendSdfDomainGhosts` (`src/flow_ibm_geometry.hpp`) repairs the ghost band on every
rank-owned non-periodic face — type 4 keeps its mirror, types 1/2/3 take the **constant normal
extension**, which is the geometric reading of "the domain plane cuts the solid": the solid
continues straight out, so the boundary face's aperture is the fluid fraction of the boundary plane
itself. A bed clear of the open faces is byte-identical either way (the aperture is 1 whichever
value the ghost carries).

### 2. A Dirichlet (outflow) row carried the literal openness 1.0, not the face's aperture

`CutcellMG::applyBoundaryOpennessFrom` wrote `1.0` over the outlet face on every level, because the
per-level periodic fill wraps the opposite boundary into that (ghost) index. With the outlet clear
of solid that is the right value. With a solid cutting the outlet it is not: the operator then says
the face is fully open while the divergence constraint weights it by the true aperture, the two
disagree by `(1 - aperture)`, and **the projection pushes mass out through solid**.

This one was *silently wrong* rather than visibly broken — and, before fix 1, invisible, because the
wrapped ghost made the aperture come out ≈ 1 too, so operator and constraint agreed while both were
wrong about the geometry.

**Fix**: carry the face's own aperture, reusing the WO-R2 item 1 save/restore/coarsen machinery that
already did exactly this for the variable-density coefficient (`mgSaveFacePlane` /
`mgRestoreFacePlane` / `mgCoarsenFacePlane`, `CutcellMG::setOutflowCoefficient`). Concretely:

* `Solver::bridgeOutflowFacePlanes` copies the HIGH-side boundary-face plane of each outflow axis
  from the g=2 openness to the g=1 MG rails — `copyInner` reaches only the inner cells, and that
  plane is a ghost index no kernel writes. (The LOW side is an inner index that was already right.)
* the plain constant-density build enables the coefficient-carrying path
  (`setOutflowCoefficient(hasOutflow_ && outflowOpCoeff_)`);
* the variable-density high-side plane (`buildRhoCoeffOutflowFace`) gains the aperture factor every
  other face of that coefficient already carried;
* the three porous builders run one index past the inner block along each axis, so their Dirichlet
  face is built by the same formula as every inner face, aperture included.

`set_outflow_operator_coefficient(False)` remains the ablation back to the literal 1.0, and the test
uses it to pin this half of the fix.

### 3. Distributed, the openness halo exchange overwrote the HIGH domain face

Found while gating the fix at np = 1/2/4, and a direct consequence of defect 2's fix making that
value matter. Ghost indices along an axis are `0 .. g-1` and `ext-g .. ext-1`; the LOW domain face
of an axis is the inner index `g` and survives, but the HIGH one is `ext-g`, the first ghost index.
`velDev_->exchange(ox_)` wraps periodically on every axis (the decomposition is periodic by
construction; the non-periodic conditions are imposed on top), so on a rank owning that global face
it came back carrying **the opposite boundary's aperture**. With a sphere cut by the inlet, the
outlet's aperture became the inlet's — the flow could not leave, and the distributed run disagreed
with the single-rank one by 66 % of the velocity scale at np = 1. A wall face hid it, because the
flux-openness step zeroes walls afterwards; a bed clear of the open faces hid it, because it wraps
1.0 onto 1.0; and before defect 2's fix the Dirichlet row overwrote the value with 1.0 regardless.

**Fix**: `buildOpennessHighFace` re-derives that one plane from the (already exchanged, already
domain-extended) SDF after the exchange, on every rank-owned non-periodic +face. Note it is not only
the operator that reads it — `divergOpen` weights the outgoing flux of the last cell by exactly that
value, so the constraint was wrong too.

The same wrap sits on `fillVelGhostsTo(..., doOutflow = false)`, whose whole purpose is to keep the
mass-conserving outflow face the projection wrote (WO-R, gate F2: it has to reach the VoF
advection): distributed, the exchange inside it destroys that plane the same way. It now saves and
restores the plane over the block's INNER transverse range — the plane's transverse ghost rows are
legitimately the neighbour's inner values, which the exchange delivers correctly, so putting stale
local values back over them would trade one wrong plane for another.

**The two halves are not separable, and the repo already had the gate that proves it.**
`tests/kokkos_mpi/test_vof_bc_mpi` drives a packing whose last sphere *cuts the +z outlet plane* —
`{8.0, 8.0, NZ, 3.6}`, with that comment on the line — and gates the composed conservation identity
`d Σ(eps_eff C) = boundary ledger` absolutely. Fixing the openness ALONE broke it, because an outlet
whose openness is its own aperture no longer pairs with an outlet velocity that is still the inlet's;
the two used to wrap together and stay mutually consistent while both were wrong about the geometry.
Measured at np = 1, where there is no decomposition at all:

| | composed budget, rel |
|---|---|
| before any of this | 1.25e-14 |
| openness half alone | **2.46e-02** |
| both halves | 2.89e-14 |

The third caller, `maxOpenDivergenceProjected`, is covered by the same fix; under MPI with an outlet
it used to return approximately the inlet velocity on every bed, converged or not.

### The configuration that is still rejected, on purpose

A cell whose pressure row is entirely closed but which a prescribed inflow still feeds is an
inconsistent row whatever the discretization: the row says nothing may leave, the constraint says
this much enters. Physically it is a pocket of fluid the solid seals off from the rest of the domain
that opens only onto the inlet. `checkSealedInflowCells` scans the inlet planes at `set_solid` time
and **throws** naming the face and the cell count, instead of handing MG-PCG a system it cannot
solve. Before fix 1 this condition fired on ordinary geometry; it now takes a genuinely sealed
pocket, which `sealedPocketRejected()` in the single-rank test constructs deliberately.

## Measured

Duct 32×16×16, μ=1, dt=0.5, 3 MG levels, MG-PCG rtol 1e-10 cap 200, 8 steps, sphere R=4.3.
`projected max|div|` is `max_open_divergence_projected()` — the residual of the constraint the
projection actually solved.

| bed                              | iters | projected max\|div\| | before the fix                        |
|----------------------------------|-------|----------------------|---------------------------------------|
| all-fluid duct                   | 17    | 3.0e-14              | 3.0e-14                               |
| sphere clear of the open faces   | 17    | 4.4e-09              | 4.4e-09 (byte-identical)              |
| sphere cutting the OUTLET        | 15    | 3.1e-09              | 1.6e-09 — *converged, and wrong*      |
| sphere cutting the INLET         | 14    | 6.1e-09              | **200 iters, capped, max\|div\| = 1.0** |

**Read the outlet row carefully.** Before the fix it *converged*: the wrapped ghost handed the
outlet an aperture of ~1 and the Dirichlet row also said 1, so operator and constraint agreed —
while both were wrong about the geometry, and the mass they conserved was leaving through solid.
That is the silently-wrong face of defect 1 and the reason defect 2 was invisible. Isolate defect 2
by ablating it with the honest aperture in place: `set_outflow_operator_coefficient(False)` on the
fixed build gives **6.7e-03**, plateaued (4.8e-03 still at 60 steps, against 2.5e-09 fixed) — the
`(1 - aperture)` disagreement, measured. That is the assertion the test carries.

**Read the right divergence.** `max_open_divergence()` refills the outflow ghost with the
zero-gradient extrapolation *before* measuring, so at a partly blocked outlet it reports how far
zero-gradient is from the mass-conserving face — a property of the boundary condition, not a solver
residual, and it does not decay (1.7e-02 on the outlet-cut bed here, 4.7e-04 on the clear one, 4.0e-07
once the clear bed reaches steady state). The original field A/B below read that diagnostic.

**Distributed** (`test_openbc_solid_mpi`, 32³, 4 steps, np 1/2/4): the distributed run reproduces
the single-rank one **exactly at np=1** (0.00e+00 relative, all four beds — clear, outlet-cut,
inlet-cut, wall-cut) and to 1e-15 at np 2/4, with the openness fields bit-identical. Before defect
3's fix the outlet-cut bed differed by 1.8e-02 and the inlet-cut bed by 6.6e-01, *at np = 1*.

### The original field evidence (kept as the record)

128³, μ=1, dt=0.78, `MGLEVELS=4`, MG-PCG rtol 1e-8 cap 300, 5000 spheres, φ=0.45:

| bed | pressure iters | capped | final `max｜div｜` |
|---|---|---|---|
| whole spheres inside [0.01, 0.99] (wall-grown) | **32.7** (max 37) | none | 9.6e-05 → **1.95e-06** over 42 steps |
| spheres **clipped by the inlet/outlet planes** | 260.8 (max 300) | **5 of 6 steps** | 4.0e-03 |

With the clipped bed the same geometry was healthy under periodic BCs (15 iterations) and six
no-slip walls (17), and an all-fluid domain was healthy under the open BCs (27) — so it was the open
boundary meeting solid *at that boundary*, not either alone. At `MGLEVELS` ≤ 2 the clipped case
diverged outright (NaN, 1e+268); `BOTTOM=smoother` also capped, so the agglomerated bottom was
exonerated; FCG capped and Chebyshev NaN'd. All of that is the inconsistent inlet row of defect 1:
an inconsistent system does not become consistent at a different MG depth or with a different
Krylov driver.

## Why it was never caught

Nothing combined `set_domain_bc` with `set_solid` in a way that put solid *on* an open face. Four
tests called both (`test_freeslip`, `test_vardensity_projection`, `test_vof_bc`,
`test_vof_collocated`, plus `test_velocitymg_bc_mpi` and `test_vof_bc_mpi`) but every one of them
keeps its geometry clear of the non-periodic faces, and none of the `verify_*_sdflow.py` domain-BC
scripts carries an immersed solid at all (the backward-facing step is deliberately an inlet profile
rather than a solid). `test_openbc_solid{,_mpi}` closes that gap: the sphere is placed clear of the
open faces, cut by the outlet, and cut by the inlet in turn.
