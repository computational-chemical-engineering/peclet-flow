# flow playbook (overrides the cce-code-health seed by id)

Read the seed first (`plugins/cce-code-health/skills/refactor-loop/playbook-seed.md`, tool pin 93a36d8), then this file.
Read `CLAUDE.md` ("Settled decisions", "Conventions and hard rules") and the register `../docs/decisions/flow.md`
before touching any solver file; a refactor never reverses one.
Gate: `PECLET_SUITE=<suite> codehealth gate`; bitwise = two reference tiers on the host/OpenMP build at ONE thread:
`state_hash` (stdout of `tests/regression/state_hash.py`: 13 single-rank cases + the np = 2 distributed step) and
`sharp_vof` (`.codehealth/sharp_vof_hash.py`: two VoF drops with the SHARP viscous interface in force, which no
state_hash case reaches; a run on the averaged path prints `SHARP INACTIVE` and fails the tier).

## Keep-out (session vof, 11-15; suite `docs/code_health/RULES.md`)

Verbatim from RULES.md:

> flow (vof, 11-15; flow main eed2fa4): keep out of src/vof/sharp_viscous.hpp, src/vof/curvature_field.hpp,
> src/flow_ibm*.hpp (all 14; vof's solids package will touch flow_ibm_vof.hpp / flow_ibm_core.hpp), the sharp-viscous
> setters/read-outs in src/flow_bindings.cpp (set_vof_viscous_interface, vof_viscous_interface,
> diagnostics.vof_viscous_interface_census/_fit), tests/study/sharp_viscous/, doc/vof_sharp_viscous_stress.md,
> doc/full_viscous_stress.md. Refactors ADJACENT to these (MAC stress elements, VoF band worklists) must be bitwise on
> state_hash AND a sharp VoF run. Paper-pinned, in main's history: f9eb1e0, 56d115f, 582ac73, 9a5d2dc, a3d1137.
> vof's live branches (never rebase/delete): sharpvisc-solids, sharpvisc, sharpvisc-perf, sharpvisc-diag, svl-fence,
> svl-mirror, svl-redist and the detached -base/-perf-ref/-pre5b worktrees. Rest of flow free.

- The path keep-outs are `forbid` entries in `config.toml`; the gate rejects a step that touches them.
- The `src/flow_bindings.cpp` keep-out is a BLOCK, not a file, so the gate cannot see it: a step that edits
  `flow_bindings.cpp` must leave the `.def(...)` entries of `set_vof_viscous_interface`, `vof_viscous_interface`,
  `vof_viscous_interface_census` and `vof_viscous_interface_fit` (and their docstrings) byte-identical. The reviewer
  checks the diff for it.
- "Adjacent" means any change to code that the sharp operator calls or that calls it: the MAC stress elements
  (`src/full_stress.hpp`), the VoF band worklists and interface census (`src/vof/advect_wy.hpp`,
  `src/vof/block_batch.hpp`, `src/vof/block_container.hpp`), the face-mu evaluation, and `src/vof/*` outside the two
  forbidden files. The `sharp_vof` tier is the required "sharp VoF run"
  and runs on every gate, so the rule is met when the gate passes; for such a step also run the ctests
  `sharp_viscous`, `sharp_viscous_mpi_np{1,2,4}`, `mirror_symmetry`, `vof_viscous_interface_auto` by name in the
  head tree and quote them in the step log (they are in `ctest_host` too; naming them makes the evidence visible).
- Never touch another session's flow worktree or branch (`git worktree list`; vof's are listed above).

## Extra gates per file (owner vof, 11-15)

- **`src/mac_cutcell_mg.hpp` (CutcellMG)**: state_hash covers the HOST path only. The class also carries the device
  FP32 V-cycle (`'auto'` = FP32 on device only, with an FP64 restart) and the distributed host paths. Every step
  there also needs: (1) bitwise on CUDA, i.e. state_hash and the sharp-VoF run on the GPU, base == head; (2) the
  MPI ctests `mg_stencil_halo_np*`, `mg_fp32_vcycle_mpi_np2`, `test_telescope_mpi`, `cg_guard_mpi` and
  `mg_vcycle_symmetry` passing at np 1/2/4.

## Rules for new code

1. Pass `Kokkos::View`s by value and capture them by value in `KOKKOS_LAMBDA`; never capture `this` in a device
   lambda (copy members to locals first; illegal on CUDA only, so the host build will not catch it; `build_cuda` does).
2. Loops use `MDRange3<Exec>` / `MDRange2<Exec>` from `src/policy.hpp`, lambda `(x, y, z)`, x fastest. A bare
   `Kokkos::Rank<` / `MDRangePolicy<` outside `policy.hpp` fails ctest `iteration_order`.
3. Never change the order of kernel launches, fences, halo exchanges or MPI calls, the RB-GS colour order, line
   relaxation sweeps, or the Krylov reduction grouping (C1–C4 fused reductions, `ccReduce3Lanes`). state_hash runs
   at one thread on the host; it cannot see an ordering change that only shows at more threads, more ranks or on the GPU.
4. Never introduce, remove or reorder atomics or reductions in a refactor (type, order, fusion, split).
5. A moved kernel keeps its execution space, memory space and execution instance (`CCExec()`); never move work to
   the host, never add a bare `Kokkos::fence()` or an exec-less `deep_copy`/`create_mirror_view` to `step()`
   (device step holds no global synchronization; ctest `device_sync_budget` checks it on device backends only, so
   the host gate does not).
6. `src/flow_ibm.hpp` keeps every member DECLARATION and the whole state block; out-of-line DEFINITIONS live in the
   thirteen domain headers. A member template called by a test or the bindings stays defined in `flow_ibm.hpp`.
   A new grid policy needs its own instantiation TU (`src/flow_solver_*.cpp`). (flow_ibm*.hpp are keep-out now
   anyway; this rule holds when they reopen.)
7. No `(float)` cast outside `// PRECISION-EXEMPT: <reason>` (ctest `no_float_operator_casts`); operator storage is
   `MReal`, never a hard-coded `float`.
8. No environment variable may change a result (ctest `no_env_knobs`); algorithm choices are `Solver` setters.
9. The MPI schedule never branches on rank-local state; every rank makes the same collective calls.
10. Name a thing by what it is, never where it runs: no `Kokkos`/`Device` qualifiers in identifiers (`View` suffix
    for device-resident duals).
11. The Python surface (`src/flow_bindings.cpp`) is unchanged by a refactor: same names, signatures, defaults,
    docstrings, and the same split between `Solver` and `s.diagnostics` (two API tiers).
12. Keep `src/` header-only Kokkos C++20 in `namespace peclet::flow`; device code in `.hpp`, never `.cu`. New
    container-free VoF math belongs in `core`, not here (and is out of scope for a refactor).
13. Format with the suite venv's clang-format 18.1.8 (flow's CI format job blocks; the six `vof-w4` files are
    excluded there and in the `clang_format` tier).

## Refactor recipes

| id | smell | when | when not | recipe |
|---|---|---|---|---|
| extract-kernel-phase | long function (operator assembly, MG level build, a solve driver) | a contiguous phase reads and writes a nameable set of Views and scalars | the phase interleaves with another phase's launches, or shares many locals that the next phase mutates; the phase needs 10+ inputs (do `launch-params-struct` first) | move the phase verbatim into an `inline` free function or a private member in the same header; same launches in the same order, same Views, same fences, same `if constexpr` branches; pass Views by value and locals by reference |
| launch-params-struct | many parameters (launchers taking 10+ Views/scalars) | the same group of 4+ Views/scalars travels together through several launchers and forms one concept (a level's operator arrays, a face field triple) | the group is used by one function only, or a member would need a different memory space | a private aggregate holding the Views BY VALUE (trivially copyable, captured by value into device lambdas); default member values reproduce today's default arguments exactly |
| host-state-owner | `[&]` host lambdas closing over a long orchestration function's mutable locals | the shared locals form named concepts (a plan, a bank of Views, mode predicates) | a later phase mutates the captured local in place without the owner knowing | a small owning struct or free function in the same header; the lambdas become calls; same launches, fences and MPI calls in the same order |
| functor-from-lambda | a kernel lambda carrying most of a launcher's CCN | the lambda body has its own branching and a name would explain it | the extraction only moves complexity, or nvcc rejects it | a named functor or `KOKKOS_INLINE_FUNCTION` with explicit by-value View members; `build_cuda` guards nvcc; the reviewer must agree it reads better |
| device-helper-extraction | a long `KOKKOS_FUNCTION` / kernel body | a sub-computation with its own name, few inputs, no register-resident accumulator crossing the seam | the seam cuts a multiply-add chain (nvcc contracts by default: the host gate, built with `-ffp-contract=off`, cannot see a GPU bit change), or needs a success-flag out-param | `KOKKOS_INLINE_FUNCTION` free helper in the same header; for a solver inner-loop helper compare SASS of the touched kernel base vs head (dem's G-sass) before accepting |
| dedup-device-predicate | the same device-side mask/predicate repeated in several kernels (fluid/cut/solid masks) | the copies are a pure predicate over by-value Views | the copies differ in tie-break, ghost handling or side effects | a device functor with by-value View members, one instance per kernel |
| guard-clauses | (seed) | host-side setup code only | device code where it changes divergence or FP order | as seed |
| replace-temp-with-query | (seed) | host-side scalars only | device code, solver inner loops | as seed |
| split-flag-argument | (seed) | a private launcher whose flag selects two disjoint kernels | the flag is threaded from a `Solver` setter through several levels | as seed |

Evidence carried over from dem (`codehealth stats`, dem-s1..s8, 38 records): extract-kernel-phase 11/12 accepted
(mean drop 7.4), launch-params-struct 5/5 (14.2), host-state-owner 6/6 (10.5), device-helper-extraction 2/3 (one
SASS reject), split-flag-argument 1/1, restructure-split-file 0/2 (dedupe first; split only if the size smell survives).
No flow records yet.

Evidence, flow F1 (WO-20, 2026-11-15, `src/mac_cutcell_mg.hpp`): extract-kernel-phase 6/6 accepted, all host
orchestration, no device lambda moved. buildAmg 59.1 -> 0.0 (five phase helpers), initMpi 92.0 -> 19.6 (telescope
stage, level layout, field allocation, debug table; `levelRatio` shared with init()), solvePCGResident 74.5 -> 48.1
(shared FP32-attempt prologue/failure path with solvePCGAttempt/solveFCGAttempt, residual/matvec/precond-dot members).
Device code untouched: SASS 836 kernels S0; GPU state_hash bitwise; MPI MG 15/15 at np 1/2/4. A branch probe outside the
gate (GraphAMG bottom at np 1/2/4 and with walls, telescoped hierarchies at np 3/4/6) was bitwise after every step:
reuse it (`branch_probe.py` pattern) for any MG step, since state_hash does not reach those paths. Gate `--base` must
be a commit with the same `.codehealth/` as the head's parent (a playbook commit after the CB fails forbid/scope).

Evidence, flow F2 (WO-21, 2026-11-16, same file): extract-kernel-phase 3/3 + dedup (host predicates) 1/1 accepted, host
orchestration only. decomposition 40.0 -> 1.0 (legacy search / auto policy / cell counts as static members), six
`can`/`evenOn` copies -> canHalve/evenBlocksOn (predict 34.2 -> 29.5), solvePCGAttempt 40.5 -> 25.4 and solveFCGAttempt
47.6 -> 34.0 (shared guessResidual/armStopTest/matvecPAp/cgUpdate/finalRemoveMean/commRank + one z = M r helper per
driver; first application passes p, so every branch keeps its copy/dot/Allreduce order), solveBiCGStab 45.5 -> 31.0
(gpMatvec, precondVcycleCopy, noteBicgExit). solvePCGResident untouched. Coverage outside the gate: `krylov_probe.py`
(predict_hierarchy over 720 configurations with the decomposition debug lines; FCG single rank; distributed FCG,
ghost BiCGStab and outflow PCG/FCG at np 2/4; the same Krylov cases on the GPU) bitwise after every step. SASS 836 S0.

## Anti-patterns (review rejects)

- **Settled-decision drift** (flow `CLAUDE.md`, `../docs/decisions/flow.md`): the collocated coupling is the
  Almgren–Bell–Colella approximate projection, NEVER Rhie–Chow; collocated pressure and forces stay INSIDE the
  implicit momentum predictor, never a face acceleration after the viscous solve (gate `collocated_stability_guard`);
  the rotational (Timmermans) pressure update; PCG not RB-GS for cut-cell pressure; backward Euler; ε-conservative
  porous momentum with the matched projection; masking excludes cut AND solid cells; velocity-MG never un-scales by
  `1/D_rescale`; the full stress as explicit conservative MAC elements on uⁿ, never the product form or the Picard
  iterate; collocated advection reads the PROJECTED face field through `ufAdvVelocity()`; double operator storage;
  c·Rᵀ prolongation under a CG driver; nested distributed MG coarse levels; advection on by default. A refactor
  that "simplifies" toward any rejected alternative is rejected even when the gate passes.
- **Reordering for tidiness**: any change to launch order, fence placement, halo/MPI call order, colouring or sweep
  order, reduction grouping, or lambda capture semantics, even when the gate agrees.
- **Space migration**: a moved kernel that runs in another execution space or instance, or a mirror/`deep_copy`
  added or removed.
- **Defaulted-argument drift**: a parameter struct whose defaults differ from the old default arguments.
- **Public surface change**: renaming, re-signing or re-documenting anything reachable from `src/flow_bindings.cpp`.
- **Call-order guard loss**: moving a check that raises on a wrong setter order (geometry-folded settings, `init_mpi`
  after the geometry) so that it no longer fires at the same call.
- **A phase helper with a parameter list as long as the phase** (seed): group the inputs first. `[&]`-capturing
  lambdas passed into a helper carry caller locals the metric does not count; the reviewer checks them.
- **Debt by lambda shuffling**: cpp@0 counts lambda bodies in the enclosing function, so moving a kernel into a
  functor lowers "debt" without lowering complexity. Accept only when the reviewer agrees the result reads better.
- **Splitting a multiply-add chain in device code**: the host gate is contraction-free by construction; the GPU is
  not. dem S7 measured seams (and combinations of seams, and dead locals) changing contracted results.
- **Moving uncovered code**: a seam in a branch no state_hash / sharp_vof case or ctest reaches is not proven by any
  gate. Cover it first (a new case belongs in the gate script, which is frozen: ask), or leave it.
