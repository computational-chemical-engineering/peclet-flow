# VoF step performance: campaign log

**APPEND-ONLY.** One dated section per work order of
[`vof_step_performance_design.md`](vof_step_performance_design.md); never edit a closed section —
a correction is a new dated entry that names the one it corrects. The raw artifacts (dumps, logs,
profiles) live outside git in `~/Codes/bubble_column_perf/` and are named in each section.

---

## 2026-09-25 — WO-0: harness, baselines, and the no-code experiments

**Build under test:** the FROZEN baseline modules of flow `ed05b6f` (= main `7fdec0d` + two doc
commits; `src/` unmodified), `~/Codes/bubble_column_perf/frozen_base/{omp,flt,cuda}`:
host-openmp with double operator storage (the default), host-openmp with
`PECLET_FLOW_OPERATOR_DOUBLE=OFF`, and nvidia-cuda (RTX 5080, sm_120); all `PECLET_FLOW_MPI=ON`.
Raw artifacts: `~/Codes/bubble_column_perf/baseline/` (driver scripts `run_hashes.sh`,
`run_dumps.sh`, `run_kp.sh`, `run_e1_timing.sh` beside them).

**Harness** (committed with this entry): `tests/study/vof_perf/` — `prof.py` (the case from a
checkpoint; `--ckpt`, `--case-dir`, `--rtol`, `--dump` now also stores the per-step pressure
iterations `iters` and every block's colour `col<id>`), `cmp.py` (bytewise per array, exit 1 on any
difference; last line = max rel over u, v, w, p, C), `run_mpi.py`, `bench_cpu.sh` (trees are
`TREE_A/B/C` variables; the `--old` / `run_peclet_old.py` branch is dropped: that pre-rename driver
existed only in a volatile scratchpad and served a comparison against flow `b0569d6` that no
longer applies), `kernel_diff.py` (per-step kernels from two profiles of different length),
`d1_tolerance.py` (D1). `tests/study/vof_surface_tension.py` and `vof_blocks_ns.py` gained
`--rtol` (defaults unchanged). The checkpoint `ckpt_t43.npz` (sha256 `d9999acb…a383779f8`) is at
`~/Codes/peclet-examples-bubble-column/benchmarks/bubble-column/data/` (gitignored there by `*.npz`)
and `snellius:/projects/0/prjs1022/peclet/data/` (same sha256).

**Environment.** The 48-core host ran at load average 60–137 throughout (other sessions), and the
RTX 5080 was shared: a production `run_peclet.py` and a `flow-vof-container` ctest battery held it at
89–91 % utilisation before our GPU timings started. **Every timing below is indicative only**; the
bitwise results are unaffected.

**Host OpenMP wait policy (a measured trap).** Under this load the default libgomp policy (spinning
barriers) is catastrophic for this workload of ~2500 small parallel regions per step:
`dump_omp8` at the default policy took **41 835 ms/step**, the same run with
`OMP_WAIT_POLICY=passive` **2 230 ms/step** — and the two dumps are **bitwise identical** (all
24 arrays; `cmp_omp8_rep.txt`). All host timings below therefore use `OMP_WAIT_POLICY=passive`
(numerics unaffected, proved by that comparison). G-PERF host numbers need a quiet host or
Snellius regardless.

### G-BIT item 1: `tests/regression/state_hash.py` (OMP_NUM_THREADS=8 OMP_PROC_BIND=false)

All cases, then `mpi` at `mpirun -np 2 --bind-to none`. Each tree ran **twice; both runs
byte-identical** on all three trees, np 1 and np 2 (`hash_<tree>{,_np2}.run2.txt`). Load 60–72.
Combined SHA-256, first 16 hex digits (full lines below):

| case | host-openmp double | host-openmp float operator | nvidia-cuda |
|---|---|---|---|
| staggered_bed | `9b72977a99389d03` | `2332ff9b2f34c87b` | `15748848338a08d3` |
| colocated_ghost | `4d4d3e382d3fb097` | `3230815167d99016` | `ba618738445dbcd0` |
| colocated_gauge_exact | `8e8733871310d969` | `6e1f63f3370b034c` | `7687786ac7ebb040` |
| colocated_plain | `7755b7efed301c56` | `f68395879f4ee1a3` | `628e98a87dbcc426` |
| colocated_embed | `6c17a5e1b4fb686a` | `b43d45b82c46741a` | `83b45c93b49c688f` |
| colocated_advect | `aedbfb476f651f49` | `cbdce4f48edc957b` | `739a644d6babb136` |
| colocated_advect_bc | `ea47f9c40a05af6c` | `b71dba897142f847` | `44c2be37d16f8833` |
| channel | `928cd3a243f1197f` | `cd6e28b3405dbdda` | `4fd52a6be32f4fe9` |
| vof_droplet | `9feeae25a28a1f2b` | `dc1a8d5adf1bcc38` | `a43216cff584672c` |
| scalar | `94ab86dcfc32b3ce` | `94ab86dcfc32b3ce` | `69ef3c8a947dc263` |
| porous | `65e528b4689ee3ce` | `b5885b54aff07fb2` | `25f0ba690c914040` |
| scene_moving | `71de68ff707fc542` | `099227c5eaa88a05` | `2fdef25e2f91970e` |
| mpi_np2 | `d9340d490e9c6e17` | `3f0c0befda1a01db` | `de53e1405fa71583` |

The float-operator tree prints `CutcellMG::solvePCG: preconditioner produced non-finite z;
returning zero correction` twice on stdout, from the `porous` case (reproducible; the known
float-storage failure, `../docs/SCALING_ISSUES.md` #1). `hash_flt.txt` therefore has 14 lines,
not 12. Files: `baseline/hash_omp.txt`, `hash_omp_np2.txt`, `hash_flt.txt`, `hash_flt_np2.txt`,
`hash_cuda.txt`, `hash_cuda_np2.txt`.

```
### omp
staggered_bed            9b72977a99389d03c75db125ce1370066d945de394205f493fb15db385b9c2d3  p=4ba82571c3f6 u=0cd6197b103d v=dff93866dbe9 w=8d97d7c47903
colocated_ghost          4d4d3e382d3fb09786df6b2456d08aa85cab5e67dbc05583515599b3612f3668  p=f8d2933b4601 u=9b48609b4695 v=e1524f350e58 w=bdf62014556e
colocated_gauge_exact    8e8733871310d9691a457527c0f70159c9db8de339c4f2c6c8b05675b5c7c572  p=95a94e0eeffc u=99fde5f9234a v=177fcabbbbe0 w=fb21ddc349f7
colocated_plain          7755b7efed301c567991684fb05dff962ef1cb9e729564fc501fe04fc04a76e1  p=24836364cd8e u=d74b30ad76f2 v=1bdad19a00b2 w=81069679eb8c
colocated_embed          6c17a5e1b4fb686a24015964992bd234642bfc6dbdd60ab8db3fbf662afd6bf8  p=ad06c1e4afee u=56b7988266a1 v=fe40def8f420 w=85391ee464ed
colocated_advect         aedbfb476f651f4951c1750721b6e420bc59d29ca1465e49bf535bc96d406d72  p=1294b7f5daf7 u=6aae80f51ec2 v=8dd1f61f291c w=7e7405190361
colocated_advect_bc      ea47f9c40a05af6cfe410c9ff0a675833617abf1b4a7d0a17e1c86d865b2dca8  p=a255d8f26f28 u=adb10ab4f21a v=505489447e4a w=d0c6db0df84c
channel                  928cd3a243f1197fce0238e35e07ff2d08339a1e263fde882f5ae8de0ab7fdb8  p=3cbf52fc65d0 u=c6da6ce49deb v=8d02e0d15572 w=b355cd751ea8
vof_droplet              9feeae25a28a1f2b5d1ecabbd41a3afca2a3c76d444ad92b1733793eb4fc88ba  C=e32910e65139 p=c096d63a5347 u=e24d0472ce80 v=2d73d16f3285 w=9a1555134506
scalar                   94ab86dcfc32b3ce2827229bc5dcac7f5525a91fc5d6ae99010a99be7b3b70fe  T=3d2310a406e6 p=4d6bc0b705ec u=6c6ac61d99b1 v=9041cb313e51 w=7984a0d63d2a
porous                   65e528b4689ee3ce0465ccfc50eecb2ceed79ca6b7b7cda298d81997bae0bea6  p=662b561405ed u=7686ae8d66d1 v=7686ae8d66d1 w=4be46a00cdd4
scene_moving             71de68ff707fc542e364b74483642d124de49ed775c62d12e6b6be8d990d2e70  p=41df1c3a2a81 u=5154d998f14a v=f48638070e01 w=751f021bbf67
mpi_np2                  d9340d490e9c6e174f0f183d344de8dc746316b0424e3f0878510e7ffd3b7139  p=0066af6cc703 u=3682bd1896d4 v=683c7c8523ca w=5fe1e3b10d07
### flt
staggered_bed            2332ff9b2f34c87ba7ce644695fc2f6b3f9ad88f0489e7c735803f00c7024ea8  p=f999883aebca u=27e65bd8c65f v=678e4f15bb3c w=f4af64e6f5ab
colocated_ghost          3230815167d9901645bb3ab484228e777663c352d70fbe3f420021ce5e2f995a  p=e0ce47818636 u=11e65232efc9 v=62e7a7b6d2a3 w=68f0c5b32664
colocated_gauge_exact    6e1f63f3370b034cfd5feb9fcc6e7bfaba694a806b21dc125eb726cf61487529  p=de30475c1e90 u=d2a8ddea13b5 v=b7d047033bd9 w=ded7cc9d9dfb
colocated_plain          f68395879f4ee1a36bf911b7f0047b683b4c0cdeab37469fc366bf6545bdfe12  p=fb0aa1b33dca u=a0c8e676453f v=5500bf2f81f9 w=756f493bfbe9
colocated_embed          b43d45b82c46741a878ffd4cce77693a998d0a004d49601611674689c1da4d01  p=4361a22aed32 u=92b9a505cf7d v=5467e8a8cd2c w=5c1045fcbec0
colocated_advect         cbdce4f48edc957ba66a8443bee648d4aa7a3e6bf1431792c4792a37e2b0358c  p=545291b02b17 u=aea7036596ca v=ccba68aea921 w=5791d8a5f7eb
colocated_advect_bc      b71dba897142f84753ce56326f0a749a8b7848ff36eefe8bc4286486876894f5  p=34d1ed362546 u=a3cb769deff0 v=1381f1540534 w=bc54d7354d02
channel                  cd6e28b3405dbdda5983988db675cc1359abd81f4d620fd20898313c72d3ea61  p=ca69ef471d1f u=13b822631c5a v=302ea4ad41ae w=40f5b2be756f
vof_droplet              dc1a8d5adf1bcc380a159e872eca612a9705ff3314a5b14e0a024822dd6ca126  C=bbc0928a5faa p=3648a292fd0e u=fa21b66ad946 v=b97da0daf389 w=f2706cb1ec02
scalar                   94ab86dcfc32b3ce2827229bc5dcac7f5525a91fc5d6ae99010a99be7b3b70fe  T=3d2310a406e6 p=4d6bc0b705ec u=6c6ac61d99b1 v=9041cb313e51 w=7984a0d63d2a
porous                   b5885b54aff07fb29659f7a00614bd52715cc965221b090f144e0d361717a761  p=198afa74c6e6 u=bff809e057d9 v=bff809e057d9 w=834f63c3c8ba
scene_moving             099227c5eaa88a05c9b46c512bc20f6bc7c1d6f2533811c37f8d4d20dc25366c  p=045359176b31 u=6a309d9d184b v=5090fd58aa54 w=ce743f04ebba
peclet::flow CutcellMG::solvePCG: preconditioner produced non-finite z; returning zero correction (reported as 500/500 iterations, i.e. a CAPPED solve)
peclet::flow CutcellMG::solvePCG: preconditioner produced non-finite z; returning zero correction (reported as 500/500 iterations, i.e. a CAPPED solve)
mpi_np2                  3f0c0befda1a01dbf9a7d6f0606e3afb7b6b339a41c27ffb5d6f3fc5b5000c80  p=77daa2cdb270 u=613325044da4 v=fb9c91bb43da w=3dfdb349376f
### cuda
staggered_bed            15748848338a08d3df25cf488df57525b8e62015cf8beb77da632a2e4bde19ce  p=e050f14a7789 u=43014d66e7f9 v=41f524741291 w=4fce30545690
colocated_ghost          ba618738445dbcd07f559f28617b603b2a01dff215dbbe5d7630b656660b8688  p=cd813fe787bf u=dd3d807ac6ca v=8ca3b00712a2 w=7f0962317d1c
colocated_gauge_exact    7687786ac7ebb04068bfad6152acee121f7e4bb04a51fd49ac9bdd3a9214e77b  p=fa24f6f00157 u=45b75cf51c8e v=b6cb46d369a1 w=b9296ec7d74f
colocated_plain          628e98a87dbcc4260ec2a33bd9ec0c723ac7f7d8d2d326038617662c762c2bbf  p=84ae31b712bb u=f9cbaa71601b v=e9bd07ca3307 w=0453889fdfe6
colocated_embed          83b45c93b49c688f657dd0f908cd6f2f645749d11b18d951d4be0842b785ea0d  p=1915c3e69fb2 u=ed924cd3f896 v=13e562247c9d w=2b2e4053323f
colocated_advect         739a644d6babb136115d48bd96e45fbabb0b4b7142a5ecab447640313a99000b  p=f0cff7378439 u=42fe8e4dc0b1 v=6d4fbdfb609d w=efe50d4f54af
colocated_advect_bc      44c2be37d16f88336d966c4e3d7f7ac9abdd4b78268acb4fbe624bc136e1a9d7  p=20aa34457aad u=2db9a1f08b0f v=39a6a12eb276 w=75d21751dd5f
channel                  4fd52a6be32f4fe9d35841bc0c739238a38030c2e0965376cf6bb2e7189a1314  p=a2ec8dde3219 u=9ce5cdba140c v=5f7728123d8a w=13271f22c2e9
vof_droplet              a43216cff584672cefe33edc413d95636a673345af7e88ec88a7031151101f44  C=462b2f458f24 p=7bb1bf0b4962 u=1cb906a19d0b v=69f0233e6acb w=3fded3becc47
scalar                   69ef3c8a947dc263413c24073b22b78c03d756e04a3b510ebf49beb4cd7b67d7  T=0fd1a012051b p=93d77603af07 u=658f2991c941 v=1bbbec3ab89f w=a891fbf3cb59
porous                   25f0ba690c9140401003c7c1313071d826794a6abf77921a2f678e1102e90860  p=3db5df26648a u=106b74d9a265 v=106b74d9a265 w=4be46a00cdd4
scene_moving             2fdef25e2f91970e119a2d64d78734278f553f369ccab895a5c33090e6cf5dc8  p=521348bc26df u=1d6095446efe v=edde80ab0fe2 w=f4e13564bcf2
mpi_np2                  de53e1405fa71583340a28ba0c54862990facf41fdbca05ad6b573cfa935e7ae  p=5ca275e30b2a u=2e56ac1c0a01 v=273be3989bdd w=21747504be98
```

### G-BIT item 2: 50-step dumps from ckpt_t43 (`prof.py 50 --pcg --dump`, default warm-up 5)

Default `prof.py` flux mode is `python` (the host round-trip mean shift; the device constraint
off); G-BIT runs must use the same command.

| dump | config | ms/step (indicative) | pressure iters/step | reproducibility |
|---|---|---|---|---|
| `dump_omp8.npz` | host 1x8, default wait policy | 41834.75 (load 67.88 70.85 72.03) | 13 x 49, 14 x 1 | `dump_omp8_rep.npz` (passive): **bitwise=True, all 24 arrays** |
| `dump_omp24.npz` | host 1x24, passive | 3139.75 (load 130.96 123.92 110.39) | 13 x 49, 14 x 1 | not repeated (brief) |
| `dump_cuda.npz` | GPU | 134.67 (load 67.88 70.85 72.03) | 13 x 49, 14 x 1 | `dump_cuda_rep.npz`: **bitwise=True, all 24 arrays** |

For reference, not a gate: host 1x8 vs 1x24 differ at 3.7e-14 relative (thread-count-dependent
reduction order; iterations identical), host 1x8 vs GPU at 4.6e-14. Baselines are therefore per
configuration.

### G-NUM item 1: N50_rtol (50 steps, rtol 1e-10 vs 1e-9, same build)

| field | host 1x8 rel | GPU rel |
|---|---|---|
| u | 3.943e-12 | 3.944e-12 |
| v | 6.247e-12 | 6.247e-12 |
| w | 5.834e-12 | 5.835e-12 |
| p | 1.499e-11 | 1.499e-11 |
| C | 1.451e-12 | 1.451e-12 |
| **N50_rtol** | **1.499e-11** | **1.499e-11** |

Pressure iterations at 1e-9: 11–12 per step (mean 11.50) against 13–14 at 1e-10, same on both
backends. Files: `dump_omp8_rtol9.npz`, `dump_cuda_rtol9.npz`, `cmp_omp8_rtol9.txt`,
`cmp_cuda_rtol9.txt`.

### Per-stage timings (`prof.py 20 --warm 3 --pcg --timing`)

`set_vof_timing` stages, ms per step. Host passive wait policy. The GPU runs shared the card with a
production run (89–91 % utilisation by others): compare with the design note's 54.1 ms/step only
after a quiet re-run.

| stage (ms/step) | host 1x24, flux python | host 1x24, flux device | GPU, flux python | GPU, flux device |
|---|---|---|---|---|
| prof.py wall | 3114.59 | 3232.88 | 261.55 | 243.58 |
| step | 3063.206 | 3190.888 | 192.135 | 242.777 |
| predictor | 40.828 | 41.186 | 0.766 | 0.746 |
| momentum_solve | 455.653 | 524.578 | 11.643 | 18.530 |
| projection | 1634.860 | 1778.237 | 67.712 | 85.387 |
| curvature | 322.279 | 294.466 | 37.349 | 43.939 |
| csf | 8.472 | 10.998 | 0.345 | 0.367 |
| block_advect | 533.598 | 460.328 | 68.322 | 86.646 |
| block_debris | 156.886 | 89.938 | 20.937 | 26.405 |
| loadavg before / after | 125.51 124.06 112.79 / 136.62 127.61 114.89 | 136.62 127.61 114.89 / 119.58 124.05 114.62 | 122.42 98.12 83.12 / 121.59 98.35 83.27 | 121.59 98.35 83.27 / 124.71 99.79 83.90 |

### Kernel profiles (20-step difference of a 10- and a 30-step run, `--warm 3 --pcg --flux device`)

GPU: `nsys profile --trace=cuda`, `nsys stats` kernel + memcpy summaries, `kernel_diff.py nsys`
(kernel = launching function; Kokkos labels: `cc_smooth` = `cutcellSmoothColor`, `mg_pfill3` =
`CutcellMG::fillWrap`, `cc_residual` = `residualCutcell`, `prolong` = `prolongAdd`, `mgmeans` /
`mgmeanr` = `CutcellMG::removeMean` for/reduce). Load 127–131, GPU shared as above. Files:
`baseline/nsys/`.

```
per step over 20 steps (nsys): 2560.0 launches, 31.577 ms kernel time
 launches/step   ms/step  kernel
          1.00    11.279  vof::VofCurvature::fallbackBatch
        312.00     4.753  cutcellSmoothColor
         48.00     2.050  ibmRbgsStencilColor
         39.00     1.235  prolongAdd
         16.00     1.151  R:vof::VofBlockSet::interfaceArea
         32.00     1.128  vof::VofCurvature::heightPass
         48.00     0.591  vof::WyAdvector::reconstructImpl
         12.00     0.539  residualVarPin
         14.00     0.527  applyCutcellOpExact
        416.00     0.504  CutcellMG::fillWrap
         48.00     0.436  vof::WyAdvector::computeFluxesImpl#2
         26.00     0.400  R:CutcellMG::dot
        173.00     0.381  Solver::fillAxis
         28.00     0.370  R:CutcellMG::removeMean
         32.00     0.362  S:vof::VofBlockSet::debrisPassBatchImpl#3
         32.00     0.337  R:vof::VofBlockSet::measure
         32.00     0.319  S:vof::VofBlockSet::debrisPassBatchImpl
         39.00     0.308  residualCutcell
          1.00     0.289  vof::VofBlockSet::debrisPassBatchImpl
          3.00     0.285  ibmBuildDiffusionVar
         96.00     0.261  S:vof::WyAdvector::reconstructImpl
         27.00     0.244  CutcellMG::axpy
        152.00     0.244  bcVelocityComp
         39.00     0.234  restrictAvg
         28.00     0.222  CutcellMG::removeMean
         32.00     0.215  vof::VofCurvature::reconstructPlanes
         14.00     0.185  R:CutcellMG::maxabs
         15.00     0.183  R:maxAbsInner
         12.00     0.171  R:maxAbsDiffInner
          3.00     0.134  Solver::buildRhsVar
memcpy per step (kind, size bin): count
   Device-to-Device                >1MiB     37.00
   Device-to-Host                 <=1KiB      1.00
   Device-to-Host                <=64KiB     20.00
   Device-to-Host                   <=8B    128.00
   Host-to-Device                <=64KiB     13.00
```

Per step on the GPU: **149 D->H** copies (128 of <= 8 B, 1 of <= 1 KiB, 20 of <= 64 KiB), **13 H->D**
(<= 64 KiB) and 37 D->D of > 1 MiB (6.7 MB each). The design note's expected `cc_smooth` 312 and
`mg_pfill3` 416 launches per step are confirmed exactly.

Host 1x8: Kokkos simple kernel timer (kokkos-tools `simple-kernel-timer`, built at
`~/Codes/bubble_column_perf/kokkos-tools/`; this version prints its table at finalize — its
`kp_reader` segfaults on no input and is not needed), `kernel_diff.py kp`, passive wait policy,
load 120–130. Times include the load; the launch counts are exact. Files: `baseline/kp/`.

```
per step over 20 steps (kp): 2537.0 launches, 2123.282 ms kernel time
 launches/step   ms/step  kernel
        312.00   378.368  peclet::flow::cc_smooth
        416.00   173.959  peclet::flow::mg_pfill3
         39.00   166.840  peclet::flow::prolong
         48.00   162.382  peclet::flow::ibm_rbgs
         39.00    96.993  peclet::flow::cc_residual
         12.00    95.227  peclet::flow::vmg_resid
          1.00    80.081  vof::curv::pv_batch
         26.00    76.164  mgdot
         28.00    75.685  mgmeanr
         14.00    63.387  peclet::flow::cc_apply_exact
        173.00    54.549  peclet::flow::ibm_pfill
         28.00    48.285  mgmeans
          7.00    40.089  peclet::flow::copyInner
         26.00    38.023  peclet::flow::restrict
         78.00    33.712  Kokkos::Impl::hostspace_parallel_zeromemset
          3.00    27.571  peclet::flow::ibm_build_diff_var
        152.00    27.415  peclet::flow::bc_vel
         37.00    27.173  Kokkos::Impl::host_space_deepcopy_double
          3.00    25.295  rhs_var
         14.00    24.237  mgmax
         48.00    23.754  vof::wy::worklist
         27.00    22.430  mgaxpy
         12.00    21.014  peclet::flow::vmg_maxabsdiff
         15.00    17.889  peclet::flow::vmg_maxabs
        110.00    17.167  vof::block::combine_block
         16.00    14.756  vof::curv::compactG
         65.00    13.643  vof::clampFill
         75.00    12.849  vof::block::move_local
          3.00    12.576  rho_minmax
         48.00    11.642  vof::wy::plic
```

Launches per step host vs GPU: `cc_smooth` 312 / 312, `mg_pfill3` 416 / 416, `cc_residual` 39 / 39,
`prolong` 39 / 39, `mgmeans` 28 / 28 (+ `mgmeanr` 28 / 28).

### E1 (design note sec. 9 Q1): pressure MG depth x bottom, host 1x8

`prof.py 20 --pcg --levels L --bottom B` (warm 5, flux python, passive wait policy).

| levels | bottom | pressure iters/step mean (min-max) | ms/step (1x8) | loadavg |
|---|---|---|---|---|
| 4 | auto | 13.00 (13-13) | 2162.03 | 119.58 124.05 114.62 / 126.27 125.18 115.55 |
| 4 | agglomerated | 13.00 (13-13) | 2267.22 | 126.27 125.18 115.55 / 125.75 125.32 116.20 |
| 5 | auto | 21.50 (20-23) | 3010.36 | 125.75 125.32 116.20 / 124.84 125.50 117.00 |
| 5 | agglomerated | 21.50 (20-23) | 2979.38 | 124.84 125.50 117.00 / 128.87 126.53 118.01 |
| 6 | auto | 22.05 (21-23) | 2886.84 | 128.87 126.53 118.01 / 127.08 126.32 118.59 |
| 6 | agglomerated | 22.05 (21-23) | 3027.57 | 127.08 126.32 118.59 / 124.31 125.73 119.04 |

The default depth (4 levels, bottom = L3) gives 13.00 iterations/step (same as `--levels 4`).
**An exact bottom one or two levels deeper does not keep parity: +8.5 and +9.05 iterations per
step.** Q1's default therefore stands (B1's bottom at the `auto` level L3 of a 4-level hierarchy).
`auto` and `agglomerated` give identical iteration counts at each depth. The ms/step column is
load-dominated and not to be read.

### D1 (sec. 5.12): started, not analysed (WO-13)

```
nohup ~/Codes/bubble_column_perf/d1/run_d1.sh > ~/Codes/bubble_column_perf/d1/d1.log 2>&1 &
# = PYTHONPATH=~/Codes/bubble_column_perf/frozen_base/cuda OMP_NUM_THREADS=4 OMP_PROC_BIND=false \
#   python -u tests/study/vof_perf/d1_tolerance.py --out ~/Codes/bubble_column_perf/d1 \
#   --cases static,hysing,column --rtols 1e-10,1e-9,1e-8,1e-7,1e-6 --column-steps 2000
```

Output: `~/Codes/bubble_column_perf/d1/{static,hysing,column}_rtol{1e-10,1e-09,1e-08,1e-07,1e-06}.json`
(15 files; resumable — an existing JSON is skipped), log `d1/d1.log` ending in `D1 DONE`. Measured
so far on the shared GPU: static ~1 min per rtol, hysing ~5 min per rtol; the column is 2000 steps
at ~0.1–0.25 s/step, so the whole study takes roughly 1.5–2.5 h. The Python API exposes no
momentum iteration count: the column records `last_momentum_residual()` and the momentum stage
time per step instead.

### BLOCKED

- **Snellius `hwloc-ls` / `taskset -cp` of the 1x24 and 8x3 layouts (Q4).** It needs a compute
  node, which is billed; not run without the user's go-ahead. The checkpoint is already on the
  project space for when it is.
