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

---

## 2026-09-25 — WO-1 … WO-5: gates (implementer session)

**Method.** Every build was frozen (`~/Codes/bubble_column_perf/frozen_<wo>/{omp,cuda,flt}`) and gated by
`~/Codes/bubble_column_perf/gbit.sh <wo> <kind>` against the WO-0 baseline files: `state_hash.py`
(12 cases, 8 threads) + its `mpi` case at np 2, compared byte for byte with `baseline/hash_<kind>*.txt`;
the 50-step bubble-column dump (`prof.py 50 --pcg --dump`) compared by `cmp.py` with
`baseline/dump_{omp8,omp24,cuda}.npz` over all 24 arrays (u v w p C dts iters col0..15). All host runs
`OMP_PROC_BIND=false OMP_WAIT_POLICY=passive` (numerics-neutral, WO-0 surprise 1). Raw output:
`~/Codes/bubble_column_perf/gates/<wo>_<kind>/`.

**Deviation (recorded).** The full ctest batteries (host `-LE bench`, the CUDA tree) were run ONCE, on
the final tree, not per work order: at load 60–130 one host battery takes hours. Every work order
got G-BIT items 1, 2 (and 4 where storage changed), which are the bitwise evidence.

| WO | state | host omp (hash, np2, d8, d24) | cuda (hash, np2, dump) | flt (hash, np2) |
|---|---|---|---|---|
| WO-1a `-ffp-contract=off` | 29c3f60 | identical / identical / bitwise / bitwise | compile commands unchanged (no flags.make carries the flag) | identical / identical |
| WO-1b `HOST_ARCH=native` | 17f5061 | identical / identical / bitwise / bitwise | not applicable (ignored on CUDA) | — |
| WO-2 A0+A4 | dbdd158 | identical / identical / bitwise / bitwise | identical / identical / bitwise | — |
| WO-3 A2, as written (TX = +o·gf, consumers read −TX) | not committed | identical / identical / bitwise / bitwise | **DIFFERS** (u 8.8e-15, p 8.1e-15 rel. after 50 steps; staggered_bed…embed + np2 hashes change) | identical / identical |
| WO-4 A3 on top of that | not committed | identical / identical / bitwise / bitwise | DIFFERS (inherited from WO-3) | — |
| WO-3 with the band sign stored (TX = −o·gf) + WO-4 + WO-5a | not committed | identical / identical / bitwise / bitwise | identical / identical / bitwise | identical / identical |
| + WO-5b A6 (final tree) | not committed | identical / identical / bitwise / bitwise | identical / identical / bitwise | — |

Pressure iterations 13.02/step (13 on 49 steps, 14 on one) in every passing dump, as the baseline.

**WO-3 STOPPED on risk R1 (§9).** As specified, the face form changes device bits; host builds
are bitwise (no contraction there). Most likely mechanism (not verified in PTX): nvcc folds the
negation of `(−TX(i+1))·φ + (−TX(i))·φ + …` into the add/sub, which changes which product the FMA
combiner fuses relative to the band form's `AE(i)·φ + AW(i)·φ + …`. Storing the band sign instead —
`TX(i) = −ox(i)·gfx` (= the old AW(i)), so `AE(i) = TX(i+sx)`, `AW(i) = TX(i)` and every consumer
expression is literally the band expression — is bitwise on host, CUDA and the float tree (rows
above). Same storage (3 arrays, 32 B/cell), same bitwise argument, sign convention flipped. That is
a change to §4.2's definition, so it is left to the note's owner; WO-3, WO-4 and WO-5 are held
uncommitted on top of it (replayable from `~/Codes/bubble_column_perf/states/r_wo{3,4,5a}` and the
scripts in the session scratchpad).

**A6 NaN-injection test** (`tests/kokkos/test_pcg_breakdown.cpp`, ctest `pcg_breakdown`): PASS on
host and CUDA — NaN pAp at iteration 3 → it 3, failed, x bitwise the maxit-3 iterate; NaN rᵀz at 3 →
it 3, failed, x bitwise the maxit-4 iterate; NaN rᵀz at the last iteration (maxit 4) → the post-loop
read reports it 3; NaN pAp at iteration 0 → it 0. The reduction-into-a-device-View assumption
(§5.6) holds: the final tree is bitwise on CUDA.

### G-PERF (indicative only: load 60–130, GPU shared with D1 and other sessions)

Host kernel timer, 1x8, 20-step difference (`perf/<wo>/kern_omp8.txt`), ms/step:

| kernel | WO-0 | WO-2 | WO-4 | WO-5a |
|---|---|---|---|---|
| total kernel time | 2123 | 1686 | 1315 | 1696 |
| launches/step | 2537 | 2535 | 2138 | 2099 |
| cc_smooth (312/step) | 378 | 331 | 316 | 428 |
| mg_pfill3 launches/step | 416 | 416 | 19 (+ serial below the cutoff) | 19 |
| mg_pfill3 ms | 174 | 203 | 5 | 6 |
| prolong | 167 | 100 | 88 | 125 |
| cc_residual | 97 | 44 | 45 | 60 |
| mgmeans | 48 | 31 | 30 | 27 |
| cc_apply_exact | 63 | 26 | 35 | 51 |

WO-2's "cc_residual, prolong, mgmeans at least halved": residual −55 %, prolong −40 %, mgmeans −37 %
(noise ±25 % at this load). WO-4's "mg_pfill3 ≤ 60 launches/step": 19 parallel launches (the fills
before prolongation and of the openness; the smaller ones run serially).

GPU (nsys, 20-step difference, `perf/<wo>_cuda/`): kernel time/step 31.6 ms (WO-0) → 30.2 (WO-3neg+4+5a)
→ 30.6 (final); the fused-wrap smoother `cutcellSmoothColorFaceWrap` 4.13 ms/step against
`cutcellSmoothColor` 4.75; `fillWrap` 416 launches/step → gone from the top 30. Synchronisations/step
(CUDA API): cudaStreamSynchronize 730 → 652 and cudaDeviceSynchronize 348 → 376 with A6 (net −50);
A5 removed the preconditioner's full-field copies: D→D memcpys > 1 MiB 37 → 11 per step. The
projection's host reads are r0 + the initial rᵀz + one 24-B packet per iteration = 15 per step
at 13 iterations (≤ iterations + 4). D→H memcpys ≥ 1 KiB: 17/step remain — the GraphAMG bottom's
per-V-cycle rhs/solution round trip, which WO-6 (B1) removes; the transfer gate cannot pass before it.

### Batteries (final tree = WO-1, WO-2 + held WO-3(band sign), WO-4, WO-5a, WO-5b)

- host-openmp (`build_omp2`, `OMP_NUM_THREADS=8 OMP_PROC_BIND=false OMP_WAIT_POLICY=passive`,
  `-LE bench -j8`): **176/176** (175 + the new `pcg_breakdown`), all `_np` MPI cases included.
- nvidia-cuda (`build_mg`, `OMP_NUM_THREADS=4`, `-LE bench -j4`): **176/176**.

### Implementation choices the note did not spell out (reversible, each isolated)

- `fillWrap`'s row-wise rewrite (§5.4) is the HOST branch only; the device keeps its
  one-thread-per-cell launch (rule H item 2: device launches unchanged; a thread per row would copy
  ~130 cells serially and uncoalesced on the GPU). Bitwise either way.
- A6 is the single-rank driver `solvePCGResident`; `distributed_` keeps solvePCG's host-scalar loop
  verbatim (§1, §4.4 "single-rank", §8 "distributed branches byte-identical"). FCG keeps host
  scalars (§5.6 names PCG and Chebyshev only); every single-rank `removeMean` (V-cycle exit,
  Chebyshev, FCG, the eigenvalue estimate) is device-resident. BiCGStab keeps its preconditioner
  copies (A5 names PCG, FCG, Chebyshev and the eigenvalue estimate).
- A5's restriction-zeroing is single-rank only; the distributed branch keeps
  `restrictAvg` + the full zero fill.

D1 (started in WO-0) finished: `~/Codes/bubble_column_perf/d1/d1.log` ends "D1 DONE" — not analysed
here (WO-13).

---

## 2026-10-02 — WO-7: C3 PV fallback, team per target (P5)

**WO-7a (core, branch `pvfit`, worktree `suite/core-pvfit`, commit `cb4c7ba`; NOT tagged or pushed
— the release session tags core).** `pvFitAdd` = `pvFitTerm` (→ `PvTerm {w, B, s[6], ok}`) +
`pvFitAccum` (the `+=` loops verbatim). ctest `vof_pvfit`: 10⁵ random cases (800 stencils × 5³;
random planes, frames, origins, isotropic + anisotropic metrics, cosMin, dW incl. 0; planted zero
normals and missing planes) — a frozen copy of the pre-split `pvFitAdd` vs the new `pvFitAdd` vs
term-to-memory + ordered fold, each in its own kernel, on the default and host spaces. Cuda and
OpenMP: 35574 accepted / 64426 rejected, 776/800 stencils with npoly ≥ 6, **0 accumulator and 0
flag mismatches**. A mutation (`w*(s*B)`) is caught (795 / 776 mismatching stencils).

**WO-7b (flow, branch `vof-pvfit`).** `curvFallbackTeam` (one warp per target, terms in team
scratch in the canonical k order, one lane folds + solves) behind `if constexpr` on the memory
space in `VofCurvature::fallbackBatch` (§5.11's launch site); the host keeps the one-thread loop.
`curvFallbackCell` now shares `curvFallbackFrame` / `curvFallbackStore` with it (expressions
verbatim). Built with `-DPECLET_SIBLING_PECLET_CORE=…/core-pvfit`: **flow main cannot build this
until core carries `pvFitTerm`** (tag + `PECLET_CORE_TAG` bump, or the shared `../core` at it).

| gate | result |
|---|---|
| G-BIT 1, state_hash 12 cases + mpi np2, CUDA | identical to main (and to the WO-0 table) |
| G-BIT 1, same, host-openmp (OMP 8) | identical to main |
| G-BIT 2, 50-step column dump, CUDA / host 1×8 / host 1×24 (passive) | `bitwise=True`, 24/24 arrays each; iters 13.06 mean, identical |
| team vs one-thread, CUDA (`test_vof_blocks` C1 gate: per-block path = team, batched = one-thread) | CSF force, colours, stats 0 differences; pv cells 2876 (periodic) / 2824 (y walls) |

The column G-BIT does **not** exercise the team kernel: the bubble column runs the C1 batched
container (`csfBatchEligible()`), whose tier 3 is `vofCurvListPass(T, 1, …)` in
`vof/block_batch.hpp` (one thread per list entry, device-side counts) — WO-8 moved the production
launch site after §5.11 was written. `fallbackBatch` now serves only the per-block container path
(cut blocks, MPI all-reduce hook, timing, debug). **Open (for the note's author):** whether and how
the team kernel goes into the batched pass. §5.11's league = Σ list lengths is host-known in
`fallbackBatch` but device-resident in the batched path (§4.6: counts stay on the device; WO-8's
≤ 3 host reads per container step). Options: (A) league = the region upper bound, teams past the
device count exit (≈ 2·10⁵ mostly-empty 32-lane teams per call on the column, each reserving 9 kB
scratch → occupancy-limited); (B) one more host read of the counts (breaks ≤ 3 reads); (C) a fixed
league of persistent teams striding over the device-counted entries (no read, no empty teams; a
new kernel shape). Recommended default: (C), else (A).

G-PERF (indicative; GPU idle at sampling, load 6–10): nsys over `test_vof_blocks`, tier-3 kernel of
the per-block path, same workload both builds, 240 launches: main `fallbackBatch` (RangePolicy)
**5.37 ms/launch** → team **1.61 ms/launch** (−70 %, ×3.3) for ≈ 2.9·10³ fallback cells per
launch. Scaled to the column's ≈ 7·10³ targets that is ≈ 3–4 ms against the ≤ 1.5 ms target — a
miss to report, and only realised on the column once the batched pass takes the team kernel.

---

## 2026-10-02 — WO-13: the D1 tolerance study, analysed (P5)

Raw data `~/Codes/bubble_column_perf/d1/{static,hysing,column}_rtol<R>.json` (driver
`tests/study/vof_perf/d1_tolerance.py`, run 2026-09-25 on the frozen CUDA module of flow
**ed05b6f**, shared GPU, load 63–132). Reference rtol 1e-10; the momentum rtol follows the
pressure rtol (its default). Relative differences are against the reference.

| rtol | static max\|u\| (4 rungs, worst) | Hysing v_max | t(v_max) | y_c final | column total vol. drift | per-bubble drift | rise, last 1000 steps | p iters/step column (min–max) | max div_proj column | momentum ms/step (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-10 | ref | ref | ref | ref | 2.8e-14 | 2.0e-13 | ref (1.002478) | 13.85 (12–16) | 1.6e-10 | 10.4 |
| 1e-9 | 2.7e-9 | −1.6e-9 | 0 | −5.3e-10 | 2.7e-13 | 2.7e-12 | 7.2e-14 | 12.16 (11–15) | 1.7e-9 | 11.6 |
| 1e-8 | 1.3e-8 | −1.6e-9 | 0 | −5.7e-10 | 5.6e-12 | 2.2e-11 | 7.7e-9 | 10.52 (9–12) | 1.7e-8 | 5.1 |
| 1e-7 | 1.8e-8 | −1.4e-9 | 0 | −5.7e-10 | 1.2e-10 | 3.6e-10 | 7.7e-9 | 9.15 (8–10) | 1.7e-7 | 5.2 |
| 1e-6 | 3.6e-8 | −1.8e-5 | 0 | −1.2e-5 | 1.6e-9 | 4.4e-9 | 7.8e-9 | 7.43 (7–9) | 1.6e-6 | 5.3 |

Pressure iterations, max per run: static 9–10 → 6 (1e-6), Hysing 18 → 10. No run capped.

**§5.12 acceptance:** (1) static max|u| ≤ 1.05× ref — every rtol (worst 1 + 3.6e-8); (2) Hysing
within 0.2 % — every rtol (worst 1.8e-5); (3) total drift ≤ max(1e-8, 2× ref) = 1e-8 and rise
within 1 % — every rtol (1.6e-9; 7.8e-9). **The loosest passing rtol is 1e-6**, the loosest
tested. Per Q8 (USER default) the solver default stays 1e-10 and only a case script may change.
DECISION (session, 2026-10-02): the published bubble-column case KEEPS 1e-10 (peclet-examples
ac35a14 reverts the 1e-6 change): 1e-6 adds a systematic ~5e-8 volume drift over the production
window where the markers conserve to 1e-12, the study predates b273031, and TBFsolver solves its
pressure exactly; revisit after the D1 rerun on main. The saving 1e-6 would give on the column:
pressure iterations −46 % (13.85 → 7.43), the momentum solve about halved (its residual stop
follows the pressure rtol: median residual 4.7e-16 → 1.9e-9 from 1e-8 on). Step times in the JSONs
(146 → 84 ms median) were taken on a shared GPU at load 63–132 and are not to be quoted.

**Caveats (read before relying on 1e-6 for a long production run):**
1. **Physics provenance.** ed05b6f predates b273031 (variable μ on the MAC control volume's own
   faces), which changed the column's physics; the production column now runs on later flow. The
   study measures rtol *sensitivity* on one build, which that fix should not move, but it is not
   re-measured. Confirmation rerun (resumable, ~1.5–2.5 h GPU; freezes main's module first):
   `nohup ~/Codes/bubble_column_perf/d1_main/run_d1_main.sh > ~/Codes/bubble_column_perf/d1_main/d1.log 2>&1 &`
   — expect the same pattern: every criterion passing at 1e-6, pressure iterations ≈ 14 → ≈ 7.5.
2. **The column window does not decorrelate.** 2000 steps from t = 43 are 3.2 time units; the
   trajectories stay together (max|Δu_x|/max|u_x| at 1e-6: 1.0e-9 at step 100, 3.8e-9 at step 1000,
   2.9e-6 at step 2000). Criterion (3)'s rise velocity is therefore a trajectory comparison, not a
   statistical one: it rules out an immediate departure, not a slow statistical bias.
3. **The volume drift at 1e-6 is systematic**, not a random walk: monotone, no sign change, about
   −8e-10 per 1000 steps. Extrapolated to a t 50–150 production window (≈ 6.3·10⁴ steps) it is
   ≈ 5e-8 relative (1e-8: ≈ 2e-10) — far below any physical effect, but larger than the 2000-step
   criterion suggests.

### WO-7c (2026-10-02): the batched container's tier 3 — persistent warp-teams (coordinator decision: option C)

`vofCurvFallbackTeams` (`src/vof/block_batch.hpp`) replaces pass 1 of `vofCurvListPass` on a
device: a fixed league of `min(concurrency()/32, 65536)` warp-teams (`kVofTier3MaxTeams`) strides
over the concatenated device-counted interfacial entries (prefix of `end − start` per job computed
in-kernel), each entry handled by `curvFallbackTeam`, with a team barrier between entries; no host
read (WO-8's ≤ 3 per container step stands), no empty teams. Host: unchanged loop.

| gate | result |
|---|---|
| 50-step column dump vs main, CUDA / host 1×8 / host 1×24 | `bitwise=True` 24/24 each (the column now runs the team kernel: 24 launches in 23 steps) |
| `ctest -R vof_blocks` (incl. `vof_blocks_mpi` np 1/2/4/8), CUDA and host | 12/12 each |

Tier 3 on the column (nsys, 20 + 3 steps, GPU at 99 % shared with P4, load ≈ 22 — indicative):
main one-thread pass 1 **14.76 ms/call** → persistent teams **7.40 ms/call** (×2.0). Still above
the ≤ 1.5 ms target; to be re-measured on a quiet GPU.

## 2026-10-02 — WO-6: B1, the geometric-Krylov bottom on GPU backends (package P3)

**Builds under test** (worktree `flow-vof-b1`, branch `vof-b1` from origin/main `7f74620`): the
frozen modules `~/Codes/bubble_column_perf/p3/frozen_base/{cuda,omp}` (origin/main, `src/`
unmodified) against `frozen_c4/cuda` (this commit) and the host tree of this commit. RTX 5080
(sm_120), `PECLET_FLOW_MPI=ON`, double operator storage. Raw output: `~/Codes/bubble_column_perf/p3/g/`,
`.../p3/phys/`, `.../p3/xfer/`. Gate driver `p3/run_gate.sh`. The GPU was shared with two other
packages' batteries throughout (90-99 % utilisation by others): **every timing below is indicative
only**; the bitwise and G-NUM results are unaffected.

**What landed.** `CutcellMG::geoBottomSolve` (`src/mac_cutcell_mg.hpp`, `GeoBottomKernel`): one
`TeamPolicy(1, T)` launch per bottom solve, FCG (Polak-Ribiere) on the bottom level's own operator,
preconditioned by one symmetric V-cycle over `sub_` (the geometric levels below the bottom, built by
init()'s `can()` + `mgChooseRatio` rule and held outside `lv_`), tau 1e-8 (relative, infinity norm,
fluid mean removed), cap 100, pre/post/bottom 2/2/12. Eligibility = §5.7 conditions 1-7
(`geoBottomIneligible()`), condition 6 evaluated once per hierarchy at its first setOpenness and
cached. Selection: `diagnostics.set_pressure_bottom_solver('auto' | 'geometric' | 'algebraic')`.
A non-finite inner scalar zeroes the bottom's x and sets a device flag that rides the A6 PCG packet
(now 4 doubles: {pAp, rn, stop, geoFlag}) or, for the other drivers, is folded in by
`lastSolveFailed()` (one scalar read, only when a geometric solve ran since the last read).
T = min(1024, team_size_max) = **640** in the module build (768 in an earlier draft; it is a
property of the compiled kernel's register use, logged under `PECLET_FLOW_MG_DEBUG`). Case:
bottom 16x12x8, sub-levels 8x6x4, 4x3x2, 2x3x2.

**Implementation choices the note did not spell out (each bitwise-neutral against the per-kernel
V-cycle, proved by the unit gate's part (a)):** the team kernel reads periodic neighbours through
the A3 wrapped indices where `smooth()` / `vcycleImpl` do (residual and matvec always, colour
passes on all-even levels) instead of a fill phase; 32-bit index arithmetic; the sub-levels'
operators are coarsened lazily, at the first geometric bottom solve after a setOpenness (same
kernels, same inputs, so a configuration that never takes the path never pays for it); the FCG's
own scalars (not part of M) fuse the x/r update with the sum of the new r. Tried and dropped:
running the smallest levels on one thread (4x slower — a single thread's dependent global-memory
round trips cost more than the barriers they save).

### Gates

| gate | result |
|---|---|
| host G-BIT: `state_hash.py` 12 cases + np2, 8 threads | **identical** to frozen_base (both files) |
| host G-BIT: 50-step bubble column 1x8 and 1x24 | **bitwise=True**, all 24 arrays, both |
| CUDA, `'algebraic'` (GraphAMG on this build), 50-step dump | **bitwise=True** vs frozen_base |
| CUDA `state_hash.py` (12 cases + np2) | **all identical** to frozen_base: no hash case reaches the geometric bottom (agglomeration needs a bottom > 4 cells on an axis; np2 is distributed) — nothing to re-baseline |
| G-NUM 1, N50 (u v w p C, 50 steps from ckpt_t43) | **max rel 1.998e-14** (v; p 9.9e-15) ≤ N50_rtol = 1.499e-11 (WO-0) — and ≤ 2.522e-09, N50 re-measured on origin/main 7f74620 (rtol 1e-9 vs 1e-10: p rel 2.522e-09, u 2.09e-10; the case now amplifies a tolerance change 170x more than at WO-0) |
| G-NUM 2 / WO-6: per-step outer iterations vs `'algebraic'` (= base) | **identical on all 50 steps** (653 = 653, 13 x 49 + 14 x 1) |
| G-NUM 3: per-step `max_open_divergence_projected()` | max ratio to the reference **1.000112** (≤ 2) |
| G-NUM 4: physics (`d1_tolerance.py --cases static,hysing --rtols 1e-10`, CUDA) | static drop max\|u\| 2.552041443788365e-03 both (identical; within 5 %); Hysing case 1 (block path): v_rise max 0.28429521057944 vs ...246 (rel 2.0e-16), its time 1.0484717840861797 both, final y_c rel -1.8e-16 (all within 0.2 %); volume drift 1.71e-14 → 1.69e-14 |
| unit gate (`geo_bottom`, CUDA and host) | (a) M with the mean removals off on both sides: **bitwise** to the per-kernel V-cycle over the same levels, periodic and walls-y (ratio-50 coefficient). (b) full M: **NOT bitwise** — 344 / 638 of 1536 cells differ by at most 4.4e-16 (2.8e-17 of max\|z\|): the exit fluid mean is a TEAM reduction (§5.7: "team reductions produce ... fluid means") and the per-kernel removeMean a Kokkos range reduction, so the summation orders differ. The note's gate as written ("M(r) equals, bitwise, the per-kernel vcycle") cannot hold; the ctest asserts (a) and a 4-eps bound on (b) and prints the strict result. **OPEN for the note's owner.** Inner FCG to tau: 13 (periodic) / 15 (walls-y) iterations, true residual 8.9e-9 / 2.1e-9 r0, no flag |
| transfer gate (nsys, 20-step difference, `--flux device --fixdt`) | **bottom transfers gone**: H→D ≥ 1 KiB 13.05 → **0**, D→H ≥ 1 KiB 19.05 → **2.00** per step. The two left are WO-8's batched-container packets inside step() (1024 B after `vofCsfForceBatch`, 2688 B after `vofBatchBox`: 16 blocks x 8 / 21 doubles), above the gate's literal 1 KiB line; not bottom transfers |
| run-to-run | two 50-step CUDA dumps of the same module bitwise identical |

### G-PERF (indicative: GPU 90-99 % busy with other packages)

`GeoBottomKernel` (nsys): **2.8-2.9 ms per bottom solve, 37-40 ms per step** at 17 inner
iterations (the bubble column's bottoms; 13-15 on the unit problem) — against the model's
0.2 ms / 2.5 ms (§4.1) and the target ≤ 3.5 ms per step: **missed by ~11x**. Wall: 74 ms/step
geometric vs 50 ms/step `'algebraic'` on the same build, back to back. **Risk R3 holds and is
larger than modelled.** The cost is per-thread FP64 throughput on the one SM, not barriers: the
unit problem's solve time rises as T falls (default T: 150 us per FCG iteration; 512: 165; 256: 170; 128: 229;
64: 432; 32: 732) — about 7 000 smoothing updates per V-cycle, each with an FP64 division, at the
5080's 1/64-rate FP64 on one SM. §4.1's premise (≈ 140 kFLOP per inner iteration, ≈ 0.2 ms per
bottom) underestimates it ~10x on this card; on H100 / MI250X (FP64 1:2) the same kernel is
expected 20-30x cheaper (§3.3), which this session could not measure.

## 2026-10-02 — WO-6 / E3 (Q2): the geometric bottom's inner tolerance

`kGeoTau` built at 1e-8, 1e-6 and 1e-5 (one CUDA module each; 1e-6 on an earlier draft of the
kernel, 1e-8 and 1e-5 on the committed source, T = 640), 50 bubble-column steps from ckpt_t43
(`prof.py 50 --pcg --dump --div`) against origin/main:

| tau | outer iterations, 50 steps | per-step diff vs reference | max rel u v w p C | max div ratio | inner iterations per bottom |
|---|---|---|---|---|---|
| reference (GraphAMG, 1e-8 2-norm) | 653 | — | — | — | — |
| 1e-8 | 653 | 0 on every step | 1.998e-14 | 1.000112 | 15-17 |
| 1e-6 | 653 | 0 on every step | 5.156e-14 | 1.000121 | — |
| 1e-5 | 653 | 0 on every step | 6.682e-14 | 1.000254 | 10-12 |

Physics at 1e-5 (`d1_tolerance.py --cases static,hysing --rtols 1e-10`): static max|u|
2.552041443788365e-03 identical; Hysing case 1 v_rise max rel 2.0e-16, its time and the final y_c
identical, volume drift identical; p_iters_max 18 = 18. **Adopted: tau = 1e-5** (the loosest of
the three with per-step outer iterations within +1 — here within 0). `GeoBottomKernel` per step
(nsys, GPU 98 % busy with other work, indicative): 43.5 ms at 1e-8 against 18.9 ms at 1e-5 in
back-to-back runs; the inner-iteration ratio (17 → 11) predicts ~1.5x.

## 2026-10-02 — WO-11: B1b, the geometric bottom with solids (package P3)

**What landed** (§5.14). Condition 6 became component labels: `GeoLabelKernel` (one team launch at
the first operator build of a hierarchy, one scalar read) runs Jacobi min-label propagation between
two buffers over the faces with coefficient > 0 joining two fluid cells (AC > 1e-30) to its fixed
point — every fluid cell ends with the smallest flat index of its component, whatever the order —
then numbers the components in label order and counts their cells. Eligible with 1..64
components; more (or none) go to GraphAMG. The kernel's bottom mean removals (of b, of r every
iteration, at M's exit, of the final x) loop over the components, one team reduction per label in
label order, then one subtraction pass; sub-level means (the "all" scope only) stay per level.
Cells with AC <= 1e-30 keep x = 0 (the x/r update is masked) and enter r as 0 (GraphAMG's identity
rows with a zero rhs: a solid cell's rhs would otherwise reach the coarse levels through the
residual and make M affine). With one component the arithmetic is WO-6's exactly.

### Gates (CUDA unless stated; raw output `~/Codes/bubble_column_perf/p3/{g,sol,xfer}/`)

| gate | result |
|---|---|
| bubble column, 50 steps (1 component) | **bitwise** to WO-6 + E3 (`frozen_e3`), all 24 arrays — the per-component path reduces exactly; T unchanged (640) |
| CUDA `state_hash.py` 12 cases | identical to origin/main: the IBM / porous hash cases are labelled (1 component) but never reach the geometric bottom (their bottoms are not agglomerated) |
| host `state_hash.py` + np2, bubble column 1x8 | identical / **bitwise** to origin/main (host untouched) |
| unit gate `geo_bottom` | + two B1b cases (solid sheet + 3 components, periodic and walls-y): device labels = host union-find (3); M with the means off bitwise to the per-kernel V-cycle; FCG 9 / 10 iterations to tau, true residual 5.5e-6 / 4.0e-6 r0, x = 0 exactly in solid cells, per-component mean of x <= 4e-16 |
| G-NUM on bottoms WITH solids (`p3/solids.py`: N = 64, levels 4, bottom 8^3, PCG rtol 1e-8, 20 steps, periodic + body force; probe geometries plus two solid slabs) | `slab1` (solid coarse layers, 1 component) and `slab2` (2 components), const and variable rho: B1b engaged; outer iterations **identical on every step** (109, 436, 111, 3678); u v w max rel diff vs GraphAMG <= 6.6e-13 against the rtol-x10 floor 2.3e-12 .. 1.6e-6; divergence ratio 1.000. WO-6 + E3 on the same cases falls back to GraphAMG: bitwise to origin/main |
| same, 1-component cut-cell beds (`cyl`, `rings`, `pack` (packing_ring.vti), const / rho slab) | iterations identical every step; velocities <= 7.8e-11 rel (floor 4e-10 .. 7e-10 on pack); fluid-cell pressure identical to 1e-15 on cyl / rings; on pack 47 cells differ by up to 1.8e-8 rel — all in 1-2-cell pockets or slivers the operator decouples, whose pressure is a free constant (velocities unaffected) |
| `'algebraic'` on these cases | bitwise to origin/main |
| transfer gate, packing_ring (`pack:slab`, nsys 20-step difference) | ≥ 1 KiB per step: H→D 13.20 → **0**, D→H 17.20 → **0** (small D→H 14.2 = 14.2) |

## 2026-10-02 — P3 HANDOFF (WO-6, E3, WO-11): state for the next implementer

**Branch `vof-b1`** (worktree `suite/flow-vof-b1`), rebased onto origin/main `4b819fb`, NOT pushed:
`7db4a75` WO-6 (B1), `1d58d56` E3 (tau = 1e-5), `ba8f769` WO-11 (B1b), + this handoff. Gate numbers
are in the three entries above; raw artifacts in `~/Codes/bubble_column_perf/p3/` (frozen modules
`frozen_{base,c4,e3,w11}`, gate driver `run_gate.sh`, transfer census `xfer/xfer.sh`, solids
driver `solids.py`, physics `phys/`).

**Batteries on the rebased tree:** host-openmp `build_omp` (`OMP_NUM_THREADS=8`, `-LE bench -j6`):
**189/189 passed** (incl. the new `geo_bottom`). CUDA `build_cuda` (`OMP_NUM_THREADS=4`, `-LE bench
-j4`): **was still running at handoff** (44/189 passed, 0 failed so far; the GPU was 98 % busy with
two other packages' batteries) — log `~/Codes/bubble_column_perf/p3/ctest_cuda.log`, ends `EXIT n`.
clang-format 18.1.8 clean on the changed C++ files (`flow_bindings.cpp` / `flow_ibm.hpp` are on the
CI exclude list).

**Open for the note's owner (not decided here):**
1. §5.7's unit gate "M(r) equals, bitwise, the per-kernel vcycle" cannot hold: §5.7 also prescribes
   team reductions for the fluid means, whose summation order differs from the per-kernel Kokkos
   range reduction (full M differs by <= 4.4e-16 = 2.8e-17 max|z|; with the mean removals off on
   both sides it IS bitwise). `geo_bottom` asserts the bitwise part and a 4-eps bound on the rest.
2. G-PERF MISSED: the bottom costs 2.8-2.9 ms per solve at tau 1e-8 (37-43 ms/step), ~19 ms/step at
   tau 1e-5, against the target <= 3.5 ms/step (shared GPU, indicative). Cost scales with 1/T
   (FP64 throughput on one SM of the 5080, risk R3), not with barriers. Needs a quiet-GPU and an
   H100 measurement before any redesign; wall time is currently WORSE than GraphAMG on the 5080
   (74 vs 50 ms/step at tau 1e-8). The coordinator should decide whether `auto` may select it on
   FP64-weak cards.
3. NAMING (Q15): `diagnostics.set_pressure_bottom_solver('auto'|'geometric'|'algebraic')` follows
   `diagnostics.set_velocity_solver`; an additive row for `suite/docs/NAMING.md`'s history list is
   not written (umbrella file).
4. Register entries (text in the WO-6 / WO-11 commit messages) to be added by the caller.

**Next commands:**
```bash
tail -n 3 ~/Codes/bubble_column_perf/p3/ctest_cuda.log        # CUDA battery verdict
# if it did not finish / was killed, rerun:
cd ~/Codes/suite/flow-vof-b1 && source ../.venv/bin/activate && export PATH=/usr/local/cuda-13.2/bin:$PATH
OMP_NUM_THREADS=4 OMP_PROC_BIND=false ctest --test-dir build_cuda -LE bench -j4 --output-on-failure \
  > ~/Codes/bubble_column_perf/p3/ctest_cuda.log 2>&1; tail -n 3 ~/Codes/bubble_column_perf/p3/ctest_cuda.log
# quiet-GPU timing of the bottom (kernel ms/step), WO-11 module:
~/Codes/bubble_column_perf/p3/xfer/xfer.sh quiet ~/Codes/bubble_column_perf/p3/frozen_w11/cuda --flux device --fixdt 1.5e-3
```
Untracked in the worktree (not for commit): `data/packing_ring.vti` (symlink to `../flow/data`,
used by `p3/solids.py`'s `pack` geometry).

## 2026-10-03 — §13 WO-D0 … WO-D4: the direct bottom (block-tridiagonal FP32 factor + FP64 FCG)

Branch `vof-b1`, NOT pushed. Commits: WO-D1 factor + M (not wired), WO-D2 `'direct'` (recorded
numerics change; its message carries gates B-E), WO-D3 B1's V-cycle retired + `Geo*` -> `Bottom*`,
WO-D2 G-PERF part 1 (bitwise-neutral), this entry. Raw artifacts: `~/Codes/bubble_column_perf/d/`
(`g*/` gate outputs, `xfer/` nsys censuses, `perf/` 300-step timings, `frozen_*` modules,
`bench_bottom_direct.cpp` + `mg_bottom_direct.profiled.hpp` = the clock64 phase profiler, not in
the tree).

**WO-D0 (quiet RTX 5080, no other GPU process; `xfer.sh … --flux device --fixdt 1.5e-3`).**
B1's `GeoBottomKernel` = **G = 19.06 ms/step** (13.05 launches, 1.46 ms per solve; 19.02 with one
co-tenant: a one-SM kernel is indeed barely slowed). Hence N = 29.1 − G = **10.0 ms**, A = 23.85 − N
= **13.8 ms**. BCs of the column: x periodic (16), **y walls (12)**, z periodic (8) -> slow axis y,
P = 12, b = 128, no border; team sizes factor 1024, solve 896. G-PERF targets from these:
factor + solve <= 2.5 ms/step, factor <= 0.6 ms, projection <= 12.5 ms, step <= 31.7 ms.

**Gates.**

| gate | result |
|---|---|
| A, `bottom_direct` (U1-U7; periodic+border P=16 b=96, periodic P=2, walls-y, B1b sheet periodic + walls-y, two-cell pocket with m_c = 1), CUDA + host | PASS. U1 FP64 residual 4e-15 .. 7e-15 (<= 1e-12), component means <= 3.5e-17 max\|x\|; U2 FP32 2e-7 .. 5e-6 (<= 1e-3); U3 bitwise over T = 32 64 128 256 896 (CUDA) and 1 2 4 (OpenMP); U4 1 FCG iteration everywhere, means <= 3.5e-17 (<= 4e-16); U5 restarts 1, 2-3 iterations; U6 flag + x = 0; U7 FP32 M vs dense FP64 A'^-1 <= 1.2e-6 |
| B, column 50 steps vs `'algebraic'` | 2.67e-14 (<= 1.499e-11); iterations identical (653); div ratio 1.0000; two runs bitwise; `'auto'` = `'direct'` bitwise |
| B, 300 steps | inner FCG 1 (3968x) / 2 (94x): max 2, mean 1.02; restarts 0 in 305 factors; no flag |
| C, solids (10 cases) | iterations identical every step; u v w within the rtol x10 floor; `'algebraic'` bitwise to pre-§13 |
| D | host state_hash + np2 + host column bitwise; CUDA state_hash unchanged; transfers: column 0 / 2 bulk, pack:slab 0 / 0, small reads = WO-11 |
| WO-D3 | CUDA column dump bitwise to WO-D2's (all arrays); host G-BIT holds |
| G-PERF part 1 | factor storage + M bitwise to WO-D1's kernel (bench dump); CUDA column dump bitwise to WO-D3's |
| E, G-PERF | **MISSED** (below) |

**Performance (quiet 5080, 300 steps, `prof.py --pcg --timing`, OMP 8).**

| | `'algebraic'` | `'direct'` (after part 1) | target |
|---|---|---|---|
| step (`s.step()`) ms | 42.6 / 42.6 | 36.1 / 37.4 | <= 31.7 |
| projection ms | 23.7 / 23.7 | 17.2 / 17.8 | <= 12.5 |
| factor kernel ms (1 per step) | — | 6.55 -> **3.08** | <= 0.6 |
| solve kernels ms/step (13.05) | — | 3.13 -> 3.10 (0.24 ms per solve) | factor + solve <= 2.5 |

N (projection minus the two bottom kernels) = 17.2 − 6.2 = 11.0 ms, consistent with WO-D0.

**Why G-PERF misses (profiled with clock64 per phase, `bench_bottom_direct`).** The §13.5 model's
1 µs per dependent phase holds only for phases whose work is short; every long dependent chain
here is L2-latency-bound (~50-100 cycles per step of a dot, with only b = 128 active threads).
Factor after part 1 (9.2 M cycles ~ 3.2 ms): W 31 %, diagonal tiles 26 % (now barrier-bound, 17
barriers per tile x 96 tiles), Q = W^T W 17 %, panel 10 %, trailing 9 %, assembly 8 %. M: forward
57 % / backward 40 %, ~4 µs per plane phase (128-term dots streaming Q_k from L2), ~80 µs per M
against 28-45 [model]; the FCG skeleton's ~8 team reductions add ~100 µs per solve. Dead ends
measured: a diagonal-major layout for L and W (coalesced in theory) was SLOWER (W 3.8 -> 6.1 M
cycles, Q 1.4 -> 2.1 M) and was reverted; `#pragma unroll 16` alone moved M by 20 % and the factor
by nothing. Kokkos caps team scratch at sharedMemPerBlock minus 8 KB (~40 KB on the 5080), so
staging a whole Q_k (64 KB at b = 128) — R-D1 lever 1 — is not available through Kokkos.

**Open for the note's owner (not decided here).**
1. G-PERF is missed by ~2.5x on the bottom (6.2 vs 2.5 ms/step) though the engine already saves
   6.5 ms/step of projection against GraphAMG and 13 against B1. Remaining bitwise-neutral levers
   (estimated): Q on the row-major W with idle threads prefetching; the M bracket (u − e g) and
   (e x) formed once by their owner thread into scratch; idle threads prefetching Q_{k+1} into L1
   during plane k. Numerics-changing levers the note lists: R-D1 lever 2 (a fixed 2-4-way split of
   each dot: more threads in flight in M and W). Whether to take lever 2, or relax the target, is
   the owner's call.
2. Readings taken where §13 was not explicit (each documented in the code): "the remaining axes"
   of the slow-axis fallback = the two axes other than the first choice, longest first, tie to the
   higher index; non-periodic boundary-crossing faces are excluded from the re-summed diagonal and
   the couplings exactly as GraphAMG excludes them; the assembled entry is 1 (diagonal) + faces in
   order + 1/m_c + delta, then cast; U1 measures the residual against the operator M factors (the
   re-summed diagonal); U3 compares M before the FCG's component-mean removal (a team reduction,
   §13.2); the test's "1-cell pocket" is a two-cell pocket whose last plane holds one cell (a
   one-cell component cannot exist: all faces closed => AC = 0 => solid).
3. NAMING row (umbrella `docs/NAMING.md` history, for the caller): "2026-10-03 — flow
   `diagnostics.set_pressure_bottom_solver('auto' | 'direct' | 'algebraic')`: `'direct'` replaces
   the never-released `'geometric'` (B1, branch-only); additive on the release line, no alias owed."
4. Register entries: §13.10 text, items 1-4 as written there; item 5's spelling `'direct'` checked
   against NAMING.md (no conflict).

## 2026-10-03 — WO-12 (D3), WO-9 (C2), WO-10 (B2): gates (package vof-perf3)

Branch `vof-perf3` (worktree `suite/flow-vof-perf3`), on origin/main 6719f01 (no code change
since d02d3b0, the baseline build `flow-main-base`). Raw output:
`~/Codes/bubble_column_perf/perf3/{wo12,wo9,wo10,hash,timing,kp}/`. Conditions throughout: the
RTX 5080 shared with the D1-on-main study (88-99 % busy), host load 23-95 on 48 cores, so every
timing is indicative; the bitwise and G-NUM gates are unaffected.

**WO-12 (D3, §5.13), recorded.** Warm power iterations (k_w = 5 from the kept v_max / v_min)
after a coefficient rebuild; guard = cap or r(3) > r0 -> cold re-estimate + redo, counted in
`diagnostics.num_pressure_chebyshev_restarts`.

| gate | result |
|---|---|
| V-cycles / step, bubble column, Chebyshev rtol 1e-10 | 44.15 -> 24.05 (solve 14.15 -> 14.05, estimate 30 -> 10) |
| projection ms / step (20 steps, min of 3 interleaved rounds) | 172.0 -> 96.8 (MG-PCG 53.8) |
| 2000 Chebyshev steps from ckpt_t43 | 0 guard firings; solve V-cycles 16.30 mean (14..20) |
| G-NUM 1 (50 steps, vs main) | max rel 2.186e-12 (p) <= N50(Chebyshev, 1e-10 vs 1e-9) 2.523e-09 |
| G-NUM 2 | per step within +-1; total 704 -> 723 = **+2.7 % (> +-2 %: literal miss)** |
| G-NUM 3 | <= 2x main at 44/50 steps; 6 steps up to 5.97x where main stopped one V-cycle later; max over the run 1.154e-10 vs main 1.193e-10 |
| G-NUM 4 (Chebyshev runs, CUDA) | static drop max|u| equal to 12 digits; Hysing vmax 0.28429521077005615 vs 0.2842952107700561, t(vmax) equal, yc equal to 2e-16; max V-cycles / solve 23 -> 16 |
| vardensity suite | hydrostatic ratios 3 / 1000 and walls-z / jump-z np 1, 2, 4 pass (np 1 bit-exact) |
| state_hash | only vof_droplet (variable-density Chebyshev) changes, CUDA and host; MG-PCG column dump bitwise |

**WO-9 (C2, §5.10), recorded.** One TeamPolicy launch (team per master block) for measure();
values deferred to read #1 / flushStats().

| gate | result |
|---|---|
| state, 50 steps (u v w p C, dts, iterations, 16 colours) | bitwise vs main on CUDA, host 1x8 and 1x24 (also with stats read every step) |
| stats vs main's per-block reductions, 50 steps x 16 blocks | CUDA rel volume 4.2e-16, area 2.7e-16, centroid 5.6e-16, moments 8.4e-16; host <= 9.2e-15. Velocity rel 1.7e-11 (CUDA) / 2.3e-10 (host) = a centroid error of 3.5e-16 / 4.7e-15: the difference quotient amplifies by \|c\|/\|Δc\| ~ 5e4 |
| state_hash | identical to WO-12's (CUDA, host) |
| VoF block ctests (23, MPI np 1-8) | pass, CUDA and host; the C1 gate now holds the measured stats to 1e-12 (velocity as a centroid error) |
| launches | moments1 + moments2 + area 48 / step -> batch_stats 1 (2.43 ms fenced, 16 teams) |
| G-PERF (shared GPU) | block_advect 31.1 -> 8.7-9.1 ms / step: the 48 fenced syncs were expensive under time-slicing; a quiet-GPU figure is owed (model -0.5 to -1.5) |

**WO-10 (B2, §5.8), recorded.** Host ccReduce3 and mgmeanr = pencil reduction.

| gate | result |
|---|---|
| CUDA | state_hash and the 50-step dump identical to WO-9 |
| host 50 steps vs WO-9 | max rel 2.543e-14 (1x8), 2.626e-14 (1x24) <= N50 host 2.522e-09; iterations identical; div ratio <= 1.000 |
| host physics | static drop and Hysing equal to WO-9 to 10 / 16 digits |
| host state_hash | 11 of 12 cases + np 2 re-baselined (porous unchanged); table in the commit |
| G-PERF | not resolvable at this host load; Snellius |

**Two WO-12 defects the batteries found, fixed in follow-up commits.**
1. The kept iterates are cross-step state; a repartition dropped them (MG rebuild), so an np = 1
   rebalance was no longer bit-exact (porous_redistribute_mpi_np1 rel 1.78e-13 vs tol 0;
   balanced_force_mpi_np1 colo-vof-rebalance V-cycles 71 vs 69). `redistribute` now carries them
   in the registry exchange (step 2b) and hands them back after the MG rebuild (5b), per the
   register rule "redistribute must carry every piece of cross-step state".
2. A first estimate on a zero right-hand side (a drop at rest) kept zero iterates; the warm
   estimate then returned bounds [0, 0] and the solve NaN, invisible to the 3-iteration guard
   because `maxabs` skips NaN (vof_collocated_mpi np 1/2/4 on CUDA: P, C NaN after step 1). Kept
   iterates are now usable only from a non-degenerate estimate (non-zero seeds, finite lmax > 0);
   a degenerate warm estimate is replaced at once by the cold one.
Neither touches a non-degenerate, non-redistributed path: state hashes (CUDA, host) and the
50-step Chebyshev column dump are unchanged by them.

**Batteries** (`ctest -LE bench`, OMP 2 per rank, load up to 147): host 192/194 and CUDA 189/194
on the tree before the follow-ups, every failure one of the two defects above; after them the
affected tests pass: host 29/29 (redistribute_mpi, porous_redistribute_mpi, balanced_force(_mpi),
vof_redistribute_mpi, vof_phase_change(_mpi), predict_weighted_mpi, telescope_mpi, vof_collocated
(_mpi), vardensity*), CUDA 32/32 (the same families); state hashes and the Chebyshev column dump
unchanged by the follow-ups on both backends.

**Readings taken where the note was not explicit (each isolated, reversible):**
1. D3 warm start only after a coefficient rebuild that drops VALID bounds; any setter
   invalidation or hierarchy rebuild -> cold. The guard's 3-iteration test aborts the solve at
   iteration 3 (the redo replaces it anyway); the abandoned V-cycles count in
   last_pressure_iterations. Under `set_pressure_warmstart` the redo restarts from the saved
   previous phi. Counter name `diagnostics.num_pressure_chebyshev_restarts` (NAMING §1.3).
2. C2's previous centroid travels by value in the launch from the host copy (current after read
   #1), not as a separate device array: same values, no extra sync path for migration.
3. B2 keeps the registered rule that reductions are never cut over to serial below 8192 cells.
4. §14 Q-H4 (fold "4 fixed x-lanes per row" into B2): arrived after WO-10 was committed
   (locally, unpushed); per the coordinator's instruction WO-10 is left as is, H-5 follows.

## STATE archive 2026-10-08 (the vof_perf_STATE.md text before the handover rewrite)

# VoF step performance campaign — STATE (rewritten in place)

**Objective.** USER DIRECTIVE 2026-09-25: peclet on par with or better than SOTA on every case it
handles; no GPU<->host transfers in the step. Yardstick: TBFsolver on the bubble column
(peclet-examples benchmarks/bubble-column), same node.

**Where we are (2026-09-25 evening).**
- Landed (flow main 7fdec0d, umbrella d0182e0): 8 bitwise perf commits (GPU 207->104 Chebyshev,
  54 ms MG-PCG), `set_superficial_velocity` (USER name), x-fastest MDRange alias `src/policy.hpp` +
  ctest `iteration_order` (register suite-wide), coupling 412b067.
- Same-node Snellius genoa (tcn538, 24 cores, 7fdec0d): TBFsolver 45-46 ms/step, peclet 186-203
  (projection 125 = 67 %); 192 cores 25-33 vs 76-81. GPU RTX 5080 peclet 54 ms.
- Design: `doc/vof_step_performance_design.md` (architect, 48c2548; brief beside it). Main line
  WO-0..WO-13 -> GPU ~17-21 ms, CPU ~80-125 ms. USER DECISION: main line first, then E2(a) =
  opt-in constant-coefficient (Dodd-Ferrante) driver solved by MG-PCG, accuracy-gated (register
  umbrella e390d73); E2(b) FFT decided later. Defaults taken: -march only site/dev builds; two
  bottom solvers (host GraphAMG, GPU geometric-Krylov); tolerance changes in the case script only;
  no float V-cycle; setter names per the note.

**Status (2026-10-02).** PAGE PUBLISHED: peclet-examples main 008c120, benchmarks/bubble-column
(the comparison; isolated bubble / in-line wake / drafting pair / resolution / Loisy free swarm;
the three peclet defects; cost). Physics findings: single bubbles + in-line wakes agree with TBF
(no smoothing) 1-4 %; swarm differs ~15 % via near-contact (peclet 1.6-2.6-cell film, TBF ~0.8-cell
overlap, both grid-set, converging slowly); Loisy E1 free array at Loisy's ratios: peclet U/U0 0.89
vs ~0.80 (0.99 at ratios 0.02). GPU now 42.7 ms/step (main 035121a+). Landed since: guard
vof_momentum+blocks (2d0a0d0). OPEN: vof_momentum isolated-bubble acceleration seen in the D/h 16
pair run (1.84) but not at D/h 20 (on/off within 7 % transient, 1.6 % settled).

**Status (2026-10-03).** LANDED on flow main: WO-0..5, WO-8 (bitwise), WO-7b/c team-per-target PV
fallback (5a34c69, bitwise, core v1.3.2 released for it; now pinned v1.4.0 by the Anderson session),
device bottom 'direct' = §13 (23a3631; GPU default; quiet RTX 5080 bubble column step 42.6 ->
36.1-37.4 ms, projection 23.7 -> ~17.5; misses §13's 12.5 ms target — L2-latency-bound factor chains;
the next lever splits dots = numerics change; B1 retired). Umbrella fce1c06 (register + NAMING).
PARKED: E2(a) (branch vof-e2, worktree flow-vof-e2): stable in the column but fails static balance
by ~5000x; USER asked to choose park vs third design round (recommended: park). D1: published case
stays rtol 1e-10; D1 rerun on main scripted (~/Codes/bubble_column_perf/d1_main/run_d1_main.sh).
Known pre-existing: PECLET_FLOW_OPERATOR_DOUBLE=OFF (non-default) fails 6 tests on main too
(cell_force_placement, collocated_stability_guard, balanced_force_restart, hydro_force_units,
vof_collocated, balanced_force) — float-storage tolerances.

**Snellius rerun 2026-10-03** (flow d02d3b0, genoa, same-node TBF): 24 cores 1x24 peclet 144 ms
(znver4 140) vs TBF 46 (3.1x, was 4.1x); 8x3 166-169 vs 45-46; 192 cores 82-85 vs 25 (unchanged).
1x24 stages: projection 84.6 (60 %), momentum 14.2, curvature 16.7, block_advect 15.0. Page cost
table updated (peclet-examples aaaeaa0, deployed). Snellius budget warning: "running low".
D1 rerun on main (a0afc9b): same pattern — rtol 1e-8: iterations 13.93 -> 10.55, column volume
drift 5.2e-12 / 2000 steps; 1e-6: 7.49, 1.5e-9. USER decision pending: published case rtol (now 1e-10).
E2(a) PARKED (register 43c78cb; branches vof-e2, vof-e2-e23-stopped2 on origin).

**CPU §14 (2026-10-04, architect; branch cpu14 = 8e5fc53, worktree flow-cpu14):** Snellius kernel
profile (job 27519212): 1x24 kernel 121 ms over 2104 launches + ~25 ms host GraphAMG bottom (serial,
1.5 ms/V-cycle). FINDINGS: the Snellius np=1 runs took the DISTRIBUTED path (init_mpi at size 1) —
single-rank fusions never ran; container kernels serial per block on host (static schedule); 8x3
imbalance = CCD-straddling placement. Plan -> 1x24 ~69 ms (60-78; TBF 46); <50 needs user options
(rtol 1e-6, float V-cycle). DECISIONS: Q-H1 'direct' bottom on host YES (recorded numerics change);
Q-H2 NO (np=1 keeps the distributed path; script fix only). Running: WO-H0..H3 (flow-cpu14),
WO-H4 container (flow-cpu14-h4), WO-12/9/10 (vof-perf3; told to fold Q-H4's 4-lane order into WO-10
if not committed). Then S-1 Snellius (needs user OK: budget low), WO-H5..H8, S-2.
Production column at case rtol 1e-8 running (peclet-examples worktree peclet-examples-rtol, branch
rtol-1e8; frozen module scratchpad/flow_prod6 = a0afc9b) -> page update.

**2026-10-04 later.** LANDED: WO-12/9/10 (flow 935ffaf; Chebyshev 44 -> 24 V-cycles/step); full
battery on main with a CORRECT MPI launcher 194/194 (fresh trees otherwise pick ParaView mpiexec =
singleton MPI tests). Page updated with the rtol 1e-8 production run (peclet-examples dee1a32; drift
0.965 +- 0.042, volume 1.3e-10). CPU §14: WO-H0..H3 (branch cpu14) + WO-H4 (cpu14-h4) DONE, all
bitwise except H-1 (host 'direct' bottom: G-NUM-H passes, G-PERF open — serially 1.35x GraphAMG,
team barriers under load); H4 container kernels 132.9 -> 23.8 ms/step at 8 threads. Branches
COMBINED as cpu14-h4 (16 commits on origin/main 935ffaf); gate running (gate_cpu14.sh). HELD until
S-1 (Snellius, needs USER OK, budget low) measures host 'direct' vs 'algebraic' and the protocol fix
(patch /home/frankp/Codes/bubble_column_perf/patches/bench_peclet_H0.patch; CCD-respecting layouts).
Builds use /home/frankp/Codes/suite/core-v140 (detached v1.4.0) via PECLET_SIBLING_PECLET_CORE.

**Next.** Snellius same-node rerun (bubble_cpu.slurm) + update the page's cost table; host MG launch
structure (CPU still ~4x TBF); WO-9/10/12; E2(a) per the user's answer.

**Old next.** Production column result -> D/h=24 + channel_18 rerun; WO-6 (B1 device
bottom, recorded decision) + WO-13 (D1 tolerance, needs correct physics); WO-7 (C3, core change
+ core tag: coordinate with the release session); E2(a) per §12; register entries of design §11 as
their WOs land; Snellius rerun (`/projects/0/prjs1022/peclet/bubble-cpu/bubble_cpu.slurm`, 15 min);
then UPDATE the page's cost table (Snellius rerun) as WO-6/E2(a) land.

**Harness.** /home/frankp/Codes/bubble_column_perf/ (ckpt_t43.npz, prof.py, cmp.py, run_mpi.py,
bench_cpu.sh, run_mirror.py). Host timings on the workstation are noise (load 50-100): use Snellius.

## 2026-10-04 — §14 WO-H0 … WO-H3 (worktree flow-cpu14, branch cpu14 on 935ffaf; not pushed)

Workstation load 35-160 on 48 cores throughout: timings below are ratios or launch counts; the
absolute numbers are not baselines. Raw output in the session scratchpad (gates, A/B, kernel
listings); the commit messages carry the same numbers.

**WO-H0** (protocol). `run_mpi.py` at np 1 skips `mpi_block`/`init_mpi`, `--rtol` (default 1e-8),
`--dump`; `bench_cpu.sh` passes RTOL (1e-8). Kernel listing of `run_mpi.py` np 1, 1x8: selfCopy 0,
cc_smooth_box 0 (the old script: both present, 13.0 vs 10.0 pressure iterations = rtol 1e-10 vs
1e-8). np 1 dump vs `prof.py --rtol 1e-8 --flux device` dump: bitwise. Baselines on 8e5fc53: rtol
1e-8 dumps 1x8 and 1x24, 10 iterations every step; L3 (GraphAMG incl. buildAmg) 0.492 / 0.384 s
per 250 V-cycles = 1.97 / 1.54 ms per V-cycle (load ~150); container kernels OMP 1 / OMP 8
(load ~150, so indicative only): batch_curv_list 207.4 / 134.6 ms/step (ratio 1.54),
gather_local_sum 1.98, batch_clamp 1.77, batch_debris_mark 1.77, batch_plic 1.42, batch_worklist
1.10, batch_flux 0.98, batch_ghost_zero 1.07, batch_update 1.13, move_local 0.49.

**WO-H1** (`'direct'` bottom on host; recorded). G-NUM-H all pass (commit message): N50-type
difference 3.3e-14, iterations identical (653; 500 at rtol 1e-8), divergence ratio <= 1.00025,
300-step FCG max 2 / mean 1.02 / restarts 0, solids battery iterations identical with velocities
<= 3.1e-12 against floors up to 4.7e-10, static drop and Hysing unchanged, host state_hash
unchanged (no case selects the direct bottom), CUDA bitwise. U3 on OpenMP: factor + M and the
whole FCG bitwise for T in {1, 2, 4, 8}. **G-PERF open:** L3 per V-cycle, 1x8 -- load 60-150:
'direct' 18.6 and 182 ms (barrier-bound single team under oversubscription) vs GraphAMG 1.5-2.0;
load ~35: 'direct' 2.3-2.5 ms at T = 8, 1.04-1.08 ms at T = 4 (OMP 4), 1.74-1.80 serial (OMP 1)
vs GraphAMG 1.29-1.50. The <= 0.35 ms gate needs the 8-lane factor to scale on a quiet node: S-1.

**WO-H2, WO-H3 (a)-(e)** (bitwise): G-BIT PASS on every commit (host state_hash 12/12 + np2, column
50 steps 1x8 and 1x24; CUDA state_hash + np2, column dump). After the rebase onto 935ffaf: CUDA
bitwise to origin/main; host state_hash + np2 identical and `--bottom-solver algebraic` bitwise to
origin/main at 1x8 and 1x24. Launches per step (1x8): ibm_pfill 159.5 -> ibm_pfill3 75.25 (/ 2.12:
66 of the 75 fills cover the two periodic axes only -- walls in y -- so / 3 is out of reach; the
two-axis sites are an implementer DECISION in their own commit); bc_vel 138.5 -> 67.25 (gate <= 30
missed: 66 of 69 calls re-impose ONE component inside the momentum sweeps). Interleaved A/B (min of
3, loaded host; 8 threads / 1 thread): prolong 1.06 / 0.85 (gate 0.45), copyInner 0.25 / 0.09
(0.15), gather_local_sum 0.37 / 0.22 (0.2), rhs_var 0.85 / 0.75 and ibm_build_diff_var 0.90 / 0.89
(0.75; controls moved 0.74-1.04) -- all to be re-measured in S-1.
Batteries (`ctest -LE bench`, final tree): host 194/194; CUDA 193/194 + collocated_stability_guard
passed alone (110 s; it timed out at 3600 s in the battery with 32 processes on the GPU).

## 2026-10-08 — Snellius S-1: host `'direct'` bottom vs `'algebraic'`; §14 lands on main

Genoa node (EPYC 9654, 24 L3s of 8 cores), flow 38e80e6 (cpu14-h4 before the rebase), core v1.4.0,
GCC 13.3 + OpenMPI 5.0.3, `-DPECLET_FLOW_HOST_ARCH=znver4` unless "generic". Protocol H-0: np 1
without `init_mpi`, case rtol 1e-8, ckpt_t43, N = 300 steps after 5 warm-up, 2 repetitions; the
number is the better of the two per-run step medians. Job 27770798 (main), 27770799 (kprof). Raw:
`bubble_column_perf/s1/summary_main.txt`, `summary_kprof.txt` (scripts beside them); remote
`/projects/0/prjs1022/peclet/bubble-cpu/s1/results/`. Rank placement audited per run (never a rank
across two CCDs except the 1x24 rows, which are one rank by construction).

| layout | bottom | ms/step | press it/step |
|---|---|---|---|
| 1x24, cores 0-23 (3 CCDs) | direct / algebraic | 93.10 / 95.81 | 10.19 |
| 1x24, OMP_WAIT_POLICY=active | direct / algebraic | 94.86 / 96.22 | 10.19 |
| 1x24, spread over 12 CCDs (every 4th core) | direct / algebraic | 83.81 / 89.38 | 10.19 |
| 6x4, CCD-aligned | algebraic (direct ineligible multi-rank until WO-H6) | 106.53 | 10.19 |
| 3x8, one rank per CCD | algebraic | 107.87 | 10.19 |
| 8x3, one rank per CCD (8 CCDs) | algebraic | 90.72 | 10.19 |
| 1x24, generic build | direct | 97.95 (znver4 is 5 % faster) | 10.19 |
| TBFsolver 8x3, same node | — | 45.45 / 46.04 (2 runs) | — |

Old published peclet number (old protocol: distributed path at np 1, rtol 1e-10): 144 ms. The
§14.4 model said ≈ 119 after H-0 and ≈ 69 (60-78) for the main line H-0…H-5 + WO-10; measured 93.1
without H-5 (H-5 not implemented).

**Verdict (orchestrator DECISION 2026-10-08):** `'direct'` is faster than GraphAMG at every
single-rank layout (contiguous −2.7 ms, active wait −1.4, spread −5.6), so H-1 stays as designed:
host `'auto'` → `'direct'` where eligible (commit 3687a09 on cpu14-h4 kept, not reverted).

**Kernel profile** (Kokkos simple kernel timer, 20 steps, rank 0). 1x24 direct: 937 launches/step,
93.1 ms kernel time; stage timers (30 steps, loop 95.6 ms/step): projection 55.3, momentum 10.6,
curvature 9.7, block_advect 7.6, predictor 3.1, csf 0.8, debris 0.6 ms. Top kernels (ms/step,
launches): `mg_bottom_factor` 16.9 (1 — the FP32 block-tridiagonal factor is rebuilt every step and
is the single largest kernel), `cc_smooth` 12.4 (240), `batch_curv_list` 7.2 (4), `mgmeanr` 3.9 (22),
`vmg_resid` 3.6 (9), `mgdot` 3.5 (20), `ibm_build_diff_var` 3.5 (3), `ibm_rbgs` 2.9, `prolong` 2.7
(30), `bc_vel` 2.3 (47), `ibm_pfill3` 2.2 (55), `mg_bottom_direct` 1.9 (10). 6x4 algebraic: 1990-2038
launches/step/rank, 68.5-71.8 ms kernel time; projection 67.7 ms of a 107.7 ms step; halo
pack/unpack/selfCopy 379 launches each.

**Trigger answers.** Q-H7: 937 < 1100 launches/step at 1x24, so F4 is NOT triggered (the step, 93,
is above 78, but the rule is AND). Q-H5: spread placement is 10 % faster than contiguous (memory
bandwidth: 12 L3s vs 3); the default stays "publish contiguous", TBFsolver is not rerun at spread —
the spread number is recorded alongside. Q-H3 (T_host): not measured by S-1. Open item (not
designed): `mg_bottom_factor` at 16.9 ms/step is the obvious next host target (factor reuse across
steps, or more lanes).

**Merge gate (cpu14-h4 rebased on main 5795fb0; two mechanical conflicts: `step()`'s domain-BC
re-imposition after the projection — `applyVelocityBcAll` kept, scalar-IBM's
`scalarCaptureOpenFaceFlux` kept after it — and this log's tail).** Baselines rebuilt at 5795fb0
(`../flow-main-base`); scripts `bubble_column_perf/gate_s1merge{,_cuda}.sh`, output
`gate_s1merge/`. Host (host-openmp, 8 threads): state_hash 12/12 + np2 identical, 50-step column
dump bitwise (24 arrays) with `--bottom-solver algebraic` (host `'direct'` differs by design, H-1).
CUDA (nvidia-cuda prefix, nvcc 13.4 — the system toolkit moved 13.2 -> 13.4 on 2026-10-08, which
broke every existing CUDA build tree's cached CUDAToolkit paths; fresh trees `build_cg134` /
`build_cuda134`): state_hash 12/12 + np2 identical, dump bitwise with `algebraic` and with the
default (`'direct'`) bottom. Batteries (`ctest -LE bench`, 231 tests after scalar-IBM): host
231/231; CUDA 227/231 at -j6, the four failures (`scalar_cutcell_g4`, `_g6_2`, `_g6_4`, `_g6_5`)
all `Cuda memory space failed to allocate` on a GPU shared with another session — rerun serially on the idle GPU: 4/4 pass
(g4 6 s, g6_2/4/5 29-31 s), so 231/231.

## 2026-10-08 — A(b): the host bottom factor, a bitwise host schedule (branch bottom-factor-host)

USER DECISION 2026-10-08: A(b) (faster factor, bitwise) first; A(a) (factor reuse across steps,
a numerics change) not in scope. Commits bb10f56 (host schedule + `teamAlgorithm` oracle hook +
ctest U8 + `bench_bottom_factor`), fbe8721 (`kBottomHostFactorTeam = 2`).

**Root cause (measured, `bench_bottom_factor`, walls-y 16x12x8: P 12, b 128 = the column's
bottom; workstation 5965WX under load ~20).** The team kernel's order is T-independent as the
header claims (factor bytes identical at T 1..24). T sweep of the old kernel, ms per factor
(median), T = 1 / 2 / 4 / 8 / 16 / 24: 11.0 / 7.9 / 5.7 / 5.6 / 6.4 / 6.5. Two costs: every scalar
is one serial, latency-bound dot product (T = 1: 11 ms), and every column of every 16-wide
diagonal tile is published by a team barrier (~170 per plane), so T stops paying at 4-8. Not the
assembly (`entry()`: 0.18 ms per factor), not memory.

**The change (host only, `if constexpr` on CCMem == HostSpace; device code untouched).** Per tile
one thread factors the diagonal tile and the panel, the team does the trailing update (~2.5
barriers per tile); W = L^{-1} by rows over contiguous cost-balanced column shares; Q = W^T W by
rows; the border by columns with no barrier between planes; the assembly scatters each face's term
to its partner. Every stored scalar is the team expression with its operands in the same order;
lanes run over independent outputs, each continuing its own accumulator in ascending order (four
terms per pass, one at a time per lane), so `omp simd` under -ffp-contract=off reorders nothing.
Host scratch layouts of Sg/L/W are by columns; the factor storage is unchanged.

**New schedule, ms per factor (same probe, loaded):** T = 1 / 2 / 3 / 4 / 6 / 8: 1.44-1.51 / 1.05-1.09
/ 0.93-1.79 / 0.86-1.70 / 1.43-1.69 / 1.43-1.88 (team algorithm at T 8: 5.2-6.7). Periodic (P 16,
b 96, border): T 2 1.59, old T 8 7.6. Phase split at T 1: assemble 0.18, diagonal tile + panel 0.12,
trailing 0.28, W 0.43, Q 0.38. Production kprof 1x24 `direct` (interleaved A/B x2, before the
rebase): `mg_bottom_factor` 6.59 / 6.53 -> 1.52 / 1.82 ms/step; the step (151 -> 143-151 ms) is
inside the load noise (`cc_smooth` 31-41 ms).

**DECISION:** `kBottomHostFactorTeam = 2` (a constant of its own; the FCG solve keeps
`kBottomHostTeam = 8`): the fastest size that is stable under load here. Alternative 4 (best on a
quiet box) pending the genoa sweep in `bubble_column_perf/s1/s1_bfac.slurm` (written, NOT
submitted: needs the user's OK). Reversible by reverting fbe8721.

**Gates (rebased on main f19ac4b, then onto 41da997 (clang-format of scalar sources only: rebuilt, bottom_direct + state_hash rechecked identical); baseline worktree at f19ac4b, host rebuilt, CUDA fresh trees
nvcc 13.4; every artifact checked newer than the rebuild start).** Factor bytes Q|Y|e|s|stat vs the
baseline: identical (walls-y 804884 B, periodic 1161236 B), host and CUDA. ctest `bottom_direct`:
U8 (host schedule vs team algorithm, FP32 + FP64, T 1/2/4, production floor and the 1e30 restart
path) bitwise on all 6 problems (negative control: swapping two terms of the 4-term pass fails U8);
U3 host and CUDA pass. Host G-BIT: state_hash 12/12 + np2 identical; 50-step column dump with
`--bottom-solver direct`, rtol 1e-8, bitwise at OMP 1, 8, 24 (24 arrays, 10.00 iterations). (The
baseline itself differs between OMP 1 and 24 at 3e-14-1e-13 relative: pre-existing, outside the
bottom.) CUDA: state_hash + np2 identical, dump bitwise, U3 pass. Host battery `ctest -LE bench`:
231/231.

**Device factor (not acted on).** USER 2026-10-08 (relayed by the orchestrator): device A(b)
wanted later — the device factor costs 7–9 ms per launch on the RTX 5080 (walls-y), out of a
36–37 ms step; same goal: bitwise-faster schedule first.
