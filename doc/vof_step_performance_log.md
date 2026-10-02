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
