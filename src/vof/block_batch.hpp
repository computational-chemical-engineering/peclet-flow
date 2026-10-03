/// @file
/// @brief flow — batched execution of the VoF block container's stages (C1,
/// `doc/vof_step_performance_design.md` §4.6, §5.9).
///
/// The block container runs one `WyAdvector` per marker on a small box (~24^3). Run block by block,
/// every stage is ~60 launches and ~8 host reads per block: on a GPU the step is launch latency and
/// pipeline drains, not work. Here each stage runs for ALL master blocks in ONE launch, through a
/// by-value job table of raw device pointers plus extents (the pattern of `DebrisTable` and
/// `VofCurvFallbackTable`); a flat index is mapped to (job, local cell) by walking the table's
/// prefix offsets. Nothing about the container's DATA changes -- every block keeps its own Views --
/// and every per-cell body is the existing inline function (`wyReconstructCell`, `wyFaceFlux`, the
/// update and ghost-fill expressions verbatim), so the state is bit-identical to the per-block
/// path.
///
/// Compaction counts stay on the device: a scan writes each job's list start and end, and the
/// consumer launches over the upper bound (the job's region) and exits at `t >= count`.
///
/// The table rides in the functor by value. With nvcc's grid-constant kernel parameters (Kokkos
/// `KOKKOS_IMPL_CUDA_USE_GRID_CONSTANT`, 32 KiB) a table of 16 jobs is passed as a kernel argument
/// -- no constant-memory staging, no host synchronisation per launch.
#ifndef PECLET_FLOW_VOF_BLOCK_BATCH_HPP
#define PECLET_FLOW_VOF_BLOCK_BATCH_HPP

#include <Kokkos_Core.hpp>

#include "mac_stencils.hpp"  // peclet::flow::SExec, SField, I3, L3
#include "vof/advect_wy.hpp"
#include "vof/curvature_field.hpp"  // VofRawField

namespace peclet::flow::vof {

/// Read-only raw accessor of an unsigned-char field (the WO-R out-of-domain mask).
struct VofRawMask {
  const unsigned char* p;
  KOKKOS_INLINE_FUNCTION unsigned char operator()(long i) const { return p[i]; }
};

/// One master block's share of a batched container launch. Every pointer is a View the block's own
/// `WyAdvector` owns; nothing is copied.
struct VofBlockJob {
  double *c, *mx, *my, *mz, *al, *fl;
  double* u[3];                  ///< face velocities uf, vf, wf
  unsigned char* cc;             ///< the frozen dilation flag
  const unsigned char* outside;  ///< the WO-R out-of-domain mask; null when there is none
  I3 e, n, o;                    ///< extended extent, inner counts, global index of inner (0,0,0)
  int g;                         ///< ghost width
  double dth;                    ///< dt / h of this block's advector
  double weps;                   ///< the advector's wisp tolerance
  bool interfaceCfl;             ///< `interfaceLocalCfl`: maxCourantInterface, else maxCourant
  bool span[3];                  ///< the block spans this globally periodic axis (wrap in-block)
  unsigned char bcOwn[6];        ///< WO-R: the global domain faces this block owns
};

/// Jobs per launch; a container with more master blocks runs each stage in chunks.
inline constexpr int kVofBlockBatch = 16;

/// The job table of one launch, by value. `off[k]` is the prefix offset of job k's range in the
/// launch's flat index (the RANGE differs per stage: the extended block, the inner region, the
/// inner region grown by one, a sweep's face range); `base` is the global index of job 0 (for the
/// per-job device arrays shared by the chunks).
struct VofBlockTable {
  VofBlockJob job[kVofBlockBatch];
  long off[kVofBlockBatch + 1];
  int nj;
  int base;
};

/// Job of flat index `t` (`off[0] <= t < off[nj]`): a walk over at most 16 offsets.
KOKKOS_INLINE_FUNCTION int vofJobOf(const long* off, long t) {
  int b = 0;
  while (t >= off[b + 1])
    ++b;
  return b;
}

/// Fill `T.off` from a per-job range size.
template <class F>
inline long vofSetOffsets(VofBlockTable& T, F size) {
  T.off[0] = 0;
  for (int k = 0; k < T.nj; ++k)
    T.off[k + 1] = T.off[k] + size(T.job[k]);
  for (int k = T.nj; k < kVofBlockBatch; ++k)
    T.off[k + 1] = T.off[T.nj];
  return T.off[T.nj];
}

inline long vofJobLen(const VofBlockJob& J) {
  return static_cast<long>(J.e.x) * J.e.y * J.e.z;
}
inline long vofJobInner(const VofBlockJob& J) {
  return static_cast<long>(J.n.x) * J.n.y * J.n.z;
}
/// The inner region grown by one cell per side: `WyAdvector::reconstructImpl`'s range.
inline long vofJobGrown1(const VofBlockJob& J) {
  return static_cast<long>(J.n.x + 2) * (J.n.y + 2) * (J.n.z + 2);
}

// ---- H-4: the host launch forms (`doc/vof_step_performance_design.md` §14.3) --------------------
//
// On a host backend GCC compiles a static `RangePolicy` to `schedule(static)`: one contiguous block
// of the flat index per thread. The device forms above therefore serialise on a host -- a list
// kernel launched over the upper bound has each block's active entries at the head of its range,
// on one thread -- and every region kernel pays 64-bit div/mod per cell. The launchers below are
// the host branches; each batched launcher selects them at compile time on its execution space's
// memory space (as `ccdetail::ccRows3` does), so a device build never instantiates them and its
// launches are untouched. Every host form visits the same cells with the same per-cell body, and
// every body writes only its own cell(s) or adds integers atomically, so the state is bitwise the
// device form's on the same host.

/// The host branches select on this (§14.2).
template <class Exec>
inline constexpr bool kVofHostExec = std::is_same_v<typename Exec::memory_space, Kokkos::HostSpace>;

/// The exact offset table of a host list launch: job k's entries are [o[k], o[k + 1]).
struct VofCountTable {
  long o[kVofBlockBatch + 1];
};

/// H-4(a): a host list kernel over the EXACT entry counts, `cnt(k)` (read directly: the count
/// Views are HostSpace), with a dynamic schedule in chunks of 16 -- so the active entries spread
/// over every thread instead of sitting at the head of one block's static share. `body(k, t)` is
/// entry t (0-based) of job k: the device form's entry `q = start + t`, which exits for `t >= cnt`.
template <class Exec, class Cnt, class F>
inline void vofHostListFor(const char* name, int nj, Cnt cnt, F body) {
  Exec().fence();  // the counts were written by an earlier launch
  VofCountTable C;
  C.o[0] = 0;
  for (int k = 0; k < nj; ++k) {
    const long n = cnt(k);
    C.o[k + 1] = C.o[k] + (n > 0 ? n : 0);
  }
  for (int k = nj; k < kVofBlockBatch; ++k)
    C.o[k + 1] = C.o[nj];
  Kokkos::RangePolicy<Exec, Kokkos::Schedule<Kokkos::Dynamic>> pol(Exec(), 0, C.o[nj]);
  pol.set_chunk_size(16);
  Kokkos::parallel_for(name, pol, [=](const long t) {
    const int k = vofJobOf(C.o, t);
    body(k, t - C.o[k]);
  });
}

/// The `[start, end)` count of job k of a list launch.
template <class Tab>
inline auto vofListCount(const Tab& T, LField start, LField end) {
  return [=](int k) { return end(T.base + k) - start(T.base + k); };
}

// The host x loops of the region kernels carry `omp simd` where `ccFor3`'s contract holds (an
// iteration writes only its own cell and reads no cell another iteration of the row writes); with
// the host's -ffp-contract=off that is bit-identical to the scalar loop. Same definition as
// `mac_cutcell.hpp`'s.
#ifndef PECLET_FLOW_OMP_SIMD
#if defined(_OPENMP)
#define PECLET_FLOW_OMP_SIMD _Pragma("omp simd")
#else
#define PECLET_FLOW_OMP_SIMD
#endif
#endif

/// The row offsets of a host region launch: job k's rows are [off[k], off[k + 1]).
struct VofRowTable {
  long off[kVofBlockBatch + 1];
};

/// H-4(b): a host region kernel over ROWS -- `rows(k)` x-runs for job k, `row(k, r)` handling job
/// k's local row r with the x loop inside -- so the job search and the index math run once per row,
/// not once per cell. Static schedule: the rows are uniform, as in `ccFor3`.
template <class Exec, class Rows, class F>
inline void vofHostRowFor(const char* name, int nj, Rows rows, F row) {
  VofRowTable R;
  R.off[0] = 0;
  for (int k = 0; k < nj; ++k)
    R.off[k + 1] = R.off[k] + rows(k);
  for (int k = nj; k < kVofBlockBatch; ++k)
    R.off[k + 1] = R.off[nj];
  Kokkos::parallel_for(name, Kokkos::RangePolicy<Exec>(Exec(), 0, R.off[nj]), [=](const long t) {
    const int k = vofJobOf(R.off, t);
    row(k, t - R.off[k]);
  });
}

// ---- stage 1: the Courant numbers ---------------------------------------------------------------

/// One team per job: `WyAdvector::maxCourant` or `maxCourantInterface` (per the job's
/// `interfaceCfl`), the same predicates and the same `fmax` chain. Max is order-free, so the team
/// reduction returns the per-block reduction's value exactly. `out(base + k)` = the job's Courant
/// number (not yet all-reduced: a block advector has no `globalMax`).
inline void vofBatchCfl(const VofBlockTable& T, SField out) {
  using Team = Kokkos::TeamPolicy<SExec>;
  Kokkos::parallel_for(
      "vof::block::batch_cfl", Team(SExec(), T.nj, Kokkos::AUTO),
      KOKKOS_LAMBDA(const typename Team::member_type& tm) {
        const int k = tm.league_rank();
        const VofBlockJob& J = T.job[k];
        const I3 e = J.e, n = J.n;
        const int g = J.g;
        const long sx = 1, sy = e.x, sz = static_cast<long>(e.x) * e.y;
        const long region = static_cast<long>(n.x) * n.y * n.z;
        const VofRawField c{J.c}, u{J.u[0]}, v{J.u[1]}, w{J.u[2]};
        const double weps = J.weps;
        const bool band0 = J.interfaceCfl;
        double m = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(tm, region),
            [&](const long r, double& acc) {
              const int x = static_cast<int>(r % n.x) + g;
              const int y = static_cast<int>((r / n.x) % n.y) + g;
              const int z = static_cast<int>(r / (static_cast<long>(n.x) * n.y)) + g;
              const long i = L3(x, y, z, e);
              if (!band0) {  // WyAdvector::maxCourant
                acc = Kokkos::fmax(acc, Kokkos::fabs(u(i)));
                acc = Kokkos::fmax(acc, Kokkos::fabs(v(i)));
                acc = Kokkos::fmax(acc, Kokkos::fabs(w(i)));
                return;
              }
              // WyAdvector::maxCourantInterface
              const double ci = c(i);
              const bool band =
                  wyIsMixed(ci, weps) || wyColourJump(c(i - sx), ci, weps) ||
                  wyColourJump(c(i + sx), ci, weps) || wyColourJump(c(i - sy), ci, weps) ||
                  wyColourJump(c(i + sy), ci, weps) || wyColourJump(c(i - sz), ci, weps) ||
                  wyColourJump(c(i + sz), ci, weps);
              if (!band)
                return;
              acc = Kokkos::fmax(acc, Kokkos::fabs(u(i)));
              acc = Kokkos::fmax(acc, Kokkos::fabs(u(i - sx)));
              acc = Kokkos::fmax(acc, Kokkos::fabs(v(i)));
              acc = Kokkos::fmax(acc, Kokkos::fabs(v(i - sy)));
              acc = Kokkos::fmax(acc, Kokkos::fabs(w(i)));
              acc = Kokkos::fmax(acc, Kokkos::fabs(w(i - sz)));
            },
            Kokkos::Max<double>(m));
        Kokkos::single(Kokkos::PerTeam(tm), [&]() {
          // the two return expressions of maxCourant / maxCourantInterface
          out(T.base + k) = band0 ? Kokkos::fmax(m, 0.0) * J.dth : m * J.dth;
        });
      });
}

// ---- stage 2: the frozen dilation flag ----------------------------------------------------------

/// `WyAdvector::freezeDilationFlag` over every job's extended block. `T.off` = extended lengths.
/// Host: the extended block's rows (H-4b).
template <class Exec = SExec>
inline void vofBatchFreeze(const VofBlockTable& T) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostRowFor<Exec>(
        "vof::block::batch_freeze", T.nj,
        [&](const int k) { return static_cast<long>(T.job[k].e.y) * T.job[k].e.z; },
        [=](const int k, const long r) {
          const VofBlockJob& J = T.job[k];
          const long i0 = r * J.e.x;  // extended row r = (r % e.y, r / e.y)
          const double* c = J.c;
          unsigned char* cc = J.cc;
          PECLET_FLOW_OMP_SIMD
          for (int x = 0; x < J.e.x; ++x)
            cc[i0 + x] = c[i0 + x] > 0.5 ? 1u : 0u;
        });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_freeze", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const VofBlockJob& J = T.job[k];
        const long i = t - T.off[k];
        J.cc[i] = J.c[i] > 0.5 ? 1u : 0u;
      });
}

// ---- stage 3a/b: the worklist and the PLIC reconstruction --------------------------------------

/// `WyAdvector::reconstructImpl`'s worklist scan, over the concatenated grown-by-one regions
/// (`T.off` = `vofJobGrown1`). The list of job k is written at `list(listBase + start(k) ...)` in
/// the per-block scan's order; `start(base + k)` / `end(base + k)` receive the job's list range
/// (absolute positions), so its count never leaves the device.
inline void vofBatchWorklist(const VofBlockTable& T, LField list, LField start, LField end) {
  Kokkos::parallel_scan(
      "vof::block::batch_worklist", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t, long& upd, const bool final) {
        const int k = vofJobOf(T.off, t);
        const VofBlockJob& J = T.job[k];
        const long r = t - T.off[k];
        const int rx = J.n.x + 2, ry = J.n.y + 2;
        const int ix = static_cast<int>(r % rx);
        const int iy = static_cast<int>((r / rx) % ry);
        const int iz = static_cast<int>(r / (static_cast<long>(rx) * ry));
        const int g = J.g;
        const long i = L3(g - 1 + ix, g - 1 + iy, g - 1 + iz, J.e);
        const bool mixed = wyIsMixed(J.c[i], J.weps);
        if (final) {
          if (r == 0)
            start(T.base + k) = upd;
          if (mixed)
            list(upd) = i;
          if (t + 1 == T.off[k + 1])
            end(T.base + k) = upd + (mixed ? 1 : 0);
        }
        if (mixed)
          ++upd;
      });
}

/// `wyReconstructCell` over each job's compacted list: launched over the upper bound (the job's
/// region), a thread past its job's count exits. Host: the exact counts (H-4a).
template <class Exec = SExec>
inline void vofBatchPlic(const VofBlockTable& T, LField list, LField start, LField end) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostListFor<Exec>("vof::block::batch_plic", T.nj, vofListCount(T, start, end),
                         [=](const int k, const long t) {
                           const long q = start(T.base + k) + t;
                           const VofBlockJob& J = T.job[k];
                           const long sy = J.e.x, sz = static_cast<long>(J.e.x) * J.e.y;
                           wyReconstructCell(VofRawField{J.c}, list(q), sy, sz, VofRawField{J.mx},
                                             VofRawField{J.my}, VofRawField{J.mz},
                                             VofRawField{J.al});
                         });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_plic", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const long s0 = start(T.base + k);
        const long q = s0 + (t - T.off[k]);
        if (q >= end(T.base + k))
          return;
        const VofBlockJob& J = T.job[k];
        const long sy = J.e.x, sz = static_cast<long>(J.e.x) * J.e.y;
        wyReconstructCell(VofRawField{J.c}, list(q), sy, sz, VofRawField{J.mx}, VofRawField{J.my},
                          VofRawField{J.mz}, VofRawField{J.al});
      });
}

// ---- stage 3c/d: fluxes and the sweep update ----------------------------------------------------

/// A sweep's face range of one job: the inner region with one extra layer on the low side of `d`.
inline long vofJobFaces(const VofBlockJob& J, int d) {
  return static_cast<long>(J.n.x + (d == 0)) * (J.n.y + (d == 1)) * (J.n.z + (d == 2));
}

/// The flux of `d`-face `p` of job `J` (the per-cell body of `vofBatchFlux`, both launch forms).
KOKKOS_INLINE_FUNCTION void vofBatchFluxCell(const VofBlockJob& J, int d, long p) {
  const I3 e = J.e;
  const long sd = d == 0 ? 1 : (d == 1 ? static_cast<long>(e.x) : static_cast<long>(e.x) * e.y);
  const VofRawField c{J.c}, mx{J.mx}, my{J.my}, mz{J.mz}, al{J.al}, u{J.u[d]};
  if (J.outside != nullptr) {
    J.fl[p] =
        wyFaceFluxBc(u(p) * J.dth, p, sd, d, c, mx, my, mz, al, VofRawMask{J.outside}, J.weps);
    return;
  }
  J.fl[p] = wyFaceFlux(u(p) * J.dth, p, sd, d, c, mx, my, mz, al, J.weps);
}

/// `WyAdvector::computeFluxesImpl` (uncut): one flux per `d`-face, `wyFaceFlux` (or, for a job
/// carrying the WO-R mask, `wyFaceFluxBc`). `T.off` = `vofJobFaces(d)`. Host: the face range's
/// rows (H-4b).
template <class Exec = SExec>
inline void vofBatchFlux(const VofBlockTable& T, int d) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostRowFor<Exec>(
        "vof::block::batch_flux", T.nj,
        [&](const int k) {
          const VofBlockJob& J = T.job[k];
          return static_cast<long>(J.n.y + (d == 1)) * (J.n.z + (d == 2));
        },
        [=](const int k, const long r) {
          const VofBlockJob& J = T.job[k];
          const int ay = J.n.y + (d == 1), g = J.g;
          const int y = static_cast<int>(r % ay) + g - (d == 1);
          const int z = static_cast<int>(r / ay) + g - (d == 2);
          const long i0 = L3(0, y, z, J.e);
          const int x0 = g - (d == 0), x1 = g + J.n.x;
          PECLET_FLOW_OMP_SIMD
          for (int x = x0; x < x1; ++x)
            vofBatchFluxCell(J, d, i0 + x);
        });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_flux", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const VofBlockJob& J = T.job[k];
        const long r = t - T.off[k];
        const int ax = J.n.x + (d == 0), ay = J.n.y + (d == 1);
        const int g = J.g;
        const int x = static_cast<int>(r % ax) + g - (d == 0);
        const int y = static_cast<int>((r / ax) % ay) + g - (d == 1);
        const int z = static_cast<int>(r / (static_cast<long>(ax) * ay)) + g - (d == 2);
        vofBatchFluxCell(J, d, L3(x, y, z, J.e));
      });
}

/// The WO-R boundary ledger (`WyAdvector::accumulateBcFaceVolume`) of the jobs carrying the mask:
/// one thread per (job, side), summing its plane of `d`-fluxes in (p0 fastest, p1) row order into
/// `out(stride (base + k) + slot0 + face)` -- a fixed order, which is a C2 order change against the
/// per-block MDRange2 reduction (recorded; `doc/vof_step_performance_design.md` §5.9). The block
/// container installs no mask today, so the path is inert on every shipped configuration.
inline void vofBatchBcLedger(const VofBlockTable& T, int d, SField out, int stride, int slot0) {
  Kokkos::parallel_for(
      "vof::block::batch_bcvol", Kokkos::RangePolicy<SExec>(SExec(), 0, 2 * T.nj),
      KOKKOS_LAMBDA(const long t) {
        const int k = static_cast<int>(t / 2), s = static_cast<int>(t % 2);
        const VofBlockJob& J = T.job[k];
        const int f = 2 * d + s;
        if (J.outside == nullptr || !J.bcOwn[f])
          return;
        const I3 e = J.e, n = J.n;
        const int g = J.g;
        const int b = (d + 1) % 3, c2 = (d + 2) % 3;
        const long st[3] = {1, e.x, static_cast<long>(e.x) * e.y};
        const long sd = st[d], sb = st[b], sc = st[c2];
        const int nn[3] = {n.x, n.y, n.z};
        const int fa = (s == 0) ? (g - 1) : (g + nn[d] - 1);
        double acc = 0.0;
        for (int p1 = g; p1 < g + nn[c2]; ++p1)
          for (int p0 = g; p0 < g + nn[b]; ++p0)
            acc += J.fl[static_cast<long>(p0) * sb + static_cast<long>(p1) * sc +
                        static_cast<long>(fa) * sd];
        out(static_cast<long>(stride) * (T.base + k) + slot0 + f) += (s == 0) ? acc : -acc;
      });
}

/// The sweep update of inner cell `i` of job `J` (the per-cell body of `vofBatchUpdate`).
KOKKOS_INLINE_FUNCTION void vofBatchUpdateCell(const VofBlockJob& J, int d, long i) {
  const I3 e = J.e;
  const long sd = d == 0 ? 1 : (d == 1 ? static_cast<long>(e.x) : static_cast<long>(e.x) * e.y);
  const double dth = J.dth;
  const VofRawField c{J.c}, fl{J.fl}, u{J.u[d]};
  const unsigned char* cc = J.cc;
  // The dilation term must scale the SAME uf by the SAME dt/h as the flux, or the exact
  // cancellation in full cells (advect_wy.hpp header) is lost to rounding.
  const double aP = u(i) * dth, aM = u(i - sd) * dth;
  const double dil = cc[i] ? (aP - aM) : 0.0;
  c(i) = c(i) + (fl(i - sd) - fl(i)) + dil;
}

/// `WyAdvector::applySweepImpl` (uncut) over the inner regions (`T.off` = `vofJobInner`); the
/// expression verbatim. Host: the inner rows (H-4b).
template <class Exec = SExec>
inline void vofBatchUpdate(const VofBlockTable& T, int d) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostRowFor<Exec>(
        "vof::block::batch_update", T.nj,
        [&](const int k) { return static_cast<long>(T.job[k].n.y) * T.job[k].n.z; },
        [=](const int k, const long r) {
          const VofBlockJob& J = T.job[k];
          const int g = J.g;
          const long i0 =
              L3(0, static_cast<int>(r % J.n.y) + g, static_cast<int>(r / J.n.y) + g, J.e);
          PECLET_FLOW_OMP_SIMD
          for (int x = g; x < g + J.n.x; ++x)
            vofBatchUpdateCell(J, d, i0 + x);
        });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_update", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const VofBlockJob& J = T.job[k];
        const long r = t - T.off[k];
        const I3 e = J.e, n = J.n;
        const int g = J.g;
        const int x = static_cast<int>(r % n.x) + g;
        const int y = static_cast<int>((r / n.x) % n.y) + g;
        const int z = static_cast<int>(r / (static_cast<long>(n.x) * n.y)) + g;
        vofBatchUpdateCell(J, d, L3(x, y, z, e));
      });
}

// ---- stage 3e: the block ghost policy (`VofBlockSet::fillBlockGhosts`), pass by pass ------------

/// Pass 1: every ghost cell of every job -> 0. `T.off` = extended lengths. Host: only the ghost
/// shell's rows, in `CutcellMG::fillWrap`'s three-slab split (H-4b) -- the z-ghost planes (every
/// y), the y-ghost rows of the inner z range, then the two x-ghost runs of each inner row.
template <class Exec = SExec>
inline void vofBatchGhostZero(const VofBlockTable& T) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostRowFor<Exec>(
        "vof::block::batch_ghost_zero", T.nj,
        [&](const int k) {
          const VofBlockJob& J = T.job[k];
          return static_cast<long>(J.e.z - J.n.z) * J.e.y +
                 static_cast<long>(J.n.z) * (J.e.y - J.n.y) + static_cast<long>(J.n.z) * J.n.y;
        },
        [=](const int k, const long r) {
          const VofBlockJob& J = T.job[k];
          const I3 e = J.e, n = J.n;
          const int g = J.g;
          const long rZ = static_cast<long>(e.z - n.z) * e.y,
                     rY = static_cast<long>(n.z) * (e.y - n.y);
          double* c = J.c;
          if (r < rZ + rY) {  // a whole ghost row
            int y, z;
            if (r < rZ) {  // z-ghost plane row
              const int l = static_cast<int>(r / e.y);
              y = static_cast<int>(r - static_cast<long>(l) * e.y);
              z = l < g ? l : n.z + l;
            } else {  // y-ghost row of the inner z range
              const long t2 = r - rZ;
              const int zi = static_cast<int>(t2 / (e.y - n.y));
              const int l = static_cast<int>(t2 - static_cast<long>(zi) * (e.y - n.y));
              y = l < g ? l : n.y + l;
              z = g + zi;
            }
            const long i0 = L3(0, y, z, e);
            PECLET_FLOW_OMP_SIMD
            for (int x = 0; x < e.x; ++x)
              c[i0 + x] = 0.0;
            return;
          }
          const long t3 = r - rZ - rY;  // the x-ghost runs of inner row (yi, zi)
          const int zi = static_cast<int>(t3 / n.y),
                    yi = static_cast<int>(t3 - static_cast<long>(zi) * n.y);
          const long i0 = L3(0, g + yi, g + zi, e);
          for (int x = 0; x < g; ++x)
            c[i0 + x] = 0.0;
          for (int x = g + n.x; x < e.x; ++x)
            c[i0 + x] = 0.0;
        });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_ghost_zero", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const VofBlockJob& J = T.job[k];
        const long i = t - T.off[k];
        const I3 e = J.e, n = J.n;
        const int g = J.g;
        const int x = static_cast<int>(i % e.x);
        const int y = static_cast<int>((i / e.x) % e.y);
        const int z = static_cast<int>(i / (static_cast<long>(e.x) * e.y));
        if (x >= g && x < g + n.x && y >= g && y < g + n.y && z >= g && z < g + n.z)
          return;
        J.c[i] = 0.0;
      });
}

/// Pass 2 on axis `a` (`colour_field.hpp::periodicFill`, one axis): only the jobs whose block spans
/// the periodic axis `a` are in `T`; `T.off` = the transverse plane sizes `dims[b] * dims[c]`.
/// Host: only the rows of the two `a`-ghost slabs (H-4b) -- for `a` = x the `g` cells at each end
/// of every extended row, for `a` = y or z whole x-rows. Each ghost cell is written once, from a
/// cell inside along `a`, as in the device form.
template <class Exec = SExec>
inline void vofBatchPeriodic(const VofBlockTable& T, int a) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostRowFor<Exec>(
        "vof::block::batch_periodic", T.nj,
        [&](const int k) {
          const I3 e = T.job[k].e;
          const int g = T.job[k].g;
          return a == 0 ? static_cast<long>(e.y) * e.z
                        : (a == 1 ? 2L * g * e.z : static_cast<long>(e.y) * 2 * g);
        },
        [=](const int k, const long r) {
          const VofBlockJob& J = T.job[k];
          const I3 e = J.e;
          const int g = J.g;
          double* f = J.c;
          if (a == 0) {  // extended row r: the g x-ghosts at each end
            const int N = e.x - 2 * g;
            const long base = r * e.x;
            for (int gl = 0; gl < g; ++gl) {
              f[base + gl] = f[base + gl + N];
              f[base + g + N + gl] = f[base + g + gl];
            }
            return;
          }
          long dst, src;
          if (a == 1) {  // ghost row (l, z): y = l (low) or g + N + (l - g) (high)
            const int N = e.y - 2 * g;
            const int z = static_cast<int>(r / (2 * g)), l = static_cast<int>(r % (2 * g));
            const int y = l < g ? l : N + l, sy = l < g ? l + N : l;  // high: y - N = g + (l - g)
            dst = L3(0, y, z, e);
            src = L3(0, sy, z, e);
          } else {  // ghost row (y, l)
            const int N = e.z - 2 * g;
            const int l = static_cast<int>(r / e.y), y = static_cast<int>(r % e.y);
            const int z = l < g ? l : N + l, sz = l < g ? l + N : l;
            dst = L3(0, y, z, e);
            src = L3(0, y, sz, e);
          }
          PECLET_FLOW_OMP_SIMD
          for (int x = 0; x < e.x; ++x)
            f[dst + x] = f[src + x];
        });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_periodic", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const VofBlockJob& J = T.job[k];
        const long r = t - T.off[k];
        const I3 e = J.e;
        const int g = J.g;
        const long st[3] = {1, e.x, static_cast<long>(e.x) * e.y};
        const int dims[3] = {e.x, e.y, e.z};
        const int b = (a + 1) % 3, c = (a + 2) % 3;
        const long sa = st[a], sb = st[b], sc = st[c];
        const int N = dims[a] - 2 * g;
        const int p0 = static_cast<int>(r % dims[b]);
        const int p1 = static_cast<int>(r / dims[b]);
        const VofRawField f{J.c};
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        for (int gl = 0; gl < g; ++gl) {
          f(base + static_cast<long>(gl) * sa) = f(base + static_cast<long>(gl + N) * sa);
          f(base + static_cast<long>(g + N + gl) * sa) = f(base + static_cast<long>(g + gl) * sa);
        }
      });
}

/// The clamp of extended cell (x, y, z) of job `J` (the per-cell body of `vofBatchClamp`).
KOKKOS_INLINE_FUNCTION void vofBatchClampCell(const VofBlockJob& J, int x, int y, int z, int q0,
                                              int q1, int q2, bool p0, bool p1, bool p2) {
  const I3 e = J.e, o = J.o;
  const int g = J.g;
  const int gx = x - g + o.x, gy = y - g + o.y, gz = z - g + o.z;
  const int cx = p0 ? gx : (gx < 0 ? 0 : (gx >= q0 ? q0 - 1 : gx));
  const int cy = p1 ? gy : (gy < 0 ? 0 : (gy >= q1 ? q1 - 1 : gy));
  const int cz = p2 ? gz : (gz < 0 ? 0 : (gz >= q2 ? q2 - 1 : gz));
  if (cx != gx || cy != gy || cz != gz)
    J.c[L3(x, y, z, e)] = J.c[L3(cx - o.x + g, cy - o.y + g, cz - o.z + g, e)];
}

/// Pass 3 (`colour_field.hpp::clampFill`, `skip = 0`): the part of each extended block outside a
/// NON-periodic domain takes the globally clamped value. `T.off` = extended lengths. Host: the
/// extended rows, skipping a row inside the domain on every non-periodic axis -- no cell of it
/// writes (H-4b). A source cell is inside on every non-periodic axis, so it is never written.
template <class Exec = SExec>
inline void vofBatchClamp(const VofBlockTable& T, I3 gs, bool px, bool py, bool pz) {
  const bool p0 = px, p1 = py, p2 = pz;
  const int q0 = gs.x, q1 = gs.y, q2 = gs.z;
  if constexpr (kVofHostExec<Exec>) {
    vofHostRowFor<Exec>(
        "vof::block::batch_clamp", T.nj,
        [&](const int k) { return static_cast<long>(T.job[k].e.y) * T.job[k].e.z; },
        [=](const int k, const long r) {
          const VofBlockJob& J = T.job[k];
          const I3 e = J.e, o = J.o;
          const int g = J.g;
          const int y = static_cast<int>(r % e.y), z = static_cast<int>(r / e.y);
          const int gy = y - g + o.y, gz = z - g + o.z, gx0 = o.x - g, gx1 = o.x - g + e.x;
          if ((p1 || (gy >= 0 && gy < q1)) && (p2 || (gz >= 0 && gz < q2)) &&
              (p0 || (gx0 >= 0 && gx1 <= q0)))
            return;  // inside on every non-periodic axis: nothing in this row writes
          for (int x = 0; x < e.x; ++x)
            vofBatchClampCell(J, x, y, z, q0, q1, q2, p0, p1, p2);
        });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_clamp", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const VofBlockJob& J = T.job[k];
        const long i = t - T.off[k];
        const I3 e = J.e;
        const int x = static_cast<int>(i % e.x);
        const int y = static_cast<int>((i / e.x) % e.y);
        const int z = static_cast<int>(i / (static_cast<long>(e.x) * e.y));
        vofBatchClampCell(J, x, y, z, q0, q1, q2, p0, p1, p2);
      });
}

// ---- stage 4: marker debris and sub-wispEps residue (`VofBlockSet::debrisPassBatchImpl`) --------

/// The debris predicate of `doc/vof_overlap_design.md` §11 on cell `i`: interfacial, and NO cell of
/// the marker at or above `cfull` within two cells (the 5^3 fit stencil). Shared by the per-block
/// scan and the batched one.
template <class SF>
KOKKOS_INLINE_FUNCTION bool vofDebrisCell(const SF& c, long i, long sy, long sz, double ieps,
                                          double cfull) {
  if (!vofIsInterface(c(i), ieps))
    return false;
  for (int oz = -2; oz <= 2; ++oz)
    for (int oy = -2; oy <= 2; ++oy)
      for (int ox = -2; ox <= 2; ++ox)
        if (c(i + ox + oy * sy + oz * sz) >= cfull)
          return false;  // attached: a cell at least half full of this marker within two
  return true;
}

/// The ATTACHED predicate (§5.3 step 2): interfacial with a cell at or above `cfull` within two.
template <class SF>
KOKKOS_INLINE_FUNCTION bool vofAttachedCell(const SF& c, long i, long sy, long sz, double ieps,
                                            double cfull) {
  if (!vofIsInterface(c(i), ieps))
    return false;
  bool attached = false;
  for (int oz = -2; oz <= 2 && !attached; ++oz)
    for (int oy = -2; oy <= 2 && !attached; ++oy)
      for (int ox = -2; ox <= 2 && !attached; ++ox)
        attached = c(i + ox + oy * sy + oz * sz) >= cfull;
  return attached;
}

/// The residue predicate (§13): `0 < |C| <= wispEps`.
KOKKOS_INLINE_FUNCTION bool vofResidueCell(double ci, double weps) {
  return ci != 0.0 && Kokkos::fabs(ci) <= weps;
}

/// One mark scan over the concatenated inner regions (`T.off` = `vofJobInner`), in each block's
/// inner-box index order -- the per-block scan's order, so every list is the per-block list.
/// `kind`: 0 debris, 1 residue, 2 attached. Each job's range lands in `start`/`end` (absolute).
inline void vofBatchMark(const VofBlockTable& T, int kind, double ieps, double cfull, double weps,
                         LField list, LField start, LField end) {
  Kokkos::parallel_scan(
      "vof::block::batch_debris_mark", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t, long& upd, const bool final) {
        const int k = vofJobOf(T.off, t);
        const VofBlockJob& J = T.job[k];
        const long r = t - T.off[k];
        const int nx = J.n.x, ny = J.n.y, g = J.g;
        const int ix = static_cast<int>(r % nx);
        const int iy = static_cast<int>((r / nx) % ny);
        const int iz = static_cast<int>(r / (static_cast<long>(nx) * ny));
        const long i = L3(g + ix, g + iy, g + iz, J.e);
        const long sy = J.e.x, sz = static_cast<long>(J.e.x) * J.e.y;
        const VofRawField c{J.c};
        const bool hit = kind == 0   ? vofDebrisCell(c, i, sy, sz, ieps, cfull)
                         : kind == 1 ? vofResidueCell(c(i), weps)
                                     : vofAttachedCell(c, i, sy, sz, ieps, cfull);
        if (final) {
          if (r == 0)
            start(T.base + k) = upd;
          if (hit)
            list(upd) = i;
          if (t + 1 == T.off[k + 1])
            end(T.base + k) = upd + (hit ? 1 : 0);
        }
        if (hit)
          ++upd;
      });
}

/// The per-job slots of the debris packet (`out(stride * (base + k) + slot)`).
enum VofDebrisSlot : int {
  kDbSumD = 0,  ///< sum_D C in list order (the removed / census debris volume)
  kDbSumR,      ///< sum_R C in list order (signed residue)
  kDbW,         ///< sum_A C(1 - C)
  kDbVA,        ///< sum_A C
  kDbDV,        ///< the volume to return (D if removing, then R); replaced by `lost` once acted
  kDbActed,     ///< 1 when the return ran (pending and VA >= kDebrisMinAttached and W > 0)
  kDbND,        ///< debris count
  kDbNR,        ///< residue count
  kDbPending,   ///< 1 when there was something to remove (removeD or nR > 0)
  kDbSlots
};

/// Flags of the pass, uniform over the blocks.
struct VofDebrisFlags {
  bool doDebris, doResidue, remove;
  double vaMin;
};

/// Step 2 of the pass: one thread per job, every sum in list order (the per-block kernel's single
/// thread and order, so the same bits). A job with nothing to remove computes only its census sum.
inline void vofBatchDebrisSums(const VofBlockTable& T, VofDebrisFlags F, LField listD, LField listR,
                               LField listA, LField sD0, LField eD0, LField sR0, LField eR0,
                               LField sA0, LField eA0, SField out, int stride, int slot0) {
  Kokkos::parallel_for(
      "vof::block::batch_debris_sums", Kokkos::RangePolicy<SExec>(SExec(), 0, T.nj),
      KOKKOS_LAMBDA(const long q) {
        const int j = T.base + static_cast<int>(q);
        const VofRawField c{T.job[q].c};
        const long nD = F.doDebris ? eD0(j) - sD0(j) : 0;
        const long nR = F.doResidue ? eR0(j) - sR0(j) : 0;
        const bool removeD = F.doDebris && F.remove && nD > 0;
        const bool pending = removeD || nR > 0;
        const long ob = static_cast<long>(stride) * j + slot0;
        double sD = 0.0, sR = 0.0, W = 0.0, VA = 0.0, dV = 0.0;
        const long d0 = F.doDebris ? sD0(j) : 0;
        for (long t = 0; t < nD; ++t)
          sD += c(listD(d0 + t));
        double acted = 0.0;
        if (pending) {
          const long r0 = F.doResidue ? sR0(j) : 0, a0 = sA0(j), nA = eA0(j) - sA0(j);
          for (long t = 0; t < nR; ++t)
            sR += c(listR(r0 + t));
          for (long t = 0; t < nA; ++t) {
            const double ca = c(listA(a0 + t));
            W += ca * (1.0 - ca);
            VA += ca;
          }
          acted = (VA < F.vaMin || !(W > 0.0)) ? 0.0 : 1.0;
          // the removed volume, in the fixed order D (if removing) then R -- the serial order
          const long nDr = removeD ? nD : 0;
          for (long t = 0; t < nDr; ++t)
            dV += c(listD(d0 + t));
          for (long t = 0; t < nR; ++t)
            dV += c(listR(r0 + t));
        }
        out(ob + kDbSumD) = sD;
        out(ob + kDbSumR) = sR;
        out(ob + kDbW) = W;
        out(ob + kDbVA) = VA;
        out(ob + kDbDV) = dV;
        out(ob + kDbActed) = acted;
        out(ob + kDbND) = static_cast<double>(nD);
        out(ob + kDbNR) = static_cast<double>(nR);
        out(ob + kDbPending) = pending ? 1.0 : 0.0;
      });
}

/// The number of act entries of job `j` (its `D` removed, then `R`, then `A`), 0 when it did not
/// act: the per-block write kernel's range.
KOKKOS_INLINE_FUNCTION long vofDebrisActCount(const VofDebrisFlags& F, LField sA0, LField eA0,
                                              SField out, long ob, int j) {
  if (out(ob + kDbActed) == 0.0)
    return 0;
  const long nD = static_cast<long>(out(ob + kDbND)), nR = static_cast<long>(out(ob + kDbNR));
  const bool removeD = F.doDebris && F.remove && nD > 0;
  const long nDr = removeD ? nD : 0, nA = eA0(j) - sA0(j);
  return nDr + nR + nA;
}

/// Act entry `t` of job `k` (the per-block write kernel's expressions verbatim); `t` past the job's
/// count, or a job that did not act, is a no-op.
KOKKOS_INLINE_FUNCTION void vofDebrisActEntry(const VofBlockTable& T, const VofDebrisFlags& F,
                                              int k, long t, LField listD, LField listR,
                                              LField listA, LField sD0, LField sR0, LField sA0,
                                              LField eA0, SField out, int stride, int slot0,
                                              SField ex, Kokkos::View<int*, SMem> capf) {
  const int j = T.base + k;
  const long ob = static_cast<long>(stride) * j + slot0;
  if (out(ob + kDbActed) == 0.0)
    return;
  const long nD = static_cast<long>(out(ob + kDbND)), nR = static_cast<long>(out(ob + kDbNR));
  const bool removeD = F.doDebris && F.remove && nD > 0;
  const long nDr = removeD ? nD : 0, nA = eA0(j) - sA0(j);
  const long nZ = nDr + nR, nTot = nZ + nA;
  if (t >= nTot)
    return;
  const VofRawField c{T.job[k].c};
  if (t < nDr) {
    c(listD(sD0(j) + t)) = 0.0;
    return;
  }
  if (t < nZ) {
    c(listR(sR0(j) + t - nDr)) = 0.0;
    return;
  }
  const double dV = out(ob + kDbDV), W = out(ob + kDbW);
  const long ta = t - nZ;
  const long i = listA(sA0(j) + ta);
  const double ca = c(i);
  const double d = dV * (ca * (1.0 - ca)) / W;
  double cn = ca + d;
  double e = 0.0;
  if (cn > 1.0) {
    e = cn - 1.0;
    cn = 1.0;
    Kokkos::atomic_max(&capf(j), 1);
  }
  ex(T.off[k] + ta) = e;
  c(i) = cn;
}

/// Step 4, the act: removed cells -> 0, attached `C += dV C(1-C)/W` capped at 1 (the per-block
/// write kernel's expressions verbatim). Over the upper bound (the inner region: D, R and A are
/// disjoint subsets of it); a job that did not act, or a thread past its count, exits. `capf(j)`
/// is raised when some cell of job j was capped. Host: the exact counts (H-4a).
template <class Exec = SExec>
inline void vofBatchDebrisWrite(const VofBlockTable& T, VofDebrisFlags F, LField listD,
                                LField listR, LField listA, LField sD0, LField sR0, LField sA0,
                                LField eA0, SField out, int stride, int slot0, SField ex,
                                Kokkos::View<int*, SMem> capf) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostListFor<Exec>(
        "vof::block::batch_debris_write", T.nj,
        [&](const int k) {
          const int j = T.base + k;
          return vofDebrisActCount(F, sA0, eA0, out, static_cast<long>(stride) * j + slot0, j);
        },
        [=](const int k, const long t) {
          vofDebrisActEntry(T, F, k, t, listD, listR, listA, sD0, sR0, sA0, eA0, out, stride, slot0,
                            ex, capf);
        });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_debris_write", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long tt) {
        const int k = vofJobOf(T.off, tt);
        vofDebrisActEntry(T, F, k, tt - T.off[k], listD, listR, listA, sD0, sR0, sA0, eA0, out,
                          stride, slot0, ex, capf);
      });
}

/// `lost`: the capped excess of an acted job summed in list order on one thread (only where a cap
/// fired, as the per-block path; 0 otherwise). Written over the job's `kDbDV` slot.
inline void vofBatchDebrisLost(const VofBlockTable& T, LField sA0, LField eA0, SField out,
                               int stride, int slot0, SField ex, Kokkos::View<int*, SMem> capf) {
  Kokkos::parallel_for(
      "vof::block::batch_debris_lost", Kokkos::RangePolicy<SExec>(SExec(), 0, T.nj),
      KOKKOS_LAMBDA(const long q) {
        const int j = T.base + static_cast<int>(q);
        const long ob = static_cast<long>(stride) * j + slot0;
        if (out(ob + kDbActed) == 0.0)
          return;
        double lost = 0.0;
        if (capf(j) > 0) {
          const long nA = eA0(j) - sA0(j), e0 = T.off[q];
          double a = 0.0;  // list order, as the serial loop accumulated it
          for (long t = 0; t < nA; ++t)
            a += ex(e0 + t);
          lost = a;
        }
        out(ob + kDbDV) = lost;
      });
}

// ---- stage 5: the bubble boxes (`VofBlockSet::bubbleBox`) ------------------------------------

/// One team per job: the tight box of `|C| > eps` over the inner region, in the block's LOCAL inner
/// frame (`bubbleBox`'s six Min/Max reductions; order-free). Written as doubles (exact integers)
/// to `out(stride (base + k) + slot0 + {0..5})` = lo x, y, z, hi x, y, z; an empty block reads
/// hi < lo (the reducers' identities), exactly as `bubbleBox` tests it.
///
/// Host (H-4c): `TeamPolicy(nj, AUTO)` is one thread per block there. Instead one row-parallel pass
/// over every job's inner rows: a row takes its six extrema over x, then integer atomic min/max
/// into the job's slots, which start at the same identities. Order-free, so bitwise.
template <class Exec = SExec>
inline void vofBatchBox(const VofBlockTable& T, double eps, SField out, int stride, int slot0) {
  if constexpr (kVofHostExec<Exec>) {
    int box[6 * kVofBlockBatch];
    for (int k = 0; k < T.nj; ++k)
      for (int a = 0; a < 6; ++a)
        box[6 * k + a] =
            a < 3 ? Kokkos::reduction_identity<int>::min() : Kokkos::reduction_identity<int>::max();
    int* bx = box;
    vofHostRowFor<Exec>(
        "vof::block::batch_bbox", T.nj,
        [&](const int k) { return static_cast<long>(T.job[k].n.y) * T.job[k].n.z; },
        [=](const int k, const long r) {
          const VofBlockJob& J = T.job[k];
          const I3 e = J.e, n = J.n;
          const int g = J.g;
          const int py = static_cast<int>(r % n.y), pz = static_cast<int>(r / n.y);
          const long i0 = L3(g, py + g, pz + g, e);
          const VofRawField c{J.c};
          int lo = -1, hi = -1;
          for (int px = 0; px < n.x; ++px)
            if (Kokkos::fabs(c(i0 + px)) > eps) {
              if (lo < 0)
                lo = px;
              hi = px;
            }
          if (lo < 0)
            return;  // no cell of the row is in the box
          int* s = bx + 6 * k;
          Kokkos::atomic_min(&s[0], lo);
          Kokkos::atomic_min(&s[1], py);
          Kokkos::atomic_min(&s[2], pz);
          Kokkos::atomic_max(&s[3], hi);
          Kokkos::atomic_max(&s[4], py);
          Kokkos::atomic_max(&s[5], pz);
        });
    Exec().fence();
    for (int k = 0; k < T.nj; ++k) {
      const long ob = static_cast<long>(stride) * (T.base + k) + slot0;
      for (int a = 0; a < 6; ++a)
        out(ob + a) = static_cast<double>(box[6 * k + a]);
    }
    return;
  }
  using Team = Kokkos::TeamPolicy<SExec>;
  Kokkos::parallel_for(
      "vof::block::batch_bbox", Team(SExec(), T.nj, Kokkos::AUTO),
      KOKKOS_LAMBDA(const typename Team::member_type& tm) {
        const int k = tm.league_rank();
        const VofBlockJob& J = T.job[k];
        const I3 e = J.e, n = J.n;
        const int g = J.g;
        const long region = static_cast<long>(n.x) * n.y * n.z;
        const VofRawField c{J.c};
        int v[6];
        for (int a = 0; a < 6; ++a) {
          int r = 0;
          auto body = [&](const long q, int& acc) {
            const int px = static_cast<int>(q % n.x);
            const int py = static_cast<int>((q / n.x) % n.y);
            const int pz = static_cast<int>(q / (static_cast<long>(n.x) * n.y));
            if (!(Kokkos::fabs(c(L3(px + g, py + g, pz + g, e))) > eps))
              return;
            const int p = (a % 3 == 0) ? px : (a % 3 == 1 ? py : pz);
            if (a < 3)
              acc = p < acc ? p : acc;
            else
              acc = p > acc ? p : acc;
          };
          if (a < 3)
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, region), body, Kokkos::Min<int>(r));
          else
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, region), body, Kokkos::Max<int>(r));
          v[a] = r;
        }
        Kokkos::single(Kokkos::PerTeam(tm), [&]() {
          const long ob = static_cast<long>(stride) * (T.base + k) + slot0;
          for (int a = 0; a < 6; ++a)
            out(ob + a) = static_cast<double>(v[a]);
        });
      });
}

// ---- stage 6: the curvature cascade and the CSF face force (`VofBlockSet::computeCsf`) ---------

/// One master block's cascade (`VofCurvature`, worklist mode) for the batched launches: its planes,
/// curvature, branch and force fields, and every argument the per-cascade passes give the shared
/// cell bodies (`wyReconstructCell`, `curvHeightCell`, `curvFallbackCell`, the clip, the census,
/// `csfFaceCurvature` + `csfFaceForce`).
struct VofCurvJob {
  double *c, *mx, *my, *mz, *al, *kap, *br;
  double* f[3];  ///< the block's CSF face force, per component
  I3 e, n;
  int g;
  double ieps, mtol, ptW, peps, dW, cmin, km;
  bool forceFb, oneDir, useFit;
  VofMetric gm;
};

struct VofCurvTable {
  VofCurvJob job[kVofBlockBatch];
  long off[kVofBlockBatch + 1];
  int nj;
  int base;
};

template <class F>
inline long vofSetOffsets(VofCurvTable& T, F size) {
  T.off[0] = 0;
  for (int k = 0; k < T.nj; ++k)
    T.off[k + 1] = T.off[k] + size(T.job[k]);
  for (int k = T.nj; k < kVofBlockBatch; ++k)
    T.off[k + 1] = T.off[T.nj];
  return T.off[T.nj];
}
inline long vofCurvInner(const VofCurvJob& J) {
  return static_cast<long>(J.n.x) * J.n.y * J.n.z;
}
/// The inner region grown by `kPvHalf`: every cell `VofCurvature::reconstructPlanes` writes.
inline long vofCurvGrown(const VofCurvJob& J) {
  return static_cast<long>(J.n.x + 2 * kPvHalf) * (J.n.y + 2 * kPvHalf) * (J.n.z + 2 * kPvHalf);
}

/// Cell `r` of job `J`'s grown (`grown`) or inner region, in the region's x-fastest order.
KOKKOS_INLINE_FUNCTION long vofCurvCell(const VofCurvJob& J, long r, bool grown) {
  const int gr = grown ? kPvHalf : 0;
  const int rx = J.n.x + 2 * gr, ry = J.n.y + 2 * gr;
  const int ix = static_cast<int>(r % rx);
  const int iy = static_cast<int>((r / rx) % ry);
  const int iz = static_cast<int>(r / (static_cast<long>(rx) * ry));
  return L3(J.g - gr + ix, J.g - gr + iy, J.g - gr + iz, J.e);
}

/// `VofCurvature::compact`: the interfacial cells of the grown (`grown`) or inner region, in the
/// per-cascade scan's order; each job's range lands in `start`/`end` (absolute list positions).
inline void vofCurvCompact(const VofCurvTable& T, bool grown, LField list, LField start,
                           LField end) {
  Kokkos::parallel_scan(
      "vof::block::batch_curv_compact", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t, long& upd, const bool final) {
        const int k = vofJobOf(T.off, t);
        const VofCurvJob& J = T.job[k];
        const long r = t - T.off[k];
        const long i = vofCurvCell(J, r, grown);
        const bool hit = vofIsInterface(J.c[i], J.ieps);
        if (final) {
          if (r == 0)
            start(T.base + k) = upd;
          if (hit)
            list(upd) = i;
          if (t + 1 == T.off[k + 1])
            end(T.base + k) = upd + (hit ? 1 : 0);
        }
        if (hit)
          ++upd;
      });
}

/// Rows of job `J`'s grown (`grown`) or inner region, for the host row forms (H-4b).
inline long vofCurvRows(const VofCurvJob& J, bool grown) {
  const int gr = grown ? kPvHalf : 0;
  return static_cast<long>(J.n.y + 2 * gr) * (J.n.z + 2 * gr);
}

/// The first cell of row `r` of job `J`'s grown or inner region, and the region's x range.
inline long vofCurvRow(const VofCurvJob& J, long r, bool grown, int& x0, int& x1) {
  const int gr = grown ? kPvHalf : 0;
  const int ry = J.n.y + 2 * gr;
  x0 = J.g - gr;
  x1 = J.g + J.n.x + gr;
  return L3(0, J.g - gr + static_cast<int>(r % ry), J.g - gr + static_cast<int>(r / ry), J.e);
}

/// `reconstructPlanes` (worklist mode), part 1: zero the four plane fields over the grown region.
/// Host: the grown rows (H-4b).
template <class Exec = SExec>
inline void vofCurvPlanesZero(const VofCurvTable& T) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostRowFor<Exec>(
        "vof::block::batch_curv_planes_zero", T.nj,
        [&](const int k) { return vofCurvRows(T.job[k], true); },
        [=](const int k, const long r) {
          const VofCurvJob& J = T.job[k];
          int x0, x1;
          const long i0 = vofCurvRow(J, r, true, x0, x1);
          PECLET_FLOW_OMP_SIMD
          for (int x = x0; x < x1; ++x) {
            J.mx[i0 + x] = 0.0;
            J.my[i0 + x] = 0.0;
            J.mz[i0 + x] = 0.0;
            J.al[i0 + x] = 0.0;
          }
        });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_curv_planes_zero", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const VofCurvJob& J = T.job[k];
        const long i = vofCurvCell(J, t - T.off[k], true);
        J.mx[i] = 0.0;
        J.my[i] = 0.0;
        J.mz[i] = 0.0;
        J.al[i] = 0.0;
      });
}

/// Part 2: `wyReconstructCell` over the grown interfacial list (upper bound: the grown region).
/// Host: the exact counts (H-4a).
template <class Exec = SExec>
inline void vofCurvPlanesList(const VofCurvTable& T, LField list, LField start, LField end) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostListFor<Exec>("vof::block::batch_curv_planes", T.nj, vofListCount(T, start, end),
                         [=](const int k, const long t) {
                           const long q = start(T.base + k) + t;
                           const VofCurvJob& J = T.job[k];
                           const long sy = J.e.x, sz = static_cast<long>(J.e.x) * J.e.y;
                           wyReconstructCell(VofRawField{J.c}, list(q), sy, sz, VofRawField{J.mx},
                                             VofRawField{J.my}, VofRawField{J.mz},
                                             VofRawField{J.al});
                         });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_curv_planes", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const long q = start(T.base + k) + (t - T.off[k]);
        if (q >= end(T.base + k))
          return;
        const VofCurvJob& J = T.job[k];
        const long sy = J.e.x, sz = static_cast<long>(J.e.x) * J.e.y;
        wyReconstructCell(VofRawField{J.c}, list(q), sy, sz, VofRawField{J.mx}, VofRawField{J.my},
                          VofRawField{J.mz}, VofRawField{J.al});
      });
}

/// `heightPass` (worklist mode), part 1: every inner cell to kappa = 0, branch = kCurvNone. Host:
/// the inner rows (H-4b).
template <class Exec = SExec>
inline void vofCurvHfReset(const VofCurvTable& T) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostRowFor<Exec>(
        "vof::block::batch_curv_hf_reset", T.nj,
        [&](const int k) { return vofCurvRows(T.job[k], false); },
        [=](const int k, const long r) {
          const VofCurvJob& J = T.job[k];
          int x0, x1;
          const long i0 = vofCurvRow(J, r, false, x0, x1);
          PECLET_FLOW_OMP_SIMD
          for (int x = x0; x < x1; ++x) {
            J.kap[i0 + x] = 0.0;
            J.br[i0 + x] = static_cast<double>(kCurvNone);
          }
        });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_curv_hf_reset", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const VofCurvJob& J = T.job[k];
        const long i = vofCurvCell(J, t - T.off[k], false);
        J.kap[i] = 0.0;
        J.br[i] = static_cast<double>(kCurvNone);
      });
}

/// Tier 3 of the batched cascade on a DEVICE (C3, `doc/vof_step_performance_design.md` §4.7,
/// §5.11; launch shape: coordinator decision 2026-10-02, option C). The interfacial counts stay on
/// the device (§4.6), so the league cannot be the number of entries; instead a FIXED league of
/// persistent warp-teams strides over the concatenated device-counted entries
/// `g = league_rank, league_rank + league_size, ...`, each entry handled by `curvFallbackTeam` (the
/// same body as `VofCurvature::fallbackBatch`'s team path: canonical-order accumulation, bit for
/// bit the one-thread `curvFallbackCell`). No host read, no empty teams.
///
/// League size: one team per hardware warp slot, `concurrency() / kVofWarp` (on CUDA = SMs x
/// resident threads per SM / 32), capped by `kVofTier3MaxTeams`. Teams beyond the resident
/// number simply start later; the stride covers every entry whatever the league.
inline constexpr int kVofTier3MaxTeams = 1 << 16;
inline void vofCurvFallbackTeams(const VofCurvTable& T, LField list, LField start, LField end) {
  using Policy = Kokkos::TeamPolicy<SExec>;
  using Terms = Kokkos::View<PvTerm*, SExec::scratch_memory_space, Kokkos::MemoryUnmanaged>;
  constexpr int side = 2 * kPvHalf + 1, nk = side * side * side;
  const int league =
      std::max(1, std::min(kVofTier3MaxTeams, static_cast<int>(SExec().concurrency() / kVofWarp)));
  Policy pol(SExec(), league, kVofWarp);
  pol.set_scratch_size(0, Kokkos::PerTeam(Terms::shmem_size(nk)));
  Kokkos::parallel_for(
      "vof::block::batch_curv_pv_team", pol, KOKKOS_LAMBDA(const typename Policy::member_type& tm) {
        Terms terms(tm.team_scratch(0), nk);
        long cum[kVofBlockBatch + 1];  // device prefix of the jobs' entry counts
        cum[0] = 0;
        for (int k = 0; k < T.nj; ++k)
          cum[k + 1] = cum[k] + (end(T.base + k) - start(T.base + k));
        for (long g = tm.league_rank(); g < cum[T.nj]; g += tm.league_size()) {
          int k = 0;
          while (g >= cum[k + 1])
            ++k;
          const VofCurvJob& J = T.job[k];
          const long i = list(start(T.base + k) + (g - cum[k]));
          const long sy = J.e.x, sz = static_cast<long>(J.e.x) * J.e.y;
          curvFallbackTeam(tm, terms, i, VofRawField{J.c}, VofRawField{J.mx}, VofRawField{J.my},
                           VofRawField{J.mz}, VofRawField{J.al}, VofRawField{J.kap},
                           VofRawField{J.br}, sy, sz, kPvHalf, J.dW, J.cmin, J.ieps, J.gm);
          tm.team_barrier();  // the single lane is done with `terms` before the next entry's map
        }
      });
}

/// Entry `q` (an absolute list position) of job `k` of a cascade list pass (see below): the body
/// shared by the device and host launch forms.
KOKKOS_INLINE_FUNCTION void vofCurvListEntry(const VofCurvTable& T, int pass, int k, long q,
                                             LField list, LField cnt) {
  const VofCurvJob& J = T.job[k];
  const long i = list(q);
  const long sy = J.e.x, sz = static_cast<long>(J.e.x) * J.e.y;
  const VofRawField c{J.c}, mx{J.mx}, my{J.my}, mz{J.mz}, al{J.al}, kap{J.kap}, br{J.br};
  if (pass == 0) {
    curvHeightCell(i, c, mx, my, mz, al, kap, br, 1L, sy, sz, J.mtol, J.ptW, J.ieps, J.forceFb,
                   J.oneDir, J.useFit, J.gm, J.peps);
    return;
  }
  if (pass == 1) {
    curvFallbackCell(i, c, mx, my, mz, al, kap, br, sy, sz, kPvHalf, J.dW, J.cmin, J.ieps, J.gm);
    return;
  }
  long* ct = &cnt(8 * (T.base + k));
  if (pass == 2) {
    const double km = J.km;
    if (!(km > 0.0))
      return;  // this cascade's clip is off (VofCurvature::clipPass returns 0)
    // `!(|k| <= km)` rather than `|k| > km`: a NaN curvature is clipped (to +km, the sign of
    // a NaN is meaningless) and counted, never silently kept.
    if (csfKappaDefined(br(i)) && !(Kokkos::fabs(kap(i)) <= km)) {
      kap(i) = (kap(i) == kap(i)) ? Kokkos::copysign(km, kap(i)) : km;
      Kokkos::atomic_add(&ct[7], 1L);
    }
    return;
  }
  const int b = static_cast<int>(br(i));
  if (b == kCurvNone)
    return;
  Kokkos::atomic_add(&ct[0], 1L);
  if (b == kCurvHf)
    Kokkos::atomic_add(&ct[1], 1L);
  else if (b == kCurvHfMixed)
    Kokkos::atomic_add(&ct[2], 1L);
  else if (b == kCurvHfFit)
    Kokkos::atomic_add(&ct[3], 1L);
  else if (b == kCurvPv)
    Kokkos::atomic_add(&ct[4], 1L);
  else if (b == kCurvPvReduced)
    Kokkos::atomic_add(&ct[5], 1L);
  else
    Kokkos::atomic_add(&ct[6], 1L);
}

/// The per-list-cell passes of the cascade over the inner interfacial list (upper bound: the inner
/// region): `pass` 0 = tiers 1-2 (`curvHeightCell`), 1 = tier 3 (`curvFallbackCell`), 2 = the
/// admissibility clip (its count into `cnt(8 (base + k) + 7)`), 3 = the branch census (its seven
/// counters into `cnt(8 (base + k) + 0..6)`, in `VofCurvature::census`'s order). The counts are
/// integer atomics: exact whatever the order. Host: the exact counts (H-4a).
template <class Exec = SExec>
inline void vofCurvListPass(const VofCurvTable& T, int pass, LField list, LField start, LField end,
                            LField cnt) {
  if constexpr (kVofHostExec<Exec>) {  // tier 3 included: the host keeps the loop
    vofHostListFor<Exec>("vof::block::batch_curv_list", T.nj, vofListCount(T, start, end),
                         [=](const int k, const long t) {
                           vofCurvListEntry(T, pass, k, start(T.base + k) + t, list, cnt);
                         });
    return;
  }
  if constexpr (!Kokkos::SpaceAccessibility<Kokkos::HostSpace, SField::memory_space>::accessible) {
    if (pass == 1) {  // tier 3 on a device: persistent warp-teams (C3); the host keeps the loop
      vofCurvFallbackTeams(T, list, start, end);
      return;
    }
  }
  Kokkos::parallel_for(
      "vof::block::batch_curv_list", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const long q = start(T.base + k) + (t - T.off[k]);
        if (q >= end(T.base + k))
          return;
        vofCurvListEntry(T, pass, k, q, list, cnt);
      });
}

/// The three CSF face-force components at the low faces of inner cell `i` of job `J` (the
/// per-cell body of `vofCsfForceBatch`).
KOKKOS_INLINE_FUNCTION void vofCsfForceCell(const VofCurvJob& J, long i, double sig, double w0,
                                            double w1, double w2) {
  const long sy = J.e.x, sz = static_cast<long>(J.e.x) * J.e.y;
  const VofRawField cv{J.c}, kp{J.kap}, kb{J.br};
  for (int cc = 0; cc < 3; ++cc) {
    const double wc = cc == 0 ? w0 : (cc == 1 ? w1 : w2);
    const long strd = (cc == 0) ? 1 : (cc == 1 ? sy : sz);
    const double dC = cv(i) - cv(i - strd);
    double f = 0.0;
    if (dC != 0.0) {
      double kf = 0.0;
      vof::csfFaceCurvature(kp(i - strd), kb(i - strd), kp(i), kb(i), kf);
      f = vof::csfFaceForce(sig, kf, dC, wc);
    }
    J.f[cc][i] = f;
  }
}

/// `VofBlockSet::buildCsfForce` for every job, the three components per cell: the V4 balanced-force
/// CSF at the LOW face of each inner cell, the per-block kernel's expressions verbatim. Host: the
/// inner rows (H-4b).
template <class Exec = SExec>
inline void vofCsfForceBatch(const VofCurvTable& T, double sig, double w0, double w1, double w2) {
  if constexpr (kVofHostExec<Exec>) {
    vofHostRowFor<Exec>(
        "vof::block::batch_csf_force", T.nj,
        [&](const int k) { return vofCurvRows(T.job[k], false); },
        [=](const int k, const long r) {
          const VofCurvJob& J = T.job[k];
          int x0, x1;
          const long i0 = vofCurvRow(J, r, false, x0, x1);
          PECLET_FLOW_OMP_SIMD
          for (int x = x0; x < x1; ++x)
            vofCsfForceCell(J, i0 + x, sig, w0, w1, w2);
        });
    return;
  }
  Kokkos::parallel_for(
      "vof::block::batch_csf_force", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        vofCsfForceCell(T.job[k], vofCurvCell(T.job[k], t - T.off[k], false), sig, w0, w1, w2);
      });
}

// ---- C2: the block statistics (§5.10) ----------------------------------------------------------

/// N double sums in one team reduction (`Kokkos::Sum<VofSumN<N>>`).
template <int N>
struct VofSumN {
  double v[N];
  KOKKOS_INLINE_FUNCTION VofSumN() {
    for (int i = 0; i < N; ++i)
      v[i] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION VofSumN& operator+=(const VofSumN& o) {
    for (int i = 0; i < N; ++i)
      v[i] += o.v[i];
    return *this;
  }
};

}  // namespace peclet::flow::vof

namespace Kokkos {
template <int N>
struct reduction_identity<peclet::flow::vof::VofSumN<N>> {
  KOKKOS_FORCEINLINE_FUNCTION static peclet::flow::vof::VofSumN<N> sum() {
    return peclet::flow::vof::VofSumN<N>();
  }
};
}  // namespace Kokkos

namespace peclet::flow::vof {

/// What `measure()` reads besides the colour: each job's previous centroid and whether it is valid
/// (the host's `prevCentroid_` / `hasPrev_`, current when the launch is issued).
struct VofStatsPrev {
  double c[kVofBlockBatch][3];
  unsigned char valid[kVofBlockBatch];
};

/// Doubles per job in the statistics packet: volume, centroid[3], velocity[3], area, moment[6].
inline constexpr int kVofStats = 14;

/// `VofBlockSet::measure` for every job in one launch, one team per job: the first moments, then
/// the central second moments and the PLIC interface area (`VofBlockSet::interfaceArea`'s body)
/// together, each as ONE team reduction over the inner box, with the per-cell expressions of the
/// per-block reductions verbatim. The summation ORDER is the team reduction's, not the per-block
/// MDRange's: a recorded change (C2) of these diagnostics, which feed no state. Job k's values land
/// at `out(off + kVofStats * (T.base + k) + ...)`; the host applies them later (deferred read).
inline void vofBatchStats(const VofBlockTable& T, const VofStatsPrev& P, double hh, double aeps,
                          double dt, SField out, long off) {
  using Team = Kokkos::TeamPolicy<SExec>;
  Kokkos::parallel_for(
      "vof::block::batch_stats", Team(SExec(), T.nj, Kokkos::AUTO),
      KOKKOS_LAMBDA(const typename Team::member_type& tm) {
        const int k = tm.league_rank();
        const VofBlockJob& J = T.job[k];
        const I3 e = J.e, n = J.n, o = J.o;
        const int g = J.g;
        const long sy = e.x, sz = static_cast<long>(e.x) * e.y;
        const long region = static_cast<long>(n.x) * n.y * n.z;
        const VofRawField c{J.c};
        double* q0 = out.data() + off + static_cast<long>(kVofStats) * (T.base + k);
        VofSumN<4> m1;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(tm, region),
            [&](const long r, VofSumN<4>& a) {
              const int x = static_cast<int>(r % n.x) + g;
              const int y = static_cast<int>((r / n.x) % n.y) + g;
              const int z = static_cast<int>(r / (static_cast<long>(n.x) * n.y)) + g;
              const double q = c(L3(x, y, z, e));
              a.v[0] += q;
              a.v[1] += q * (x - g + o.x + 0.5) * hh;
              a.v[2] += q * (y - g + o.y + 0.5) * hh;
              a.v[3] += q * (z - g + o.z + 0.5) * hh;
            },
            Kokkos::Sum<VofSumN<4>>(m1));
        const double v = m1.v[0];
        if (v <= 0.0) {  // measure()'s empty-marker branch: everything but the volume is zero
          Kokkos::single(Kokkos::PerTeam(tm), [&]() {
            q0[0] = v;
            for (int d = 1; d < kVofStats; ++d)
              q0[d] = 0.0;
          });
          return;
        }
        const double cx = m1.v[1] / v, cy = m1.v[2] / v, cz = m1.v[3] / v;
        VofSumN<7> m2;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(tm, region),
            [&](const long r, VofSumN<7>& a) {
              const int x = static_cast<int>(r % n.x) + g;
              const int y = static_cast<int>((r / n.x) % n.y) + g;
              const int z = static_cast<int>(r / (static_cast<long>(n.x) * n.y)) + g;
              const long i = L3(x, y, z, e);
              const double q = c(i);
              const double px = (x - g + o.x + 0.5) * hh - cx, py = (y - g + o.y + 0.5) * hh - cy,
                           pz = (z - g + o.z + 0.5) * hh - cz;
              a.v[0] += q * px * px;
              a.v[1] += q * py * py;
              a.v[2] += q * pz * pz;
              a.v[3] += q * px * py;
              a.v[4] += q * px * pz;
              a.v[5] += q * py * pz;
              if (!(q > aeps) || !(q < 1.0 - aeps))  // interfaceArea: mixed cells only
                return;
              double st[27];
              for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                  for (int dx = -1; dx <= 1; ++dx)
                    st[vof::plicSt(dx + 1, dy + 1, dz + 1)] = c(i + dx + dy * sy + dz * sz);
              double m[3];
              vof::mycNormal(st, m);
              const double al = vof::plicAlpha(m[0], m[1], m[2], q);
              double pv[8][3], ctr[3], ar = 0.0;
              const int nv = vof::plicPolygon(m[0], m[1], m[2], al, pv);
              vof::polygonAreaCentroid(pv, nv, ctr, ar);
              a.v[6] += ar;
            },
            Kokkos::Sum<VofSumN<7>>(m2));
        Kokkos::single(Kokkos::PerTeam(tm), [&]() {
          const double nc[3] = {cx, cy, cz};
          q0[0] = v;
          for (int d = 0; d < 3; ++d) {
            q0[1 + d] = nc[d];
            q0[4 + d] = (P.valid[k] && dt > 0.0) ? (nc[d] - P.c[k][d]) / dt : 0.0;
          }
          q0[7] = m2.v[6];
          const double iv = 1.0 / v;
          for (int d = 0; d < 6; ++d)
            q0[8 + d] = m2.v[d] * iv;
        });
      });
}

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_BLOCK_BATCH_HPP
