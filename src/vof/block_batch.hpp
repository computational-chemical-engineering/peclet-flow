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
inline void vofBatchFreeze(const VofBlockTable& T) {
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
/// region), a thread past its job's count exits.
inline void vofBatchPlic(const VofBlockTable& T, LField list, LField start, LField end) {
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

/// `WyAdvector::computeFluxesImpl` (uncut): one flux per `d`-face, `wyFaceFlux` (or, for a job
/// carrying the WO-R mask, `wyFaceFluxBc`). `T.off` = `vofJobFaces(d)`.
inline void vofBatchFlux(const VofBlockTable& T, int d) {
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
        const I3 e = J.e;
        const long sd =
            d == 0 ? 1 : (d == 1 ? static_cast<long>(e.x) : static_cast<long>(e.x) * e.y);
        const long p = L3(x, y, z, e);
        const VofRawField c{J.c}, mx{J.mx}, my{J.my}, mz{J.mz}, al{J.al}, u{J.u[d]};
        if (J.outside != nullptr) {
          J.fl[p] = wyFaceFluxBc(u(p) * J.dth, p, sd, d, c, mx, my, mz, al, VofRawMask{J.outside},
                                 J.weps);
          return;
        }
        J.fl[p] = wyFaceFlux(u(p) * J.dth, p, sd, d, c, mx, my, mz, al, J.weps);
      });
}

/// The WO-R boundary ledger (`WyAdvector::accumulateBcFaceVolume`) of the jobs carrying the mask:
/// one thread per (job, side), summing its plane of `d`-fluxes in (p0 fastest, p1) row order into
/// `bcv(6 (base + k) + face)` -- a fixed order, which is a C2 order change against the per-block
/// MDRange2 reduction (recorded; `doc/vof_step_performance_design.md` §5.9). The block container
/// installs no mask today, so the path is inert on every shipped configuration.
inline void vofBatchBcLedger(const VofBlockTable& T, int d, SField bcv) {
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
        bcv(6 * (T.base + k) + f) += (s == 0) ? acc : -acc;
      });
}

/// `WyAdvector::applySweepImpl` (uncut) over the inner regions (`T.off` = `vofJobInner`); the
/// expression verbatim.
inline void vofBatchUpdate(const VofBlockTable& T, int d) {
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
        const long sd =
            d == 0 ? 1 : (d == 1 ? static_cast<long>(e.x) : static_cast<long>(e.x) * e.y);
        const double dth = J.dth;
        const VofRawField c{J.c}, fl{J.fl}, u{J.u[d]};
        const unsigned char* cc = J.cc;
        const long i = L3(x, y, z, e);
        // The dilation term must scale the SAME uf by the SAME dt/h as the flux, or the exact
        // cancellation in full cells (advect_wy.hpp header) is lost to rounding.
        const double aP = u(i) * dth, aM = u(i - sd) * dth;
        const double dil = cc[i] ? (aP - aM) : 0.0;
        c(i) = c(i) + (fl(i - sd) - fl(i)) + dil;
      });
}

// ---- stage 3e: the block ghost policy (`VofBlockSet::fillBlockGhosts`), pass by pass ------------

/// Pass 1: every ghost cell of every job -> 0. `T.off` = extended lengths.
inline void vofBatchGhostZero(const VofBlockTable& T) {
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
inline void vofBatchPeriodic(const VofBlockTable& T, int a) {
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

/// Pass 3 (`colour_field.hpp::clampFill`, `skip = 0`): the part of each extended block outside a
/// NON-periodic domain takes the globally clamped value. `T.off` = extended lengths.
inline void vofBatchClamp(const VofBlockTable& T, I3 gs, bool px, bool py, bool pz) {
  const bool p0 = px, p1 = py, p2 = pz;
  const int q0 = gs.x, q1 = gs.y, q2 = gs.z;
  Kokkos::parallel_for(
      "vof::block::batch_clamp", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long t) {
        const int k = vofJobOf(T.off, t);
        const VofBlockJob& J = T.job[k];
        const long i = t - T.off[k];
        const I3 e = J.e, o = J.o;
        const int g = J.g;
        const int x = static_cast<int>(i % e.x);
        const int y = static_cast<int>((i / e.x) % e.y);
        const int z = static_cast<int>(i / (static_cast<long>(e.x) * e.y));
        const int gx = x - g + o.x, gy = y - g + o.y, gz = z - g + o.z;
        const int cx = p0 ? gx : (gx < 0 ? 0 : (gx >= q0 ? q0 - 1 : gx));
        const int cy = p1 ? gy : (gy < 0 ? 0 : (gy >= q1 ? q1 - 1 : gy));
        const int cz = p2 ? gz : (gz < 0 ? 0 : (gz >= q2 ? q2 - 1 : gz));
        if (cx != gx || cy != gy || cz != gz)
          J.c[L3(x, y, z, e)] = J.c[L3(cx - o.x + g, cy - o.y + g, cz - o.z + g, e)];
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

/// Step 4, the act: removed cells -> 0, attached `C += dV C(1-C)/W` capped at 1 (the per-block
/// write kernel's expressions verbatim). Over the upper bound (the inner region: D, R and A are
/// disjoint subsets of it); a job that did not act, or a thread past its count, exits. `capf(j)`
/// is raised when some cell of job j was capped.
inline void vofBatchDebrisWrite(const VofBlockTable& T, VofDebrisFlags F, LField listD,
                                LField listR, LField listA, LField sD0, LField sR0, LField sA0,
                                LField eA0, SField out, int stride, int slot0, SField ex,
                                Kokkos::View<int*, SMem> capf) {
  Kokkos::parallel_for(
      "vof::block::batch_debris_write", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
      KOKKOS_LAMBDA(const long tt) {
        const int k = vofJobOf(T.off, tt);
        const int j = T.base + k;
        const long ob = static_cast<long>(stride) * j + slot0;
        if (out(ob + kDbActed) == 0.0)
          return;
        const long t = tt - T.off[k];
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

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_BLOCK_BATCH_HPP
