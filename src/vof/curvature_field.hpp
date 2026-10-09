/// @file
/// @brief flow — VoF rung V3 (WO-O): the view-level driver for the curvature cascade.
///
/// `vof/curvature.hpp` holds the promotable, container-free kernels (WO-D signature rule). This
/// file is their block-walking driver: it owns the PLIC scratch fields, walks the colour block,
/// and produces the two output fields
///
///   * `kappa`        — 2H in units of 1/h (cell units), positive for a convex blob of liquid;
///   * `kappa_branch` — which tier of the cascade produced it (`CurvatureBranch`), as a double so
///                      it rides the ordinary field registry / IO / redistribute path.
///
/// Both live on the colour advector's **g = 3** block, which is exactly the reach the cascade
/// needs and no more:
///
///   * tier 1/2 (height functions) read a 3 x 3 x 7 patch of colour — transverse +/-1, column
///     +/-3;
///   * tier 3 (the 5^3 PV fit) reads a PLIC plane at +/-2, and each plane's MYC normal reads
///     colour at +/-1 around it — again +/-3.
///
/// So **no new halo machinery**: `IbmSolver::vofFillGhosts` on the colour field is the only
/// exchange this rung needs, and the whole computation is a pure local stencil, hence bitwise
/// decomposition-independent by construction (there is not one reduction in it).
///
/// The one deliberate cost: the PLIC planes are rebuilt here over the inner region grown by 2,
/// rather than reusing `WyAdvector`'s (`reconstruct()` covers inner+1 only, and it is a validated
/// kernel body that hard rule 1 forbids editing).
#ifndef PECLET_FLOW_VOF_CURVATURE_FIELD_HPP
#define PECLET_FLOW_VOF_CURVATURE_FIELD_HPP

#include <algorithm>
#include <chrono>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <vector>

#include "mac_stencils.hpp"  // peclet::flow::SExec, SField, I3, L3
#include "policy.hpp"
#include "vof/advect_wy.hpp"  // wyIsMixed, wyReconstructCell
#include "vof/curvature.hpp"
#include "vof/surface_tension.hpp"  // csfKappaDefined

namespace peclet::flow::vof {

/// `wyIsMixed` with a wisp threshold: the cell carries an interface only while `eps < C < 1 - eps`.
/// `eps = 0` is `wyIsMixed` exactly (`C > 0 && C < 1`). See `VofCurvature::interfaceEps` for the
/// measurement that put it there.
KOKKOS_INLINE_FUNCTION bool vofIsInterface(double c, double eps) {
  return c > eps && c < 1.0 - eps;
}

/// WO-V9 — the per-cell bodies of the two cascade tiers, lifted VERBATIM out of the kernels they
/// used to be written inline in, so the dense (whole-region) and the compacted (interfacial-cell
/// list) kernels below can share ONE body instead of two copies that could drift apart. The only
/// substitution is the cell index: `i` is a parameter instead of `L3(x, y, z, e)`, and tier 3's
/// neighbour index `L3(x+ox, y+oy, z+oz, e)` becomes the identical `i + ox + oy*sy + oz*sz` (the
/// same integers, by the definition of L3). Nothing else changed, which is what lets the ctest
/// gate the compaction as bitwise.
template <class SF>
KOKKOS_INLINE_FUNCTION void curvHeightCell(long i, SF c, SF mx, SF my, SF mz, SF al, SF kap, SF br,
                                           long s0, long s1, long s2, double mtol, double ptW,
                                           double ieps, bool forceFb, bool oneDir, bool useFit,
                                           VofMetric g, double peps) {
  const long st[3] = {s0, s1, s2};
  kap(i) = 0.0;
  if (!vofIsInterface(c(i), ieps)) {
    br(i) = static_cast<double>(kCurvNone);
    return;
  }
  if (forceFb) {
    br(i) = -1.0;
    return;
  }
  // Order the three column directions by the cell's own |n_d|. The normal is used as a
  // CATEGORICAL selector only — it is never differenced (see curvature.hpp).
  const double am[3] = {Kokkos::fabs(mx(i)), Kokkos::fabs(my(i)), Kokkos::fabs(mz(i))};
  int ord[3] = {0, 1, 2};
  for (int p = 1; p < 3; ++p)  // insertion sort, descending
    for (int q = p; q > 0 && am[ord[q]] > am[ord[q - 1]]; --q) {
      const int t = ord[q];
      ord[q] = ord[q - 1];
      ord[q - 1] = t;
    }
  const int ntry = oneDir ? 1 : 3;
  for (int t = 0; t < ntry; ++t) {
    const int d = ord[t];
    const int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
    const long sd = st[d], s1 = st[d1], s2 = st[d2];
    double hh[9];
    int orient0 = 0;
    bool ok = true;
    for (int q = 0; q < 3 && ok; ++q)
      for (int p = 0; p < 3 && ok; ++p) {
        const long base = i + (p - 1) * s1 + (q - 1) * s2;
        double col[kHfColumn];
        for (int k = 0; k < kHfColumn; ++k)
          col[k] = c(base + (k - kHfColumn / 2) * sd);
        double h;
        int orient;
        if (!hfColumnHeight(col, kHfColumn, h, orient, mtol, peps)) {
          ok = false;
          break;
        }
        if (orient0 == 0)
          orient0 = orient;
        else if (orient != orient0) {
          ok = false;  // the nine columns do not describe one single-valued surface
          break;
        }
        hh[p + 3 * q] = h;
      }
    if (!ok)
      continue;
    // Phase 3 (V2.3): the heights are in cells along `d` and the patch spans cells along
    // `d1`/`d2`, so the metric turns them into the physical graph. Unit metric == today.
    kap(i) = hfPatchKappa(hh, g, d);
    br(i) = static_cast<double>(t == 0 ? kCurvHf : kCurvHfMixed);
    return;
  }

  // Tier 2b — the MIXED height function. No direction gives nine consistent columns; pool
  // the interface positions of whichever of the 27 columns DO close and fit a paraboloid
  // through them in the target's own frame. Every cell it reads is inside the same
  // 3x3x7-per-direction footprint tier 1 already used, so it costs no extra halo.
  if (useFit) {
    const double m0 = mx(i), m1 = my(i), m2 = mz(i);
    const double mi[3] = {m0, m1, m2};
    double nn[3] = {0.0, 0.0, 0.0};
    // PHYSICAL frame (Phase 3, V2.4); identity at the unit metric.
    if (vofPhysNormalInv(mi, g, nn) > 0.0) {
      double t1[3], t2[3];
      curvFrame(nn, t1, t2);
      double vtx[8][3], ctr[3], area;
      const int nvt = plicPolygon(m0, m1, m2, al(i), vtx);
      polygonAreaCentroid(vtx, nvt, ctr, area);
      const double org[3] = {ctr[0] - 0.5, ctr[1] - 0.5, ctr[2] - 0.5};
      PtFit pf;
      ptFitInit(pf);
      for (int d = 0; d < 3; ++d) {
        const int d1 = (d + 1) % 3, d2 = (d + 2) % 3;
        const long sd = st[d], s1 = st[d1], s2 = st[d2];
        for (int q = -1; q <= 1; ++q)
          for (int p = -1; p <= 1; ++p) {
            const long base = i + p * s1 + q * s2;
            double col[kHfColumn];
            for (int k = 0; k < kHfColumn; ++k)
              col[k] = c(base + (k - kHfColumn / 2) * sd);
            double hv;
            int orient;
            if (!hfColumnHeight(col, kHfColumn, hv, orient, mtol, peps))
              continue;
            double X[3];
            X[d1] = static_cast<double>(p);
            X[d2] = static_cast<double>(q);
            X[d] = orient * hv;  // the interface POSITION along d (h is the signed height)
            ptFitAdd(pf, X, org, t1, t2, nn, ptW, g);
          }
      }
      double a[6];
      bool red = false;
      if (pf.npt >= 6 && ptFitSolve(pf, a, red) && !red) {
        kap(i) = paraboloidKappa(a);
        br(i) = static_cast<double>(kCurvHfFit);
        return;
      }
    }
  }
  br(i) = -1.0;  // to the fallback
}

/// Tier 3's last step: solve the accumulated PV system and write the target's kappa and branch.
template <class SF>
KOKKOS_INLINE_FUNCTION void curvFallbackStore(long i, const PvFit& fit, SF kap, SF br) {
  double a[6];
  bool red = false;
  if (!pvFitSolve(fit, a, red)) {
    kap(i) = 0.0;
    br(i) = static_cast<double>(kCurvNoEstimate);
    return;
  }
  kap(i) = paraboloidKappa(a);
  br(i) = static_cast<double>(red ? kCurvPvReduced : kCurvPv);
}

// ---- tier 3 at cost (design G, `doc/vof_curvature_cost_design.md` §5.2-§5.4) -----------------
//
// Each interfacial cell's PLIC polygon is built ONCE per curvature pass, in the planes pass, into a
// per-cell cache indexed by a slot map, and every target reads it (it used to be rebuilt by each of
// the ~16 targets whose 5^3 stencil holds the cell: 63 % of tier 3). The tier-3 body is one
// template over an ENTRY ACCESSOR, so the compacted path (`VofPvCached`: the slot map + the cache)
// and the dense compaction oracle (`VofPvOnTheFly`: the entry built from the planes on the spot)
// share one body; both hand out bit-identical entries because both run `pvPolygonBuild` on the
// same plane.

/// The cache-backed accessor: `slot[j] >= 0` iff cell `j` of the grown region is interfacial, and
/// `cache[slot[j]]` is its entry (written by the planes pass of the same cascade).
struct VofPvCached {
  const int* slot;
  const PvPolygon* cache;
  KOKKOS_INLINE_FUNCTION bool present(long j) const { return slot[j] >= 0; }
  KOKKOS_INLINE_FUNCTION const PvPolygon& entry(long j) const { return cache[slot[j]]; }
};

/// The on-the-fly accessor of the dense mode (`useWorklist = false`, no list and no slot): the
/// interfacial test is `vofIsInterface(c(j))`, and the entry is built from the planes.
template <class SF>
struct VofPvOnTheFly {
  SF c, mx, my, mz, al;
  double ieps;
  KOKKOS_INLINE_FUNCTION bool present(long j) const { return vofIsInterface(c(j), ieps); }
  KOKKOS_INLINE_FUNCTION PvPolygon entry(long j) const {
    PvPolygon P;
    pvPolygonBuild(mx(j), my(j), mz(j), al(j), P);
    return P;
  }
};

/// Tier 3's fit frame for a target with PLIC normal `(m0, m1, m2)` and cached polygon `P`: the unit
/// PHYSICAL normal `nn` (V2.4; identity at h = 1), the tangents `t1`, `t2`, and the origin `org` =
/// the target cell's own PLIC centroid in target-centred cell units. false when there is no
/// normal. The pre-cache `curvFallbackFrame` with `plicPolygon` replaced by the cached polygon: the
/// same expressions in the same order (bitwise).
KOKKOS_INLINE_FUNCTION bool curvFallbackFrameCached(double m0, double m1, double m2,
                                                    const PvPolygon& P, const VofMetric& g,
                                                    double nn[3], double t1[3], double t2[3],
                                                    double org[3]) {
  const double mi[3] = {m0, m1, m2};
  nn[0] = nn[1] = nn[2] = 0.0;
  if (!(vofPhysNormalInv(mi, g, nn) > 0.0))
    return false;
  curvFrame(nn, t1, t2);
  double ctr[3], area;
  polygonAreaCentroid(P.v, P.nv, ctr, area);
  org[0] = ctr[0] - 0.5;
  org[1] = ctr[1] - 0.5;
  org[2] = ctr[2] - 0.5;
  return true;
}

/// One stencil cell's PV term from its cache entry (design G §5.3, V5): the support prefilter,
/// then `pvTermNormal` and `pvTermPolygon` — `pvFitTerm` bit for bit wherever it accepts, and a
/// rejection wherever it rejects (core `test_vof_pvcache` T2).
KOKKOS_INLINE_FUNCTION bool curvPvTermCached(PvTerm& t, const PvPolygon& P, double mx, double my,
                                             double mz, const double off[3], const double org[3],
                                             const double t1[3], const double t2[3],
                                             const double nn[3], double dW, double cmin,
                                             const VofMetric& g) {
  t.ok = false;
  if (pvOutsideSupport(P, off, org, dW, g))
    return false;
  double np[3];
  if (!pvTermNormal(mx, my, mz, t1, t2, nn, cmin, g, np))
    return false;
  if (P.nv < 3)
    return false;
  return pvTermPolygon(t, np, P.v, P.nv, off, org, t1, t2, nn, dW, g);
}

/// Tier 3 for one target, one thread (host): the frame from the target's entry, the 5^3 stencil in
/// canonical order, lower-triangle accumulation, the solve. `acc` is `VofPvCached` or
/// `VofPvOnTheFly`.
template <class SF, class Acc>
KOKKOS_INLINE_FUNCTION void curvFallbackCell(long i, const Acc& acc, SF mx, SF my, SF mz, SF kap,
                                             SF br, long sy, long sz, int gr, double dW,
                                             double cmin, VofMetric g) {
  if (br(i) >= 0.0)
    return;

  double nn[3], t1[3], t2[3], org[3];
  const auto& Pi = acc.entry(i);
  if (!curvFallbackFrameCached(mx(i), my(i), mz(i), Pi, g, nn, t1, t2, org)) {
    kap(i) = 0.0;
    br(i) = static_cast<double>(kCurvNoEstimate);
    return;
  }

  PvFit fit;
  pvFitInit(fit);
  for (int oz = -gr; oz <= gr; ++oz)
    for (int oy = -gr; oy <= gr; ++oy)
      for (int ox = -gr; ox <= gr; ++ox) {
        const long j = i + (long)ox + (long)oy * sy + (long)oz * sz;
        if (!acc.present(j))
          continue;
        const double off[3] = {static_cast<double>(ox), static_cast<double>(oy),
                               static_cast<double>(oz)};
        PvTerm t;
        if (curvPvTermCached(t, acc.entry(j), mx(j), my(j), mz(j), off, org, t1, t2, nn, dW, cmin,
                             g))
          pvFitAccumLower(fit, t);
      }
  curvFallbackStore(i, fit, kap, br);
}

/// `curvFallbackCell` for one target on a whole team (C3, `doc/vof_step_performance_design.md`
/// §4.7, §5.11): every lane computes the frame (deterministic, so no broadcast), the lanes map the
/// `(2gr+1)^3` stencil offsets to `PvTerm`s in `terms` (team scratch) in the canonical index
/// `k = ((oz+gr)(2gr+1) + (oy+gr))(2gr+1) + (ox+gr)`, and ONE lane folds the accepted terms in
/// `k` order and solves. The map runs in parallel; the reduction keeps `curvFallbackCell`'s order,
/// so the result is bit for bit the one-thread body's. Every lane must call it (it holds a team
/// barrier).
template <class Member, class SF, class Acc, class Scratch>
KOKKOS_INLINE_FUNCTION void curvFallbackTeam(const Member& tm, Scratch terms, long i,
                                             const Acc& acc, SF mx, SF my, SF mz, SF kap, SF br,
                                             long sy, long sz, int gr, double dW, double cmin,
                                             VofMetric g) {
  if (br(i) >= 0.0)  // uniform across the team: nothing writes br(i) before the barrier below
    return;
  double nn[3], t1[3], t2[3], org[3];
  const bool framed =
      curvFallbackFrameCached(mx(i), my(i), mz(i), acc.entry(i), g, nn, t1, t2, org);
  const int side = 2 * gr + 1, nk = side * side * side;
  if (framed)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nk), [&](const int k) {
      const int ox = k % side - gr, oy = (k / side) % side - gr, oz = k / (side * side) - gr;
      const long j = i + (long)ox + (long)oy * sy + (long)oz * sz;
      PvTerm t;
      t.ok = false;
      if (acc.present(j)) {
        const double off[3] = {static_cast<double>(ox), static_cast<double>(oy),
                               static_cast<double>(oz)};
        curvPvTermCached(t, acc.entry(j), mx(j), my(j), mz(j), off, org, t1, t2, nn, dW, cmin, g);
      }
      terms(k) = t;
    });
  tm.team_barrier();  // the terms are in scratch, and every lane has read br(i)
  Kokkos::single(Kokkos::PerTeam(tm), [&]() {
    if (!framed) {
      kap(i) = 0.0;
      br(i) = static_cast<double>(kCurvNoEstimate);
      return;
    }
    PvFit fit;
    pvFitInit(fit);
    for (int k = 0; k < nk; ++k)
      if (terms(k).ok)
        pvFitAccumLower(fit, terms(k));
    curvFallbackStore(i, fit, kap, br);
  });
}

/// The team size of a device tier-3 launch: one warp / wavefront.
#if defined(KOKKOS_ENABLE_HIP)
inline constexpr int kVofWarp = 64;
#else
inline constexpr int kVofWarp = 32;
#endif

/// A raw-pointer field accessor with the `View<double*>` call syntax, for kernels that walk SEVERAL
/// blocks' fields in one launch (a device-side array of `View`s would carry reference counts).
/// The cell bodies above are templates on the field type and use nothing but `operator()`.
struct VofRawField {
  double* p;
  KOKKOS_INLINE_FUNCTION double& operator()(long i) const { return p[i]; }
};

/// One block's share of a batched tier-3 launch (`VofCurvature::fallbackBatch`): its fields, its
/// compacted interfacial list, and every argument its own `fallbackPass` would pass to
/// `curvFallbackCell`.
struct VofCurvFallbackJob {
  double *c, *mx, *my, *mz, *al, *kap, *br;
  const int* slot;         ///< the cascade's slot map (design G §5.2)
  const PvPolygon* cache;  ///< ... and its per-cell polygon cache
  const long* list;
  long n, sy, sz;
  int gr;
  double dW, cmin, ieps;
  VofMetric gm;
};

/// The job table of one batched launch, passed to the kernel BY VALUE (namespace scope: nvcc's
/// extended lambdas may not capture a function-local type).
template <int N>
struct VofCurvFallbackTable {
  VofCurvFallbackJob job[N];
  long off[N + 1];
  int nj;
};

/// Curvature of the colour field on an extended (inner + ghost) block.
class VofCurvature {
 public:
  using IField = Kokkos::View<int*, SMem>;
  using PolyCache = Kokkos::View<PvPolygon*, SMem>;
  /// Per-call census of the cascade over this block's inner region (LOCAL; a distributed caller
  /// sums them itself — the driver stays MPI-free, exactly as `WyAdvector` does).
  struct Stats {
    long interfacial = 0;  ///< cells carrying an interface (`vofIsInterface`, i.e. 0 < C < 1
                           ///< unless `interfaceEps` was raised) — the cells a kappa is asked for
    long hf = 0;           ///< tier 1: HF in the preferred direction
    long hfMixed = 0;      ///< tier 2a: HF in one of the other two directions
    long hfFit = 0;        ///< tier 2b: mixed HF -- paraboloid through the consistent columns
    long pv = 0;           ///< tier 3: PV paraboloid fit, full 6-parameter model
    long pvReduced = 0;    ///< tier 3 with the rank-deficient 3-parameter model
    long noEstimate = 0;   ///< NO estimate produced (must be 0 on every gated case)
    /// Cells whose curvature the admissibility clip (`kappaMax`) bounded this call. 0 on every
    /// resolved interface; a non-zero count says the cascade was asked for a curvature it cannot
    /// resolve (`R < 2 Delta`) -- debris, a pinch-off tail, a sub-cell drop.
    long clipped = 0;
  };

  /// @param ghost must be >= 3 (the column reach); the colour block's `kVofG` is 3.
  void init(int nx, int ny, int nz, int ghost) {
    if (ghost < kHfColumn / 2)
      throw std::invalid_argument(
          "peclet::flow::vof::VofCurvature: ghost width must be >= 3 (the height-function column "
          "reaches 3 cells and the 5^3 PV stencil's MYC normals reach 3 cells)");
    if (nx < 1 || ny < 1 || nz < 1)
      throw std::invalid_argument("peclet::flow::vof::VofCurvature: empty block");
    n_ = I3{nx, ny, nz};
    g_ = ghost;
    e_ = I3{nx + 2 * ghost, ny + 2 * ghost, nz + 2 * ghost};
    len_ = static_cast<long>(e_.x) * e_.y * e_.z;
    mx_ = SField("vof::curv::mx", len_);
    my_ = SField("vof::curv::my", len_);
    mz_ = SField("vof::curv::mz", len_);
    alpha_ = SField("vof::curv::alpha", len_);
    kappa_ = SField("vof::curv::kappa", len_);
    branch_ = SField("vof::curv::branch", len_);
    // WO-V9: the compaction lists. The grown one covers every cell `reconstructPlanes` writes; the
    // inner one every cell the cascade runs on. Sized at the full region (worst case: an entirely
    // interfacial block) so a scan can never overflow them.
    const long grown = (long)(n_.x + 2 * kPvHalf) * (n_.y + 2 * kPvHalf) * (n_.z + 2 * kPvHalf);
    listG_ = LField("vof::curv::listG", grown);
    listI_ = LField("vof::curv::listI", (long)n_.x * n_.y * n_.z);
    // Design G §5.2: the slot map over the extended block (read only inside the grown region,
    // where the planes pass writes it) and the per-cell polygon cache, sized to listG_'s capacity
    // on the first worklist planes pass (`ensureCache_`; a block-container cascade whose passes are
    // batched never runs one and never allocates it).
    slot_ = IField("vof::curv::slot", len_);
    cache_ = PolyCache();
  }

  bool ready() const { return kappa_.extent(0) != 0; }

  // ---- WO-V9: compaction, and the per-pass timers --------------------------------------------
  //
  // The cascade is the most divergent kernel in the VoF pipeline: on a resolved droplet the
  // interfacial cells are ~0.5 % of the block, so a dense `parallel_for` puts at most one or two
  // active lanes in a warp and the other thirty run the guard and idle for the whole height
  // function.  Compacting the interfacial cells into a contiguous list first makes those warps
  // full.  It is a pure re-ordering — every cell reads the same neighbours and writes the same
  // value — so the two paths are BIT-IDENTICAL and the ctest gates that.
  //
  // The timers follow `WyAdvector`'s rule exactly: fence at the boundaries only when armed.
  struct Timing {
    double planes = 0.0;    ///< the PLIC pass over the grown region
    double height = 0.0;    ///< tiers 1/2, the height-function cascade
    double fallback = 0.0;  ///< tier 3, the PLIC-volumetric paraboloid
    double census = 0.0;    ///< the branch census reduction
    double compact = 0.0;   ///< the two parallel_scans (0 when the compaction is off)
    double clip = 0.0;      ///< the curvature admissibility clip
    long calls = 0;
  };
  bool useWorklist = true;
  bool timingOn = false;
  Timing tm;
  void resetTiming() { tm = Timing(); }
  const Timing& timing() const { return tm; }
  double tick_() const {
    if (!timingOn)
      return 0.0;
    Kokkos::fence();
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  void addT_(double& acc, double t0) {
    if (!timingOn)
      return;
    Kokkos::fence();
    acc +=
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() -
        t0;
  }
  I3 inner() const { return n_; }
  I3 extent() const { return e_; }
  int ghost() const { return g_; }
  SField kappa() const { return kappa_; }
  SField branch() const { return branch_; }
  /// The PLIC planes the cascade reconstructed (`planeM(d)`, `planeAlpha()`): for the block
  /// container's batched cascade (`vof/block_batch.hpp`), which drives the passes itself.
  SField planeM(int d) const { return d == 0 ? mx_ : (d == 1 ? my_ : mz_); }
  SField planeAlpha() const { return alpha_; }
  /// The slot map of the polygon cache (design G §5.2), for the batched cascade, which writes it
  /// with positions into its own chunk-level cache.
  IField slotMap() const { return slot_; }

  // ---- tunables (all measured knobs, defaults are the literature values) ---------------------
  /// The anisotropic cell metric (Phase 3). Default `{1,1,1}` == the pre-Phase-3 arithmetic.
  /// `Solver::pushVofMetric()` sets it; the Wendland supports below are scaled by `metric.maxH()`
  /// so the 5^3 stencil's farthest cell along the LONG axis stays inside the support exactly as
  /// `d = 2.5` cells does on a cubic grid.
  VofMetric metric;
  /// Wendland support width `d` of the PV fit, in cell units (Han et al. §5: 2.5 with S = 5).
  double weightWidth = kPvWeightWidth;
  /// Tolerance of the PURITY test that ends a height-function column (`hfIsFull`/`hfIsEmpty`),
  /// floored by `core`'s `kHfPureEps = 1e-10`.
  ///
  /// **Set this to the tolerance the COLOUR FIELD was advected at** (`WyAdvector::wispEps`);
  /// `Solver::computeVofCurvature` does. The two must agree about what a pure cell is. When the
  /// advector is the looser of the two it treats a cell at `1 - O(1e-9)` as pure -- fluxing it
  /// algebraically instead of reconstructing it -- so the colour field accumulates bulk cells in
  /// the band between the two tolerances, the column walk finds no pure end there, and the whole
  /// height-function tier degrades into the paraboloid fallback. It degrades SILENTLY: a fallback
  /// is a valid answer, so nothing fails, the curvature just gets worse as the run goes on.
  /// Measured on the mode-2 droplet (48^3, R = 8, mu = 0.0025, 2.5 periods) with the advector at
  /// 1e-8 and this at the 1e-10 floor: the HF tier fell 790 -> 134 cells over the run (37 % ->
  /// 89 % fallback) as 425 bulk cells drifted into the band, and the mode-2 damping rate came out
  /// 2.3x the shared-tolerance value.
  double pureEps = 0.0;
  /// Tolerance of the column monotonicity test (`hfColumnHeight`). Tight enough to reject a
  /// sub-grid blob inside a column, loose enough to ignore transport round-off.
  double monoTol = 1e-6;
  /// Wendland support width of the tier-2b height-position fit, in cell units.
  double ptWeightWidth = kPtWeightWidth;
  /// Minimum `n_p . n` for a stencil polygon to enter the PV fit (`pvFitAdd`).
  double cosMin = 0.2;
  /// **Wisp threshold on the interfacial predicate.** A cell counts as carrying an interface only
  /// while `eps < C < 1 - eps`. DEFAULT 0, which is `wyIsMixed` verbatim (`C > 0 && C < 1`) and
  /// therefore byte-identical to the V3 rung as gated.
  ///
  /// Why it exists (a V4 finding, WO-P). Weymouth-Yue leaves ROUND-OFF colour residue in cells the
  /// sweeps touched — measured on a static droplet after 10 steps: `C` down to `-3e-35` and some
  /// 5200 extra cells at `0 < C < 1e-30` on a 64^3 grid, i.e. more cells than the interface itself
  /// has. Those cells satisfy `wyIsMixed` and the cascade dutifully produces a curvature for them,
  /// **from a PLIC polygon of area ~0**: measured `|kappa|` up to **1.2e+08** where the physical
  /// value is 0.125. On its own that is only a wrong number in a field nobody was reading. Under
  /// V4 it is fatal: a face between such a cell and a REAL interfacial cell has `dC = O(1)` and a
  /// face curvature `(kappa_real + 1e8)/2`, so the surface-tension force at that one face is eight
  /// orders too large. Measured on a 32^3 static droplet, unguarded: `max|u|` 4.5e-4 at step 1 ->
  /// **2.7e-1** by step 20, and at 96^3 the run trips the Weymouth-Yue CFL cap outright, while the
  /// SAME run with the curvature frozen at its (clean) initial value stays bounded at 4e-3.
  ///
  /// `Solver::setSurfaceTension` therefore sets this to 1e-8 — the same wisp threshold
  /// `WyAdvector::diagnostics` already reports against. It changes nothing on an exact colour field
  /// (no cell of any gated V3 case lies in the band) and it is the guard the literature assumes
  /// when it says wisp cleanup is unavoidable once surface tension is on (VOF_PLAN §6, after
  /// Arrufat et al. 2021).
  double interfaceEps = 0.0;
  /// DIAGNOSTIC ONLY — disable tiers 1 and 2 so every interfacial cell goes to the PV fit. Used by
  /// the ctest to measure the fallback branch on its own; never enable it in production.
  bool debugForceFallback = false;
  /// DIAGNOSTIC ONLY — disable tier 2 (the non-preferred column directions), so the branch census
  /// separates "the preferred direction alone" from "the direction cascade".
  bool debugSingleDirection = false;
  /// **Tier 2b, and it ships OFF — a measured decision, see the WO-O findings entry.**
  ///
  /// The mixed height function (a paraboloid through the interface positions of whichever columns
  /// closed, `PtFit`) is implemented and correct, and on the exact-fraction sphere it takes over
  /// exactly the 19.5-59.6 % of interfacial cells that tier 1 cannot serve. Measured on that
  /// sphere at 16/32/64 with everything else identical:
  ///
  ///     tier 2b ON   L1 2.83e-2 / 7.27e-3 / 4.21e-3   max 6.08e-2 / 4.64e-2 / 6.07e-2
  ///     tier 2b OFF  L1 3.03e-2 / 5.94e-3 / 1.32e-3   max 5.01e-2 / 1.32e-2 / 3.79e-3
  ///     order        L1 1.37 vs 2.26                  max 0.00 vs 1.86
  ///
  /// i.e. it destroys the convergence of the MAX error, on the very cells the PV fallback handles
  /// at second order. The mechanism is structural, not a parameter choice (four Wendland widths
  /// from 1.5 to 6.0 cells were swept; none converges in the max): **its data set is the columns
  /// the height function could close, which is a slope-SELECTED subset.** At a cell whose normal
  /// sits near an octant diagonal the corner columns on the steep side are exactly the ones that
  /// fail, so the surviving points sample the interface asymmetrically about the target and the
  /// quadratic fit acquires a lever-arm bias. That selection depends on the normal direction and
  /// not on h, so the bias is scale invariant — which is precisely the flat max-error curve above.
  /// The PV fallback is immune because a PLIC polygon exists in every mixed cell whatever the
  /// slope, so its 5^3 data set is symmetric.
  ///
  /// Turning it on is therefore a measurement, not a configuration. Kept because it is the WO's
  /// specified tier 2 and because the mechanism above is worth being able to re-measure.
  bool useMixedHeightFit = false;
  /// **Curvature admissibility clip** (`doc/vof_overlap_design.md` §5.1, §11): after the cascade,
  /// every inner cell carrying a curvature (`csfKappaDefined`) gets `|kappa| <= kappaMax`, in INDEX
  /// units (1/hRef). `0` (the class DEFAULT) is off -- the single-field cascade never clips (§11:
  /// there it would bound LEGITIMATE curvature: under-resolved droplets, contact-line cells).
  /// The block container's prototype sets it ON. Negative means `1 / metric.minH()`, which is 1
  /// when `hRef = min h`:
  /// `|kappa| <= 1/Delta` is a sphere of `R >= 2 Delta`, the smallest the 7-cell height function
  /// can see at all, and it bounds every CSF face force by `sigma |dC| / Delta^2` -- the scale the
  /// Brackbill capillary step is built to hold. The value is written only where it CHANGES, so an
  /// unclipped field is bit-identical.
  ///
  /// Why it exists: an interfacial cell of a marker with no body of that marker near it (DEBRIS,
  /// left where two markers overlapped) has no pure end in any height-function column, and the
  /// PLIC-volumetric fit is then made from a handful of corner-sliver polygons of total area ~0.
  /// The paraboloid through them is arbitrary: measured |kappa| = 273 and 428 per cell against a
  /// true 0.4 on channel_18, face forces 3-100x the physical ones, a one-step velocity blow-up.
  double kappaMax = 0.0;
  /// Copy EVERY tunable above (the metric, the estimator knobs, the predicates, the clip, the
  /// diagnostic switches) from `src` -- the one place a prototype reaches a cascade, so a new
  /// tunable is added here and nowhere else (review finding 1: the block path once dropped
  /// `pureEps`; the block ctest pins every field through this function).
  void copyTunablesFrom(const VofCurvature& src) {
    metric = src.metric;
    weightWidth = src.weightWidth;
    pureEps = src.pureEps;
    monoTol = src.monoTol;
    ptWeightWidth = src.ptWeightWidth;
    cosMin = src.cosMin;
    interfaceEps = src.interfaceEps;
    debugForceFallback = src.debugForceFallback;
    debugSingleDirection = src.debugSingleDirection;
    useMixedHeightFit = src.useMixedHeightFit;
    kappaMax = src.kappaMax;
    useWorklist = src.useWorklist;
  }
  /// The bound the clip applies this call (0 = off).
  double kappaClipValue() const { return kappaMax < 0.0 ? 1.0 / metric.minH() : kappaMax; }

  /// Compute the curvature over the inner region from a colour field on the SAME extended block.
  /// `c`'s ghosts must be valid on entry (the caller's exchange); nothing here communicates.
  Stats compute(SField c) {
    computeBegin(c);
    const double t3 = tick_();
    fallbackPass(c);
    addT_(tm.fallback, t3);
    return computeEnd();
  }

  /// `compute()` in three parts, for a caller that runs MANY cascades (the block container):
  /// `computeBegin` (compaction, planes, tiers 1-2) on every cascade, ONE `fallbackBatch` launch of
  /// tier 3 over all of them, then `computeEnd` (clip + census) on every cascade. Each cell's
  /// tier-3 body reads only its own block's planes/colour and writes only its own kappa/branch, so
  /// the batch computes exactly what the per-cascade `fallbackPass` calls would: bit for bit.
  void computeBegin(SField c) {
    if (!ready())
      throw std::runtime_error("peclet::flow::vof::VofCurvature::compute: init() not called");
    const double t0 = tick_();
    if (useWorklist)
      compact(c);
    addT_(tm.compact, t0);
    const double t1 = tick_();
    reconstructPlanes(c);
    addT_(tm.planes, t1);
    const double t2 = tick_();
    heightPass(c);
    addT_(tm.height, t2);
  }
  Stats computeEnd() {
    const double t5 = tick_();
    const long nclip = clipPass();
    addT_(tm.clip, t5);
    const double t4 = tick_();
    Stats s = census();
    addT_(tm.census, t4);
    s.clipped = nclip;
    ++tm.calls;
    return s;
  }
  /// This cascade's tier-3 share for `fallbackBatch` (worklist mode only: the job walks the
  /// compacted interfacial list `computeBegin` just built).
  VofCurvFallbackJob fallbackJob(SField c) const {
    VofCurvFallbackJob j;
    j.c = c.data();
    j.mx = mx_.data();
    j.my = my_.data();
    j.mz = mz_.data();
    j.al = alpha_.data();
    j.kap = kappa_.data();
    j.br = branch_.data();
    j.slot = slot_.data();
    j.cache = cache_.data();
    j.list = listI_.data();
    j.n = nI_;
    j.sy = e_.x;
    j.sz = static_cast<long>(e_.x) * e_.y;
    j.gr = kPvHalf;
    j.dW = weightWidth * metric.maxH();
    j.cmin = cosMin;
    j.ieps = interfaceEps;
    j.gm = metric;
    return j;
  }
  /// Jobs per launch of `fallbackBatch` (the table rides in the kernel's functor by value).
  static constexpr int kFallbackBatch = 16;
  /// Tier 3 of many cascades in as few launches as possible: one kernel per `kFallbackBatch`
  /// jobs over their concatenated interfacial lists. The per-cascade launches each put ~1e3
  /// threads (one wave, a few warps per SM) on the device for the full latency of the 5^3 fit,
  /// one block after the other; batched they share that latency. Pure re-scheduling (see
  /// `computeBegin`).
  static void fallbackBatch(const std::vector<VofCurvFallbackJob>& jobs) {
    using Table = VofCurvFallbackTable<kFallbackBatch>;
    for (std::size_t j0 = 0; j0 < jobs.size(); j0 += kFallbackBatch) {
      Table T;
      T.nj = static_cast<int>(std::min<std::size_t>(kFallbackBatch, jobs.size() - j0));
      T.off[0] = 0;
      for (int k = 0; k < T.nj; ++k) {
        T.job[k] = jobs[j0 + k];
        T.off[k + 1] = T.off[k] + T.job[k].n;
      }
      for (int k = T.nj; k < kFallbackBatch; ++k)
        T.off[k + 1] = T.off[T.nj];
      if (T.off[T.nj] == 0)
        continue;
      if constexpr (!Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                                SField::memory_space>::accessible) {
        // C3 (§5.11): one team (a warp) per target on a device; the host keeps one thread per
        // target below.
        using Policy = Kokkos::TeamPolicy<SExec>;
        using Terms = Kokkos::View<PvTerm*, SExec::scratch_memory_space, Kokkos::MemoryUnmanaged>;
        int gr = 0;
        for (int k = 0; k < T.nj; ++k)
          gr = std::max(gr, T.job[k].gr);
        const int nk = (2 * gr + 1) * (2 * gr + 1) * (2 * gr + 1);
        Policy pol(SExec(), T.off[T.nj], kVofWarp);
        pol.set_scratch_size(0, Kokkos::PerTeam(Terms::shmem_size(nk)));
        Kokkos::parallel_for(
            "vof::curv::pv_batch_team", pol, KOKKOS_LAMBDA(const typename Policy::member_type& tm) {
              const long t = tm.league_rank();
              int b = 0;
              while (t >= T.off[b + 1])
                ++b;
              const VofCurvFallbackJob& J = T.job[b];
              const int side = 2 * J.gr + 1;
              Terms terms(tm.team_scratch(0), side * side * side);
              curvFallbackTeam(tm, terms, J.list[t - T.off[b]], VofPvCached{J.slot, J.cache},
                               VofRawField{J.mx}, VofRawField{J.my}, VofRawField{J.mz},
                               VofRawField{J.kap}, VofRawField{J.br}, J.sy, J.sz, J.gr, J.dW,
                               J.cmin, J.gm);
            });
        continue;
      }
      Kokkos::parallel_for(
          "vof::curv::pv_batch", Kokkos::RangePolicy<SExec>(SExec(), 0, T.off[T.nj]),
          KOKKOS_LAMBDA(long t) {
            int b = 0;
            while (t >= T.off[b + 1])
              ++b;
            const VofCurvFallbackJob& J = T.job[b];
            curvFallbackCell(J.list[t - T.off[b]], VofPvCached{J.slot, J.cache}, VofRawField{J.mx},
                             VofRawField{J.my}, VofRawField{J.mz}, VofRawField{J.kap},
                             VofRawField{J.br}, J.sy, J.sz, J.gr, J.dW, J.cmin, J.gm);
          });
    }
  }

  /// The two compaction scans: `listG_` = every interfacial cell of the GROWN region (what
  /// `reconstructPlanes` reconstructs), `listI_` = every interfacial cell of the INNER region
  /// (what the cascade runs on).  Same predicate as the passes, so the compacted and dense paths
  /// visit exactly the same cells.
  void compact(SField c) {
    const I3 e = e_, n = n_;
    const int g = g_, gr = kPvHalf;
    const double ieps = interfaceEps;
    {
      const int rx = n.x + 2 * gr, ry = n.y + 2 * gr, rz = n.z + 2 * gr;
      const long region = (long)rx * ry * rz;
      LField list = listG_;
      long cnt = 0;
      Kokkos::parallel_scan(
          "vof::curv::compactG", Kokkos::RangePolicy<SExec>(SExec(), 0, region),
          KOKKOS_LAMBDA(const long r, long& upd, const bool final) {
            const int ix = (int)(r % rx);
            const int iy = (int)((r / rx) % ry);
            const int iz = (int)(r / ((long)rx * ry));
            const long i = L3(g - gr + ix, g - gr + iy, g - gr + iz, e);
            if (vofIsInterface(c(i), ieps)) {
              if (final)
                list(upd) = i;
              ++upd;
            }
          },
          cnt);
      Kokkos::fence();
      nG_ = cnt;
    }
    {
      const long region = (long)n.x * n.y * n.z;
      const int nx = n.x, ny = n.y;
      LField list = listI_;
      long cnt = 0;
      Kokkos::parallel_scan(
          "vof::curv::compactI", Kokkos::RangePolicy<SExec>(SExec(), 0, region),
          KOKKOS_LAMBDA(const long r, long& upd, const bool final) {
            const int ix = (int)(r % nx);
            const int iy = (int)((r / nx) % ny);
            const int iz = (int)(r / ((long)nx * ny));
            const long i = L3(g + ix, g + iy, g + iz, e);
            if (vofIsInterface(c(i), ieps)) {
              if (final)
                list(upd) = i;
              ++upd;
            }
          },
          cnt);
      Kokkos::fence();
      nI_ = cnt;
    }
  }

  // The four passes below are PUBLIC only because nvcc forbids an extended __host__ __device__
  // lambda inside a private or protected member function ("The enclosing parent function ... cannot
  // have private or protected access within its class"). They are implementation detail; call
  // `compute()`. `WyAdvector` is public for the same reason.
  //
  /// PLIC planes over the inner region grown by `kPvHalf` — every cell the 5^3 fit can read.
  /// Reads colour at +/-1 around those, i.e. +/-(kPvHalf+1) = +/-3 overall.
  void reconstructPlanes(SField c) {
    const I3 e = e_, n = n_;
    const int g = g_, gr = kPvHalf;
    const long sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    SField mx = mx_, my = my_, mz = mz_, al = alpha_;
    const double ieps = interfaceEps;
    if (useWorklist) {
      // The same two things the dense kernel does, as two coherent kernels: a branchless zeroing
      // sweep (a pure store, bandwidth-bound) and the MYC reconstruction over the compacted list.
      // Every cell ends with exactly the value the dense kernel would have written. Design G §5.2:
      // the zero sweep also clears the slot map, and the list kernel builds each cell's polygon
      // cache entry from the plane it just wrote and records its slot.
      ensureCache_();
      IField slot = slot_;
      PolyCache cache = cache_;
      Kokkos::parallel_for(
          "vof::curv::planes_zero",
          MDRange3<SExec>(SExec(), {g - gr, g - gr, g - gr},
                          {g + n.x + gr, g + n.y + gr, g + n.z + gr}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = L3(x, y, z, e);
            mx(i) = 0.0;
            my(i) = 0.0;
            mz(i) = 0.0;
            al(i) = 0.0;
            slot(i) = -1;
          });
      LField list = listG_;
      Kokkos::parallel_for(
          "vof::curv::planes_list", Kokkos::RangePolicy<SExec>(SExec(), 0, nG_),
          KOKKOS_LAMBDA(long t) {
            const long i = list(t);
            wyReconstructCell(c, i, sy, sz, mx, my, mz, al);
            pvPolygonBuild(mx(i), my(i), mz(i), al(i), cache(t));
            slot(i) = static_cast<int>(t);
          });
      return;
    }
    Kokkos::parallel_for(
        "vof::curv::planes",
        MDRange3<SExec>(SExec(), {g - gr, g - gr, g - gr},
                        {g + n.x + gr, g + n.y + gr, g + n.z + gr}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e);
          if (!vofIsInterface(c(i), ieps)) {
            mx(i) = 0.0;
            my(i) = 0.0;
            mz(i) = 0.0;
            al(i) = 0.0;
            return;
          }
          wyReconstructCell(c, i, sy, sz, mx, my, mz, al);
        });
  }

  /// Tiers 1 and 2. Writes `kappa`/`branch` for every inner cell; a cell the height functions
  /// cannot serve is left with `branch = -1` for the fallback pass.
  void heightPass(SField c) {
    const I3 e = e_, n = n_;
    const int g = g_;
    const long st[3] = {1, e_.x, static_cast<long>(e_.x) * e_.y};
    SField mx = mx_, my = my_, mz = mz_, al = alpha_, kap = kappa_, br = branch_;
    const double mtol = monoTol, ptW = ptWeightWidth * metric.maxH(), ieps = interfaceEps;
    const double peps = pureEps;
    const VofMetric gm = metric;  // `g` is the ghost width in this scope
    const bool forceFb = debugForceFallback, oneDir = debugSingleDirection,
               useFit = useMixedHeightFit;
    const long s0 = st[0], s1 = st[1], s2 = st[2];
    if (useWorklist) {
      // Reset every inner cell (branchless: a pure pair of stores), then run the cascade over the
      // compacted interfacial list.  A non-interfacial cell ends at kappa = 0, branch = kCurvNone
      // and an interfacial one at whatever the shared body writes -- the dense kernel's outcome,
      // cell for cell.
      Kokkos::parallel_for(
          "vof::curv::hf_reset", MDRange3<SExec>(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = L3(x, y, z, e);
            kap(i) = 0.0;
            br(i) = static_cast<double>(kCurvNone);
          });
      LField list = listI_;
      Kokkos::parallel_for(
          "vof::curv::hf_list", Kokkos::RangePolicy<SExec>(SExec(), 0, nI_), KOKKOS_LAMBDA(long t) {
            curvHeightCell(list(t), c, mx, my, mz, al, kap, br, s0, s1, s2, mtol, ptW, ieps,
                           forceFb, oneDir, useFit, gm, peps);
          });
      return;
    }
    Kokkos::parallel_for(
        "vof::curv::hf", MDRange3<SExec>(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          curvHeightCell(L3(x, y, z, e), c, mx, my, mz, al, kap, br, s0, s1, s2, mtol, ptW, ieps,
                         forceFb, oneDir, useFit, gm, peps);
        });
  }

  /// Tier 3, over the cells the height pass could not serve. A plain guarded `parallel_for` rather
  /// than a compacted worklist: the branch is a minority at any usable resolution, so almost every
  /// warp exits at the guard, and the fallback's larger local state is confined to this kernel.
  void fallbackPass(SField c) {
    const I3 e = e_, n = n_;
    const int g = g_, gr = kPvHalf;
    SField mx = mx_, my = my_, mz = mz_, al = alpha_, kap = kappa_, br = branch_;
    const double dW = weightWidth * metric.maxH(), cmin = cosMin, ieps = interfaceEps;
    const VofMetric gm = metric;  // `g` is the ghost width in this scope
    const long sy = e_.x, sz = static_cast<long>(e_.x) * e_.y;
    if (useWorklist) {
      // Tier 3 is a SUBSET of the interfacial cells (those tier 1/2 could not serve), so the
      // interfacial list already removes the whole empty domain; the branch guard inside the body
      // removes the rest.
      LField list = listI_;
      const VofPvCached acc{slot_.data(), cache_.data()};
      Kokkos::parallel_for(
          "vof::curv::pv_list", Kokkos::RangePolicy<SExec>(SExec(), 0, nI_), KOKKOS_LAMBDA(long t) {
            curvFallbackCell(list(t), acc, mx, my, mz, kap, br, sy, sz, gr, dW, cmin, gm);
          });
      return;
    }
    const VofPvOnTheFly<SField> acc{c, mx, my, mz, al, ieps};
    Kokkos::parallel_for(
        "vof::curv::pv", MDRange3<SExec>(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          curvFallbackCell(L3(x, y, z, e), acc, mx, my, mz, kap, br, sy, sz, gr, dW, cmin, gm);
        });
  }

  /// The admissibility clip (see `kappaMax`), over the same cells the cascade ran on: the
  /// compacted interfacial list in worklist mode, the inner region in dense mode. Returns the
  /// number of cells it bounded (an integer reduction: deterministic).
  long clipPass() {
    const double km = kappaClipValue();
    if (!(km > 0.0))
      return 0;
    const I3 e = e_, n = n_;
    const int g = g_;
    SField kap = kappa_, br = branch_;
    long cnt = 0;
    if (useWorklist) {
      LField list = listI_;
      Kokkos::parallel_reduce(
          "vof::curv::clip_list", Kokkos::RangePolicy<SExec>(SExec(), 0, nI_),
          KOKKOS_LAMBDA(long t, long& acc) {
            const long i = list(t);
            // `!(|k| <= km)` rather than `|k| > km`: a NaN curvature is clipped (to +km, the sign
            // of a NaN is meaningless) and counted, never silently kept.
            if (csfKappaDefined(br(i)) && !(Kokkos::fabs(kap(i)) <= km)) {
              kap(i) = (kap(i) == kap(i)) ? Kokkos::copysign(km, kap(i)) : km;
              ++acc;
            }
          },
          cnt);
    } else {
      Kokkos::parallel_reduce(
          "vof::curv::clip", MDRange3<SExec>(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z}),
          KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
            const long i = L3(x, y, z, e);
            // `!(|k| <= km)` rather than `|k| > km`: a NaN curvature is clipped (to +km, the sign
            // of a NaN is meaningless) and counted, never silently kept.
            if (csfKappaDefined(br(i)) && !(Kokkos::fabs(kap(i)) <= km)) {
              kap(i) = (kap(i) == kap(i)) ? Kokkos::copysign(km, kap(i)) : km;
              ++acc;
            }
          },
          cnt);
    }
    Kokkos::fence();
    return cnt;
  }

  Stats census() const {
    const I3 e = e_, n = n_;
    const int g = g_;
    SField br = branch_;
    Stats s;
    Kokkos::parallel_reduce(
        "vof::curv::census", MDRange3<SExec>(SExec(), {g, g, g}, {g + n.x, g + n.y, g + n.z}),
        KOKKOS_LAMBDA(int x, int y, int z, long& ni, long& n1, long& n2, long& n2b, long& n3,
                      long& n4, long& n5) {
          const int b = static_cast<int>(br(L3(x, y, z, e)));
          if (b == kCurvNone)
            return;
          ++ni;
          if (b == kCurvHf)
            ++n1;
          else if (b == kCurvHfMixed)
            ++n2;
          else if (b == kCurvHfFit)
            ++n2b;
          else if (b == kCurvPv)
            ++n3;
          else if (b == kCurvPvReduced)
            ++n4;
          else
            ++n5;
        },
        s.interfacial, s.hf, s.hfMixed, s.hfFit, s.pv, s.pvReduced, s.noEstimate);
    Kokkos::fence();
    return s;
  }

 private:
  I3 n_{0, 0, 0}, e_{0, 0, 0};
  int g_ = 0;
  long len_ = 0;
  SField mx_, my_, mz_, alpha_, kappa_, branch_;
  LField listG_, listI_;  // WO-V9: the compacted interfacial-cell lists
  long nG_ = 0, nI_ = 0;  // ... and their lengths from the last compact()
  IField slot_;           // design G §5.2: grown-list position of each interfacial cell, else -1
  PolyCache cache_;       // ... and each one's polygon, in grown-list order

  /// Allocate the polygon cache at `listG_`'s capacity. Uninitialized: the planes pass writes
  /// every entry a later pass reads.
  void ensureCache_() {
    const long cap = static_cast<long>(listG_.extent(0));
    if (cap >= (1L << 31))
      throw std::runtime_error(
          "peclet::flow::vof::VofCurvature: grown list capacity >= 2^31 (the slot map is int)");
    if (static_cast<long>(cache_.extent(0)) < cap)
      cache_ = PolyCache(
          Kokkos::view_alloc(std::string("vof::curv::cache"), Kokkos::WithoutInitializing), cap);
  }
};

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_CURVATURE_FIELD_HPP
