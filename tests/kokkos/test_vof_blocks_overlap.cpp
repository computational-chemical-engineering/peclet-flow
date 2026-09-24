// flow — the overlap-robustness rules of the block VoF container (doc/vof_overlap_design.md §5,
// §11-§13), gated one mechanism at a time (review doc/vof_overlap_review.md findings 1-3).
//
//   K  the admissibility clip is LIVE ON THE BLOCK PATH: a detached cell of marker B (C = 1e-6) two
//      cells outside B's resolved surface gets |kappa| = 4.4 from the block cascade without the
//      clip and <= 1 with it (Solver, enable_vof_blocks_from_colors + enable_vof_block_csf); the
//      face force on the speck's own faces moves by at most sigma |dC| / Delta^2.
//   P  every tunable of the block cascade prototype (VofBlockSet::curvProto) and the advector's
//      wispEps / metric reach every block: at seeding, after a re-centring re-allocates the block,
//      and after setWispEps / setKappaMax on live blocks. (Finding 1: pureEps once did not.)
//   W  §12.3: a marker straddling the periodic x seam whose box must grow to the whole axis is
//      re-centred WITH the wrap -- fullAxis > 0, nothing discarded, volume exact, and (wispEps 0)
//      the union equal to a global-field WyAdvector to round-off.
//   R  §13: 1000 kinematic steps of a translating marker at block wispEps 1e-8 conserve the volume
//      to 1e-14 relative with nothing discarded (the residue is returned), where the pre-§13
//      container leaked ~1e-10.
#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

#include "flow_ibm.hpp"
#include "vof/block_container.hpp"
#include "vof/block_exchange.hpp"
#include "vof_advect_scenes.hpp"

using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SField;
using peclet::flow::vof::VofBlockExchange;
using peclet::flow::vof::VofBlockSet;
using peclet::flow::vof::VofBox;
using peclet::flow::vof::VofCurvature;
using peclet::flow::vof::WyAdvector;

namespace {

int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

constexpr int G = 3;

double sumInner(SField f, I3 e, I3 n) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  double s = 0.0;
  for (int z = 0; z < n.z; ++z)
    for (int y = 0; y < n.y; ++y)
      for (int x = 0; x < n.x; ++x)
        s += h(L3(x + G, y + G, z + G, e));
  return s;
}

/// A serial (single-rank) block exchange over a whole-grid patch `pa`.
void serialExchange(VofBlockSet& set, WyAdvector& pa, I3 gs) {
  auto ex = std::make_shared<VofBlockExchange>();
  std::vector<VofBox> rb(1);
  rb[0].lo[0] = rb[0].lo[1] = rb[0].lo[2] = 0;
  rb[0].hi[0] = gs.x;
  rb[0].hi[1] = gs.y;
  rb[0].hi[2] = gs.z;
  ex->init(gs, set.periodic(), rb, 0);
  VofBlockExchange::Patch pp;
  pp.e = pa.extent();
  pp.n = pa.inner();
  pp.o = I3{0, 0, 0};
  pp.g = G;
  ex->setPatch(pp, pa.faceU(), pa.faceV(), pa.faceW());
  set.setExchange(ex);
}

// ------------------------------------------------------------------------------------------ K
double sphereFrac(double cx, double cy, double cz, double R, int x, int y, int z) {
  const int NS = 8;
  int in = 0;
  for (int k = 0; k < NS; ++k)
    for (int j = 0; j < NS; ++j)
      for (int i = 0; i < NS; ++i) {
        const double px = x + (i + 0.5) / NS - cx, py = y + (j + 0.5) / NS - cy,
                     pz = z + (k + 0.5) / NS - cz;
        in += (px * px + py * py + pz * pz < R * R) ? 1 : 0;
      }
  return static_cast<double>(in) / (NS * NS * NS);
}

void gateClipOnBlockPath() {
  std::printf(
      "\n=== K  the admissibility clip is live on the BLOCK path (a speck at B's fringe)\n");
  constexpr int NX = 48, NY = 32, NZ = 32;
  constexpr double R = 5.0, SIGMA = 320.0, BX = 32.3, CY = 16.2, CZ = 16.1;
  const std::array<int, 6> box = {18, 7, 7, 41, 26, 26};
  // (25, 15, 16): 1.84 cells outside B's surface -- measured |kappa| 4.41 unclipped
  constexpr int SX = 25, SY = 15, SZ = 16;
  const long nx = box[3] - box[0], ny = box[4] - box[1];
  auto loc = [&](int x, int y, int z) {
    return (x - box[0]) + nx * ((y - box[1]) + ny * static_cast<long>(z - box[2]));
  };
  auto build = [&](peclet::flow::IbmSolver& s, bool speck, bool clip) {
    s.setRho(1.0);
    s.setMu(0.5);
    s.setPressureGeometry(std::vector<double>(static_cast<std::size_t>(NX) * NY * NZ, 10.0));
    s.enableVof();
    s.setVof(std::vector<double>(static_cast<std::size_t>(NX) * NY * NZ, 0.0));
    s.setSurfaceTension(SIGMA);
    if (!clip)
      s.setVofBlockKappaClip(false);
    std::vector<double> c;
    for (int z = box[2]; z < box[5]; ++z)
      for (int y = box[1]; y < box[4]; ++y)
        for (int x = box[0]; x < box[3]; ++x)
          c.push_back(sphereFrac(BX, CY, CZ, R, x, y, z));
    if (speck) {
      CHECK(c[loc(SX, SY, SZ)] == 0.0);  // the cell really is detached
      c[loc(SX, SY, SZ)] = 1e-6;
    }
    s.enableVofBlocksFromColours({box}, {c});
    s.enableVofBlockCsf();  // computes the block curvature and the CSF force
  };
  peclet::flow::IbmSolver ref(NX, NY, NZ), off(NX, NY, NZ), on(NX, NY, NZ);
  build(ref, false, true);
  build(off, true, false);
  build(on, true, true);
  const double kOff = off.vofBlockKappa(0)[loc(SX, SY, SZ)];
  const double kOn = on.vofBlockKappa(0)[loc(SX, SY, SZ)];
  const long clipOn = on.vofBlockCurvatureStats().clipped,
             clipOff = off.vofBlockCurvatureStats().clipped,
             clipRef = ref.vofBlockCurvatureStats().clipped;
  // face force on the speck's six faces (low face of (SX,SY,SZ) and of its +1 neighbours)
  const long gi = SX + static_cast<long>(NX) * (SY + static_cast<long>(NY) * SZ);
  const long st[3] = {1, NX, static_cast<long>(NX) * NY};
  double dFon = 0.0, dFoff = 0.0, dFallOn = 0.0;
  for (int c = 0; c < 3; ++c) {
    const auto fr = ref.getVofBlockForce(c), fo = off.getVofBlockForce(c),
               fn = on.getVofBlockForce(c);
    for (long f : {gi, gi + st[c]}) {
      dFon = std::fmax(dFon, std::fabs(fn[f] - fr[f]));
      dFoff = std::fmax(dFoff, std::fabs(fo[f] - fr[f]));
    }
    for (std::size_t i = 0; i < fr.size(); ++i)
      dFallOn = std::fmax(dFallOn, std::fabs(fn[i] - fr[i]));
  }
  const double bound = SIGMA * 1e-6;  // sigma max|dC_speck| / Delta^2 on the speck's own faces
  std::printf(
      "  speck kappa: clip off %.4f, clip on %.4f; clipped: on %ld, off %ld, no-speck %ld\n", kOff,
      kOn, clipOn, clipOff, clipRef);
  std::printf(
      "  |F - F_ref| on the speck's faces: clip on %.3e, clip off %.3e (bound %.3e); "
      "anywhere, clip on: %.3e\n",
      dFon, dFoff, bound, dFallOn);
  CHECK(std::fabs(kOff) > 1.0);  // the degenerate fit really is unbounded without the clip
  CHECK(std::fabs(kOn) <= 1.0);
  CHECK(clipOn >= 1);
  CHECK(clipOff == 0);
  CHECK(clipRef == 0);  // a resolved marker never fires it
  CHECK(dFon <= bound * (1.0 + 1e-12));
}

// ------------------------------------------------------------------------------------------ P
/// Every tunable `VofCurvature::copyTunablesFrom` carries, compared field by field.
bool sameTunables(const VofCurvature& a, const VofCurvature& b) {
  bool ok = true;
  for (int d = 0; d < 3; ++d)
    ok = ok && a.metric.h[d] == b.metric.h[d];
  return ok && a.weightWidth == b.weightWidth && a.pureEps == b.pureEps && a.monoTol == b.monoTol &&
         a.ptWeightWidth == b.ptWeightWidth && a.cosMin == b.cosMin &&
         a.interfaceEps == b.interfaceEps && a.debugForceFallback == b.debugForceFallback &&
         a.debugSingleDirection == b.debugSingleDirection &&
         a.useMixedHeightFit == b.useMixedHeightFit && a.kappaMax == b.kappaMax &&
         a.useWorklist == b.useWorklist;
}

void gatePrototypePropagation() {
  std::printf("\n=== P  the block prototype's tunables reach every block cascade and advector\n");
  const I3 gs{40, 40, 40};
  WyAdvector pa;
  pa.init(gs.x, gs.y, gs.z, 1.0, G);
  vofscene::Block blk = vofscene::blockOf(pa, I3{0, 0, 0});
  vofscene::fillUniform(pa, blk, 1.0, 0.0, 0.0);
  VofBlockSet set;
  set.init(gs, {true, true, true}, 0, 1, 1.0);
  serialExchange(set, pa, gs);
  // every tunable at a NON-default value, so a field the copy forgets is caught
  const peclet::flow::vof::VofMetric m{{1.0, 1.0, 1.0 + 1e-9}};
  set.setMetric(m);
  VofCurvature& p = set.curvProto;
  p.weightWidth = 2.4;
  p.monoTol = 2e-6;
  p.ptWeightWidth = 1.9;
  p.cosMin = 0.21;
  p.interfaceEps = 3e-8;
  p.debugSingleDirection = true;
  p.useMixedHeightFit = true;
  p.useWorklist = false;
  p.kappaMax = 1.7;
  set.setWispEps(2e-9);  // sets the advectors AND curvProto.pureEps
  const VofCurvature defaults;
  CHECK(!sameTunables(p, defaults));
  set.enableCsf(1.0);
  set.seedSphere(8.3, 20.1, 20.2, 5.0);
  set.scatter(pa.colour());
  auto allReached = [&](const char* when) {
    bool ok = true;
    for (const auto& b : set.blocks()) {
      if (!b.mine())
        continue;
      const bool c = sameTunables(b.curvature(), set.curvProto);
      const bool a =
          b.advector().wispEps == set.wispEps && b.advector().metric.h[2] == set.metric().h[2];
      std::printf("  %-34s block %ld: cascade tunables %s, advector wispEps/metric %s\n", when,
                  b.id, c ? "match" : "MISMATCH", a ? "match" : "MISMATCH");
      ok = ok && c && a;
    }
    return ok;
  };
  CHECK(allReached("after seeding"));
  long nRe = 0;
  for (long s = 0; s < 40; ++s) {  // 8 cells of translation: several re-centrings
    set.advect(0.2, pa.colour());
    nRe += set.blocks()[0].stats().recentred ? 1 : 0;
  }
  std::printf("  %ld re-centrings\n", nRe);
  CHECK(nRe > 0);
  CHECK(allReached("after re-centring"));
  set.setWispEps(4e-9);
  set.setKappaMax(0.9);
  CHECK(set.curvProto.pureEps == 4e-9 && set.curvProto.kappaMax == 0.9);
  CHECK(allReached("after setWispEps/setKappaMax"));
}

// ------------------------------------------------------------------------------------------ W
void gateWrapRecentre() {
  std::printf("\n=== W  §12.3 lossless periodic-wrap re-centring (a seam-straddling marker)\n");
  for (double wisp : {0.0, 1e-8}) {
    const int NX = 20, NY = 40, NZ = 40;
    const double h = 1.0, R = 5.0, cx = 1.3, cy = 12.3, cz = 20.1, dt = 0.2;
    WyAdvector pa, ref;
    pa.init(NX, NY, NZ, h, G);
    ref.init(NX, NY, NZ, h, G);
    const I3 e = pa.extent(), n = pa.inner();
    vofscene::Block blk = vofscene::blockOf(pa, I3{0, 0, 0});
    vofscene::fillUniform(pa, blk, 0.0, 1.0, 0.0);
    vofscene::fillUniform(ref, blk, 0.0, 1.0, 0.0);
    ref.exchange = [e](SField f) { vofscene::periodicFill(f, e, G, true, true, true); };
    ref.wispEps = wisp;
    {
      SField c = ref.colour();
      vofscene::forEachExtended(
          blk, KOKKOS_LAMBDA(long i, int gx, int gy, int gz) {
            c(i) =
                peclet::flow::vof::sphereCellFraction(cx, cy, cz, R, gx * h, gy * h, gz * h, h, 4) +
                peclet::flow::vof::sphereCellFraction(cx + NX, cy, cz, R, gx * h, gy * h, gz * h, h,
                                                      4) +
                peclet::flow::vof::sphereCellFraction(cx - NX, cy, cz, R, gx * h, gy * h, gz * h, h,
                                                      4);
          });
      ref.syncGhosts();
    }
    VofBlockSet set;
    set.init(I3{NX, NY, NZ}, {true, true, true}, 0, 1, h);
    set.bubbleEps = 0.0;
    set.setWispEps(wisp);
    serialExchange(set, pa, I3{NX, NY, NZ});
    set.seedSphere(cx, cy, cz, R);
    set.scatter(pa.colour());
    const int lo0 = set.blocks()[0].box.lo[0], hi0 = set.blocks()[0].box.hi[0];
    const double v0 = set.blocks()[0].stats().volume;
    double worst = 0.0;
    long nRe = 0;
    for (long s = 0; s < 60; ++s) {
      ref.advect(dt, s);
      set.advect(dt, pa.colour());
      nRe += set.blocks()[0].stats().recentred ? 1 : 0;
      auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pa.colour());
      auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref.colour());
      for (int z = 0; z < NZ; ++z)
        for (int y = 0; y < NY; ++y)
          for (int x = 0; x < NX; ++x) {
            const long i = L3(x + G, y + G, z + G, e);
            worst = std::fmax(worst, std::fabs(hu(i) - hr(i)));
          }
    }
    const auto& st = set.blocks()[0].stats();
    const double dv = std::fabs(st.volume - v0) / v0;
    std::printf(
        "  wispEps %.0e: box x [%d,%d) -> [%d,%d); %ld re-centrings, fullAxis %ld, "
        "discarded %.3e, volume rel %.3e, max|union - global| %.3e\n",
        wisp, lo0, hi0, st.lo[0], st.hi[0], nRe, st.fullAxis, st.discarded, dv, worst);
    CHECK(lo0 < 0);  // seeded across the seam
    CHECK(st.lo[0] == 0 && st.hi[0] == NX);
    CHECK(st.fullAxis > 0);
    CHECK(st.discarded == 0.0);
    CHECK(dv <= 1e-14);
    if (wisp == 0.0)
      CHECK(worst <= 1e-14);  // lossless: the union IS the global field to round-off
    (void)n;
  }
}

// ------------------------------------------------------------------------------------------ R
void gateResidueReturn() {
  std::printf("\n=== R  §13 residue return: 1000 kinematic steps, volume exact at wispEps 1e-8\n");
  const I3 gs{48, 48, 48};
  const long steps = 1000;
  double dv[2] = {0, 0}, disc[2] = {0, 0}, res[2] = {0, 0};
  const double wisps[2] = {0.0, 1e-8};
  for (int k = 0; k < 2; ++k) {
    WyAdvector pa;
    pa.init(gs.x, gs.y, gs.z, 1.0, G);
    vofscene::Block blk = vofscene::blockOf(pa, I3{0, 0, 0});
    vofscene::fillUniform(pa, blk, 0.9, 0.6, 0.3);
    VofBlockSet set;
    set.init(gs, {true, true, true}, 0, 1, 1.0);
    set.setWispEps(wisps[k]);
    serialExchange(set, pa, gs);
    set.seedSphere(24.2, 24.1, 23.9, 8.0);
    set.scatter(pa.colour());
    const double v0 = set.blocks()[0].stats().volume;
    for (long s = 0; s < steps; ++s)
      set.advect(0.2, pa.colour());
    const auto& st = set.blocks()[0].stats();
    dv[k] = (st.volume - v0) / v0;
    disc[k] = st.discarded;
    res[k] = st.residueReturned;
    std::printf("  wispEps %.0e: volume rel change %+.3e, discarded %.3e, residueReturned %.3e\n",
                wisps[k], dv[k], disc[k], res[k]);
  }
  CHECK(std::fabs(dv[1]) <= 1e-14);
  CHECK(disc[1] == 0.0);
  CHECK(res[1] != 0.0);  // the pass ran: residue exists and was returned
  CHECK(res[0] == 0.0);  // wispEps 0 = the W0 container verbatim: the pass never runs
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("VoF block container: overlap-robustness rules (vof_overlap_design), backend %s\n",
                peclet::flow::SExec::name());
    gateClipOnBlockPath();
    gatePrototypePropagation();
    gateWrapRecentre();
    gateResidueReturn();
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
                failures == 1 ? "" : "s");
  }
  Kokkos::finalize();
  return failures ? 1 : 0;
}
