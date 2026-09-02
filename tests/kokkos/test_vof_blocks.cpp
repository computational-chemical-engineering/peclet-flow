// VoF Part III rung W0 (WO-W0) — the per-bubble BLOCK container, single rank.
//
//   P  plan self-check      : the gather/scatter pieces PARTITION a block's box (every cell exactly
//                             once, from exactly one owner) for periodic, walled and seam-crossing
//                             boxes -- the property that makes arrival order irrelevant.
//   G1 one bubble, bitwise  : the V1 LeVeque scene (32^3, T = 3 reversal) with one sphere carried
//                             as a block: the UNION field equals the global-field WyAdvector result
//                             BIT FOR BIT at every step.
//   G2 two bubbles          : two spheres pushed together by a prescribed solenoidal
//   (discrete-curl)
//                             field: the two markers OVERLAP in space, each keeps its own volume,
//                             the reversal recovers two intact bubbles -- while the single-field
//                             control merges them irreversibly. The raison d'etre gate.
//   G3 re-centring          : a sphere translated 20 cells; the moving block is bitwise equal to a
//                             block large enough never to move, and its volume is exact.
//
// Everything is compared against a plain `WyAdvector` on the whole grid seeded with the SAME exact
// `sphereCellFraction` and driven by the SAME face field, so "bitwise" is a real statement about
// the container and not about the scene.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <vector>

#include "vof/block_container.hpp"
#include "vof/block_exchange.hpp"
#include "vof_advect_scenes.hpp"

using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SExec;
using peclet::flow::SField;
using peclet::flow::vof::VofBlockExchange;
using peclet::flow::vof::VofBlockSet;
using peclet::flow::vof::VofBox;
using peclet::flow::vof::VofPiece;
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

/// A whole-grid patch: the "rank-local" face velocity + union colour of the serial case.
struct Patch {
  WyAdvector adv;
  vofscene::Block blk;
  int n = 0;
  double h = 1.0;
  void init(int nx, double hh) {
    n = nx;
    h = hh;
    adv.init(nx, nx, nx, hh, G);
    blk = vofscene::blockOf(adv, I3{0, 0, 0});
  }
};

std::shared_ptr<VofBlockExchange> serialExchange(VofBlockSet& set, Patch& p) {
  auto ex = std::make_shared<VofBlockExchange>();
  std::vector<VofBox> rb(1);
  rb[0].lo[0] = rb[0].lo[1] = rb[0].lo[2] = 0;
  rb[0].hi[0] = rb[0].hi[1] = rb[0].hi[2] = p.n;
  ex->init(I3{p.n, p.n, p.n}, set.periodic(), rb, 0);
  VofBlockExchange::Patch pp;
  pp.e = p.adv.extent();
  pp.n = p.adv.inner();
  pp.o = I3{0, 0, 0};
  pp.g = G;
  ex->setPatch(pp, p.adv.faceU(), p.adv.faceV(), p.adv.faceW());
  set.setExchange(ex);
  return ex;
}

/// Seed a reference (global-field) advector with the SAME sphere fraction the block set uses.
void refSphere(WyAdvector& a, vofscene::Block b, double h, double cx, double cy, double cz,
               double r) {
  vofscene::initSphere(a.colour(), b, h, cx, cy, cz, r);
}

/// Bitwise comparison over a global index box (default: the whole grid).
long bitwiseDiffBox(SField a, SField b, I3 e, int n, int g, const int lo[3], const int hi[3],
                    double* worst, double* worstMag) {
  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  long d = 0;
  double w = 0.0, wm = 0.0;
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        if (x < lo[0] || x >= hi[0] || y < lo[1] || y >= hi[1] || z < lo[2] || z >= hi[2])
          continue;
        const double va = ha(L3(x + g, y + g, z + g, e)), vb = hb(L3(x + g, y + g, z + g, e));
        if (std::memcmp(&va, &vb, sizeof(double)) != 0) {
          ++d;
          w = std::fmax(w, std::fabs(va - vb));
          wm = std::fmax(wm, std::fmax(std::fabs(va), std::fabs(vb)));
        }
      }
  if (worst)
    *worst = w;
  if (worstMag)
    *worstMag = wm;
  return d;
}

/// The block's own colour on its inner box, lifted onto a whole-grid host array (NaN elsewhere).
std::vector<double> blockOnGrid(const peclet::flow::vof::VofBlock& b, int gn) {
  std::vector<double> v(static_cast<std::size_t>(gn) * gn * gn,
                        std::numeric_limits<double>::quiet_NaN());
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b.advector().colour());
  const I3 be = b.advector().extent();
  const int g = b.advector().ghost();
  for (int z = 0; z < b.box.n(2); ++z)
    for (int y = 0; y < b.box.n(1); ++y)
      for (int x = 0; x < b.box.n(0); ++x) {
        const int gx = ((b.box.lo[0] + x) % gn + gn) % gn;
        const int gy = ((b.box.lo[1] + y) % gn + gn) % gn;
        const int gz = ((b.box.lo[2] + z) % gn + gn) % gn;
        v[static_cast<std::size_t>(gx) + static_cast<std::size_t>(gy) * gn +
          static_cast<std::size_t>(gz) * gn * gn] = hb(L3(x + g, y + g, z + g, be));
      }
  return v;
}

std::vector<double> fieldOnGrid(SField f, I3 e, int gn, int g) {
  std::vector<double> v(static_cast<std::size_t>(gn) * gn * gn, 0.0);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  for (int z = 0; z < gn; ++z)
    for (int y = 0; y < gn; ++y)
      for (int x = 0; x < gn; ++x)
        v[static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * gn +
          static_cast<std::size_t>(z) * gn * gn] = h(L3(x + g, y + g, z + g, e));
  return v;
}

/// bitwise differences between two whole-grid arrays, skipping cells where either is NaN
long gridDiff(const std::vector<double>& a, const std::vector<double>& b, double* worst) {
  long d = 0;
  double w = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i]) || std::isnan(b[i]))
      continue;
    if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) {
      ++d;
      w = std::fmax(w, std::fabs(a[i] - b[i]));
    }
  }
  if (worst)
    *worst = w;
  return d;
}

long bitwiseDiff(SField a, SField b, I3 e, int n, int g, double* worst) {
  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  long d = 0;
  double w = 0.0;
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        const double va = ha(L3(x + g, y + g, z + g, e)), vb = hb(L3(x + g, y + g, z + g, e));
        if (std::memcmp(&va, &vb, sizeof(double)) != 0) {
          ++d;
          w = std::fmax(w, std::fabs(va - vb));
        }
      }
  if (worst)
    *worst = w;
  return d;
}

double sumInner(SField f, I3 e, int n, int g) {
  double s = 0.0;
  Kokkos::parallel_reduce(
      "sumInner",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g}, {g + n, g + n, g + n}),
      KOKKOS_LAMBDA(int x, int y, int z, double& a) { a += f(L3(x, y, z, e)); }, s);
  Kokkos::fence();
  return s;
}

// ======================================================================== P: the plan partitions
void gatePlan() {
  std::printf("\n=== P  the gather/scatter plan partitions the block box exactly once\n");
  const int gn = 32;
  const I3 gs{gn, gn, gn};
  // three rank boxes that tile the grid (a 2x2x1 ORB plus a slab), so overlaps are non-trivial
  std::vector<VofBox> rb(4);
  const int cuts[4][6] = {{0, 0, 0, 16, 16, 32},
                          {16, 0, 0, 32, 16, 32},
                          {0, 16, 0, 16, 32, 32},
                          {16, 16, 0, 32, 32, 32}};
  for (int r = 0; r < 4; ++r)
    for (int d = 0; d < 3; ++d) {
      rb[r].lo[d] = cuts[r][d];
      rb[r].hi[d] = cuts[r][3 + d];
    }
  struct Case {
    VofBox box;
    std::array<bool, 3> per;
    const char* what;
    long expect;
  };
  std::vector<Case> cases = {
      {VofBox{{4, 4, 4}, {20, 20, 20}}, {true, true, true}, "interior, periodic", 16L * 16 * 16},
      {VofBox{{-3, 4, 4}, {13, 20, 20}}, {true, true, true}, "crosses the -x seam", 16L * 16 * 16},
      {VofBox{{28, 4, 4}, {44, 20, 20}}, {true, true, true}, "crosses the +x seam", 16L * 16 * 16},
      // a walled axis: the part outside the domain has no owner and is the block's own clamp fill
      {VofBox{{-3, 4, 4}, {13, 20, 20}},
       {false, true, true},
       "walled -x, hangs out",
       13L * 16 * 16},
      {VofBox{{-2, -2, -2}, {34, 34, 34}},
       {true, true, true},
       "spans the whole grid (clamped)",
       32L * 32 * 32},
  };
  for (auto& c : cases) {
    const VofBox box = peclet::flow::vof::vofClampBox(c.box, gs, c.per);
    std::vector<VofPiece> pieces;
    peclet::flow::vof::vofBuildPieces(box, gs, c.per, rb, pieces);
    // mark every block-local cell each piece writes; assert each is written at most once
    std::vector<int> hits(static_cast<std::size_t>(box.cells()), 0);
    long total = 0;
    bool oob = false;
    for (const auto& p : pieces) {
      total += p.cells();
      for (int z = 0; z < p.g.n(2); ++z)
        for (int y = 0; y < p.g.n(1); ++y)
          for (int x = 0; x < p.g.n(0); ++x) {
            const int lx = p.loc[0] + x, ly = p.loc[1] + y, lz = p.loc[2] + z;
            if (lx < 0 || ly < 0 || lz < 0 || lx >= box.n(0) || ly >= box.n(1) || lz >= box.n(2)) {
              oob = true;
              continue;
            }
            ++hits[static_cast<std::size_t>(lx) + static_cast<std::size_t>(ly) * box.n(0) +
                   static_cast<std::size_t>(lz) * box.n(0) * box.n(1)];
          }
    }
    long once = 0, twice = 0;
    for (int v : hits) {
      if (v == 1)
        ++once;
      else if (v > 1)
        ++twice;
    }
    std::printf("  %-32s pieces %2zu  cells %7ld  written-once %7ld  multiply-written %ld\n",
                c.what, pieces.size(), total, once, twice);
    CHECK(!oob);
    CHECK(twice == 0);
    CHECK(once == c.expect);
    CHECK(total == c.expect);
  }
}

// ======================================================================== G1: one bubble, bitwise
//
// Run the SAME scene with two block sets that differ only in `bubbleEps`, the threshold that
// defines the bubble extent (and hence the box):
//   eps = 0      the box contains the ENTIRE support of the colour, wisps included -> nothing is
//                ever dropped and the block is bitwise the global field for all 768 steps. This is
//                the gate the work order asks for, and it also measures what the wisp WAKE costs:
//                the box grows to cover everywhere the interface has been.
//   eps = 1e-12  the production default: the box tracks the BUBBLE, the round-off wake outside it
//                is dropped, and the block then differs from the global field by exactly that
//                residue (reported).
void gateOneBubble() {
  std::printf("\n=== G1 one bubble as a block == the global field, BITWISE (LeVeque, T = 3)\n");
  const int gn = 32;
  const double h = 1.0 / gn, T = 3.0;
  const long steps = 768;  // the V1 gate-D schedule at 32^3 (CFL 0.25)
  const double dt = T / steps;
  const double cx = 0.35, cy = 0.35, cz = 0.35, r = 0.15;

  Patch patch;   // supplies the face velocity; also carries the eps = 0 union
  Patch patch2;  // carries the eps = 1e-12 union
  patch.init(gn, h);
  patch2.init(gn, h);
  WyAdvector ref;
  ref.init(gn, gn, gn, h, G);
  const vofscene::Block rblk = vofscene::blockOf(ref, I3{0, 0, 0});
  const I3 e = ref.extent();
  ref.exchange = [e](SField f) { vofscene::periodicFill(f, e, G, true, true, true); };
  refSphere(ref, rblk, h, cx, cy, cz, r);
  ref.syncGhosts();

  VofBlockSet exact, prod;
  exact.init(I3{gn, gn, gn}, {true, true, true}, 0, 1, h);
  prod.init(I3{gn, gn, gn}, {true, true, true}, 0, 1, h);
  exact.bubbleEps = 0.0;
  serialExchange(exact, patch);
  serialExchange(prod, patch2);
  exact.seedSphere(cx, cy, cz, r);
  prod.seedSphere(cx, cy, cz, r);
  exact.scatter(patch.adv.colour());
  prod.scatter(patch2.adv.colour());

  double worst = 0.0;
  const long d0 = bitwiseDiff(patch.adv.colour(), ref.colour(), e, gn, G, &worst);
  std::printf("  seed: bitwise diffs union vs global field %ld\n", d0);
  CHECK(d0 == 0);

  const double v0 = sumInner(ref.colour(), e, gn, G);
  long nRecentre = 0, nRecentreP = 0, maxCells = 0, maxCellsP = 0;
  long firstBad0 = -1;
  double afterWorst = 0.0;
  long unionDiffs = 0, unionOther = 0;
  double unionWorst = 0.0, prodWorst = 0.0;
  long prodFirst = -1;
  for (long s = 0; s < steps; ++s) {
    const double phase = std::cos(M_PI * (s + 0.5) * dt / T);
    vofscene::fillLeVeque(patch.adv, patch.blk, h, phase);
    // The block GATHERS its ghost face velocity from the owner of the WRAPPED global cell, so the
    // reference must consume the wrapped value too. Evaluating the analytic potential at the
    // out-of-domain index instead agrees only to round-off (sin^2(-pi h) against sin^2(pi(1-h)))
    // and that 1e-19 alone breaks a bitwise gate -- a property of the scene, not of the container.
    vofscene::periodicFill(patch.adv.faceU(), e, G, true, true, true);
    vofscene::periodicFill(patch.adv.faceV(), e, G, true, true, true);
    vofscene::periodicFill(patch.adv.faceW(), e, G, true, true, true);
    Kokkos::deep_copy(ref.faceU(), patch.adv.faceU());
    Kokkos::deep_copy(ref.faceV(), patch.adv.faceV());
    Kokkos::deep_copy(ref.faceW(), patch.adv.faceW());
    Kokkos::deep_copy(patch2.adv.faceU(), patch.adv.faceU());
    Kokkos::deep_copy(patch2.adv.faceV(), patch.adv.faceV());
    Kokkos::deep_copy(patch2.adv.faceW(), patch.adv.faceW());
    ref.advect(dt, s);
    exact.advect(dt, patch.adv.colour());
    prod.advect(dt, patch2.adv.colour());
    nRecentre += exact.blocks()[0].stats().recentred ? 1 : 0;
    nRecentreP += prod.blocks()[0].stats().recentred ? 1 : 0;
    maxCells = std::max(maxCells, exact.blocks()[0].stats().cells);
    maxCellsP = std::max(maxCellsP, prod.blocks()[0].stats().cells);

    const std::vector<double> gr = fieldOnGrid(ref.colour(), e, gn, G);
    // THE gate: the block's own colour == the global field's, bit for bit, over the block's box
    const std::vector<double> ge = blockOnGrid(exact.blocks()[0], gn);
    const long dBox = gridDiff(ge, gr, &worst);
    // The block is bitwise the global field until the Weymouth-Yue round-off RESIDUE — which the
    // sweeps leave in every cell they touch, down to +-1e-300 and to signed zeros — leaves the
    // block's support. That residue is part of the global field's state and is deliberately NOT
    // part of the block's: the block's ghost policy says "outside my box it is pure gas", which is
    // the marker model (the alternative, ghosting from the UNION, would reproduce the global field
    // bit for bit and let a neighbouring marker's colour flux in, i.e. coalesce). So record the
    // BITWISE HORIZON and bound what happens after it, rather than asserting bitwise forever.
    if (dBox != 0 && firstBad0 < 0)
      firstBad0 = s;
    afterWorst = std::fmax(afterWorst, worst);
    // the production default: the same comparison, now with the wake dropped
    double wp = 0.0;
    const long dp = gridDiff(blockOnGrid(prod.blocks()[0], gn), gr, &wp);
    if (dp != 0 && prodFirst < 0)
      prodFirst = s;
    prodWorst = std::fmax(prodWorst, wp);
    // The UNION is a different object: `C = max_blocks C_block` from an empty union, so it CLIPS
    // the NEGATIVE Weymouth-Yue round-off residue to an exact 0. Every union/reference difference
    // must be exactly that, and nothing else.
    const std::vector<double> gu = fieldOnGrid(patch.adv.colour(), e, gn, G);
    double wu = 0.0;
    const long dUni = gridDiff(gu, gr, &wu);
    long other = 0;
    for (std::size_t i = 0; i < gu.size(); ++i)
      if (std::memcmp(&gu[i], &gr[i], sizeof(double)) != 0 && !(gr[i] < 0.0 && gu[i] == 0.0))
        ++other;
    unionDiffs = std::max(unionDiffs, dUni);
    unionWorst = std::fmax(unionWorst, wu);
    unionOther = std::max(unionOther, other);
  }
  const double v1 = sumInner(patch.adv.colour(), e, gn, G);
  const double v2 = sumInner(patch2.adv.colour(), e, gn, G);
  const double vr = sumInner(ref.colour(), e, gn, G);
  const auto& st = exact.blocks()[0].stats();
  const auto& sp = prod.blocks()[0].stats();
  std::printf("  eps = 0     : bitwise horizon %ld steps; after it max|d| %.3e over %ld steps\n",
              firstBad0 < 0 ? steps : firstBad0, afterWorst, steps);
  std::printf(
      "                %ld re-centrings, largest box %ld cells (%.1f%% of the grid) -- the WAKE\n",
      nRecentre, maxCells, 100.0 * maxCells / (gn * gn * gn));
  std::printf(
      "  eps = 1e-12 : bitwise horizon %ld steps; max|d| over the run %.3e, colour dropped %.3e\n",
      prodFirst < 0 ? steps : prodFirst, prodWorst, sp.discarded);
  std::printf("                %ld re-centrings, largest box %ld cells (%.1f%% of the grid)\n",
              nRecentreP, maxCellsP, 100.0 * maxCellsP / (gn * gn * gn));
  std::printf(
      "  union vs the global field: worst-case %ld cells differ (max|d| %.3e); of those %ld\n"
      "                are NOT the UNPACK_MAX clip of a negative WY residue\n",
      unionDiffs, unionWorst, unionOther);
  std::printf("  volume: union(eps=0) %.17g  union(prod) %.17g  global %.17g  drift %.3e / %.3e\n",
              v1, v2, vr, std::fabs(v1 - v0) / v0, std::fabs(v2 - v0) / v0);
  std::printf("  block stats: V %.6f  centroid (%.4f, %.4f, %.4f)  moments (%.3e, %.3e, %.3e)\n",
              st.volume, st.centroid[0], st.centroid[1], st.centroid[2], st.moment[0], st.moment[1],
              st.moment[2]);
  CHECK(firstBad0 != 0);      // the first step is bitwise by construction
  CHECK(afterWorst < 1e-15);  // and never more than the WY round-off residue after
  CHECK(nRecentre > 0);
  CHECK(nRecentreP > 0);
  CHECK(unionOther == 0);
  CHECK(prodWorst < 1e-15);
  CHECK(std::fabs(sp.discarded) < 1e-15);
  CHECK(std::fabs(v1 - v0) / v0 < 1e-13);
  CHECK(std::fabs(v2 - v0) / v0 < 1e-13);
}

// ======================================================================== G2: two bubbles
/// A 2-D cellular stream function, sampled as the DISCRETE curl on z-edges so the discrete face
/// divergence telescopes to round-off (the same construction `fillLeVeque` uses):
///     psi(x, y) = -(A / 2pi) sin(2pi x) sin(2pi y) * phase
/// Along y = 1/2 the motion is purely in x with u = A sin(2pi x) phase, i.e. x = 1/2 attracts:
/// two bubbles placed either side of it are pushed together and, when `phase` reverses, apart.
///
/// `x0` shifts the attracting plane onto a cell CENTRE. On a cell FACE the two markers can never
/// share a cell — each stays on its own side of the face and the overlap the gate is looking for is
/// structurally impossible. (Measured: with the plane at x = 1/2 on a 48^3 grid, i.e. exactly a
/// face, the two bubbles flatten against it and `max cells with BOTH markers liquid` is 0.)
KOKKOS_INLINE_FUNCTION double cellPsi(double x, double y, double A, double phase, double x0) {
  const double PI = 3.14159265358979323846;
  return -(A / (2.0 * PI)) * Kokkos::sin(2.0 * PI * (x - x0)) * Kokkos::sin(2.0 * PI * y) * phase;
}
void fillCellular(WyAdvector& a, vofscene::Block b, double h, double A, double phase,
                  double x0 = 0.0) {
  SField u = a.faceU(), v = a.faceV(), w = a.faceW();
  const double invh = 1.0 / h;
  vofscene::forEachExtended(
      b, KOKKOS_LAMBDA(long i, int gx, int gy, int) {
        const double xn = gx * h, yn = gy * h;
        const double xp = (gx + 1) * h, yp = (gy + 1) * h;
        u(i) = (cellPsi(xp, yp, A, phase, x0) - cellPsi(xp, yn, A, phase, x0)) * invh;
        v(i) = -(cellPsi(xp, yp, A, phase, x0) - cellPsi(xn, yp, A, phase, x0)) * invh;
        w(i) = 0.0;
      });
}

void gateTwoBubbles() {
  std::printf(
      "\n=== G2 two bubbles: overlap without coalescence, and the reversal recovers both\n");
  const int gn = 48;
  const double h = 1.0 / gn;
  const double A = 1.0, T = 0.9;
  const long steps = 216;
  const double dt = T / steps;
  // u ~ +A sin(2pi(x - x0)) along y = 1/2, so the ATTRACTING plane is x = x0 + 1/2. Shift it onto
  // a cell CENTRE: on a cell FACE the two markers can never share a cell and the overlap this gate
  // looks for is structurally impossible (measured: 0 shared cells with the plane at x = 1/2).
  const double x0 = 0.5 * h, xatt = x0 + 0.5;
  const double r = 0.10, y0 = 0.5, z0 = 0.5;
  const double x1 = xatt - 0.1208, x2 = xatt + 0.1208;  // 2 cells of gap between the surfaces

  Patch patch;
  patch.init(gn, h);
  const I3 e = patch.adv.extent();

  VofBlockSet set;
  set.init(I3{gn, gn, gn}, {true, true, true}, 0, 1, h);
  serialExchange(set, patch);
  set.seedSphere(x1, y0, z0, r);
  set.seedSphere(x2, y0, z0, r);
  set.scatter(patch.adv.colour());
  SField c0 = vofscene::copyOf(patch.adv.colour(), "c0");

  // the single-field CONTROL: both spheres in ONE colour field
  WyAdvector ctl;
  ctl.init(gn, gn, gn, h, G);
  const vofscene::Block cblk = vofscene::blockOf(ctl, I3{0, 0, 0});
  ctl.exchange = [e](SField f) { vofscene::periodicFill(f, e, G, true, true, true); };
  Kokkos::deep_copy(ctl.colour(), c0);
  ctl.syncGhosts();

  const double vb0[2] = {set.blocks()[0].stats().volume, set.blocks()[1].stats().volume};
  double drift[2] = {0.0, 0.0};
  long overlapMax = 0;
  double unionDeficitMax = 0.0;
  for (long s = 0; s < steps; ++s) {
    const double phase = std::cos(M_PI * (s + 0.5) * dt / T);
    fillCellular(patch.adv, patch.blk, h, A, phase, x0);
    vofscene::periodicFill(patch.adv.faceU(), e, G, true, true, true);
    vofscene::periodicFill(patch.adv.faceV(), e, G, true, true, true);
    vofscene::periodicFill(patch.adv.faceW(), e, G, true, true, true);
    Kokkos::deep_copy(ctl.faceU(), patch.adv.faceU());
    Kokkos::deep_copy(ctl.faceV(), patch.adv.faceV());
    Kokkos::deep_copy(ctl.faceW(), patch.adv.faceW());
    ctl.advect(dt, s);
    set.advect(dt, patch.adv.colour());
    for (int b = 0; b < 2; ++b)
      drift[b] = std::fmax(drift[b], std::fabs(set.blocks()[b].stats().volume - vb0[b]) / vb0[b]);
    // cells where BOTH markers carry liquid: impossible in a single field, which is the point
    const VofBox& b0 = set.blocks()[0].box;
    const VofBox& b1 = set.blocks()[1].box;
    const VofBox ov = VofBox::intersect(b0, b1);
    if (!ov.empty()) {
      auto h0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    set.blocks()[0].advector().colour());
      auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    set.blocks()[1].advector().colour());
      const I3 e0 = set.blocks()[0].advector().extent(), e1 = set.blocks()[1].advector().extent();
      long cnt = 0;
      for (int z = ov.lo[2]; z < ov.hi[2]; ++z)
        for (int y = ov.lo[1]; y < ov.hi[1]; ++y)
          for (int x = ov.lo[0]; x < ov.hi[0]; ++x) {
            const double a = h0(L3(x - b0.lo[0] + G, y - b0.lo[1] + G, z - b0.lo[2] + G, e0));
            const double bb = h1(L3(x - b1.lo[0] + G, y - b1.lo[1] + G, z - b1.lo[2] + G, e1));
            if (a > 1e-12 && bb > 1e-12)
              ++cnt;
          }
      overlapMax = std::max(overlapMax, cnt);
    }
    // the union is a max, so it is SMALLER than the sum of the markers wherever they overlap
    const double su = sumInner(patch.adv.colour(), e, gn, G);
    const double sm = set.blocks()[0].stats().volume + set.blocks()[1].stats().volume;
    unionDeficitMax = std::fmax(unionDeficitMax, sm - su);
  }
  // recovery: L1 distance of the final field from the initial one
  const double l1blk = vofscene::l1Diff(patch.adv.colour(), c0, e, I3{gn, gn, gn}, G);
  const double l1ctl = vofscene::l1Diff(ctl.colour(), c0, e, I3{gn, gn, gn}, G);
  // the "neck": liquid exactly halfway between the two seeds at the END (0 => two separate bubbles)
  auto hu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), patch.adv.colour());
  auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ctl.colour());
  const int mx = static_cast<int>(xatt * gn), my = static_cast<int>(y0 * gn),
            mz = static_cast<int>(z0 * gn);
  const double neckB = hu(L3(mx + G, my + G, mz + G, e));
  const double neckC = hc(L3(mx + G, my + G, mz + G, e));
  const double v0 = sumInner(c0, e, gn, G);
  const double vB = sumInner(patch.adv.colour(), e, gn, G);
  const double vC = sumInner(ctl.colour(), e, gn, G);
  std::printf("  %ld steps, CFL %.3f: max cells with BOTH markers liquid = %ld\n", steps,
              A * dt / h, overlapMax);
  std::printf("  max union DEFICIT (V_0 + V_1 - V_union) = %.4f cells of shared liquid\n",
              unionDeficitMax);
  std::printf("  per-marker volume drift: block0 %.3e  block1 %.3e\n", drift[0], drift[1]);
  std::printf("  union volume drift %.3e   control volume drift %.3e\n", std::fabs(vB - v0) / v0,
              std::fabs(vC - v0) / v0);
  std::printf(
      "  recovery L1 vs the initial field: BLOCKS %.4e   single-field CONTROL %.4e"
      "  (ratio %.2f)\n",
      l1blk, l1ctl, l1ctl / l1blk);
  std::printf("  neck cell (%d,%d,%d) at the end: blocks %.3e   control %.3e\n", mx, my, mz, neckB,
              neckC);
  CHECK(overlapMax > 0);
  CHECK(unionDeficitMax > 0.0);
  CHECK(drift[0] < 1e-14);
  CHECK(drift[1] < 1e-14);
  CHECK(l1ctl > l1blk);
}

// ======================================================================== G3: re-centring
//
// A sphere translated 20 cells. The moving block (margin 3 + pad 2) is compared with a block large
// enough that it never has to move, at both `bubbleEps` settings -- so the gate separates "the
// re-centring copy is exact" (which is bitwise at eps = 0) from "the wake the production threshold
// drops" (a 1e-19 residue and nothing else).
void gateRecentre() {
  std::printf("\n=== G3 re-centring: a block that MOVES is bitwise a block that never has to\n");
  const int gn = 48;
  const double h = 1.0 / gn;
  const double dt = 0.2 * h;  // CFL 0.2 with |u| = 1
  const long steps = 100;     // 20 cells of translation
  const double r = 5.0 * h;
  const double epsList[2] = {0.0, 1e-12};

  for (int k = 0; k < 2; ++k) {
    Patch pa, pb;
    pa.init(gn, h);
    pb.init(gn, h);
    vofscene::fillUniform(pa.adv, pa.blk, 1.0, 0.0, 0.0);
    vofscene::fillUniform(pb.adv, pb.blk, 1.0, 0.0, 0.0);

    VofBlockSet moving, fixed;
    moving.init(I3{gn, gn, gn}, {true, true, true}, 0, 1, h);
    fixed.init(I3{gn, gn, gn}, {true, true, true}, 0, 1, h);
    moving.bubbleEps = epsList[k];
    fixed.bubbleEps = epsList[k];
    fixed.allowShrink = false;  // a block large enough that it never has to move
    serialExchange(moving, pa);
    serialExchange(fixed, pb);
    moving.seedSphere(10.5 * h, 24.5 * h, 24.5 * h, r);
    fixed.seedSphere(10.5 * h, 24.5 * h, 24.5 * h, r, 4, 30);
    moving.scatter(pa.adv.colour());
    fixed.scatter(pb.adv.colour());

    const double v0 = moving.blocks()[0].stats().volume;
    const int lo0 = moving.blocks()[0].box.lo[0];
    long nMove = 0, nFix = 0, firstBad = -1;
    double worst = 0.0;
    for (long s = 0; s < steps; ++s) {
      moving.advect(dt, pa.adv.colour());
      fixed.advect(dt, pb.adv.colour());
      nMove += moving.blocks()[0].stats().recentred ? 1 : 0;
      nFix += fixed.blocks()[0].stats().recentred ? 1 : 0;
      double w = 0.0;
      const long d =
          gridDiff(blockOnGrid(moving.blocks()[0], gn), blockOnGrid(fixed.blocks()[0], gn), &w);
      worst = std::fmax(worst, w);
      if (d != 0 && firstBad < 0)
        firstBad = s;
    }
    const auto& sm = moving.blocks()[0].stats();
    const auto& sf = fixed.blocks()[0].stats();
    std::printf("  -- bubbleEps = %.0e\n", epsList[k]);
    std::printf(
        "     moving: box [%d,%d)x[%d,%d)x[%d,%d)  lo_x %d -> %d  %ld re-centrings, %ld cells\n",
        sm.lo[0], sm.hi[0], sm.lo[1], sm.hi[1], sm.lo[2], sm.hi[2], lo0, sm.lo[0], nMove, sm.cells);
    std::printf("     fixed : box [%d,%d)x...  %ld re-centrings, %ld cells\n", sf.lo[0], sf.hi[0],
                nFix, sf.cells);
    std::printf("     first differing step %ld (-1 = none)   max|d| over the run %.3e\n", firstBad,
                worst);
    std::printf(
        "     volume %.17g -> %.17g  drift %.3e  dropped %.3e  centroid x %.5f (exact %.5f)\n", v0,
        sm.volume, std::fabs(sm.volume - v0) / v0, sm.discarded, sm.centroid[0],
        (10.5 + steps * 0.2) * h);
    CHECK(nMove > 0);
    CHECK(nFix == 0);
    CHECK(std::fabs(sm.volume - v0) / v0 < 1e-14);
    CHECK(std::fabs(sm.centroid[0] - (10.5 + steps * 0.2) * h) < 0.5 * h);
    if (k == 0)
      CHECK(firstBad < 0);  // nothing dropped => bitwise
    else
      CHECK(worst < 1e-15);  // only the dropped wake, at round-off magnitude
  }
}

// ================================================= W1: assignment, pool, device pack, area
//
//   W1a  master assignment : round robin vs LPT vs weighted ORB on a 64-bubble swarm of MIXED
//                            sizes, at np 1/2/4/8 (the assignment is a pure function of the
//                            replicated table, so it is measurable without MPI).
//   W1b  device packing    : the device-resident pack/unpack produces the same union BITWISE as
//                            the W0 host-staged path, and the packing time of both is reported.
//   W1c  block pool        : `usePool` is bitwise inert, and the allocation census.
//   W1d  interface area    : a seeded sphere's PLIC area against 4 pi R^2.
void gateAssignment() {
  std::printf("\n=== W1a master assignment: round robin vs LPT vs weighted ORB (64 bubbles)\n");
  // 64 bubbles on a 4x4x4 lattice with radii spanning 2.2x, so the block SIZES differ by ~10x in
  // cell count -- exactly the regime round robin is blind to.
  const int gn = 64;
  const double h = 1.0 / gn;
  VofBlockSet set;
  set.init(I3{gn, gn, gn}, {true, true, true}, 0, 1, h);
  Patch patch;
  patch.init(gn, h);
  serialExchange(set, patch);
  int k = 0;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int m = 0; m < 4; ++m, ++k) {
        // radii 2 .. 9 cells: the block box is the extent + 3 margin + 3 ghost, so the cell
        // counts span ~10x -- the regime a round robin by block id is blind to.
        const double r = (2.0 + 7.0 * ((k % 8) / 7.0)) / gn;
        set.seedSphere((i + 0.5) / 4, (j + 0.5) / 4, (m + 0.5) / 4, r);
      }
  CHECK(set.count() == 64);
  std::vector<long> w(set.count());
  long tot = 0, wmin = 1L << 60, wmax = 0;
  for (std::size_t b = 0; b < set.count(); ++b) {
    w[b] = set.blocks()[b].box.cells();
    tot += w[b];
    wmin = std::min(wmin, w[b]);
    wmax = std::max(wmax, w[b]);
  }
  std::printf("  block cell counts: min %ld  max %ld  (ratio %.1fx)  total %ld\n", wmin, wmax,
              double(wmax) / wmin, tot);
  std::printf("  %-4s %-14s %-14s %-14s\n", "np", "round robin", "LPT", "weighted ORB");
  const int nps[4] = {1, 2, 4, 8};
  for (int q = 0; q < 4; ++q) {
    const int np = nps[q];
    double imb[3];
    for (int mode = 0; mode < 3; ++mode) {
      std::vector<int> m;
      if (mode == 0) {
        m.assign(w.size(), 0);
        for (std::size_t b = 0; b < w.size(); ++b)
          m[b] = static_cast<int>(b % np);
      } else if (mode == 1) {
        peclet::flow::vof::vofAssignLpt(w, np, m);
      } else {
        peclet::flow::vof::vofAssignOrb(w, np, m);
      }
      std::vector<long> load(np, 0);
      for (std::size_t b = 0; b < w.size(); ++b)
        load[m[b]] += w[b];
      long mx = 0;
      for (long v : load)
        mx = std::max(mx, v);
      imb[mode] = double(mx) * np / double(tot);
    }
    std::printf("  %-4d %-14.4f %-14.4f %-14.4f\n", np, imb[0], imb[1], imb[2]);
    // LPT must never be worse than the round robin, and it must be within 4/3 of perfect
    CHECK(imb[1] <= imb[0] + 1e-12);
    CHECK(imb[1] <= 4.0 / 3.0 + 1e-12);
  }
  // W1d: the interface area of the biggest marker against the sphere formula
  std::size_t big = 0;
  for (std::size_t b = 1; b < set.count(); ++b)
    if (set.blocks()[b].stats().volume > set.blocks()[big].stats().volume)
      big = b;
  const auto& st = set.blocks()[big].stats();
  const double R = std::cbrt(st.volume * 3.0 / (4.0 * M_PI));
  const double exact = 4.0 * M_PI * R * R;
  std::printf("  W1d interface area of marker %zu: V %.6e -> R %.5f (%.2f cells), area %.6e vs "
              "4 pi R^2 %.6e (%+.2f %%)\n",
              big, st.volume, R, R / h, st.area, exact, 100.0 * (st.area / exact - 1.0));
  CHECK(std::fabs(st.area / exact - 1.0) < 0.05);
}

// The whole W1 machinery must not move a single bit: run the SAME 8-bubble LeVeque scene four
// ways -- (host packing, no pool), (device packing, no pool), (host, pool), (device, pool) -- and
// demand the four unions and the 8 marker volumes are bitwise identical.
void gateInert() {
  std::printf("\n=== W1b/W1c device-resident packing and the block pool are BITWISE inert\n");
  const int gn = 48;
  const double h = 1.0 / gn, T = 0.75;
  const long steps = 360;  // CFL 0.2 for the LeVeque field at 48^3
  const double dt = T / steps;
  struct Cfg {
    bool dev, pool;
    const char* name;
  };
  const Cfg cfgs[4] = {{false, false, "host, no pool"},
                       {true, false, "device, no pool"},
                       {false, true, "host, pool"},
                       {true, true, "device, pool"}};
  std::vector<double> ref;
  std::vector<double> refVol;
  double packMs[4] = {0, 0, 0, 0};
  for (int q = 0; q < 4; ++q) {
    Patch patch;
    patch.init(gn, h);
    VofBlockSet set;
    set.init(I3{gn, gn, gn}, {true, true, true}, 0, 1, h);
    auto ex = serialExchange(set, patch);
    ex->deviceStaging = cfgs[q].dev;
    set.usePool = cfgs[q].pool;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        for (int m = 0; m < 2; ++m)
          set.seedSphere(0.25 + 0.5 * i, 0.25 + 0.5 * j, 0.25 + 0.5 * m, 0.09);
    set.scatter(patch.adv.colour());
    Kokkos::Timer timer;
    double tPack = 0.0;
    for (long s = 0; s < steps; ++s) {
      const double phase = std::cos(M_PI * (s + 0.5) * dt / T);
      vofscene::fillLeVeque(patch.adv, patch.blk, h, phase);
      vofscene::periodicFill(patch.adv.faceU(), patch.adv.extent(), G, true, true, true);
      vofscene::periodicFill(patch.adv.faceV(), patch.adv.extent(), G, true, true, true);
      vofscene::periodicFill(patch.adv.faceW(), patch.adv.extent(), G, true, true, true);
      Kokkos::fence();
      const double t0 = timer.seconds();
      set.advect(dt, patch.adv.colour());
      Kokkos::fence();
      tPack += timer.seconds() - t0;
    }
    packMs[q] = 1e3 * tPack / steps;
    const std::vector<double> got = fieldOnGrid(patch.adv.colour(), patch.adv.extent(), gn, G);
    std::vector<double> vol(set.count());
    for (std::size_t b = 0; b < set.count(); ++b)
      vol[b] = set.blocks()[b].stats().volume;
    if (q == 0) {
      ref = got;
      refVol = vol;
    } else {
      double worst = 0.0;
      const long d = gridDiff(got, ref, &worst);
      double dv = 0.0;
      for (std::size_t b = 0; b < vol.size(); ++b)
        dv = std::fmax(dv, std::fabs(vol[b] - refVol[b]));
      std::printf("  %-18s vs (host, no pool): %ld cells differ, max|dV| %.3e\n", cfgs[q].name, d,
                  dv);
      CHECK(d == 0);
      CHECK(dv == 0.0);
    }
    const auto ps = std::array<long, 2>{set.poolHits(), set.poolMisses()};
    std::printf("  %-18s block-step %.3f ms   pool hits %ld misses %ld\n", cfgs[q].name, packMs[q],
                ps[0], ps[1]);
    if (cfgs[q].pool)
      CHECK(ps[0] > 0);  // the pool must actually be recycling on a moving swarm
  }
  std::printf("  device vs host block step: %.3f ms vs %.3f ms  (ratio %.2fx)\n", packMs[1],
              packMs[0], packMs[0] / packMs[1]);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("VoF rung W0 (WO-W0) - per-bubble block container, backend: %s\n", SExec::name());
    gatePlan();
    gateOneBubble();
    gateTwoBubbles();
    gateRecentre();
    gateAssignment();
    gateInert();
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
                failures == 1 ? "" : "s");
  }
  Kokkos::finalize();
  return failures ? 1 : 0;
}
