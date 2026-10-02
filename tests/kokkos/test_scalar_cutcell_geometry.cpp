// Cut-cell scalar geometry record (doc/scalar_ibm_design.md §2, §3; WO-2), single rank.
//
//   ghosts  — §2.5 prerequisite: sdf_ carries BOTH ghost layers on every single-rank path — the
//             periodic wrap, and extendSdfDomainGhosts' constant extension (wall) and mirror
//             ('slip'). Fails if layer 2 is stale on any of them.
//   G-geom (e) spheres R/h in {8, 16, 32}, 3 random offsets: |V_sphere / V_exact - 1| <= 2 (h/R)^2
//             and |sum |A_phi| / 4 pi R^2 - 1| <= 3 (h/R)^2, each with order >= 1.8 (RMS over the
//             offsets, between successive resolutions); every cut cell `single` with rho >= 0.9;
//             sealed volume <= 1e-6 of the fluid volume (ruling 2026-10-03; the count is
//             reported); R0 = 100 %.
//   G-geom (f) |A^snap - areaPL| <= 2e-3 max A per cell.
//   G-geom (g) anisotropic h' = (1, 1, 2): (e) with h = max h', R0 >= 99.9 % and no R2.
//   pipe    — G7's pipe (fluid inside, z periodic, nz = 4), R/h in {16, 32, 64}: R0 = 100 %, 0
//   sealed.
//
// The sphere is SOLID (the G1 configuration), so the measured volume is sum (1 - kappa) V over
// the inner cells. Per-cell PL records (rho, kind, areaPL) are recomputed on the host from a
// mirror of sdf_ through the SAME sample assembly (scg::cellSamples) and core kernel, and the
// device record's kappa must equal them bitwise.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "flow_ibm.hpp"

namespace {
int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

namespace cs = peclet::core::scheme;
namespace scg = peclet::flow::scg;
using IbmSolver = peclet::flow::IbmSolver;
constexpr int G = IbmSolver::G;

// ---------------------------------------------------------------------------------------------
// ghosts: expected global index of a ghost sample beyond the domain on one axis
// ---------------------------------------------------------------------------------------------
int expectIndex(int i, int n, int bcLo, int bcHi) {
  if (i >= 0 && i < n)
    return i;
  const int t = i < 0 ? bcLo : bcHi;
  if (t == 0)
    return ((i % n) + n) % n;  // periodic
  if (t == 4)
    return i < 0 ? -1 - i : 2 * n - 1 - i;  // mirror about the face
  return i < 0 ? 0 : n - 1;                 // constant normal extension
}

// sdf_ ghost layers 1 and 2 on every face slab (transverse inner range) against the expected
// global sample. `bc` per face (0 periodic, 1 wall, 4 slip).
void checkGhosts(const char* label, const int bc[6]) {
  const int nx = 12, ny = 10, nz = 8;
  auto f = [&](int x, int y, int z) { return 1000.0 * x + 37.0 * y + 1.25 * z - 300.0; };
  std::vector<double> sdf((std::size_t)nx * ny * nz);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x)
        sdf[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny] = f(x, y, z);
  IbmSolver s(nx, ny, nz);
  for (int face = 0; face < 6; ++face)
    if (bc[face] != 0)
      s.setDomainBc(face, bc[face], 0.0, 0.0, 0.0);
  s.setSolid(sdf, false);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.fieldView("sdf"));
  const int ex = nx + 2 * G, ey = ny + 2 * G;
  const int n[3] = {nx, ny, nz};
  long bad = 0, checked = 0;
  for (int a = 0; a < 3; ++a)
    for (int side = 0; side < 2; ++side)
      for (int layer = 1; layer <= G; ++layer) {
        const int ia = side ? n[a] - 1 + layer : -layer;
        const int b = (a + 1) % 3, c = (a + 2) % 3;
        for (int jb = 0; jb < n[b]; ++jb)
          for (int jc = 0; jc < n[c]; ++jc) {
            int g3[3];
            g3[a] = ia;
            g3[b] = jb;
            g3[c] = jc;
            const double got =
                h((long)(g3[0] + G) + (long)(g3[1] + G) * ex + (long)(g3[2] + G) * (long)ex * ey);
            const int ex3[3] = {expectIndex(g3[0], nx, bc[0], bc[1]),
                                expectIndex(g3[1], ny, bc[2], bc[3]),
                                expectIndex(g3[2], nz, bc[4], bc[5])};
            ++checked;
            if (got != f(ex3[0], ex3[1], ex3[2]))
              ++bad;
          }
      }
  std::printf("  ghosts %-28s: %ld / %ld ghost samples (layers 1-2) stale\n", label, bad, checked);
  CHECK(bad == 0);
}

// ---------------------------------------------------------------------------------------------
// one geometry: the record + host recomputation
// ---------------------------------------------------------------------------------------------
struct Measure {
  double solidVol = 0.0, fluidVol = 0.0, area = 0.0;  // physical (hRef = 1)
  long cutCells = 0, notSingle = 0, lowRho = 0, sealed = 0, facets = 0, r0 = 0, r2 = 0;
  double minRho = 1.0, maxSnapRes = 0.0;             // (f): max |A^snap - areaPL| / max A
  double sealedMaxKappa = 0.0, sealedVolFrac = 0.0;  // evidence for the '0 sealed' clause
  long kappaMismatch = 0, countMismatch = 0;
};

Measure measure(IbmSolver& s, const int n[3], const double hp[3]) {
  Measure m;
  const auto& rec = s.scalarCutGeometry();
  const auto c = s.scalarCutCensus();
  const int ex = n[0] + 2 * G, ey = n[1] + 2 * G;
  const long sy = ex, sz = (long)ex * ey;
  auto sdf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.fieldView("sdf"));
  auto kap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rec.kappa);
  auto sax = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rec.sax);
  auto say = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rec.say);
  auto saz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rec.saz);
  auto alpha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rec.fac.alpha);
  auto rung = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rec.fac.rungF);
  const double vol = hp[0] * hp[1] * hp[2];
  const double maxA = std::fmax(hp[1] * hp[2], std::fmax(hp[0] * hp[2], hp[0] * hp[1]));
  long nfacHost = 0;
  for (int z = G; z < n[2] + G; ++z)
    for (int y = G; y < n[1] + G; ++y)
      for (int x = G; x < n[0] + G; ++x) {
        const long i = x + y * sy + z * sz;
        double corner[8], face[6], centre;
        scg::cellSamples(sdf, i, sy, sz, corner, face, centre);
        const cs::CutCellGeometry g = cs::cutCellGeometryFanTet(corner, face, centre, hp);
        if (std::memcmp(&g.kappa, &kap(i), sizeof(double)) != 0)
          ++m.kappaMismatch;
        m.fluidVol += g.kappa * vol;
        m.solidVol += (1.0 - g.kappa) * vol;
        double ap[6];
        scg::cellSnapped(sax, say, saz, i, sy, sz, ap);
        double area[2][3], cen[2][3];
        nfacHost += scg::cellFacets(g, ap, hp, area, cen);
        if (g.nFacet >= 1) {
          ++m.cutCells;
          const double rho = std::sqrt(g.areaPL[0] * g.areaPL[0] + g.areaPL[1] * g.areaPL[1] +
                                       g.areaPL[2] * g.areaPL[2]) /
                             g.areaSum;
          m.minRho = std::fmin(m.minRho, rho);
          if (g.kind != cs::CutCellKind::single)
            ++m.notSingle;
          if (rho < 0.9)
            ++m.lowRho;
          double aSnap[3];
          cs::facetAreaVector(ap, hp, aSnap);
          for (int d = 0; d < 3; ++d)
            m.maxSnapRes = std::fmax(m.maxSnapRes, std::fabs(aSnap[d] - g.areaPL[d]) / maxA);
        }
      }
  for (long f = 0; f < rec.fac.n; ++f) {
    m.area += alpha(f) * vol;
    if (rung(f) == cs::kProbeR0)
      ++m.r0;
    if (rung(f) == cs::kProbeR2)
      ++m.r2;
  }
  m.facets = rec.fac.n;
  m.sealed = c.numSealed;
  m.sealedVolFrac = c.sealedVolume / m.fluidVol;  // hRef = 1: physical = internal
  {
    auto unk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rec.unknown);
    for (int z = G; z < n[2] + G; ++z)
      for (int y = G; y < n[1] + G; ++y)
        for (int x = G; x < n[0] + G; ++x) {
          const long i = x + y * sy + z * sz;
          if (kap(i) > 0.0 && unk(i) < 0.5)
            m.sealedMaxKappa = std::fmax(m.sealedMaxKappa, kap(i));
        }
  }
  if (nfacHost != rec.fac.n)
    ++m.countMismatch;
  CHECK(c.numFacets == rec.fac.n);
  CHECK(c.rungs[0] + c.rungs[1] + c.rungs[2] + c.rungs[3] == rec.fac.n);
  return m;
}

// Sphere of radius R (physical = internal, hRef = 1) on cells n over extent n*hp, centre at the
// box centre + off. SOLID inside.
Measure sphereCase(const int n[3], const double hp[3], double R, const double off[3]) {
  const std::array<double, 3> extent{n[0] * hp[0], n[1] * hp[1], n[2] * hp[2]};
  IbmSolver s(n[0], n[1], n[2], extent, {0.0, 0.0, 0.0}, {(long)n[0], (long)n[1], (long)n[2]});
  std::vector<double> sdf((std::size_t)n[0] * n[1] * n[2]);
  const double c[3] = {0.5 * extent[0] + off[0], 0.5 * extent[1] + off[1],
                       0.5 * extent[2] + off[2]};
  for (int z = 0; z < n[2]; ++z)
    for (int y = 0; y < n[1]; ++y)
      for (int x = 0; x < n[0]; ++x) {
        const double dx = (x + 0.5) * hp[0] - c[0], dy = (y + 0.5) * hp[1] - c[1],
                     dz = (z + 0.5) * hp[2] - c[2];
        sdf[(std::size_t)x + (std::size_t)y * n[0] + (std::size_t)z * n[0] * n[1]] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - R;
      }
  s.setSolid(sdf, false);
  return measure(s, n, hp);
}

// G-geom (e) / (g): returns 0 on pass.
// `strictR0`: R0 = 100 % (the isotropic set). Otherwise (h' = (1,1,2)) R0 >= 99.9 % and no R2 —
// orchestrator ruling 2026-10-03 (doc/scalar_ibm_log.md, WO-2): a tiny corner facet's snapped area
// vector can point exactly along the coarse axis and its probe take R1b.
void sphereSet(const char* label, const double hp[3], bool strictR0) {
  const double h = std::fmax(hp[0], std::fmax(hp[1], hp[2]));
  const double ratios[3] = {8.0, 16.0, 32.0};
  std::mt19937_64 rng(4242);
  std::uniform_real_distribution<double> u(-0.5, 0.5);
  double rmsV[3], rmsA[3];
  for (int r = 0; r < 3; ++r) {
    const double R = ratios[r] * h;
    int n[3];
    for (int a = 0; a < 3; ++a)
      n[a] = (int)std::lround((2.0 * R + 8.0 * h) / hp[a]);
    const double Vex = 4.0 / 3.0 * M_PI * R * R * R, Aex = 4.0 * M_PI * R * R;
    const double bound2 = (h / R) * (h / R);
    double sV = 0.0, sA = 0.0;
    for (int o = 0; o < 3; ++o) {
      const double off[3] = {u(rng) * h, u(rng) * h, u(rng) * h};
      const Measure m = sphereCase(n, hp, R, off);
      const double eV = m.solidVol / Vex - 1.0, eA = m.area / Aex - 1.0;
      sV += eV * eV;
      sA += eA * eA;
      std::printf(
          "  %s R/h=%2.0f off %d: eV %+.3e (bound %.2e)  eA %+.3e (bound %.2e)  cut %ld "
          "facets %ld R0 %ld R2 %ld  min rho %.4f  not-single %ld  sealed %ld (max kappa %.1e, "
          "vol/fluid %.1e)  (f) %.2e\n",
          label, ratios[r], o, eV, 2.0 * bound2, eA, 3.0 * bound2, m.cutCells, m.facets, m.r0, m.r2,
          m.minRho, m.notSingle, m.sealed, m.sealedMaxKappa, m.sealedVolFrac, m.maxSnapRes);
      CHECK(std::fabs(eV) <= 2.0 * bound2);
      CHECK(std::fabs(eA) <= 3.0 * bound2);
      CHECK(m.notSingle == 0 && m.lowRho == 0);
      // '0 sealed' restated (orchestrator ruling 2026-10-03): the sealed volume sum kappa V over
      // sealed cells <= 1e-6 of the fluid volume, the §9 threshold. The count stays reported.
      CHECK(m.sealedVolFrac <= 1e-6);
      CHECK(m.facets > 0);
      if (strictR0)
        CHECK(m.r0 == m.facets);
      else
        CHECK((double)m.r0 >= 0.999 * (double)m.facets && m.r2 == 0);
      CHECK(m.maxSnapRes <= 2e-3);
      CHECK(m.kappaMismatch == 0 && m.countMismatch == 0);
    }
    rmsV[r] = std::sqrt(sV / 3.0);
    rmsA[r] = std::sqrt(sA / 3.0);
  }
  for (int r = 0; r + 1 < 3; ++r) {
    const double oV = std::log2(rmsV[r] / rmsV[r + 1]), oA = std::log2(rmsA[r] / rmsA[r + 1]);
    std::printf("  %s order %2.0f->%2.0f: volume %.3f  area %.3f\n", label, ratios[r],
                ratios[r + 1], oV, oA);
    CHECK(oV >= 1.8);
    CHECK(oA >= 1.8);
  }
}

// G7's pipe: fluid inside r < R, z periodic with nz = 4, R/h in {16, 32, 64}.
void pipeSet() {
  std::mt19937_64 rng(77);
  std::uniform_real_distribution<double> u(-0.5, 0.5);
  for (double R : {16.0, 32.0, 64.0}) {
    const int nxy = (int)std::lround(2.0 * R + 8.0);
    const int n[3] = {nxy, nxy, 4};
    const double hp[3] = {1.0, 1.0, 1.0};
    for (int o = 0; o < 3; ++o) {
      const double cx = 0.5 * nxy + u(rng), cy = 0.5 * nxy + u(rng);
      IbmSolver s(n[0], n[1], n[2], {(double)n[0], (double)n[1], (double)n[2]}, {0.0, 0.0, 0.0},
                  {(long)n[0], (long)n[1], (long)n[2]});
      std::vector<double> sdf((std::size_t)n[0] * n[1] * n[2]);
      for (int z = 0; z < n[2]; ++z)
        for (int y = 0; y < n[1]; ++y)
          for (int x = 0; x < n[0]; ++x) {
            const double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
            sdf[(std::size_t)x + (std::size_t)y * n[0] + (std::size_t)z * n[0] * n[1]] =
                R - std::sqrt(dx * dx + dy * dy);
          }
      s.setSolid(sdf, false);
      const Measure m = measure(s, n, hp);
      const double eV = m.fluidVol / (M_PI * R * R * n[2]) - 1.0;
      std::printf(
          "  pipe R/h=%2.0f off %d: eV %+.3e  facets %ld R0 %ld  min rho %.4f  sealed %ld  "
          "(f) %.2e\n",
          R, o, eV, m.facets, m.r0, m.minRho, m.sealed, m.maxSnapRes);
      CHECK(m.r0 == m.facets && m.facets > 0);
      CHECK(m.sealed == 0 && m.notSingle == 0);
      CHECK(m.maxSnapRes <= 2e-3);
      CHECK(m.kappaMismatch == 0 && m.countMismatch == 0);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  {
    const int periodic[6] = {0, 0, 0, 0, 0, 0};
    const int walls[6] = {1, 1, 4, 4, 0, 0};  // x walls (constant extension), y slip (mirror)
    const int mixed[6] = {0, 0, 1, 4, 4, 1};
    checkGhosts("periodic wrap", periodic);
    checkGhosts("x wall + y slip", walls);
    checkGhosts("y wall/slip + z slip/wall", mixed);

    const double iso[3] = {1.0, 1.0, 1.0}, aniso[3] = {1.0, 1.0, 2.0};
    sphereSet("iso  ", iso, true);
    sphereSet("aniso", aniso, false);
    pipeSet();
  }
  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
