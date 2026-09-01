// Correctness of the Kokkos cut-cell face openness (peclet::flow::buildOpenness) vs a host
// replication on a sphere SDF, plus properties: an all-fluid SDF gives openness 1, an all-solid SDF
// gives 0. SDF sign convention: negative inside solid, positive in fluid. Runs on whatever backend
// Kokkos was built for.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "mac_cutcell.hpp"
#include "mac_cutcell_mg.hpp"  // MReal / FPV / FPC (the operator storage type under test)
#include "mac_pressure.hpp"

using namespace peclet::flow;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    C3 ext{16, 16, 16};
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const std::size_t n = (std::size_t)ext.x * ext.y * ext.z;
    auto IDX = [&](int x, int y, int z) {
      return (std::size_t)x + (std::size_t)y * ext.x + (std::size_t)z * (std::size_t)ext.x * ext.y;
    };

    // solid sphere radius 5 at (8,8,8): sdf = dist - R  (>0 fluid outside, <0 solid inside)
    std::vector<double> hsdf(n);
    const double cx = 8, cy = 8, cz = 8, R = 5;
    for (int z = 0; z < ext.z; ++z)
      for (int y = 0; y < ext.y; ++y)
        for (int x = 0; x < ext.x; ++x)
          hsdf[IDX(x, y, z)] =
              std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz)) - R;

    auto upC = [&](const char* nm, std::vector<double>& h) {
      CCField v(nm, n);
      auto m = Kokkos::create_mirror_view(v);
      for (std::size_t i = 0; i < n; ++i)
        m(i) = h[i];
      Kokkos::deep_copy(v, m);
      return v;
    };
    auto getC = [&](CCField v) {
      std::vector<double> o(n);
      auto m = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(m, v);
      for (std::size_t i = 0; i < n; ++i)
        o[i] = m(i);
      return o;
    };

    CCField sdf = upC("sdf", hsdf), ox("ox", n), oy("oy", n), oz("oz", n);
    buildOpenness(ox, oy, oz, sdf, ext, dx, dy, dz);
    auto gox = getC(ox), goy = getC(oy), goz = getC(oz);

    // host reference (same functions)
    auto close = [](double a, double b) {
      return std::fabs(a - b) <= 1e-10 * (1.0 + std::fabs(b));
    };
    int bad = 0;
    // Host reference replicates the same math on the host SDF vector.
    auto sampleH = [&](double X, double Y, double Z) {
      double fx = std::floor(X), fy = std::floor(Y), fz = std::floor(Z);
      double wx = X - fx, wy = Y - fy, wz = Z - fz;
      int x0 = (int)fx, y0 = (int)fy, z0 = (int)fz;
      auto cl = [&](int v, int nn) { return v < 0 ? 0 : (v >= nn ? nn - 1 : v); };
      int x1 = cl(x0 + 1, ext.x), y1 = cl(y0 + 1, ext.y), z1 = cl(z0 + 1, ext.z);
      x0 = cl(x0, ext.x);
      y0 = cl(y0, ext.y);
      z0 = cl(z0, ext.z);
      auto F = [&](int xx, int yy, int zz) { return hsdf[IDX(xx, yy, zz)]; };
      double c00 = F(x0, y0, z0) * (1 - wx) + F(x1, y0, z0) * wx,
             c10 = F(x0, y1, z0) * (1 - wx) + F(x1, y1, z0) * wx;
      double c01 = F(x0, y0, z1) * (1 - wx) + F(x1, y0, z1) * wx,
             c11 = F(x0, y1, z1) * (1 - wx) + F(x1, y1, z1) * wx;
      double c0 = c00 * (1 - wy) + c10 * wy, c1 = c01 * (1 - wy) + c11 * wy;
      return c0 * (1 - wz) + c1 * wz;
    };
    auto faceH = [&](double X, double Y, double Z, int type) {
      double sd = sampleH(X, Y, Z);
      if (sd <= 0)
        return 0.0;
      double e = 1.0;
      return ccFractionCore(sd, sampleH(X + e, Y, Z), sampleH(X - e, Y, Z), sampleH(X, Y + e, Z),
                            sampleH(X, Y - e, Z), sampleH(X, Y, Z + e), sampleH(X, Y, Z - e), type,
                            dx, dy, dz);
    };
    for (int z = 0; z < ext.z; ++z)
      for (int y = 0; y < ext.y; ++y)
        for (int x = 0; x < ext.x; ++x) {
          std::size_t i = IDX(x, y, z);
          if (!close(gox[i], faceH(x - 0.5, y, z, 1)))
            ++bad;
          if (!close(goy[i], faceH(x, y - 0.5, z, 2)))
            ++bad;
          if (!close(goz[i], faceH(x, y, z - 0.5, 3)))
            ++bad;
        }
    if (bad) {
      std::fprintf(stderr, "FAIL: %d openness faces differ\n", bad);
      status = 1;
    }

    // count cut faces (0<open<1) — the sphere surface must produce some
    int cut = 0, open1 = 0, closed0 = 0;
    for (std::size_t i = 0; i < n; ++i) {
      double v = gox[i];
      if (v > 1e-9 && v < 1 - 1e-9)
        ++cut;
      else if (v > 1 - 1e-9)
        ++open1;
      else
        ++closed0;
    }

    // properties
    std::vector<double> allf(n, 100.0), alls(n, -100.0);
    CCField sf = upC("sf", allf), so = upC("so", alls), a("a", n), b("b", n), c("c", n);
    buildOpenness(a, b, c, sf, ext, dx, dy, dz);
    auto ga = getC(a);
    for (std::size_t i = 0; i < n; ++i)
      if (!close(ga[i], 1.0))
        ++bad;
    buildOpenness(a, b, c, so, ext, dx, dy, dz);
    auto ga2 = getC(a);
    for (std::size_t i = 0; i < n; ++i)
      if (!close(ga2[i], 0.0))
        ++bad;
    if (bad && !status) {
      std::fprintf(stderr, "FAIL: all-fluid/all-solid property\n");
      status = 1;
    }

    if (!status)
      std::printf(
          "[mac_cutcell] PASS: openness matches host (%d cut / %d open / %d closed x-faces) + "
          "properties (exec: %s)\n",
          cut, open1, closed0, CCExec::name());
  }

  // ---------------------------------------------------------------------------------------
  // P1 (docs/DEFECT_CORRECTION_PLAN.md) — the exact flux-form level-0 apply annihilates the
  // constant vector BITWISE, where the stored (float) bands do not.
  //
  // A = -div(open grad) is singular by construction: its null space is the constants, and the
  // pressure solve deflates exactly that mode. buildCutcellOp forms AC as the float-rounded sum
  // of the six face terms, so A*1 comes out at eps_f32 * max|AC| instead of 0 -- the defect WO-M
  // measured as a resolution-independent residual floor at ~5e-9 with a 1e4-1e5 rebound.
  // applyCutcellOpExact re-derives t_f = open_f * gf from the double openness and applies
  // sum_f t_f (x_i - x_nbr), in which every difference vanishes identically for constant x.
  //
  // Beds: the two the regression suite ships (tests/regression/sdflow_regression.py) -- the
  // jittered 2x2x2 sphere packing (centres lifted from that generator's fixed seed 12345) and the
  // three Raschig rings -- at N = 32 and 64, because what matters here is the aperture spectrum,
  // and those beds span ~3 decades of it.
  {
    // random_spheres: fractional centres of sdf_random_spheres(N, seed=12345); N-independent.
    static const double kSphereC[8][3] = {
        {0.16457049781272215, 0.32582370748774664, 0.19776029572245485},
        {0.23444960590393615, 0.24547940157936873, 0.70554692087486348},
        {0.16793243789302340, 0.78893356813158244, 0.27166348678329372},
        {0.13282821621926860, 0.89084457926273108, 0.80810981434511542},
        {0.70443676917452958, 0.30413189645273508, 0.22198280960076699},
        {0.74635862887577831, 0.29733066067115205, 0.67459991201161940},
        {0.78455145086375566, 0.83393873968342314, 0.32933788364396716},
        {0.73201808908205368, 0.80417516048550364, 0.65270503594906770}};
    // hollow_rings: (centre fraction, axis) of sdf_hollow_rings.
    static const double kRingC[3][3] = {{0.30, 0.32, 0.30}, {0.70, 0.68, 0.55}, {0.45, 0.50, 0.78}};
    static const int kRingA[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    auto minimg = [](double d, double N) { return d - N * std::round(d / N); };

    // SDF of the two beds at a point, in cell units on the periodic N^3 box.
    auto sdfSpheres = [&](double X, double Y, double Z, double N) {
      const double R = 0.18 * N;
      double best = 1e30;
      for (int s = 0; s < 8; ++s) {
        const double dx = minimg(X - kSphereC[s][0] * N, N), dy = minimg(Y - kSphereC[s][1] * N, N),
                     dz = minimg(Z - kSphereC[s][2] * N, N);
        best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz) - R);
      }
      return best;
    };
    auto sdfRings = [&](double X, double Y, double Z, double N) {
      const double rO = 0.22 * N, rI = 0.12 * N, H = 0.34 * N;
      double best = 1e30;
      for (int r = 0; r < 3; ++r) {
        const double ax = kRingA[r][0], ay = kRingA[r][1], az = kRingA[r][2];  // already unit
        const double dx = minimg(X - kRingC[r][0] * N, N), dy = minimg(Y - kRingC[r][1] * N, N),
                     dz = minimg(Z - kRingC[r][2] * N, N);
        const double zc = dx * ax + dy * ay + dz * az;  // axial coordinate
        const double rx = dx - zc * ax, ry = dy - zc * ay, rz = dz - zc * az;
        const double rho = std::sqrt(rx * rx + ry * ry + rz * rz);
        best = std::min(best, std::max(std::max(rI - rho, rho - rO), std::fabs(zc) - 0.5 * H));
      }
      return best;
    };

    const char* bedName[2] = {"random_spheres", "hollow_rings"};
    const int grids[2] = {32, 64};
    for (int bed = 0; bed < 2 && !status; ++bed)
      for (int gi = 0; gi < 2 && !status; ++gi) {
        const int N = grids[gi];
        const int G = 1;  // CutcellMG::G — the level-0 ghost width
        const C3 e{N + 2 * G, N + 2 * G, N + 2 * G};
        const std::size_t nn = (std::size_t)e.x * e.y * e.z;
        // The bed SDFs are periodic, so evaluating them on the ghosted block directly IS the
        // periodic ghost fill CutcellMG::fillOpenness performs (up to last-bit differences in the
        // minimum image, which affect both operators identically and cannot affect the identity).
        std::vector<double> hs(nn);
        for (int z = 0; z < e.z; ++z)
          for (int y = 0; y < e.y; ++y)
            for (int x = 0; x < e.x; ++x) {
              const double X = (x - G) + 0.5, Y = (y - G) + 0.5, Z = (z - G) + 0.5;
              hs[(std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y] =
                  bed == 0 ? sdfSpheres(X, Y, Z, (double)N) : sdfRings(X, Y, Z, (double)N);
            }
        CCField sdfB("sdf_bed", nn);
        {
          auto m = Kokkos::create_mirror_view(sdfB);
          for (std::size_t i = 0; i < nn; ++i)
            m(i) = hs[i];
          Kokkos::deep_copy(sdfB, m);
        }
        CCField box("box", nn), boy("boy", nn), boz("boz", nn);
        buildOpenness(box, boy, boz, sdfB, e, 1.0, 1.0, 1.0);

        FPV AC("AC", nn), AW("AW", nn), AE("AE", nn), AS("AS", nn), AN("AN", nn), AB("AB", nn),
            AT("AT", nn);
        buildCutcellOp(AC, AW, AE, AS, AN, AB, AT, CCConst(box), CCConst(boy), CCConst(boz), e, G,
                       1.0, 1.0, 1.0);

        // x == 1 (the null vector). A*1 must be 0; the flux form must give it BITWISE.
        CCField one("one", nn), yBand("yBand", nn), yEx("yEx", nn);
        Kokkos::deep_copy(one, 1.0);
        applyCutcellOp(yBand, CCConst(one), FPC(AC), FPC(AW), FPC(AE), FPC(AS), FPC(AN), FPC(AB),
                       FPC(AT), e, G);
        applyCutcellOpExact(yEx, CCConst(one), CCConst(box), CCConst(boy), CCConst(boz), e, G, 1.0,
                            1.0, 1.0);

        auto hb = Kokkos::create_mirror_view(yBand), hx = Kokkos::create_mirror_view(yEx);
        auto hAC = Kokkos::create_mirror_view(AC);
        auto hox = Kokkos::create_mirror_view(box), hoy = Kokkos::create_mirror_view(boy),
             hoz = Kokkos::create_mirror_view(boz);
        Kokkos::deep_copy(hb, yBand);
        Kokkos::deep_copy(hx, yEx);
        Kokkos::deep_copy(hAC, AC);
        Kokkos::deep_copy(hox, box);
        Kokkos::deep_copy(hoy, boy);
        Kokkos::deep_copy(hoz, boz);

        double maxBand = 0.0, maxExact = 0.0, maxAC = 0.0;
        long nSolid = 0, nSolidBad = 0, nCut = 0;
        for (int z = G; z < e.z - G; ++z)
          for (int y = G; y < e.y - G; ++y)
            for (int x = G; x < e.x - G; ++x) {
              const std::size_t i =
                  (std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y;
              maxBand = std::max(maxBand, std::fabs(hb(i)));
              maxExact = std::max(maxExact, std::fabs(hx(i)));
              maxAC = std::max(maxAC, (double)hAC(i));
              const double sum = hox(i) + hox(i + 1) + hoy(i) + hoy(i + e.x) + hoz(i) +
                                 hoz(i + (std::size_t)e.x * e.y);
              if (sum == 0.0) {  // fully closed (solid): both forms must be exactly 0
                ++nSolid;
                if (hx(i) != 0.0 || hb(i) != 0.0)
                  ++nSolidBad;
              } else if (sum < 6.0) {
                ++nCut;
              }
            }

        // THE GATE: bitwise zero. Not a tolerance — every term of the flux form is t_f*(1-1) = 0.
        if (maxExact != 0.0) {
          std::fprintf(stderr, "FAIL: %s N=%d exact A*1 = %.17g, expected bitwise 0\n",
                       bedName[bed], N, maxExact);
          status = 1;
        }
        if (nSolidBad) {
          std::fprintf(stderr, "FAIL: %s N=%d %ld solid cells with nonzero A*1\n", bedName[bed], N,
                       nSolidBad);
          status = 1;
        }
        std::printf(
            "[cutcell/P1] %-15s N=%2d  max|A_exact*1| = %.3e (bitwise 0)   "
            "max|A_bands*1| = %.3e  (rel to max AC %.3e: %.2e)   %ld cut / %ld solid cells\n",
            bedName[bed], N, maxExact, maxBand, maxAC, maxBand / maxAC, nCut, nSolid);

        // Same operator, not just a null-space trick: on a non-constant field the two forms must
        // agree to the float rounding of the stored bands (this is what "the bands are a rounded
        // copy of the exact operator" means quantitatively).
        CCField xr("xr", nn);
        {
          auto m = Kokkos::create_mirror_view(xr);
          unsigned s32 = 22801u + 7919u * (unsigned)(bed * 4 + gi);
          for (std::size_t i = 0; i < nn; ++i) {
            s32 = s32 * 1664525u + 1013904223u;  // LCG: deterministic across backends
            m(i) = (double)(s32 >> 8) / (double)(1u << 24) - 0.5;
          }
          Kokkos::deep_copy(xr, m);
        }
        applyCutcellOp(yBand, CCConst(xr), FPC(AC), FPC(AW), FPC(AE), FPC(AS), FPC(AN), FPC(AB),
                       FPC(AT), e, G);
        applyCutcellOpExact(yEx, CCConst(xr), CCConst(box), CCConst(boy), CCConst(boz), e, G, 1.0,
                            1.0, 1.0);
        Kokkos::deep_copy(hb, yBand);
        Kokkos::deep_copy(hx, yEx);
        double dmax = 0.0, ymax = 0.0;
        for (int z = G; z < e.z - G; ++z)
          for (int y = G; y < e.y - G; ++y)
            for (int x = G; x < e.x - G; ++x) {
              const std::size_t i =
                  (std::size_t)x + (std::size_t)y * e.x + (std::size_t)z * (std::size_t)e.x * e.y;
              dmax = std::max(dmax, std::fabs(hb(i) - hx(i)));
              ymax = std::max(ymax, std::fabs(hx(i)));
            }
        const double rel = dmax / ymax;
        // sizeof(MReal) picks the bound: a double build stores the exact coefficient, so the two
        // forms then differ only by summation order.
        const double tol = sizeof(MReal) == sizeof(float) ? 1e-5 : 1e-13;
        if (!(rel < tol)) {
          std::fprintf(stderr, "FAIL: %s N=%d bands vs exact rel diff %.3e >= %.1e\n", bedName[bed],
                       N, rel, tol);
          status = 1;
        }
        std::printf("[cutcell/P1] %-15s N=%2d  bands vs exact on random x: rel %.3e (< %.0e)\n",
                    bedName[bed], N, rel, tol);
      }
    if (!status)
      std::printf("[cutcell/P1] PASS: exact flux-form apply annihilates the constants bitwise\n");
  }

  Kokkos::finalize();
  return status;
}
