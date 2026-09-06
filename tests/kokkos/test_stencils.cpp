// Correctness of the Kokkos MAC stencils (peclet::flow::poisSweep RB-GS + divergence) vs a host
// replication, plus a smoother sanity check (Poisson residual decreases over sweeps). Extended
// (inner+ghost=1) block, x-fastest. Within a Red-Black colour the updates are independent, so
// device (parallel) and host (sequential, colour 0 then 1) match exactly. Runs on whatever backend
// Kokkos was built for.
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "cut_cell_ibm.hpp"
#include "mac_stencils.hpp"
#include "mac_velocity_mg.hpp"

using namespace peclet::flow;

static long l3(int x, int y, int z, I3 e) {
  return (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int g = 1;
    I3 inner{18, 14, 10};
    I3 e{inner.x + 2 * g, inner.y + 2 * g, inner.z + 2 * g};
    I3 og{0, 0, 0};  // single block: global origin 0
    const std::size_t n = (std::size_t)e.x * e.y * e.z;

    std::mt19937 rng(8);
    std::uniform_real_distribution<double> uf(-1.0, 1.0);
    std::vector<double> hphi(n), hd(n), hu(n), hv(n), hw(n);
    for (std::size_t i = 0; i < n; ++i) {
      hphi[i] = uf(rng);
      hd[i] = uf(rng);
      hu[i] = uf(rng);
      hv[i] = uf(rng);
      hw[i] = uf(rng);
    }

    auto up = [&](const char* nm, std::vector<double>& h) {
      SField v(nm, n);
      auto m = Kokkos::create_mirror_view(v);
      for (std::size_t i = 0; i < n; ++i)
        m(i) = h[i];
      Kokkos::deep_copy(v, m);
      return v;
    };
    SField phi = up("phi", hphi), d = up("d", hd), u = up("u", hu), v = up("v", hv),
           w = up("w", hw);
    SField dout("dout", n);

    // device: one RB-GS sweep + divergence
    poisSweep(phi, d, e, og, g);
    divergence(u, v, w, dout, e, g);
    auto hp = Kokkos::create_mirror_view(phi);
    Kokkos::deep_copy(hp, phi);
    auto hdo = Kokkos::create_mirror_view(dout);
    Kokkos::deep_copy(hdo, dout);

    // host reference: identical RB-GS sweep (colour 0 then 1) + divergence
    std::vector<double> rphi = hphi;
    for (int color = 0; color < 2; ++color)
      for (int z = g; z < e.z - g; ++z)
        for (int y = g; y < e.y - g; ++y)
          for (int x = g; x < e.x - g; ++x) {
            if ((((x + og.x) + (y + og.y) + (z + og.z)) & 1) != color)
              continue;
            long i = l3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
            double s = rphi[i + sx] + rphi[i - sx] + rphi[i + sy] + rphi[i - sy] + rphi[i + sz] +
                       rphi[i - sz];
            rphi[i] = (s - hd[i]) / 6.0;
          }
    auto close = [](double a, double b) {
      return std::fabs(a - b) <= 1e-10 * (1.0 + std::fabs(b));
    };
    int bad = 0;
    for (std::size_t i = 0; i < n; ++i)
      if (!close(hp(i), rphi[i]))
        ++bad;
    for (int z = g; z < e.z - g; ++z)
      for (int y = g; y < e.y - g; ++y)
        for (int x = g; x < e.x - g; ++x) {
          long i = l3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
          double rd = (hu[i + sx] - hu[i]) + (hv[i + sy] - hv[i]) + (hw[i + sz] - hw[i]);
          if (!close(hdo(i), rd))
            ++bad;
        }
    if (bad) {
      std::fprintf(stderr, "FAIL: %d cells differ (sweep/divergence)\n", bad);
      status = 1;
    }

    // smoother sanity: many sweeps on Lap(phi)=d (periodic-ish, ghosts left as data) -> residual
    // drops.
    SField phi2 = up("phi2", hphi);
    auto resid = [&]() {
      auto m = Kokkos::create_mirror_view(phi2);
      Kokkos::deep_copy(m, phi2);
      double r = 0.0;
      for (int z = g; z < e.z - g; ++z)
        for (int y = g; y < e.y - g; ++y)
          for (int x = g; x < e.x - g; ++x) {
            long i = l3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
            double s = m(i + sx) + m(i - sx) + m(i + sy) + m(i - sy) + m(i + sz) + m(i - sz);
            double res = (s - 6.0 * m(i)) - hd[i];  // Lap(phi)-d
            r += res * res;
          }
      return std::sqrt(r);
    };
    double r0 = resid();
    for (int it = 0; it < 100; ++it)
      poisSweep(phi2, d, e, og, g);
    double r1 = resid();
    if (!(r1 < 0.2 * r0)) {
      std::fprintf(stderr, "FAIL: smoother did not reduce residual (%.3e -> %.3e)\n", r0, r1);
      status = 1;
    }

    if (!status)
      std::printf(
          "[mac_stencils] PASS: RB-GS sweep + divergence match host; residual %.2e -> %.2e (exec: "
          "%s)\n",
          r0, r1, SExec::name());
  }
  // ------------------------------------------------------------------------------------------
  // Phase 2 commit C1 (doc/anisotropic_metric.md §2, §9 row C1): the per-axis viscous fold.
  //   A_C = idiag + 2*((bx + by) + bz),  AW = AE = -bx, AS = AN = -by, AB = AT = -bz
  // in EXACTLY that association order, for BOTH assemblers of the constant-coefficient momentum
  // operator: `ibmBuildDiffusion` (the fine stencil) and `buildConstAniso` (the velocity-MG
  // levels).  Two statements are gated:
  //   (a) an anisotropic metric w = (1, 4, 1/4) produces exactly those per-axis bands;
  //   (b) at w = (1,1,1) the diagonal is BITWISE the pre-Phase-2 expression `idiag + 6.0*beta`
  //       — (beta+beta) is exact, +beta rounds once, and the *2 is exact, so 2*fl(3 beta) ==
  //       fl(6 beta).  That is the whole isotropic-identity argument, checked at kernel level.
  {
    using peclet::flow::buildConstAniso;
    using peclet::flow::C3;
    using peclet::flow::FPV;
    using peclet::flow::ibmBuildDiffusion;
    using peclet::flow::MReal;

    const int ex = 6, ey = 5, ez = 4;
    const std::size_t n = (std::size_t)ex * ey * ez;
    const C3 e3{ex, ey, ez};
    const double mu = 0.1, idiag = 1.0 / 50.0;

    auto bands = [&](FPV AC, FPV AW, FPV AE, FPV AS, FPV AN, FPV AB, FPV AT, MReal out[7]) {
      FPV all[7] = {AC, AW, AE, AS, AN, AB, AT};
      for (int k = 0; k < 7; ++k) {
        auto m = Kokkos::create_mirror_view(all[k]);
        Kokkos::deep_copy(m, all[k]);
        out[k] = m(n / 2);
      }
    };
    FPV AC("AC", n), AW("AW", n), AE("AE", n), AS("AS", n), AN("AN", n), AB("AB", n), AT("AT", n);

    // (a) anisotropic metric
    {
      const double w[3] = {1.0, 4.0, 0.25};
      const double bx = mu * w[0], by = mu * w[1], bz = mu * w[2];
      const MReal wantC = (MReal)(idiag + 2.0 * ((bx + by) + bz));
      const MReal wantX = (MReal)(-bx), wantY = (MReal)(-by), wantZ = (MReal)(-bz);
      const MReal want[7] = {wantC, wantX, wantX, wantY, wantY, wantZ, wantZ};
      const char* nm[7] = {"AC", "AW", "AE", "AS", "AN", "AB", "AT"};
      const char* who[2] = {"ibmBuildDiffusion", "buildConstAniso"};
      for (int k = 0; k < 2; ++k) {
        if (k == 0)
          ibmBuildDiffusion(AC, AW, AE, AS, AN, AB, AT, ex, ey, ez, bx, by, bz, idiag);
        else
          buildConstAniso(AC, AW, AE, AS, AN, AB, AT, e3, bx, by, bz, idiag);
        MReal got[7];
        bands(AC, AW, AE, AS, AN, AB, AT, got);
        for (int b = 0; b < 7; ++b)
          if (!(got[b] == want[b])) {
            std::fprintf(stderr, "FAIL: %s aniso %s = %.17g, want %.17g\n", who[k], nm[b],
                         (double)got[b], (double)want[b]);
            status = 1;
          }
        if (!status)
          std::printf(
              "[aniso_fold] %-17s w=(1,4,0.25): AC %.17g  AW/AE %.17g  AS/AN %.17g  AB/AT %.17g\n",
              who[k], (double)got[0], (double)got[1], (double)got[3], (double)got[5]);
      }
    }

    // (b) isotropic metric -> bitwise the pre-Phase-2 expression
    {
      const double beta = mu;
      const MReal legacyC = (MReal)(idiag + 6.0 * beta), legacyO = (MReal)(-beta);
      const MReal want[7] = {legacyC, legacyO, legacyO, legacyO, legacyO, legacyO, legacyO};
      const char* nm[7] = {"AC", "AW", "AE", "AS", "AN", "AB", "AT"};
      const char* who[2] = {"ibmBuildDiffusion", "buildConstAniso"};
      for (int k = 0; k < 2; ++k) {
        if (k == 0)
          ibmBuildDiffusion(AC, AW, AE, AS, AN, AB, AT, ex, ey, ez, beta, beta, beta, idiag);
        else
          buildConstAniso(AC, AW, AE, AS, AN, AB, AT, e3, beta, beta, beta, idiag);
        MReal got[7];
        bands(AC, AW, AE, AS, AN, AB, AT, got);
        for (int b = 0; b < 7; ++b)
          if (!(got[b] == want[b])) {
            std::fprintf(stderr, "FAIL: %s isotropic %s = %.17g, want (legacy) %.17g\n", who[k],
                         nm[b], (double)got[b], (double)want[b]);
            status = 1;
          }
        if (!status)
          std::printf("[aniso_fold] %-17s w=(1,1,1):    AC %.17g == idiag+6*beta %.17g (bitwise)\n",
                      who[k], (double)got[0], (double)legacyC);
      }
    }
    if (!status)
      std::printf("[aniso_fold] PASS: per-axis viscous fold; isotropic == legacy bitwise\n");
  }

  Kokkos::finalize();
  return status;
}
