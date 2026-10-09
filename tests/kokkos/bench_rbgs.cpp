// Throughput benchmark of the Kokkos RB-GS Poisson sweep (the dominant solver kernel), for the
// Kokkos-vs-native-CUDA efficiency comparison. Times many full Red-Black sweeps on a fixed grid and
// reports ns/sweep + effective bandwidth. Pair with tests/cuda_bench/bench_rbgs.cu (identical
// kernel, plain nvcc) run on the same GPU.
//
// `bench_rbgs p0 [rounds] [passes]` instead runs the WO-P0 instruments of
// doc/vof_projection_cost_design.md §12 (an instrument, not a gate):
//   * t_L, the cost of an empty parallel_for (launch + join) at the running thread count;
//   * one level-0 colour pass at 128x96x64 (x, z periodic, walls in y) with variable-density face
//     weights, in four forms:
//       (a) today's FP64 wrap pass -- the production cutcellSmoothColorFaceWrap;
//       (b) the same pass with the row ends peeled (§9; host only), checked bitwise against (a);
//       (c) D-lite: FP32 face weights, FP64 arithmetic and iterates, flux/correction form (§4.3);
//       (d) the FP32 flux pass, the §4.4.3 bodies exactly as written.
//     The rounds interleave the four forms; min, median and max ms per pass are printed.
// The coefficients are a DOCUMENTED SYNTHETIC field, not a solver state: rho = 1 (liquid) and
// rho = 0.02 inside 24 spheres of radius 6 cells (a smoothed one-cell interface), the staggered
// c_f = open_f * rho0 / rho_f with the arithmetic face mean and rho0 = 1 (buildRhoCoeff), open = 1
// except the two y-wall faces (open 0), metric gf = 1. Run it at OMP_NUM_THREADS=1, 8 and 24.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "mac_cutcell_mg.hpp"
#include "mac_stencils.hpp"

namespace pf = peclet::flow;

namespace p0bench {

using VReal = float;  // the FP32 V-cycle data type of §4.4.2 (instrument copy)
using FV = Kokkos::View<VReal*, pf::CCMem>;
using FVC = Kokkos::View<const VReal*, pf::CCMem>;

int runRbgs(int N, int K) {
  const int g = 1;
  pf::I3 e{N + 2 * g, N + 2 * g, N + 2 * g}, og{0, 0, 0};
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  pf::SField phi("phi", n), d("d", n);
  Kokkos::deep_copy(phi, 1.0);
  Kokkos::deep_copy(d, 0.5);

  // warmup
  for (int i = 0; i < 10; ++i)
    pf::poisSweep(phi, pf::SConst(d), e, og, g);
  Kokkos::fence();

  double best = 1e30;
  for (int rep = 0; rep < 5; ++rep) {
    Kokkos::Timer t;
    for (int i = 0; i < K; ++i)
      pf::poisSweep(phi, pf::SConst(d), e, og, g);
    Kokkos::fence();
    double ms = t.seconds() * 1e3;
    if (ms < best)
      best = ms;
  }
  const double ns_per_sweep = best * 1e6 / K;
  const double cells = (double)N * N * N;
  const double ns_per_cell = ns_per_sweep / cells;
  const double gbps =
      cells * 64.0 / (ns_per_sweep);  // 64 B/cell (6 nbr + d + write), bytes/ns = GB/s
  std::printf("KOKKOS  N=%d sweeps=%d : %.3f ns/sweep, %.4f ns/cell, %.1f GB/s (exec %s)\n", N, K,
              ns_per_sweep, ns_per_cell, gbps, Kokkos::DefaultExecutionSpace::name());
  return 0;
}

constexpr bool kHost = std::is_same_v<pf::CCMem, Kokkos::HostSpace>;

// (b) The peeled FP64 wrap pass of §9: per row the y/z neighbour offsets once, the first and last
// cell of the row scalar with ccWrapNbrs (today's body), the interior with i +- 1. Host only.
template <class Exec = pf::CCExec>
void smoothPeeledFp64(pf::CCField phi, pf::CCConst b, pf::FPC AC, pf::FPC AFX, pf::FPC AFY,
                      pf::FPC AFZ, pf::C3 e, pf::C3 n, pf::C3 og, int g, int color) {
  if constexpr (std::is_same_v<typename Exec::memory_space, Kokkos::HostSpace>) {
    const int nyi = e.y - 2 * g, nzi = e.z - 2 * g;
    auto pencil = [=](long t) {
      const int ly = g + (int)(t % nyi), lz = g + (int)(t / nyi);
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
      const int lx0 = g + ((P ^ (g & 1)) & 1), lxLast = g + n.x - 1;
      const long dym = (ly == g) ? (long)(n.y - 1) * sy : -sy;
      const long dyp = (ly == g + n.y - 1) ? -(long)(n.y - 1) * sy : sy;
      const long dzm = (lz == g) ? (long)(n.z - 1) * sz : -sz;
      const long dzp = (lz == g + n.z - 1) ? -(long)(n.z - 1) * sz : sz;
      const long row = (long)ly * sy + (long)lz * sz;
      auto scalar = [&](int lx) {
        const long i = (long)lx + row;
        const pf::CcNbrs w = pf::ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
        pf::cutcellSmoothFaceCell(phi, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, w.xp, w.xm, w.yp, w.ym,
                                  w.zp, w.zm);
      };
      int lo = lx0;
      if (lo == g) {
        scalar(lo);
        lo += 2;
      }
      const bool lastIsColour = ((lxLast - lx0) & 1) == 0;
      const int hi = lastIsColour ? lxLast : lxLast + 1;  // exclusive
      PECLET_FLOW_OMP_SIMD
      for (int lx = lo; lx < hi; lx += 2) {
        const long i = (long)lx + row;
        pf::cutcellSmoothFaceCell(phi, b, AC, AFX, AFY, AFZ, i, sx, sy, sz, i + 1, i - 1, i + dyp,
                                  i + dym, i + dzp, i + dzm);
      }
      if (lastIsColour && lxLast > lx0)
        scalar(lxLast);
    };
    Kokkos::parallel_for("bench::smooth_peeled", Kokkos::RangePolicy<Exec>(0, (long)nyi * nzi),
                         pencil);
  }
}

// (b) == (a) bitwise check (host only).
template <class Exec = pf::CCExec>
void checkPeeled(pf::CCField xa, pf::CCField xb, pf::CCConst b, pf::FPC AC, pf::FPC AFX,
                 pf::FPC AFY, pf::FPC AFZ, pf::C3 e, pf::C3 n, pf::C3 og, int g) {
  if constexpr (std::is_same_v<typename Exec::memory_space, Kokkos::HostSpace>) {
    for (int c = 0; c < 2; ++c) {
      pf::cutcellSmoothColorFaceWrap(xa, b, AC, AFX, AFY, AFZ, e, n, og, g, c);
      smoothPeeledFp64(xb, b, AC, AFX, AFY, AFZ, e, n, og, g, c);
    }
    long ndiff = 0;
    for (long i = 0; i < (long)xa.extent(0); ++i)
      ndiff += std::memcmp(&xa(i), &xb(i), sizeof(double)) != 0;
    std::printf("P0 (b) peeled vs (a) production after 2 passes: %ld differing cells -> %s\n",
                ndiff, ndiff == 0 ? "BITWISE" : "DIFFERS");
  }
}

// (c) D-lite cell: FP32 faces, FP64 arithmetic and iterates, flux/correction form.
template <class XV, class BV, class WV>
KOKKOS_INLINE_FUNCTION void smoothLiteCell(const XV& x, const BV& b, const WV& WX, const WV& WY,
                                           const WV& WZ, long i, long sx, long sy, long sz,
                                           const pf::CcNbrs& w) {
  const double D = (((((double)WX(i + sx) + (double)WX(i)) + (double)WY(i + sy)) + (double)WY(i)) +
                    (double)WZ(i + sz)) +
                   (double)WZ(i);
  if (D == 0.0)
    return;
  const double xi = x(i);
  const double q = (double)WX(i) * (xi - x(w.xm)) + (double)WX(i + sx) * (xi - x(w.xp)) +
                   (double)WY(i) * (xi - x(w.ym)) + (double)WY(i + sy) * (xi - x(w.yp)) +
                   (double)WZ(i) * (xi - x(w.zm)) + (double)WZ(i + sz) * (xi - x(w.zp));
  x(i) = xi + (b(i) - q) / D;
}

// (d) The FP32 flux pass, §4.4.3 verbatim.
template <class XV, class BV, class WV>
KOKKOS_INLINE_FUNCTION void smoothFp32Cell(const XV& x, const BV& b, const WV& WX, const WV& WY,
                                           const WV& WZ, long i, long sx, long sy, long sz,
                                           const pf::CcNbrs& w) {
  const VReal D = ((((WX(i + sx) + WX(i)) + WY(i + sy)) + WY(i)) + WZ(i + sz)) + WZ(i);
  if (D == 0.0f)
    return;
  const VReal xi = x(i);
  const VReal q = WX(i) * (xi - x(w.xm)) + WX(i + sx) * (xi - x(w.xp)) + WY(i) * (xi - x(w.ym)) +
                  WY(i + sy) * (xi - x(w.yp)) + WZ(i) * (xi - x(w.zm)) +
                  WZ(i + sz) * (xi - x(w.zp));
  x(i) = xi + (b(i) - q) / D;
}

template <bool Fp32, class XV, class BV, class WV>
KOKKOS_INLINE_FUNCTION void fluxCell(const XV& x, const BV& b, const WV& WX, const WV& WY,
                                     const WV& WZ, long i, long sx, long sy, long sz,
                                     const pf::CcNbrs& w) {
  if constexpr (Fp32)
    smoothFp32Cell(x, b, WX, WY, WZ, i, sx, sy, sz, w);
  else
    smoothLiteCell(x, b, WX, WY, WZ, i, sx, sy, sz, w);
}

// One colour pass of a flux-form body in the production wrap launch forms (host pencil, device
// MDRange with the colour test), so (c)/(d) differ from (a) only in their bodies and operands.
template <bool Fp32, class XV, class BV>
void smoothFluxPass(XV x, BV b, FVC WX, FVC WY, FVC WZ, pf::C3 e, pf::C3 n, pf::C3 og, int g,
                    int color) {
  pf::CCExec space;
  if (kHost) {  // a plain if: nvcc rejects first captures inside if-constexpr
    const int nyi = e.y - 2 * g, nzi = e.z - 2 * g;
    auto pencil = KOKKOS_LAMBDA(long t) {
      const int ly = g + (int)(t % nyi), lz = g + (int)(t / nyi);
      const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
      const int P = (color + og.x + og.y + ly + og.z + lz) & 1;
      PECLET_FLOW_OMP_SIMD
      for (int lx = g + ((P ^ (g & 1)) & 1); lx < e.x - g; lx += 2) {
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const pf::CcNbrs w = pf::ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
        fluxCell<Fp32>(x, b, WX, WY, WZ, i, sx, sy, sz, w);
      }
    };
    Kokkos::parallel_for("bench::smooth_flux",
                         Kokkos::RangePolicy<pf::CCExec>(space, 0, (long)nyi * nzi), pencil);
  } else {
    using MD = pf::MDRange3<pf::CCExec>;
    Kokkos::parallel_for(
        "bench::smooth_flux", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
        KOKKOS_LAMBDA(int lx, int ly, int lz) {
          if (((og.x + lx + og.y + ly + og.z + lz) & 1) != color)
            return;
          const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)lx + (long)ly * sy + (long)lz * sz;
          const pf::CcNbrs w = pf::ccWrapNbrs(lx, ly, lz, n, g, i, sy, sz);
          fluxCell<Fp32>(x, b, WX, WY, WZ, i, sx, sy, sz, w);
        });
  }
}

void loadavg(const char* tag) {
  FILE* f = std::fopen("/proc/loadavg", "r");
  char buf[128] = {0};
  if (f) {
    if (!std::fgets(buf, sizeof buf, f))
      buf[0] = 0;
    std::fclose(f);
  }
  std::printf("P0 %s loadavg %s", tag, buf[0] ? buf : "?\n");
}

struct Stat {
  double mn, med, mx;
};
Stat stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return {v.front(), v[v.size() / 2], v.back()};
}

int runP0(int rounds, int passes) {
  const int conc = pf::CCExec().concurrency();
  std::printf("P0 exec %s concurrency %d rounds %d passes/round %d\n", pf::CCExec::name(), conc,
              rounds, passes);
  loadavg("start");

  // ---- t_L: empty parallel_for (launch + join), RangePolicy over `conc` and over 6144 rows ----
  for (long len : {(long)conc, 6144L}) {
    std::vector<double> us;
    const int K = 2000;
    for (int r = 0; r < rounds; ++r) {
      Kokkos::fence();
      Kokkos::Timer t;
      for (int k = 0; k < K; ++k) {
        Kokkos::parallel_for("bench::empty", Kokkos::RangePolicy<pf::CCExec>(0, len),
                             KOKKOS_LAMBDA(long){});
        Kokkos::fence();
      }
      us.push_back(t.seconds() * 1e6 / K);
    }
    const Stat s = stats(us);
    std::printf("P0 t_L empty parallel_for len %ld: min %.2f med %.2f max %.2f us/launch\n", len,
                s.mn, s.med, s.mx);
  }

  // ---- the level-0 pass ----
  const int g = 1;
  const pf::C3 n{128, 96, 64}, e{n.x + 2, n.y + 2, n.z + 2}, og{0, 0, 0};
  const long ne = (long)e.x * e.y * e.z;
  std::vector<double> rho(ne), ox(ne), oy(ne), oz(ne), bh(ne, 0.0), xh(ne, 0.0);
  auto wrap = [](int l, int nn) { return ((l - 1) % nn + nn) % nn; };  // ghost -> inner coord
  // 24 bubbles on a fixed pseudo-random lattice (deterministic LCG).
  unsigned long long s = 12345;
  auto urand = [&s]() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(s >> 11) * (1.0 / 9007199254740992.0);
  };
  std::vector<double> cx, cy, cz;
  for (int k = 0; k < 24; ++k) {
    cx.push_back(urand() * n.x);
    cy.push_back(8 + urand() * (n.y - 16));
    cz.push_back(urand() * n.z);
  }
  for (int lz = 0; lz < e.z; ++lz)
    for (int ly = 0; ly < e.y; ++ly)
      for (int lx = 0; lx < e.x; ++lx) {
        const double X = wrap(lx, n.x) + 0.5, Y = std::clamp(ly - 1, 0, n.y - 1) + 0.5,
                     Z = wrap(lz, n.z) + 0.5;
        double C = 0;  // gas fraction
        for (int k = 0; k < 24; ++k) {
          double dx = std::fabs(X - cx[k]), dz = std::fabs(Z - cz[k]);
          dx = std::min(dx, n.x - dx);
          dz = std::min(dz, n.z - dz);
          const double r = std::sqrt(dx * dx + (Y - cy[k]) * (Y - cy[k]) + dz * dz);
          C = std::max(C, std::clamp(6.5 - r, 0.0, 1.0));
        }
        rho[(long)lx + (long)ly * e.x + (long)lz * e.x * e.y] = 1.0 - C + 0.02 * C;
      }
  const long sy = e.x, sz = (long)e.x * e.y;
  for (int lz = 1; lz < e.z; ++lz)
    for (int ly = 1; ly < e.y; ++ly)
      for (int lx = 1; lx < e.x; ++lx) {
        const long i = (long)lx + ly * sy + lz * sz;
        ox[i] = 1.0 / (0.5 * (rho[i] + rho[i - 1]));
        oy[i] = (ly == 1 || ly == n.y + 1) ? 0.0 : 1.0 / (0.5 * (rho[i] + rho[i - sy]));
        oz[i] = 1.0 / (0.5 * (rho[i] + rho[i - sz]));
      }
  for (int lz = 1; lz <= n.z; ++lz)
    for (int ly = 1; ly <= n.y; ++ly)
      for (int lx = 1; lx <= n.x; ++lx) {
        const long i = (long)lx + ly * sy + lz * sz;
        bh[i] = urand() - 0.5;
        xh[i] = urand() - 0.5;
      }
  auto up = [ne](const std::vector<double>& h, const char* nm) {
    pf::CCField d(nm, ne);
    auto m = Kokkos::create_mirror_view(d);
    for (long i = 0; i < ne; ++i)
      m(i) = h[i];
    Kokkos::deep_copy(d, m);
    return d;
  };
  pf::CCField dox = up(ox, "ox"), doy = up(oy, "oy"), doz = up(oz, "oz"), b = up(bh, "b"),
              x0 = up(xh, "x0");
  pf::FPV AC("AC", ne), AFX("AFX", ne), AFY("AFY", ne), AFZ("AFZ", ne);
  pf::buildCutcellOpFace(AC, AFX, AFY, AFZ, pf::CCConst(dox), pf::CCConst(doy), pf::CCConst(doz), e,
                         g, 1.0, 1.0, 1.0);
  FV WX("WX", ne), WY("WY", ne), WZ("WZ", ne), xf("xf", ne), bf("bf", ne);
  Kokkos::parallel_for(
      "bench::fp32_data", Kokkos::RangePolicy<pf::CCExec>(0, ne), KOKKOS_LAMBDA(long i) {
        WX(i) = static_cast<VReal>(dox(i) * 1.0);
        WY(i) = static_cast<VReal>(doy(i) * 1.0);
        WZ(i) = static_cast<VReal>(doz(i) * 1.0);
        xf(i) = static_cast<VReal>(x0(i));
        bf(i) = static_cast<VReal>(b(i));
      });
  pf::CCField xa("xa", ne), xb("xb", ne), xc("xc", ne);
  Kokkos::deep_copy(xa, x0);
  Kokkos::deep_copy(xb, x0);
  Kokkos::deep_copy(xc, x0);
  Kokkos::fence();

  // (b) == (a) bitwise after two colour passes from the same start (host).
  checkPeeled(xa, xb, pf::CCConst(b), pf::FPC(AC), pf::FPC(AFX), pf::FPC(AFY), pf::FPC(AFZ), e, n,
              og, g);

  const char* names[4] = {"(a) fp64 wrap (production)", "(b) fp64 peeled", "(c) D-lite fp32 faces",
                          "(d) fp32 flux"};
  std::vector<double> ms[4];
  auto pass = [&](int v, int c) {
    switch (v) {
      case 0:
        pf::cutcellSmoothColorFaceWrap(xa, pf::CCConst(b), pf::FPC(AC), pf::FPC(AFX), pf::FPC(AFY),
                                       pf::FPC(AFZ), e, n, og, g, c);
        break;
      case 1:
        smoothPeeledFp64(xb, pf::CCConst(b), pf::FPC(AC), pf::FPC(AFX), pf::FPC(AFY), pf::FPC(AFZ),
                         e, n, og, g, c);
        break;
      case 2:
        smoothFluxPass<false>(xc, pf::CCConst(b), FVC(WX), FVC(WY), FVC(WZ), e, n, og, g, c);
        break;
      default:
        smoothFluxPass<true>(xf, FVC(bf), FVC(WX), FVC(WY), FVC(WZ), e, n, og, g, c);
    }
  };
  for (int v = 0; v < 4; ++v)  // warm-up
    for (int k = 0; k < 4; ++k)
      if (kHost || v != 1)
        pass(v, k & 1);
  Kokkos::fence();
  for (int r = 0; r < rounds; ++r)
    for (int v = 0; v < 4; ++v) {
      if (!kHost && v == 1)
        continue;
      Kokkos::fence();
      Kokkos::Timer t;
      for (int k = 0; k < passes; ++k)
        pass(v, k & 1);
      Kokkos::fence();
      ms[v].push_back(t.seconds() * 1e3 / passes);
    }
  const double upd = 0.5 * (double)n.x * n.y * n.z;
  double amin = 0;
  for (int v = 0; v < 4; ++v) {
    if (ms[v].empty()) {
      std::printf("P0 pass %-28s n/a (host-only variant)\n", names[v]);
      continue;
    }
    const Stat st = stats(ms[v]);
    if (v == 0)
      amin = st.mn;
    std::printf(
        "P0 pass %-28s min %.4f med %.4f max %.4f ms/pass | %.3f ns/updated cell | (a)/this %.3f\n",
        names[v], st.mn, st.med, st.mx, st.mn * 1e6 / upd, amin / st.mn);
  }
  loadavg("end");
  return 0;
}

}  // namespace p0bench

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  if (argc > 1 && std::strcmp(argv[1], "p0") == 0) {
    rc = p0bench::runP0(argc > 2 ? std::atoi(argv[2]) : 5, argc > 3 ? std::atoi(argv[3]) : 40);
  } else {
    rc = p0bench::runRbgs((argc > 1) ? std::atoi(argv[1]) : 128,
                          (argc > 2) ? std::atoi(argv[2]) : 200);
  }
  Kokkos::finalize();
  return rc;
}
