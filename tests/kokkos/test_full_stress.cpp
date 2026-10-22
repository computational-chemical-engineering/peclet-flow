// flow — the transpose half of the full viscous stress (src/full_stress.hpp): kernel vs a host
// oracle, and the a-priori operator gate G1 of doc/full_viscous_stress.md §13:
//
//   (o) the kernel against a straightforward host transcription of §5.1/§5.4 (random fields,
//       isotropic + anisotropic metric, all-fluid + random solid mask, BC row skip): bitwise on a
//       host backend, <= 1e-13 relative on a device backend (FMA contraction);
//   (a) rigid rotation, h ratio 1 : 1 : 1 and 1 : 1.5 : 2, mu sharp (cell-sampled sphere,
//       kappa = 0.1) and smeared (tanh, width 2h): max|L+T| <= 1e-12 max|L| and
//       max|L| > 1e3 max|L+T| (the defect exists and is cancelled);
//   (b) pure strain u = (g x, -g y, 0): (L+T)_x = 2 w_x g' (mu(i) - mu(i-s_x)), (L+T)_y likewise,
//       L alone half of it, to a relative 1e-13;
//   (c) simple shear u_x = g y: T_y = w_y g' (h_y/h_x)(beta_y(i,i+s_x) - beta_y(i,i-s_x)), L_y = 0;
//   (d) 100 random periodic (u, mu), mu ratio up to 1e3: telescoping |sum T_c| <= 1e-13 sum|T_c|,
//       W-symmetry |<v,Tu> - <u,Tv>| <= 1e-12 |<v,Tu>|, <u,(K+-T)u> >= -1e-12 <u,Ku> with K the
//       ibmBuildDiffusionVar bands at idiag = 0; with a random solid mask, T symmetry only;
//   (e) smooth mu and a solenoidal trigonometric u on the unit periodic cube: max-norm truncation
//       error of (L+T) against the analytic div(2 mu D), order >= 1.9 for 32->64 and 64->128; the
//       Laplacian alone does not converge (order < 0.5 on both pairs);
//   (f) uniform mu: muRangeInner returns min == max.
//
// L = -K u is the existing Laplacian part (the 7-band kernel, metric weights w_a); T carries the
// weight of the component axis w_c. Index velocity v_a = u_a t_ref / h_a (t_ref = 1), w_a =
// (h_ref/h_a)^2 with h_ref = min h.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <type_traits>
#include <vector>

#include "cut_cell_ibm.hpp"
#include "face_props.hpp"
#include "full_stress.hpp"

namespace {
int failures = 0;
#define CHECK(cond, ...)                                                              \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "CHECK failed: %s at %s:%d  ", #cond, __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);                                              \
      std::fprintf(stderr, "\n");                                                     \
      ++failures;                                                                     \
    }                                                                                 \
  } while (0)

using namespace peclet::flow;
using DV = Kokkos::View<double*, CCMem>;
constexpr bool kHost = std::is_same_v<CCMem, Kokkos::HostSpace>;
constexpr double kPi = 3.14159265358979323846;

struct Box {
  int n[3];  // inner cells
  int g = 2;
  double h[3];
  int e[3];
  long st[3];
  long cells;
  double w[3];
  Box(int nx, int ny, int nz, double hx, double hy, double hz) {
    n[0] = nx, n[1] = ny, n[2] = nz;
    h[0] = hx, h[1] = hy, h[2] = hz;
    for (int a = 0; a < 3; ++a)
      e[a] = n[a] + 2 * g;
    st[0] = 1, st[1] = e[0], st[2] = (long)e[0] * e[1];
    cells = (long)e[0] * e[1] * e[2];
    const double hr = std::min(hx, std::min(hy, hz));
    for (int a = 0; a < 3; ++a)
      w[a] = (hr / h[a]) * (hr / h[a]);
  }
  long idx(int x, int y, int z) const { return (long)x + (long)y * st[1] + (long)z * st[2]; }
  bool inner(int x, int y, int z) const {
    return x >= g && x < e[0] - g && y >= g && y < e[1] - g && z >= g && z < e[2] - g;
  }
  C3 c3() const { return C3{e[0], e[1], e[2]}; }
  // Physical position of node (x, y, z) of component c (c = 3: the cell centre).
  void pos(int c, int x, int y, int z, double* X) const {
    const int q[3] = {x, y, z};
    for (int a = 0; a < 3; ++a)
      X[a] = (q[a] - g + ((a == c) ? 0.0 : 0.5)) * h[a];
  }
  // Periodic wrap of the ghost cells (host vector).
  void wrap(std::vector<double>& f) const {
    for (int z = 0; z < e[2]; ++z)
      for (int y = 0; y < e[1]; ++y)
        for (int x = 0; x < e[0]; ++x) {
          if (inner(x, y, z))
            continue;
          auto wr = [&](int q, int a) { return g + (((q - g) % n[a]) + n[a]) % n[a]; };
          f[idx(x, y, z)] = f[idx(wr(x, 0), wr(y, 1), wr(z, 2))];
        }
  }
};

DV toDev(const std::vector<double>& h, const char* name) {
  DV d(name, h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i)
    m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}
std::vector<double> toHost(const DV& d) {
  auto m = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(m, d);
  std::vector<double> h(d.extent(0));
  for (std::size_t i = 0; i < h.size(); ++i)
    h[i] = m(i);
  return h;
}

VarFaceProps props(const Box& b, const DV& mu, int c) {
  VarFaceProps fp;
  fp.haveMu = true;
  fp.mu = CCConst(mu);
  fp.msc = b.st[c];
  fp.harmMu = false;
  fp.haveRho = false;
  fp.rhoIdtC = 0.0;
  return fp;
}

// T_c from the kernel (rs = 1, bb = 0 on entry).
std::vector<double> kernelT(const Box& b, int c, const std::vector<double>* v, const DV& mu,
                            const std::vector<std::uint8_t>* reg = nullptr, bool skipLo = false) {
  DV bb("bb", b.cells), rs("rs", b.cells);
  Kokkos::deep_copy(rs, 1.0);
  const CCConst vv[3] = {CCConst(toDev(v[0], "v0")), CCConst(toDev(v[1], "v1")),
                         CCConst(toDev(v[2], "v2"))};
  if (reg) {
    RegConst rr[3];
    for (int a = 0; a < 3; ++a) {
      RegMask r("reg", b.cells);
      auto m = Kokkos::create_mirror_view(r);
      for (long i = 0; i < b.cells; ++i)
        m(i) = reg[a][i];
      Kokkos::deep_copy(r, m);
      rr[a] = r;
    }
    addFullStressRhs<VarFaceProps, true>(c, bb, CCConst(rs), vv, props(b, mu, c), b.c3(), b.g,
                                         b.w[c], skipLo, rr);
  } else {
    addFullStressRhs<VarFaceProps, false>(c, bb, CCConst(rs), vv, props(b, mu, c), b.c3(), b.g,
                                          b.w[c], skipLo);
  }
  Kokkos::fence();
  return toHost(bb);
}

// Host oracle: §5.1 / §5.4 transcribed literally.
std::vector<double> oracleT(const Box& b, int c, const std::vector<double>* v,
                            const std::vector<double>& mu,
                            const std::vector<std::uint8_t>* reg = nullptr, bool skipLo = false) {
  std::vector<double> T(b.cells, 0.0);
  const long sc = b.st[c];
  auto beta = [&](long i, long j) {
    if (j - i == sc)
      return mu[i];
    if (i - j == sc)
      return mu[j];
    return 0.5 * (0.5 * (mu[i] + mu[j]) + 0.5 * (mu[i - sc] + mu[j - sc]));
  };
  auto R = [&](int a, long i) { return reg ? (bool)reg[a][i] : true; };
  auto N = [&](long k) {
    return R(c, k) && R(c, k + sc) ? beta(k, k + sc) * (v[c][k + sc] - v[c][k]) : 0.0;
  };
  auto Sp = [&](long i, int a) {
    const long sa = b.st[a];
    const bool inc = R(c, i) && R(c, i + sa) && R(a, i + sa) && R(a, i + sa - sc);
    return inc ? beta(i, i + sa) * (v[a][i + sa] - v[a][i + sa - sc]) : 0.0;
  };
  auto Sm = [&](long i, int a) {
    const long sa = b.st[a];
    const bool inc = R(c, i) && R(c, i - sa) && R(a, i) && R(a, i - sc);
    return inc ? beta(i, i - sa) * (v[a][i] - v[a][i - sc]) : 0.0;
  };
  const int a1 = (c == 0) ? 1 : 0, a2 = (c == 2) ? 1 : 2;
  for (int z = b.g; z < b.e[2] - b.g; ++z)
    for (int y = b.g; y < b.e[1] - b.g; ++y)
      for (int x = b.g; x < b.e[0] - b.g; ++x) {
        const int q[3] = {x, y, z};
        if (skipLo && q[c] == b.g)
          continue;
        const long i = b.idx(x, y, z);
        if (!R(c, i))
          continue;
        T[i] = b.w[c] * (((N(i) - N(i - sc)) + (Sp(i, a1) - Sm(i, a1))) + (Sp(i, a2) - Sm(i, a2)));
      }
  return T;
}

// L_c = -K_c v_c from the production 7-band kernel at idiag = 0 (metric weights on the bands).
std::vector<double> laplacian(const Box& b, int c, const std::vector<double>& vc, const DV& mu) {
  DV AC("AC", b.cells), AW("AW", b.cells), AE("AE", b.cells), AS("AS", b.cells), AN("AN", b.cells),
      AB("AB", b.cells), AT("AT", b.cells);
  ibmBuildDiffusionVar(AC, AW, AE, AS, AN, AB, AT, b.e[0], b.e[1], b.e[2], b.g, props(b, mu, c),
                       b.w[0], b.w[1], b.w[2]);
  Kokkos::fence();
  const auto ac = toHost(AC), aw = toHost(AW), ae = toHost(AE), as = toHost(AS), an = toHost(AN),
             ab = toHost(AB), at = toHost(AT);
  std::vector<double> L(b.cells, 0.0);
  for (int z = b.g; z < b.e[2] - b.g; ++z)
    for (int y = b.g; y < b.e[1] - b.g; ++y)
      for (int x = b.g; x < b.e[0] - b.g; ++x) {
        const long i = b.idx(x, y, z);
        L[i] = -(ac[i] * vc[i] + aw[i] * vc[i - 1] + ae[i] * vc[i + 1] + as[i] * vc[i - b.st[1]] +
                 an[i] * vc[i + b.st[1]] + ab[i] * vc[i - b.st[2]] + at[i] * vc[i + b.st[2]]);
      }
  return L;
}

double maxAbsInner(const Box& b, const std::vector<double>& f) {
  double m = 0.0;
  for (int z = b.g; z < b.e[2] - b.g; ++z)
    for (int y = b.g; y < b.e[1] - b.g; ++y)
      for (int x = b.g; x < b.e[0] - b.g; ++x)
        m = std::max(m, std::fabs(f[b.idx(x, y, z)]));
  return m;
}

// --- (o) kernel vs oracle ------------------------------------------------------------------------
void testOracle() {
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> U(-1.0, 1.0), M(0.0, 1.0);
  double worst = 0.0;
  long bitDiff = 0;
  for (int aniso = 0; aniso < 2; ++aniso)
    for (int solid = 0; solid < 2; ++solid)
      for (int skip = 0; skip < 2; ++skip) {
        Box b(7, 6, 5, 1.0, aniso ? 1.5 : 1.0, aniso ? 2.0 : 1.0);
        std::vector<double> v[3], mu(b.cells);
        std::vector<std::uint8_t> reg[3];
        for (int a = 0; a < 3; ++a) {
          v[a].resize(b.cells);
          reg[a].resize(b.cells);
          for (long i = 0; i < b.cells; ++i) {
            v[a][i] = U(rng);
            reg[a][i] = M(rng) < 0.8 ? 1 : 0;
          }
        }
        for (long i = 0; i < b.cells; ++i)
          mu[i] = std::pow(10.0, 3.0 * M(rng));
        const DV muD = toDev(mu, "mu");
        for (int c = 0; c < 3; ++c) {
          const auto k = kernelT(b, c, v, muD, solid ? reg : nullptr, skip);
          const auto o = oracleT(b, c, v, mu, solid ? reg : nullptr, skip);
          const double s = maxAbsInner(b, o);
          for (long i = 0; i < b.cells; ++i) {
            if (k[i] != o[i])
              ++bitDiff;
            worst = std::max(worst, std::fabs(k[i] - o[i]) / s);
          }
        }
      }
  std::printf("(o) kernel vs host oracle: %ld differing values, max rel %.3e (%s)\n", bitDiff,
              worst, kHost ? "host: bitwise required" : "device: <= 1e-13");
  if (kHost)
    CHECK(bitDiff == 0, "kernel != oracle bitwise on host");
  else
    CHECK(worst <= 1e-13, "kernel vs oracle %.3e", worst);
}

// mu fields on a non-periodic box: sharp sphere (cell-sampled) or tanh, ratio kappa = 0.1.
std::vector<double> muSphere(const Box& b, bool sharp) {
  std::vector<double> mu(b.cells);
  const double L[3] = {b.n[0] * b.h[0], b.n[1] * b.h[1], b.n[2] * b.h[2]};
  const double R = 0.3 * std::min(L[0], std::min(L[1], L[2]));
  const double hm = std::max(b.h[0], std::max(b.h[1], b.h[2]));
  const double mo = 1.0, mi = 0.1;
  for (int z = 0; z < b.e[2]; ++z)
    for (int y = 0; y < b.e[1]; ++y)
      for (int x = 0; x < b.e[0]; ++x) {
        double X[3];
        b.pos(3, x, y, z, X);
        const double r = std::sqrt((X[0] - 0.47 * L[0]) * (X[0] - 0.47 * L[0]) +
                                   (X[1] - 0.52 * L[1]) * (X[1] - 0.52 * L[1]) +
                                   (X[2] - 0.5 * L[2]) * (X[2] - 0.5 * L[2]));
        const double C =
            sharp ? (r < R ? 1.0 : 0.0) : 0.5 * (1.0 - std::tanh((r - R) / (2.0 * hm)));
        mu[b.idx(x, y, z)] = mi * C + mo * (1.0 - C);
      }
  return mu;
}

template <class F>
void fillVel(const Box& b, std::vector<double>* v, F u) {
  for (int c = 0; c < 3; ++c) {
    v[c].assign(b.cells, 0.0);
    for (int z = 0; z < b.e[2]; ++z)
      for (int y = 0; y < b.e[1]; ++y)
        for (int x = 0; x < b.e[0]; ++x) {
          double X[3];
          b.pos(c, x, y, z, X);
          v[c][b.idx(x, y, z)] = u(c, X) / b.h[c];  // t_ref = 1
        }
  }
}

// --- (a) rigid rotation -------------------------------------------------------------------------
void testRotation() {
  for (int aniso = 0; aniso < 2; ++aniso)
    for (int sharp = 0; sharp < 2; ++sharp) {
      Box b(20, 18, 16, 0.1, aniso ? 0.15 : 0.1, aniso ? 0.2 : 0.1);
      const auto mu = muSphere(b, sharp);
      const DV muD = toDev(mu, "mu");
      const double Om[3] = {0.7, -1.1, 1.3}, X0[3] = {0.9, 1.1, 0.7};
      std::vector<double> v[3];
      fillVel(b, v, [&](int c, const double* X) {
        const double r[3] = {X[0] - X0[0], X[1] - X0[1], X[2] - X0[2]};
        const double u[3] = {Om[1] * r[2] - Om[2] * r[1], Om[2] * r[0] - Om[0] * r[2],
                             Om[0] * r[1] - Om[1] * r[0]};
        return u[c];
      });
      double mL = 0.0, mS = 0.0;
      for (int c = 0; c < 3; ++c) {
        const auto L = laplacian(b, c, v[c], muD), T = kernelT(b, c, v, muD);
        std::vector<double> S(b.cells);
        for (long i = 0; i < b.cells; ++i)
          S[i] = L[i] + T[i];
        mL = std::max(mL, maxAbsInner(b, L));
        mS = std::max(mS, maxAbsInner(b, S));
      }
      std::printf("(a) rotation %s %s: max|L| %.3e  max|L+T| %.3e  ratio %.3e\n",
                  aniso ? "aniso 1:1.5:2" : "iso", sharp ? "sharp" : "tanh ", mL, mS, mS / mL);
      CHECK(mS <= 1e-12 * mL && mL > 1e3 * mS, "rotation not cancelled");
    }
}

// --- (b) pure strain, (c) simple shear ----------------------------------------------------------
void testStrainShear() {
  for (int aniso = 0; aniso < 2; ++aniso) {
    Box b(12, 10, 8, 0.1, aniso ? 0.15 : 0.1, aniso ? 0.2 : 0.1);
    const auto mu = muSphere(b, true);
    const DV muD = toDev(mu, "mu");
    const double gam = 0.375;
    // (b) u = (g x, -g y, 0)
    std::vector<double> v[3];
    fillVel(b, v, [&](int c, const double* X) {
      return c == 0 ? gam * X[0] : (c == 1 ? -gam * X[1] : 0.0);
    });
    double errS = 0.0, errL = 0.0;
    for (int c = 0; c < 2; ++c) {
      const auto L = laplacian(b, c, v[c], muD), T = kernelT(b, c, v, muD);
      const double sg = (c == 0) ? gam : -gam;
      double scale = 0.0;
      for (int z = b.g; z < b.e[2] - b.g; ++z)
        for (int y = b.g; y < b.e[1] - b.g; ++y)
          for (int x = b.g; x < b.e[0] - b.g; ++x) {
            const long i = b.idx(x, y, z);
            const double ex = b.w[c] * sg * (mu[i] - mu[i - b.st[c]]);
            scale = std::max(scale, std::fabs(ex));
            errS = std::max(errS, std::fabs((L[i] + T[i]) - 2.0 * ex));
            errL = std::max(errL, std::fabs(L[i] - ex));
          }
      errS /= scale;
      errL /= scale;
    }
    // (c) u_x = g y
    fillVel(b, v, [&](int c, const double* X) { return c == 0 ? gam * X[1] : 0.0; });
    const auto Ly = laplacian(b, 1, v[1], muD), Ty = kernelT(b, 1, v, muD);
    double errT = 0.0, scale = 0.0, mLy = 0.0;
    for (int z = b.g; z < b.e[2] - b.g; ++z)
      for (int y = b.g; y < b.e[1] - b.g; ++y)
        for (int x = b.g; x < b.e[0] - b.g; ++x) {
          const long i = b.idx(x, y, z);
          const VarFaceProps fp = props(b, muD, 1);  // host-side beta via the same formula
          (void)fp;
          const long sy = b.st[1];
          auto beta = [&](long p, long q) {
            return 0.5 * (0.5 * (mu[p] + mu[q]) + 0.5 * (mu[p - sy] + mu[q - sy]));
          };
          const double ex = b.w[1] * gam * (b.h[1] / b.h[0]) * (beta(i, i + 1) - beta(i, i - 1));
          scale = std::max(scale, std::fabs(ex));
          errT = std::max(errT, std::fabs(Ty[i] - ex));
          mLy = std::max(mLy, std::fabs(Ly[i]));
        }
    errT /= scale;
    std::printf(
        "(b) strain %s: |L+T - 2x|/s %.3e  |L - x|/s %.3e   (c) shear: |T_y - id|/s %.3e  "
        "max|L_y| %.3e\n",
        aniso ? "aniso" : "iso  ", errS, errL, errT, mLy);
    CHECK(errS <= 1e-13 && errL <= 1e-13, "pure strain identity");
    CHECK(errT <= 1e-13 && mLy == 0.0, "simple shear identity");
  }
}

// --- (d) random periodic: telescoping, W-symmetry, energy bounds ---------------------------------
void testRandom() {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> U(-1.0, 1.0), M(0.0, 1.0);
  double wTel = 0.0, wSym = 0.0, wEn = 1e300, wSymS = 0.0;
  for (int t = 0; t < 100; ++t) {
    const bool aniso = (t % 2) == 1;
    Box b(6, 8, 6, 1.0, aniso ? 1.5 : 1.0, aniso ? 2.0 : 1.0);
    const double ratio = std::pow(10.0, 3.0 * M(rng));  // up to 1e3
    std::vector<double> mu(b.cells), u[3], v[3];
    for (long i = 0; i < b.cells; ++i)
      mu[i] = 1.0 + (ratio - 1.0) * (M(rng) < 0.5 ? M(rng) : 0.0);
    b.wrap(mu);
    for (int c = 0; c < 3; ++c) {
      u[c].resize(b.cells);
      v[c].resize(b.cells);
      for (long i = 0; i < b.cells; ++i) {
        u[c][i] = U(rng);
        v[c][i] = U(rng);
      }
      b.wrap(u[c]);
      b.wrap(v[c]);
    }
    const DV muD = toDev(mu, "mu");
    double vTu = 0.0, uTv = 0.0, uKu = 0.0, uTu = 0.0, sAbs = 0.0;
    for (int c = 0; c < 3; ++c) {
      const auto Tu = kernelT(b, c, u, muD), Tv = kernelT(b, c, v, muD);
      const auto Lu = laplacian(b, c, u[c], muD);
      double sum = 0.0, sab = 0.0;
      for (long i = 0; i < b.cells; ++i) {
        sum += Tu[i];
        sab += std::fabs(Tu[i]);
        vTu += v[c][i] * Tu[i] / b.w[c];
        uTv += u[c][i] * Tv[i] / b.w[c];
        uTu += u[c][i] * Tu[i] / b.w[c];
        uKu += -u[c][i] * Lu[i] / b.w[c];
      }
      wTel = std::max(wTel, std::fabs(sum) / sab);
      sAbs += sab;
    }
    wSym = std::max(wSym, std::fabs(vTu - uTv) / std::fabs(vTu));
    wEn = std::min(wEn, std::min(uKu + uTu, uKu - uTu) / uKu);
    // with a random solid mask: symmetry only
    std::vector<std::uint8_t> reg[3];
    for (int a = 0; a < 3; ++a) {
      reg[a].resize(b.cells);
      std::vector<double> r(b.cells);
      for (long i = 0; i < b.cells; ++i)
        r[i] = M(rng) < 0.85 ? 1.0 : 0.0;
      b.wrap(r);
      for (long i = 0; i < b.cells; ++i)
        reg[a][i] = (std::uint8_t)r[i];
    }
    double svTu = 0.0, suTv = 0.0;
    for (int c = 0; c < 3; ++c) {
      const auto Tu = kernelT(b, c, u, muD, reg), Tv = kernelT(b, c, v, muD, reg);
      for (long i = 0; i < b.cells; ++i) {
        svTu += v[c][i] * Tu[i] / b.w[c];
        suTv += u[c][i] * Tv[i] / b.w[c];
      }
    }
    wSymS = std::max(wSymS, std::fabs(svTu - suTv) / std::fabs(svTu));
  }
  std::printf(
      "(d) 100 random periodic: telescoping %.3e (<=1e-13)  W-symmetry %.3e (<=1e-12)  "
      "min <u,(K+-T)u>/<u,Ku> %.3e (>=-1e-12)  solid-mask symmetry %.3e (<=1e-12)\n",
      wTel, wSym, wEn, wSymS);
  CHECK(wTel <= 1e-13, "telescoping");
  CHECK(wSym <= 1e-12, "W-symmetry");
  CHECK(wEn >= -1e-12, "energy bound");
  CHECK(wSymS <= 1e-12, "solid-mask symmetry");
}

// --- (e) smooth convergence ---------------------------------------------------------------------
struct Smooth {
  static constexpr double mu0 = 1.0, k = 2.0 * kPi;
  // mu = mu0 (1 + 0.5 sin kx sin ky cos kz)
  static double mu(const double* X) {
    return mu0 * (1.0 + 0.5 * std::sin(k * X[0]) * std::sin(k * X[1]) * std::cos(k * X[2]));
  }
  static void gradMu(const double* X, double* g) {
    const double sx = std::sin(k * X[0]), cx = std::cos(k * X[0]), sy = std::sin(k * X[1]),
                 cy = std::cos(k * X[1]), sz = std::sin(k * X[2]), cz = std::cos(k * X[2]);
    g[0] = mu0 * 0.5 * k * cx * sy * cz;
    g[1] = mu0 * 0.5 * k * sx * cy * cz;
    g[2] = -mu0 * 0.5 * k * sx * sy * sz;
  }
  // u_c = A_c * (sin along c, cos along the others), A = (1, 1, -2): div u = 0.
  static constexpr double A[3] = {1.0, 1.0, -2.0};
  static double u(int c, const double* X) {
    double p = A[c];
    for (int a = 0; a < 3; ++a)
      p *= (a == c) ? std::sin(k * X[a]) : std::cos(k * X[a]);
    return p;
  }
  // d u_c / d x_a
  static double du(int c, int a, const double* X) {
    double p = A[c] * k;
    for (int b = 0; b < 3; ++b) {
      if (b == a)
        p *= (b == c) ? std::cos(k * X[b]) : -std::sin(k * X[b]);
      else
        p *= (b == c) ? std::sin(k * X[b]) : std::cos(k * X[b]);
    }
    return p;
  }
  // div(2 mu D)_c = mu Lap u_c + sum_a d_a mu (d_a u_c + d_c u_a)   (div u = 0, Lap u = -3k^2 u)
  static double full(int c, const double* X) {
    double g[3];
    gradMu(X, g);
    double f = mu(X) * (-3.0 * k * k) * u(c, X);
    for (int a = 0; a < 3; ++a)
      f += g[a] * (du(c, a, X) + du(a, c, X));
    return f;
  }
};

void testConvergence() {
  const int Ns[3] = {32, 64, 128};
  double eF[3], eL[3];
  for (int r = 0; r < 3; ++r) {
    const int N = Ns[r];
    const double h = 1.0 / N;
    Box b(N, N, N, h, h, h);
    std::vector<double> mu(b.cells), v[3];
    for (int z = 0; z < b.e[2]; ++z)
      for (int y = 0; y < b.e[1]; ++y)
        for (int x = 0; x < b.e[0]; ++x) {
          double X[3];
          b.pos(3, x, y, z, X);
          mu[b.idx(x, y, z)] = Smooth::mu(X);
        }
    fillVel(b, v,
            [](int c, const double* X) { return Smooth::u(c, X); });  // periodic by construction
    const DV muD = toDev(mu, "mu");
    eF[r] = eL[r] = 0.0;
    for (int c = 0; c < 3; ++c) {
      const auto L = laplacian(b, c, v[c], muD), T = kernelT(b, c, v, muD);
      for (int z = b.g; z < b.e[2] - b.g; ++z)
        for (int y = b.g; y < b.e[1] - b.g; ++y)
          for (int x = b.g; x < b.e[0] - b.g; ++x) {
            const long i = b.idx(x, y, z);
            double X[3];
            b.pos(c, x, y, z, X);
            const double ex = Smooth::full(c, X);
            // index units: (L+T) = h * div(2 mu D) with t_ref = 1, w = 1
            eF[r] = std::max(eF[r], std::fabs((L[i] + T[i]) / h - ex));
            eL[r] = std::max(eL[r], std::fabs(L[i] / h - ex));
          }
    }
  }
  const double oF1 = std::log2(eF[0] / eF[1]), oF2 = std::log2(eF[1] / eF[2]);
  const double oL1 = std::log2(eL[0] / eL[1]), oL2 = std::log2(eL[1] / eL[2]);
  std::printf(
      "(e) smooth: full err %.3e %.3e %.3e  order %.3f %.3f (>=1.9);  Laplacian err %.3e "
      "%.3e %.3e  order %.3f %.3f (<0.5)\n",
      eF[0], eF[1], eF[2], oF1, oF2, eL[0], eL[1], eL[2], oL1, oL2);
  CHECK(oF1 >= 1.9 && oF2 >= 1.9, "full-stress order");
  CHECK(oL1 < 0.5 && oL2 < 0.5, "Laplacian alone converges?");
}

// --- (f) uniform mu -----------------------------------------------------------------------------
void testUniform() {
  Box b(6, 5, 4, 1.0, 1.0, 1.0);
  std::vector<double> mu(b.cells, 0.37);
  double lo = 0.0, hi = 0.0;
  muRangeInner(CCConst(toDev(mu, "mu")), b.c3(), b.g, lo, hi);
  // a value in the read ring (g-1) and one outside it (0) must be seen / ignored
  mu[b.idx(b.g - 1, b.g, b.g)] = 0.5;
  mu[b.idx(0, 0, 0)] = 9.0;
  double lo2 = 0.0, hi2 = 0.0;
  muRangeInner(CCConst(toDev(mu, "mu")), b.c3(), b.g, lo2, hi2);
  std::printf("(f) uniform: min %.17g max %.17g (equal: %s); ring value seen: max %.3g\n", lo, hi,
              lo == hi ? "yes" : "NO", hi2);
  CHECK(lo == hi && lo == 0.37, "uniform range");
  CHECK(lo2 == 0.37 && hi2 == 0.5, "range covers [g-1, e-g+1) only");
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    testOracle();
    testRotation();
    testStrainShear();
    testRandom();
    testConvergence();
    testUniform();
  }
  Kokkos::finalize();
  std::printf("full_stress_operator: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
