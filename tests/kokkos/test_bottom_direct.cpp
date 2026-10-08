/// @file
/// @brief ctest `bottom_direct`: the device pressure bottom's block-tridiagonal direct
/// preconditioner (doc/vof_step_performance_design.md §13, gate A, U1-U7).
///
/// Problems (§13.8 A), every one at a face-coefficient ratio of 50 on a single-rank two-level
/// hierarchy whose agglomerated bottom is the level under test:
///   periodic        16x12x8 all periodic: slow axis x, P = 16, b = 96, with the border;
///   periodic-P2     2x2x2 all periodic: slow axis z (the tie rule), P = 2 (treated non-periodic),
///                   every in-plane axis of length 2 (the doubled faces);
///   walls-y         16x12x8, walls on y: slow axis y, P = 12, b = 128, no border;
///   solids          the B1b solid sheet (three fluid components), periodic;
///   solids-walls-y  the same with walls on y;
///   pocket          periodic, plus a closed two-cell pocket along the slow axis whose last plane
///                   holds ONE cell (m_c = 1: the augmentation is a single diagonal entry).
///                   (The bottom operator cannot hold a one-cell component: a cell all of whose
///                   faces are closed has AC = 0 and is solid.)
/// Checks:
///   U1  the FP64 instantiation: one M on a compatible r gives ||A x - r||inf / ||r||inf <= 1e-12
///       with A the operator M factors (the stored face form with its diagonal re-summed in FP64,
///       §13.4.1), and per-component means of x <= 1e-15 max|x|;
///   U2  the FP32 (production) instantiation: the same residual <= 1e-3;
///   U3  the factor storage (Q, Y, e, s) and M(r) before the FCG's mean removal (a team
///       reduction on a device, §13.2: "the FCG's dots stay team reductions") are BITWISE
///       identical over the team sizes {32, 64, 128, 256, T_max} (device) or {1, 2, 4, 8}
///       (OpenMP); on OpenMP also the WHOLE production FCG solve (§14 H-1: the host FCG's
///       reductions are single-lane, so its bits are independent of T);
///   U4  the production FCG reaches tau in <= 3 iterations, raises no flag, its true residual
///       (stored operator) is <= 10 tau r0, x = 0 exactly on solids, per-component means of x
///       <= 4e-16 max|x|;
///   U5  a first-attempt pivot floor of 1e30 (test hook): restarts = 1, the FCG reaches tau, no
///       flag;
///   U6  a NaN face coefficient: the flag is raised (lastSolveFailed) and x = 0;
///   U7  float M against a host FP64 dense Cholesky solve of A' (the augmented matrix), on the
///       16x12x8 problems: max |z - x_ref| / max |x_ref| <= 1e-3;
///   U8  (OpenMP) the host schedule of the factor (§14 H-1, A(b)) against the team algorithm, its
///       oracle, at team sizes {1, 2, 4}: the factor storage (Q, Y, e, s, stat) BITWISE identical,
///       FP32 and FP64, with the production pivot floor and with a first-attempt floor of 1e30
///       (every pivot fails: the restart path).
/// Host sums use long double so the checks measure the device result, not the host's rounding.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <random>
#include <type_traits>
#include <vector>

#include "mac_cutcell_mg.hpp"

using namespace peclet::flow;

namespace {

constexpr bool kHost = std::is_same_v<CCMem, Kokkos::HostSpace>;

std::vector<double> toHost(CCField f) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  return std::vector<double>(h.data(), h.data() + h.extent(0));
}
std::vector<double> toHostOp(FPV f) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  std::vector<double> v(h.extent(0));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = (double)h(i);
  return v;
}
CCField toDevice(const std::vector<double>& v, const char* name) {
  CCField f(name, v.size());
  auto h = Kokkos::create_mirror_view(f);
  for (std::size_t i = 0; i < v.size(); ++i)
    h(i) = v[i];
  Kokkos::deep_copy(f, h);
  return f;
}
template <class V>
bool sameBits(const V& a, const V& b) {
  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  return ha.span() == hb.span() &&
         std::memcmp(ha.data(), hb.data(), ha.span() * sizeof(typename V::value_type)) == 0;
}

// The bottom level on the host: its face form, labels (union-find, an independent check of the
// device ones), the re-summed diagonal and the scaling of §13.4.1, and the planes.
struct HostBottom {
  C3 e{0, 0, 0}, n{0, 0, 0};
  int bc[6];
  std::vector<double> AC, AX, AY, AZ;
  std::vector<int> comp;  // ext-indexed, -1 = solid
  int nc = 0;
  std::vector<double> d, s;              // ext-indexed
  long idx(int x, int y, int z) const {  // 0-based inner coordinates
    return (long)(x + 1) + (long)(y + 1) * e.x + (long)(z + 1) * e.x * e.y;
  }
  // face f6 of inner cell c: false if it crosses a non-periodic domain face
  bool face(const int c[3], int f6, int nb[3], double& af) const {
    const int a = f6 >> 1, side = f6 & 1, nn[3] = {n.x, n.y, n.z};
    nb[0] = c[0];
    nb[1] = c[1];
    nb[2] = c[2];
    int v = c[a] + (side ? 1 : -1);
    if (v < 0 || v >= nn[a]) {
      if (bc[f6] != 0)
        return false;
      v = (v + nn[a]) % nn[a];
    }
    nb[a] = v;
    const long i = idx(c[0], c[1], c[2]);
    const long st = (a == 0) ? 1 : (a == 1 ? e.x : (long)e.x * e.y);
    const std::vector<double>& F = (a == 0) ? AX : (a == 1 ? AY : AZ);
    af = F[side ? i + st : i];
    return true;
  }
  template <class Fn>
  void forCells(Fn fn) const {
    for (int z = 0; z < n.z; ++z)
      for (int y = 0; y < n.y; ++y)
        for (int x = 0; x < n.x; ++x) {
          const int c[3] = {x, y, z};
          fn(c, idx(x, y, z));
        }
  }
  void build() {
    std::vector<long> parent(AC.size());
    for (std::size_t i = 0; i < parent.size(); ++i)
      parent[i] = (long)i;
    auto find = [&](long a) {
      while (parent[a] != a)
        a = parent[a] = parent[parent[a]];
      return a;
    };
    forCells([&](const int c[3], long i) {
      if (!(AC[i] > 1e-30))
        return;
      for (int f6 = 0; f6 < 6; ++f6) {
        int nb[3];
        double af;
        if (face(c, f6, nb, af) && af < 0.0) {
          const long j = idx(nb[0], nb[1], nb[2]);
          if (AC[j] > 1e-30)
            parent[find(i)] = find(j);
        }
      }
    });
    comp.assign(AC.size(), -1);
    std::vector<int> id(AC.size(), -1);
    nc = 0;
    forCells([&](const int*, long i) {
      if (!(AC[i] > 1e-30))
        return;
      const long r = find(i);
      if (id[r] < 0)
        id[r] = nc++;
      comp[i] = id[r];
    });
    d.assign(AC.size(), 0.0);
    s.assign(AC.size(), 0.0);
    forCells([&](const int c[3], long i) {
      double sum = 0.0;
      for (int f6 = 0; f6 < 6; ++f6) {
        int nb[3];
        double af;
        if (face(c, f6, nb, af))
          sum += af;
      }
      d[i] = -sum;
      s[i] = (comp[i] >= 0 && d[i] > 0.0) ? 1.0 / std::sqrt(d[i]) : 0.0;
    });
  }
  // (A x)_i with the re-summed diagonal (the matrix M factors), fluid rows
  double applyResummed(const std::vector<double>& x, const int c[3], long i) const {
    double v = d[i] * x[i];
    for (int f6 = 0; f6 < 6; ++f6) {
      int nb[3];
      double af;
      if (face(c, f6, nb, af))
        v += af * x[idx(nb[0], nb[1], nb[2])];
    }
    return v;
  }
  // (A x)_i with the STORED operator, read the way the device kernels read it (wrapped indices)
  double applyStored(const std::vector<double>& x, const int c[3], long i) const {
    const int nn[3] = {n.x, n.y, n.z};
    auto at = [&](int dx, int dy, int dz) {
      const int q[3] = {(c[0] + dx + nn[0]) % nn[0], (c[1] + dy + nn[1]) % nn[1],
                        (c[2] + dz + nn[2]) % nn[2]};
      return x[idx(q[0], q[1], q[2])];
    };
    const long sy = e.x, sz = (long)e.x * e.y;
    return AC[i] * x[i] + AX[i + 1] * at(1, 0, 0) + AX[i] * at(-1, 0, 0) +
           AY[i + sy] * at(0, 1, 0) + AY[i] * at(0, -1, 0) + AZ[i + sz] * at(0, 0, 1) +
           AZ[i] * at(0, 0, -1);
  }
  // per-component means of x (long double) -> max |mean| ; also max |x| on fluid, max |x| solid
  void means(const std::vector<double>& x, double& meanMax, double& xmax, double& xsolid) const {
    std::vector<long double> sum(nc, 0.0L);
    std::vector<long> cnt(nc, 0);
    xmax = xsolid = meanMax = 0.0;
    forCells([&](const int*, long i) {
      if (comp[i] >= 0) {
        sum[comp[i]] += (long double)x[i];
        ++cnt[comp[i]];
        xmax = std::max(xmax, std::fabs(x[i]));
      } else {
        xsolid = std::max(xsolid, std::fabs(x[i]));
      }
    });
    for (int c = 0; c < nc; ++c)
      meanMax = std::max(meanMax, (double)std::fabs(sum[c] / (long double)cnt[c]));
  }
};

// A compatible rhs: random on the fluid cells, each component's mean removed (long double), 0 on
// solids and ghosts.
std::vector<double> compatibleRhs(const HostBottom& H, unsigned seed) {
  std::vector<double> v(H.AC.size(), 0.0);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  std::vector<long double> sum(H.nc, 0.0L);
  std::vector<long> cnt(H.nc, 0);
  H.forCells([&](const int*, long i) {
    const double r = U(rng);
    if (H.comp[i] >= 0) {
      v[i] = r;
      sum[H.comp[i]] += r;
      ++cnt[H.comp[i]];
    }
  });
  H.forCells([&](const int*, long i) {
    if (H.comp[i] >= 0)
      v[i] = (double)((long double)v[i] - sum[H.comp[i]] / (long double)cnt[H.comp[i]]);
  });
  return v;
}

double residualResummed(const HostBottom& H, const std::vector<double>& x,
                        const std::vector<double>& r) {
  double rn = 0.0, r0 = 0.0;
  H.forCells([&](const int c[3], long i) {
    if (H.comp[i] < 0)
      return;
    r0 = std::max(r0, std::fabs(r[i]));
    rn = std::max(rn, std::fabs(H.applyResummed(x, c, i) - r[i]));
  });
  return rn / r0;
}

// x_ref = A'^{-1} r on the host in FP64: the scaled augmented matrix A~' of §13.4.1-2 assembled
// densely from the host copies (planes from the solver; labels, k_c, m_c host-computed), a dense
// Cholesky, x = S y, then each component's mean removed (long double).
std::vector<double> denseReference(const HostBottom& H, const BottomPlanes& pl,
                                   const std::vector<double>& r) {
  const int nn = pl.P * pl.b;
  auto plane = [&](const int c[3]) { return c[pl.s]; };
  auto fidx = [&](const int c[3]) { return c[pl.s] * pl.b + c[pl.a0] + pl.n[pl.a0] * c[pl.a1]; };
  std::vector<int> kc(H.nc, -1), mc(H.nc, 0);
  H.forCells([&](const int c[3], long i) {
    if (H.comp[i] >= 0)
      kc[H.comp[i]] = std::max(kc[H.comp[i]], plane(c));
  });
  H.forCells([&](const int c[3], long i) {
    if (H.comp[i] >= 0 && plane(c) == kc[H.comp[i]])
      ++mc[H.comp[i]];
  });
  std::vector<double> A((std::size_t)nn * nn, 0.0), rhs(nn, 0.0);
  std::vector<long> ext(nn);
  H.forCells([&](const int c[3], long i) {
    const int fi = fidx(c);
    ext[fi] = i;
    rhs[fi] = H.s[i] * r[i];
    A[(std::size_t)fi * nn + fi] += 1.0;
    for (int f6 = 0; f6 < 6; ++f6) {
      int nb[3];
      double af;
      if (H.face(c, f6, nb, af)) {
        const long j = H.idx(nb[0], nb[1], nb[2]);
        A[(std::size_t)fi * nn + fidx(nb)] += H.s[i] * af * H.s[j];
      }
    }
  });
  for (int fi = 0; fi < nn; ++fi)
    for (int fj = 0; fj < nn; ++fj) {
      const long i = ext[fi], j = ext[fj];
      const int ci = H.comp[i];
      if (ci >= 0 && H.comp[j] == ci && fi / pl.b == kc[ci] && fj / pl.b == kc[ci])
        A[(std::size_t)fi * nn + fj] += 1.0 / mc[ci];
    }
  for (int j = 0; j < nn; ++j) {  // in-place Cholesky (lower)
    double p = A[(std::size_t)j * nn + j];
    for (int m = 0; m < j; ++m)
      p -= A[(std::size_t)j * nn + m] * A[(std::size_t)j * nn + m];
    const double ljj = std::sqrt(p);
    A[(std::size_t)j * nn + j] = ljj;
    for (int i = j + 1; i < nn; ++i) {
      double v = A[(std::size_t)i * nn + j];
      for (int m = 0; m < j; ++m)
        v -= A[(std::size_t)i * nn + m] * A[(std::size_t)j * nn + m];
      A[(std::size_t)i * nn + j] = v / ljj;
    }
  }
  for (int i = 0; i < nn; ++i) {  // L y = rhs
    double v = rhs[i];
    for (int m = 0; m < i; ++m)
      v -= A[(std::size_t)i * nn + m] * rhs[m];
    rhs[i] = v / A[(std::size_t)i * nn + i];
  }
  for (int i = nn - 1; i >= 0; --i) {  // L^T y = .
    double v = rhs[i];
    for (int m = i + 1; m < nn; ++m)
      v -= A[(std::size_t)m * nn + i] * rhs[m];
    rhs[i] = v / A[(std::size_t)i * nn + i];
  }
  std::vector<double> x(H.AC.size(), 0.0);
  for (int fi = 0; fi < nn; ++fi)
    x[ext[fi]] = (H.comp[ext[fi]] >= 0) ? H.s[ext[fi]] * rhs[fi] : 0.0;
  std::vector<long double> sum(H.nc, 0.0L);
  std::vector<long> cnt(H.nc, 0);
  H.forCells([&](const int*, long i) {
    if (H.comp[i] >= 0) {
      sum[H.comp[i]] += x[i];
      ++cnt[H.comp[i]];
    }
  });
  H.forCells([&](const int*, long i) {
    if (H.comp[i] >= 0)
      x[i] = (double)((long double)x[i] - sum[H.comp[i]] / (long double)cnt[H.comp[i]]);
  });
  return x;
}

struct Case {
  const char* name;
  int bc[6];
  int nx, ny, nz;
  bool solids, pocket;
};

int runCase(const Case& cs) {
  const int nx = cs.nx, ny = cs.ny, nz = cs.nz;
  CutcellMG mg;
  mg.setBoundaryConditions(cs.bc);
  mg.init(nx, ny, nz, 2);
  mg.setAgglomerationMode(1);
  const C3 e{nx + 2, ny + 2, nz + 2};
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  std::vector<double> hx(n), hy(n), hz(n);
  const double lr = std::log(50.0), k = 2.0 * M_PI;
  for (int z = 0; z < e.z; ++z)
    for (int y = 0; y < e.y; ++y)
      for (int x = 0; x < e.x; ++x) {
        const std::size_t i = x + (std::size_t)y * e.x + (std::size_t)z * e.x * e.y;
        const double s = 0.5 + 0.5 * std::sin(k * x / nx) * std::cos(k * (y + 0.5 * z) / ny);
        const double t = 0.5 + 0.5 * std::cos(k * (x + z) / nz);
        hx[i] = std::exp(lr * s) / 50.0;  // in [0.02, 1]: ratio 50
        hy[i] = std::exp(lr * t) / 50.0;
        hz[i] = std::exp(lr * (1.0 - s)) / 50.0;
        const int ix = x - 1, iy = y - 1, iz = z - 1;  // the low face of fine cell (ix, iy, iz)
        if (cs.solids) {  // the B1b sheet: coarse x = 3 solid, coarse faces x = 0 and 8 closed
          if (ix == 0 || ix == 16 || ix == 6 || ix == 8)
            hx[i] = 0.0;
          if (ix == 6 || ix == 7)
            hy[i] = hz[i] = 0.0;
        }
        if (cs.pocket) {  // fine box [20,24) x [10,12) x [6,8) = coarse cells (10..11, 5, 3)
          const bool inX = ix >= 20 && ix < 24, inY = iy >= 10 && iy < 12, inZ = iz >= 6 && iz < 8;
          if ((ix == 20 || ix == 24) && inY && inZ)
            hx[i] = 0.0;
          if ((iy == 10 || iy == 12) && inX && inZ)
            hy[i] = 0.0;
          if ((iz == 6 || iz == 8) && inX && inY)
            hz[i] = 0.0;
        }
      }
  CCField ox = toDevice(hx, "ox"), oy = toDevice(hy, "oy"), oz = toDevice(hz, "oz");
  mg.setOpenness(CCConst(ox), CCConst(oy), CCConst(oz), 1.0, 1.0, 1.0);
  mg.bottomForceForTest();
  CutcellMG::Level& bt = mg.level(mg.nLevels() - 1);
  HostBottom H;
  H.e = bt.ext;
  H.n = bt.inner;
  for (int f = 0; f < 6; ++f)
    H.bc[f] = cs.bc[f];
  H.AC = toHostOp(bt.AC);
  H.AX = toHostOp(bt.AFX);
  H.AY = toHostOp(bt.AFY);
  H.AZ = toHostOp(bt.AFZ);
  H.build();
  const BottomPlanes pl = mg.directPlanes();
  const char* why = mg.directBottomIneligible();
  printf(
      "[%s] bottom %dx%dx%d: slow axis %d, P = %d, b = %d, border %d; %d fluid component(s) "
      "(host %d); %s\n",
      cs.name, bt.inner.x, bt.inner.y, bt.inner.z, pl.s, pl.P, pl.b, pl.border,
      mg.bottomComponents(), H.nc, why ? why : "eligible");
  int fails = 0;
  if (!pl.valid() || mg.bottomComponents() != H.nc) {
    printf("[%s] FAIL: no plane ordering, or the device labels disagree\n", cs.name);
    return 1;
  }
  const std::vector<double> r = compatibleRhs(H, 7);
  CCField rd = toDevice(r, "r");

  {  // U1: FP64 exactness
    auto D = mg.directFactorForTest<double>(0);
    CCField z("z", bt.n);
    mg.directApplyForTest(D, z, rd, 0);
    const auto x = toHost(z);
    const double res = residualResummed(H, x, r);
    double mm, xm, xs;
    H.means(x, mm, xm, xs);
    const bool ok = res <= 1e-12 && mm <= 1e-15 * xm && xs == 0.0;
    printf(
        "[%s] U1 FP64 M: residual %.3e, max|component mean| %.2e max|x|, max|x| solid %.1e -> "
        "%s\n",
        cs.name, res, mm / xm, xs, ok ? "ok" : "FAIL");
    fails += !ok;
  }
  std::vector<double> zf;
  {  // U2: FP32 accuracy
    auto D = mg.directFactorForTest<float>(0);
    CCField z("z", bt.n);
    mg.directApplyForTest(D, z, rd, 0);
    zf = toHost(z);
    const double res = residualResummed(H, zf, r);
    const bool ok = res <= 1e-3;
    printf("[%s] U2 FP32 M: residual %.3e -> %s\n", cs.name, res, ok ? "ok" : "FAIL");
    fails += !ok;
  }
  {  // U3: team-size independence (factor storage; M before the mean removal)
    std::vector<int> Ts =
        kHost ? std::vector<int>{1, 2, 4, 8} : std::vector<int>{32, 64, 128, 256, 0};
    BottomDirect<float> ref;
    CCField zref("zref", bt.n);
    int Tref = -1;
    bool ok = true;
    std::vector<int> used;
    for (int T : Ts) {
      int Tf = 0;
      auto D = mg.directFactorForTest<float>(T, kBottomPivotTol, &Tf);
      CCField z("z", bt.n);
      const int Ta = mg.directApplyForTest(D, z, rd, T, /*noMean=*/true);
      if ((T != 0 && (Tf != T || Ta != T))) {
        printf("[%s] U3 team size %d not available (factor %d, solve %d): skipped\n", cs.name, T,
               Tf, Ta);
        continue;
      }
      used.push_back(Ta);
      if (Tref < 0) {
        ref = D;
        Kokkos::deep_copy(zref, z);
        Tref = Ta;
        continue;
      }
      const bool same = sameBits(D.Q, ref.Q) && sameBits(D.e, ref.e) && sameBits(D.s, ref.s) &&
                        (!pl.border || sameBits(D.Y, ref.Y)) && sameBits(z, zref);
      if (!same)
        printf("[%s] U3 team size %d differs from %d\n", cs.name, Ta, Tref);
      ok = ok && same;
    }
    ok = ok && used.size() >= 2;
    printf("[%s] U3 team sizes", cs.name);
    for (int T : used)
      printf(" %d", T);
    printf(": factor + M %s -> %s\n", ok ? "bitwise identical" : "DIFFER (or < 2 sizes)",
           ok ? "ok" : "FAIL");
    fails += !ok;
  }
  if (kHost) {  // U8: the host schedule against the team algorithm, its bitwise oracle
    bool ok = true;
    std::vector<int> used;
    auto agree = [&](const auto& A, const auto& B) {
      return sameBits(A.Q, B.Q) && sameBits(A.e, B.e) && sameBits(A.s, B.s) &&
             (!pl.border || sameBits(A.Y, B.Y)) && sameBits(A.stat, B.stat);
    };
    for (int T : {1, 2, 4}) {
      for (double tau : {kBottomPivotTol, 1e30}) {  // 1e30: every first-attempt pivot fails
        int Th = 0, Tt = 0, Th64 = 0, Tt64 = 0;
        const auto Dh = mg.directFactorForTest<float>(T, tau, &Th);
        const auto Dt = mg.directFactorForTest<float>(T, tau, &Tt, /*teamAlgorithm=*/true);
        const auto Dh64 = mg.directFactorForTest<double>(T, tau, &Th64);
        const auto Dt64 = mg.directFactorForTest<double>(T, tau, &Tt64, /*teamAlgorithm=*/true);
        if (Th != T || Tt != T || Th64 != T || Tt64 != T)
          continue;
        if (tau == kBottomPivotTol)
          used.push_back(T);
        const bool same = agree(Dh, Dt) && agree(Dh64, Dt64);
        if (!same)
          printf("[%s] U8 team size %d, first pivot floor %.0e: host schedule differs\n", cs.name,
                 T, tau);
        ok = ok && same;
      }
    }
    ok = ok && used.size() >= 2;
    printf(
        "[%s] U8 host schedule vs team algorithm (FP32 + FP64, with and without a restart), "
        "team sizes",
        cs.name);
    for (int T : used)
      printf(" %d", T);
    printf(": %s -> %s\n", ok ? "bitwise identical" : "DIFFER (or < 2 sizes)", ok ? "ok" : "FAIL");
    fails += !ok;
  }
  if (kHost) {  // U3, host: the whole FCG solve is bitwise across T (single-lane reductions)
    std::vector<double> hb(bt.n, 0.0);
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    H.forCells([&](const int*, long i) { hb[i] = U(rng); });
    CCField b = toDevice(hb, "b");
    CCField xref("xref", bt.n);
    int itRef = -1, Tref = -1;
    bool ok = true;
    std::vector<int> used;
    for (int T : {1, 2, 4, 8}) {
      Kokkos::deep_copy(bt.rhs, b);
      int Tu = 0;
      const int it = mg.directSolveTeamForTest(T, &Tu);
      if (Tu != T) {
        printf("[%s] U3 FCG team size %d not available (ran %d): skipped\n", cs.name, T, Tu);
        continue;
      }
      used.push_back(T);
      if (Tref < 0) {
        Kokkos::deep_copy(xref, bt.x);
        itRef = it;
        Tref = T;
        continue;
      }
      const bool same = it == itRef && sameBits(bt.x, xref);
      if (!same)
        printf("[%s] U3 FCG team size %d differs from %d (iterations %d vs %d)\n", cs.name, T, Tref,
               it, itRef);
      ok = ok && same;
    }
    ok = ok && used.size() >= 2 && !mg.lastSolveFailed();
    printf("[%s] U3 FCG team sizes", cs.name);
    for (int T : used)
      printf(" %d", T);
    printf(": the whole solve (%d iterations) %s -> %s\n", itRef,
           ok ? "bitwise identical" : "DIFFERS (or < 2 sizes)", ok ? "ok" : "FAIL");
    fails += !ok;
  }
  auto fcg = [&](const char* tag, int itMax, int restartsWant) {
    std::vector<double> hb(bt.n, 0.0);  // a random rhs WITH component means (the kernel removes)
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    H.forCells([&](const int*, long i) { hb[i] = U(rng); });
    {
      CCField b = toDevice(hb, "b");
      Kokkos::deep_copy(bt.rhs, b);
    }
    const int it = mg.directSolveForTest();
    const int restarts = mg.directRestarts();
    const bool flag = mg.lastSolveFailed();
    const auto x = toHost(bt.x);
    std::vector<long double> bsum(H.nc, 0.0L);
    std::vector<long> bcnt(H.nc, 0);
    H.forCells([&](const int*, long i) {
      if (H.comp[i] >= 0) {
        bsum[H.comp[i]] += hb[i];
        ++bcnt[H.comp[i]];
      }
    });
    double r0 = 0.0, rn = 0.0;
    H.forCells([&](const int c[3], long i) {
      if (H.comp[i] < 0)
        return;
      const double bp =
          (double)((long double)hb[i] - bsum[H.comp[i]] / (long double)bcnt[H.comp[i]]);
      r0 = std::max(r0, std::fabs(bp));
      rn = std::max(rn, std::fabs(bp - H.applyStored(x, c, i)));
    });
    double mm, xm, xs;
    H.means(x, mm, xm, xs);
    const bool ok = it >= 1 && it <= itMax && !flag && restarts == restartsWant &&
                    rn <= 10.0 * kBottomTau * r0 && xs == 0.0 && mm <= 4e-16 * xm;
    printf(
        "[%s] %s FCG: %d iteration(s), restarts %d, flag %d, true residual %.3e r0, max|x| "
        "solid %.1e, max|component mean| %.2e max|x| -> %s\n",
        tag, cs.name, it, restarts, flag ? 1 : 0, rn / r0, xs, mm / xm, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
  };
  fails += fcg("U4", 3, 0);        // U4
  mg.directPivotTolForTest(1e30);  // U5: the first attempt fails at every pivot
  mg.directMarkStaleForTest();
  fails += fcg("U5", kBottomCap - 1, 1);
  mg.directPivotTolForTest(kBottomPivotTol);
  mg.directMarkStaleForTest();
  {  // U6: a non-finite face coefficient
    auto hA = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bt.AFX);
    const long i0 = 1 + (long)1 * bt.ext.x + (long)1 * bt.ext.x * bt.ext.y;
    long i = i0;
    while (!(hA(i) < 0) && i < (long)bt.n - 1)  // the first open x face
      ++i;
    const auto keep = hA(i);
    hA(i) = std::numeric_limits<MReal>::quiet_NaN();
    Kokkos::deep_copy(bt.AFX, hA);
    mg.directMarkStaleForTest();
    Kokkos::deep_copy(bt.rhs, rd);
    mg.directSolveForTest();
    const bool flag = mg.lastSolveFailed();
    const auto x = toHost(bt.x);
    double xmax = 0.0;
    for (double v : x)
      xmax = std::max(xmax, std::fabs(v));
    const bool ok = flag && xmax == 0.0;
    printf("[%s] U6 NaN coefficient: flag %d, max|x| %.1e -> %s\n", cs.name, flag ? 1 : 0, xmax,
           ok ? "ok" : "FAIL");
    fails += !ok;
    hA(i) = keep;
    Kokkos::deep_copy(bt.AFX, hA);
    mg.directMarkStaleForTest();
  }
  if (bt.inner.x == 16 && bt.inner.y == 12 && bt.inner.z == 8) {  // U7
    const std::vector<double> xr = denseReference(H, pl, r);
    double dm = 0.0, am = 0.0;
    H.forCells([&](const int*, long i) {
      am = std::max(am, std::fabs(xr[i]));
      dm = std::max(dm, std::fabs(zf[i] - xr[i]));
    });
    const bool ok = dm <= 1e-3 * am;
    printf("[%s] U7 FP32 M vs dense FP64 A'^-1: max diff %.3e max|x_ref| -> %s\n", cs.name, dm / am,
           ok ? "ok" : "FAIL");
    fails += !ok;
  }
  return fails;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    const Case cases[] = {
        {"periodic", {0, 0, 0, 0, 0, 0}, 32, 24, 16, false, false},
        {"periodic-P2", {0, 0, 0, 0, 0, 0}, 4, 4, 4, false, false},
        {"walls-y", {0, 0, 1, 1, 0, 0}, 32, 24, 16, false, false},
        {"solids", {0, 0, 0, 0, 0, 0}, 32, 24, 16, true, false},
        {"solids-walls-y", {0, 0, 1, 1, 0, 0}, 32, 24, 16, true, false},
        {"pocket", {0, 0, 0, 0, 0, 0}, 32, 24, 16, false, true},
    };
    for (const Case& c : cases)
      fails += runCase(c);
  }
  Kokkos::finalize();
  printf(fails ? "bottom_direct: FAIL (%d)\n" : "bottom_direct: PASS\n", fails);
  return fails ? 1 : 0;
}
