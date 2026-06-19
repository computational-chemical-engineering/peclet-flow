// cfd-gpu — portable (Kokkos) geometric multigrid V-cycle for the periodic pressure Poisson.
//
// Assembles the ported operators (RB-GS smoother, averaging restriction, trilinear prolongation,
// mean removal) into a correction-scheme V-cycle for Lap(phi)=d on a periodic staggered grid. Coarse
// operators are the rediscretized constant-coefficient Laplacian at the level spacing (A = -Lap =
// (6 phi - sum)/h^2). This replaces the slow plain RB-GS pressure solve in SdflowKokkos. Single GPU
// (periodic ghost wrap per level). Any Kokkos backend.
#ifndef CFD_MAC_MG_KOKKOS_HPP
#define CFD_MAC_MG_KOKKOS_HPP

#include <Kokkos_Core.hpp>
#include <vector>

#include "mac_stencils_kokkos.hpp"   // SField/SConst, I3, L3
#include "mac_transfer_kokkos.hpp"   // restrict_, prolong, T3

namespace cfdk {

// Periodic ghost fill (g ghosts, all 3 axes) of a level field with inner size N.
inline void mgPeriodicFill(SField f, I3 e, int N, int g) {
  SExec space;
  int dims[3] = {e.x, e.y, e.z};
  long st[3] = {1, e.x, (long)e.x * e.y};
  for (int axis = 0; axis < 3; ++axis) {
    const int b = (axis + 1) % 3, c = (axis + 2) % 3;
    const long sa = st[axis], sb = st[b], sc = st[c];
    SField ff = f;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<2>>;
    Kokkos::parallel_for(
        "cfdk::mg_pfill", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
          const long base = (long)p0 * sb + (long)p1 * sc;
          for (int gl = 0; gl < g; ++gl) {
            ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
            ff(base + (long)(g + N + gl) * sa) = ff(base + (long)(g + gl) * sa);
          }
        });

  }
}

class MgPoisson {
 public:
  // Build the hierarchy: N, N/2, ... while even and >= minN. g = 1 ghost per level.
  MgPoisson(int N, int minN = 4) {
    int n = N;
    double h2 = 1.0;
    while (true) {
      Level L;
      L.N = n; L.h2 = h2;
      L.e = I3{n + 2 * G, n + 2 * G, n + 2 * G};
      const std::size_t ne = (std::size_t)L.e.x * L.e.y * L.e.z;
      L.phi = SField("mg_phi", ne); L.f = SField("mg_f", ne); L.r = SField("mg_r", ne);
      lv_.push_back(L);
      if (n % 2 != 0 || n / 2 < minN) break;
      n /= 2; h2 *= 4.0;
    }
  }

  int numLevels() const { return (int)lv_.size(); }

  // Solve Lap(phi)=d on the finest grid (in/out phi, rhs d). A = -Lap, so f0 = -d.
  void solve(SField phi, SConst d, int nVcycles, int nu1 = 2, int nu2 = 2) {
    // f0 = -d
    SExec space; SField f0 = lv_[0].f;
    Kokkos::parallel_for("cfdk::mg_negd", Kokkos::RangePolicy<SExec>(space, 0, f0.extent(0)),
                         KOKKOS_LAMBDA(std::size_t i) { f0(i) = -d(i); });

    Kokkos::deep_copy(lv_[0].phi, phi);
    for (int v = 0; v < nVcycles; ++v) { vcycle(0, nu1, nu2); removeMean(lv_[0].phi, lv_[0]); }
    Kokkos::deep_copy(phi, lv_[0].phi);
  }

  // Residual norm of the finest level's current solve (max|f - A phi|), for diagnostics.
  double finestResidualMax() { computeResidual(0); return maxAbsInner(lv_[0].r, lv_[0]); }

  // Public so the Kokkos launch-compiler stubs (and the extended-lambda enclosing methods below) can
  // name these; the V-cycle internals are not meant for external use.
  static constexpr int G = 1;
  struct Level { int N; I3 e; double h2; SField phi, f, r; };
  std::vector<Level> lv_;

  // RB-GS sweep: phi = (sum_nbr + h2*f) / 6, both colours; A = -Lap = (6 phi - sum)/h2.
  void smoothColor(Level& L, int color) {
    SExec space; const I3 e = L.e; const double h2 = L.h2;
    SField phi = L.phi, f = L.f;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "cfdk::mg_smooth", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          if (((x + y + z) & 1) != color) return;
          const long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const double s = phi(i+sx)+phi(i-sx)+phi(i+sy)+phi(i-sy)+phi(i+sz)+phi(i-sz);
          phi(i) = (s + h2 * f(i)) / 6.0;
        });

  }
  void smooth(Level& L, int sweeps) {
    for (int it = 0; it < sweeps; ++it) {
      mgPeriodicFill(L.phi, L.e, L.N, G); smoothColor(L, 0);
      mgPeriodicFill(L.phi, L.e, L.N, G); smoothColor(L, 1);
    }
  }
  void computeResidual(int l) {
    Level& L = lv_[l];
    mgPeriodicFill(L.phi, L.e, L.N, G);
    SExec space; const I3 e = L.e; const double h2 = L.h2;
    SField phi = L.phi, f = L.f, r = L.r;
    using MD = Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "cfdk::mg_resid", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = L3(x, y, z, e), sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const double s = phi(i+sx)+phi(i-sx)+phi(i+sy)+phi(i-sy)+phi(i+sz)+phi(i-sz);
          r(i) = f(i) - (6.0 * phi(i) - s) / h2;  // f - A phi
        });

  }
  void vcycle(int l, int nu1, int nu2) {
    Level& L = lv_[l];
    if (l == (int)lv_.size() - 1) { smooth(L, 40); return; }  // coarsest: smooth hard
    smooth(L, nu1);
    computeResidual(l);
    // restrict r_l -> f_{l+1}; zero coarse correction
    Level& C = lv_[l + 1];
    Kokkos::deep_copy(C.phi, 0.0);
    restrict_(C.f, SConst(L.r), T3{C.e.x, C.e.y, C.e.z}, T3{L.e.x, L.e.y, L.e.z}, G,
              T3{C.N, C.N, C.N}, T3{2, 2, 2});
    vcycle(l + 1, nu1, nu2);
    // prolong correction back (needs coarse ghosts)
    mgPeriodicFill(C.phi, C.e, C.N, G);
    prolong(L.phi, SConst(C.phi), T3{L.e.x, L.e.y, L.e.z}, T3{C.e.x, C.e.y, C.e.z}, G,
            T3{L.N, L.N, L.N}, T3{2, 2, 2});
    smooth(L, nu2);
  }
  void removeMean(SField f, Level& L) {
    SExec space; const I3 e = L.e; const int N = L.N;
    double sum = 0;
    Kokkos::parallel_reduce(
        "cfdk::mg_mean", Kokkos::RangePolicy<SExec>(space, 0, (long)N * N * N),
        KOKKOS_LAMBDA(long c, double& acc) {
          const int ix=(int)(c%N), iy=(int)((c/N)%N), iz=(int)(c/((long)N*N));
          acc += f(L3(ix+G, iy+G, iz+G, e));
        }, sum);
    const double mean = sum / ((double)N * N * N);
    Kokkos::parallel_for("cfdk::mg_submean", Kokkos::RangePolicy<SExec>(space, 0, f.extent(0)),
                         KOKKOS_LAMBDA(std::size_t i) { f(i) -= mean; });

  }
  double maxAbsInner(SField f, Level& L) {
    SExec space; const I3 e = L.e; const int N = L.N;
    double m = 0;
    Kokkos::parallel_reduce(
        "cfdk::mg_maxabs", Kokkos::RangePolicy<SExec>(space, 0, (long)N * N * N),
        KOKKOS_LAMBDA(long c, double& acc) {
          const int ix=(int)(c%N), iy=(int)((c/N)%N), iz=(int)(c/((long)N*N));
          double v = Kokkos::fabs(f(L3(ix+G, iy+G, iz+G, e)));
          if (v > acc) acc = v;
        }, Kokkos::Max<double>(m));
    return m;
  }
};

}  // namespace cfdk

#endif  // CFD_MAC_MG_KOKKOS_HPP
