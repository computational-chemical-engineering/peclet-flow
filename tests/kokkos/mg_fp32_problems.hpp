/// @file
/// @brief The FP32 V-cycle test problems (column, cyl, slab, pack) and their right-hand side,
/// shared by the ctests `mg_fp32_vcycle` and `pressure_restart` (problem descriptions:
/// test_mg_fp32_vcycle.cpp's header).
#pragma once

#include <cmath>
#include <Kokkos_Core.hpp>
#include <random>
#include <string>
#include <vector>

#include "mac_cutcell_mg.hpp"

namespace peclet_flow_fp32_problems {

using namespace peclet::flow;

template <class V>
inline std::vector<typename V::non_const_value_type> toHost(const V& f) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  return std::vector<typename V::non_const_value_type>(h.data(), h.data() + h.extent(0));
}
inline CCField toDevice(const std::vector<double>& v, const char* name) {
  CCField f(name, v.size());
  auto h = Kokkos::create_mirror_view(f);
  for (std::size_t i = 0; i < v.size(); ++i)
    h(i) = v[i];
  Kokkos::deep_copy(f, h);
  return f;
}

struct Problem {
  std::string name;
  int nx = 0, ny = 0, nz = 0, levels = 0;
  int bc[6] = {0, 0, 0, 0, 0, 0};
  std::vector<double> ox, oy, oz;  // ext-indexed, inner faces set (setOpenness fills the ghosts)
  C3 ext() const { return C3{nx + 2, ny + 2, nz + 2}; }
  long idx(int x, int y, int z) const {  // 0-based inner coordinates
    return (long)(x + 1) + (long)(y + 1) * (nx + 2) + (long)(z + 1) * (long)(nx + 2) * (ny + 2);
  }
};

// open(x, y, z): the face openness at a physical face centre; rho(x, y, z) at a cell centre.
template <class Open, class Rho>
inline Problem makeProblem(const std::string& name, int nx, int ny, int nz, int levels,
                           const int bc[6], Open open, Rho rho) {
  Problem P;
  P.name = name;
  P.nx = nx;
  P.ny = ny;
  P.nz = nz;
  P.levels = levels;
  for (int f = 0; f < 6; ++f)
    P.bc[f] = bc[f];
  const C3 e = P.ext();
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  P.ox.assign(n, 0.0);
  P.oy.assign(n, 0.0);
  P.oz.assign(n, 0.0);
  auto wrap = [](int v, int m) { return (v + m) % m; };
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const double cx = x + 0.5, cy = y + 0.5, cz = z + 0.5;
        const double r0 = rho(cx, cy, cz);
        const long i = P.idx(x, y, z);
        // the minus face of cell (x, y, z) against its periodic minus neighbour
        P.ox[i] = open(cx - 0.5, cy, cz) / (0.5 * (r0 + rho(wrap(x - 1, nx) + 0.5, cy, cz)));
        P.oy[i] = open(cx, cy - 0.5, cz) / (0.5 * (r0 + rho(cx, wrap(y - 1, ny) + 0.5, cz)));
        P.oz[i] = open(cx, cy, cz - 0.5) / (0.5 * (r0 + rho(cx, cy, wrap(z - 1, nz) + 0.5)));
      }
  return P;
}

inline double clamp01(double v) {
  return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

inline Problem columnProblem() {
  const int bc[6] = {0, 0, 1, 1, 0, 0};
  const double B[5][4] = {
      {32, 20, 16, 6}, {80, 30, 40, 5}, {100, 60, 20, 7}, {50, 70, 48, 6}, {16, 48, 56, 4}};
  auto rho = [B](double x, double y, double z) {
    for (const auto& b : B) {
      const double dx = x - b[0], dy = y - b[1], dz = z - b[2];
      if (dx * dx + dy * dy + dz * dz < b[3] * b[3])
        return 0.02;
    }
    return 1.0;
  };
  return makeProblem("column", 128, 96, 64, 5, bc, [](double, double, double) { return 1.0; }, rho);
}

inline Problem cylProblem() {
  const int bc[6] = {0, 0, 0, 0, 0, 0};
  const double N = 32, c = N / 2, R = 0.25 * N;
  auto open = [=](double x, double, double z) {
    return clamp01(0.5 + std::sqrt((x - c) * (x - c) + (z - c) * (z - c)) - R);
  };
  auto rho = [=](double x, double y, double z) {
    const double dx = x - 6, dy = y - c, dz = z - 6;
    return dx * dx + dy * dy + dz * dz < 25.0 ? 1e4 : 1.0;
  };
  return makeProblem("cyl", 32, 32, 32, 4, bc, open, rho);
}

// Review finding 4: a WALLED stratified problem, so that the G-D3 checks (a)-(e) run against walls
// (the column is periodic in z; cyl and pack cap in FP64): a sharp heavy-below slab, density
// ratio 1e3, walls at +-z. FP64 FCG converges here (52 iterations) and is required to; FP64 PCG
// does not (true residual 172 after 200 iterations at 1e3, 0.085 at 1e2: the V-cycle is not an
// SPD preconditioner against a stratified wall), so for PCG (a)-(e) are n/a and (b) runs.
inline Problem slabProblem() {
  const int bc[6] = {0, 0, 0, 0, 1, 1};
  auto rho = [](double, double, double z) { return z < 16.0 ? 1e3 : 1.0; };
  return makeProblem("slab", 32, 32, 32, 4, bc, [](double, double, double) { return 1.0; }, rho);
}

inline Problem packProblem() {
  const int bc[6] = {0, 0, 0, 0, 1, 1};
  std::vector<double> C;
  std::mt19937 gen(7);
  std::uniform_real_distribution<double> jit(-2.0, 2.0);
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i) {
        C.push_back(8 + 16 * i + jit(gen));
        C.push_back(8 + 16 * j + jit(gen));
        C.push_back(8 + 16 * k + jit(gen));
      }
  auto open = [C](double x, double y, double z) {
    double d = 1e30;
    for (std::size_t s = 0; s < C.size(); s += 3) {
      auto mi = [](double v) { return v - 64.0 * std::round(v / 64.0); };
      const double dx = mi(x - C[s]), dy = mi(y - C[s + 1]), dz = mi(z - C[s + 2]);
      d = std::min(d, std::sqrt(dx * dx + dy * dy + dz * dz) - 6.5);
    }
    return clamp01(0.5 + d);
  };
  auto rho = [](double, double, double z) { return z < 32.0 ? 1e4 : 1.0; };
  return makeProblem("pack", 64, 64, 64, 5, bc, open, rho);
}

inline void setUp(CutcellMG& mg, const Problem& P, int mode) {
  mg.init(P.nx, P.ny, P.nz, P.levels);
  mg.setBoundaryConditions(P.bc);
  mg.setVcyclePrecision(mode);
  mg.setOpenness(CCConst(toDevice(P.ox, "ox")), CCConst(toDevice(P.oy, "oy")),
                 CCConst(toDevice(P.oz, "oz")), 1.0, 1.0, 1.0);
}

inline std::vector<double> rhsOf(const Problem& P) {
  const C3 e = P.ext();
  const std::size_t n = (std::size_t)e.x * e.y * e.z;
  std::vector<double> hb(n, 0.0);
  double sum = 0.0;
  long cnt = 0;
  for (int z = 0; z < P.nz; ++z)
    for (int y = 0; y < P.ny; ++y)
      for (int x = 0; x < P.nx; ++x) {
        const double v = std::sin(0.3 * x + 0.1) * std::cos(0.2 * y) + 0.1 * std::sin(0.5 * z);
        hb[P.idx(x, y, z)] = v;
        sum += v;
        ++cnt;
      }
  for (int z = 0; z < P.nz; ++z)
    for (int y = 0; y < P.ny; ++y)
      for (int x = 0; x < P.nx; ++x)
        hb[P.idx(x, y, z)] -= sum / (double)cnt;
  return hb;
}

}  // namespace peclet_flow_fp32_problems
