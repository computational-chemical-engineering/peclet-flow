/// @file
/// @brief ctest `mg_fp32_vcycle`: the FP32 V-cycle preconditioner's data, eligibility and kernels
/// (doc/vof_projection_cost_design.md §4, gate G-D1 of §13).
///
/// Problems (single rank; the face coefficient handed to setOpenness is c_f = open_f / rho_f with
/// rho_f the arithmetic face mean, the form the projection builds):
///   column  128x96x64, walls on y, x and z periodic: the bubble column's grid with five gas
///           spheres at rho_g / rho_l = 0.02 (a synthetic stand-in for the VoF case's field);
///   cyl     32^3 periodic, the probe's cylinder along y (R = 0.25 N) with face openness
///           clamp(0.5 + sdf, 0, 1) at the face centre, and a sharp heavy blob at ratio 1e4;
///   pack    64^3, walls on z: a 4x4x4 lattice of spheres (R = 6.5, deterministic jitter) with
///           the same face openness, and a sharp heavy slab (z < 32) at ratio 1e4 (a synthetic
///           stand-in for the probe's packing_ring.vti bed, which a C++ ctest cannot read).
/// Checks, on every non-bottom level of each problem:
///   (b) |u^T W v - v^T W u| <= 1e-6 |u| |v| max D for 10 random pairs (W: the flux form on the
///       stored FP32 weights, evaluated in long double on the host);
///   (c) max_f |w_f / t_f - 1| <= 2^-24 (t_f = open_f * gf from the level's own operands), and
///       the FP32 decoupled set (all six weights 0) equals the FP64 AC < 1e-30 set;
///   the setter's rebuild from the stored openness gives the fused build's weights bitwise; the
///   default 'fp64' allocates no FP32 data;
///   (d) eligibility is false (and names the condition) on an outflow, an overlay, an odd-dimension
///       and a single-level configuration and on a face weight below 1e-30 (the distributed case
///       is checked in the MPI ctest cutcellmg_aniso_mpi);
///   (f) on the ineligible outflow and odd-dimension configurations a PCG solve with 'auto' is
///       bitwise (x, iterations) to 'fp64' and reports FP64.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <random>
#include <string>
#include <vector>

#include "mac_cutcell_mg.hpp"

using namespace peclet::flow;

namespace {

template <class V>
std::vector<typename V::non_const_value_type> toHost(const V& f) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
  return std::vector<typename V::non_const_value_type>(h.data(), h.data() + h.extent(0));
}
CCField toDevice(const std::vector<double>& v, const char* name) {
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
Problem makeProblem(const std::string& name, int nx, int ny, int nz, int levels, const int bc[6],
                    Open open, Rho rho) {
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

double clamp01(double v) {
  return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

Problem columnProblem() {
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

Problem cylProblem() {
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

Problem packProblem() {
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

void setUp(CutcellMG& mg, const Problem& P, int mode) {
  mg.init(P.nx, P.ny, P.nz, P.levels);
  mg.setBoundaryConditions(P.bc);
  mg.setVcyclePrecision(mode);
  mg.setOpenness(CCConst(toDevice(P.ox, "ox")), CCConst(toDevice(P.oy, "oy")),
                 CCConst(toDevice(P.oz, "oz")), 1.0, 1.0, 1.0);
}

// A level's FP32 data on the host, with the wrap neighbours of ccWrapNbrs.
struct HostLevel {
  C3 e{0, 0, 0}, n{0, 0, 0};
  int g = 1;
  std::vector<float> WX, WY, WZ;
  std::vector<double> ox, oy, oz, AC;
  double gf[3] = {1, 1, 1};
  explicit HostLevel(CutcellMG::Level& lv) {
    e = lv.ext;
    n = lv.inner;
    g = lv.g;
    WX = toHost(lv.WX);
    WY = toHost(lv.WY);
    WZ = toHost(lv.WZ);
    ox = toHost(lv.ox);
    oy = toHost(lv.oy);
    oz = toHost(lv.oz);
    auto ac = toHost(lv.AC);
    AC.assign(ac.begin(), ac.end());
    gf[0] = 1.0 * (1.0 / (double)(lv.cfac.x * lv.cfac.x));
    gf[1] = 1.0 * (1.0 / (double)(lv.cfac.y * lv.cfac.y));
    gf[2] = 1.0 * (1.0 / (double)(lv.cfac.z * lv.cfac.z));
  }
  template <class Fn>
  void forInner(Fn fn) const {
    for (int lz = g; lz < e.z - g; ++lz)
      for (int ly = g; ly < e.y - g; ++ly)
        for (int lx = g; lx < e.x - g; ++lx) {
          const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
          fn(lx, ly, lz, i, ccWrapNbrs(lx, ly, lz, n, g, i, e.x, (long)e.x * e.y));
        }
  }
  // (W v)_i = sum_f w_f (v_i - v_j), long double
  long double applyDot(const std::vector<double>& u, const std::vector<double>& v) const {
    const long sy = e.x, sz = (long)e.x * e.y;
    long double s = 0;
    forInner([&](int, int, int, long i, const CcNbrs& w) {
      const long double q =
          (long double)WX[i] * (v[i] - v[w.xm]) + (long double)WX[i + 1] * (v[i] - v[w.xp]) +
          (long double)WY[i] * (v[i] - v[w.ym]) + (long double)WY[i + sy] * (v[i] - v[w.yp]) +
          (long double)WZ[i] * (v[i] - v[w.zm]) + (long double)WZ[i + sz] * (v[i] - v[w.zp]);
      s += (long double)u[i] * q;
    });
    return s;
  }
  float Dsum(long i) const {
    const long sy = e.x, sz = (long)e.x * e.y;
    return ((((WX[i + 1] + WX[i]) + WY[i + sy]) + WY[i]) + WZ[i + sz]) + WZ[i];
  }
};

std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.3g", v);
  return b;
}
int fails = 0;
void check(bool ok, const std::string& what) {
  std::printf("  %-72s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
  if (!ok)
    ++fails;
}

// (b), (c) and the rebuild identity on every non-bottom level of P.
void dataChecks(const Problem& P) {
  CutcellMG mg;
  setUp(mg, P, CutcellMG::kVcycleFp32);
  const int nl = mg.nLevels();
  check(mg.fp32VcycleIneligible(true) == nullptr, P.name + ": eligible");
  std::vector<std::vector<float>> fused;
  for (int L = 0; L + 1 < nl; ++L) {
    CutcellMG::Level& lv = mg.level(L);
    HostLevel h(lv);
    char tag[96];
    std::snprintf(tag, sizeof tag, "%s L%d %dx%dx%d", P.name.c_str(), L, h.n.x, h.n.y, h.n.z);
    // (c) the rounding of each face weight, and the decoupled sets
    double worst = 0.0;
    bool sign = true;
    for (std::size_t i = 0; i < h.WX.size(); ++i) {
      const double t[3] = {h.ox[i] * h.gf[0], h.oy[i] * h.gf[1], h.oz[i] * h.gf[2]};
      const float w[3] = {h.WX[i], h.WY[i], h.WZ[i]};
      for (int a = 0; a < 3; ++a) {
        if (t[a] != 0.0)
          worst = std::max(worst, std::fabs((double)w[a] / t[a] - 1.0));
        else if (w[a] != 0.0f)
          worst = 1.0;
        if (w[a] < 0.0f)
          sign = false;
      }
    }
    long n32 = 0, n64 = 0, h32 = 0, h64 = 0;
    mg.fp32DecoupledCounts(L, n32, n64);
    h.forInner([&](int, int, int, long i, const CcNbrs&) {
      h32 += h.Dsum(i) == 0.0f ? 1 : 0;
      h64 += h.AC[i] < 1e-30 ? 1 : 0;
    });
    check(lv.wOk && sign && worst <= 0x1p-24,
          std::string(tag) + ": (c) max|w/t - 1| = " + num(worst) + ", w >= 0");
    check(n32 == n64 && h32 == n32 && h64 == n64, std::string(tag) + ": (c) decoupled sets " +
                                                      std::to_string(n32) +
                                                      " == " + std::to_string(n64));
    // (b) symmetry of the stored-weight flux form
    std::mt19937 gen(11 + L);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    double maxD = 0.0;
    h.forInner(
        [&](int, int, int, long i, const CcNbrs&) { maxD = std::max(maxD, (double)h.Dsum(i)); });
    double worstSym = 0.0;
    for (int k = 0; k < 10; ++k) {
      std::vector<double> u(h.WX.size(), 0.0), v(h.WX.size(), 0.0);
      long double nu = 0, nv = 0;
      h.forInner([&](int, int, int, long i, const CcNbrs&) {
        u[i] = uni(gen);
        v[i] = uni(gen);
        nu += (long double)u[i] * u[i];
        nv += (long double)v[i] * v[i];
      });
      const long double d = std::fabs(h.applyDot(u, v) - h.applyDot(v, u));
      worstSym = std::max(worstSym, (double)(d / (std::sqrt(nu) * std::sqrt(nv) * maxD)));
    }
    check(worstSym <= 1e-6,
          std::string(tag) + ": (b) |u'Wv - v'Wu| / (|u||v| max D) = " + num(worstSym));
    fused.push_back(h.WX);
    fused.back().insert(fused.back().end(), h.WY.begin(), h.WY.end());
    fused.back().insert(fused.back().end(), h.WZ.begin(), h.WZ.end());
  }
  // the setter's rebuild from the stored openness == the fused build, bitwise
  mg.setVcyclePrecision(CutcellMG::kVcycleAuto);
  bool same = true;
  for (int L = 0; L + 1 < nl; ++L) {
    CutcellMG::Level& lv = mg.level(L);
    std::vector<float> w = toHost(lv.WX), wy = toHost(lv.WY), wz = toHost(lv.WZ);
    w.insert(w.end(), wy.begin(), wy.end());
    w.insert(w.end(), wz.begin(), wz.end());
    same = same && lv.wOk && w.size() == fused[L].size() &&
           std::memcmp(w.data(), fused[L].data(), w.size() * sizeof(float)) == 0;
  }
  check(same, P.name + ": the setter's rebuild == the fused build, bitwise");
  // 'fp64': no FP32 data on a fresh hierarchy
  CutcellMG m64;
  setUp(m64, P, CutcellMG::kVcycleFp64);
  bool none = true;
  for (int L = 0; L < m64.nLevels(); ++L)
    none = none && m64.level(L).WX.extent(0) == 0 && !m64.level(L).wOk;
  check(none, P.name + ": 'fp64' allocates no FP32 data");
}

// A PCG solve of P from x = 0 on a fresh hierarchy; mode as given.
struct Solve {
  int it = -1;
  bool fp32 = false;
  std::vector<double> x;
};
Solve solve(const Problem& P, int mode) {
  CutcellMG mg;
  setUp(mg, P, mode);
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
  CCField b = toDevice(hb, "b"), x("x", n), r("r", n), p("p", n), z("z", n), Ap("Ap", n);
  Solve out;
  out.it = mg.solvePCG(b, x, r, p, z, Ap, 200, 1e-8, 2, 2, 12);
  out.fp32 = mg.lastVcycleFp32();
  out.x = toHost(x);
  return out;
}

void eligibilityChecks() {
  auto ones = [](double, double, double) { return 1.0; };
  auto uni = [](double, double, double) { return 1.0; };
  // outflow
  const int bcOut[6] = {0, 0, 0, 0, 1, 3};
  Problem out = makeProblem("outflow", 16, 16, 16, 3, bcOut, ones, uni);
  // odd inner dimension on a non-bottom level (18 -> 9 on level 1 of 3)
  const int bcPer[6] = {0, 0, 0, 0, 0, 0};
  Problem odd = makeProblem("odd", 24, 20, 18, 3, bcPer, ones, uni);
  // a single level
  Problem one = makeProblem("one-level", 16, 16, 16, 1, bcPer, ones, uni);
  // a face weight below 1e-30 (one face of the periodic box)
  Problem tiny = makeProblem("tiny-face", 16, 16, 16, 3, bcPer, ones, uni);
  tiny.ox[tiny.idx(5, 5, 5)] = 1e-35;
  struct Case {
    const Problem* P;
    const char* want;
  } cases[] = {{&out, "outflow"}, {&odd, "odd"}, {&one, "single level"}, {&tiny, "face weight"}};
  for (const Case& c : cases) {
    CutcellMG mg;
    setUp(mg, *c.P, CutcellMG::kVcycleAuto);
    const char* why = mg.fp32VcycleIneligible(true);
    check(why && std::string(why).find(c.want) != std::string::npos,
          c.P->name + ": (d) ineligible: " + (why ? why : "(eligible!)"));
  }
  {  // the overlay (test hook) and a non-Krylov caller, on an otherwise eligible problem
    Problem ok = makeProblem("eligible", 16, 16, 16, 3, bcPer, ones, uni);
    CutcellMG mg;
    setUp(mg, ok, CutcellMG::kVcycleAuto);
    check(mg.fp32VcycleIneligible(true) == nullptr, "eligible 16^3: eligible");
    const char* why = mg.fp32VcycleIneligible(false);
    check(why && std::string(why).find("PCG or FCG") != std::string::npos,
          std::string("eligible 16^3, another driver: (d) ") + (why ? why : "(eligible!)"));
    mg.debugSetOverlaySolve(true);
    why = mg.fp32VcycleIneligible(true);
    check(why && std::string(why).find("overlay") != std::string::npos,
          std::string("eligible 16^3 + overlay: (d) ") + (why ? why : "(eligible!)"));
    mg.debugSetOverlaySolve(false);
  }
  // (f) 'auto' on the ineligible solvable configurations is bitwise to 'fp64'
  for (const Problem* P : {&out, &odd}) {
    const Solve a = solve(*P, CutcellMG::kVcycleFp64), b = solve(*P, CutcellMG::kVcycleAuto);
    const bool bits = a.x.size() == b.x.size() &&
                      std::memcmp(a.x.data(), b.x.data(), a.x.size() * sizeof(double)) == 0;
    check(bits && a.it == b.it && !a.fp32 && !b.fp32,
          P->name + ": (f) 'auto' == 'fp64' bitwise, it " + std::to_string(a.it) + " / " +
              std::to_string(b.it) + ", FP64 reported");
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("mg_fp32_vcycle (doc/vof_projection_cost_design.md §4, G-D1)\n");
    for (const Problem& P : {columnProblem(), cylProblem(), packProblem()})
      dataChecks(P);
    eligibilityChecks();
    std::printf("%s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails, fails == 1 ? "" : "s");
  }
  Kokkos::finalize();
  return fails ? 1 : 0;
}
