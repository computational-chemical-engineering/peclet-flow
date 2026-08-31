// WO-M — DENSE ASSEMBLY OF THE MULTIGRID V-CYCLE PRECONDITIONER.
//
// Why this exists. WO-H closed the wall-bounded MG-PCG stall and, in doing so, measured a
// *residual* failure it could not fix: on a high-contrast coefficient the V-cycle preconditioner M
// goes INDEFINITE (a negative LDL pivot of sym(M) from density ratio ~1e3 walled and ~1e4 even
// fully periodic), which no choice of the CG beta survives. WO-H attributed that to the arithmetic
// coarsening of the face coefficient (`coarsenOpenAvg`) — VOF_PLAN's S3 item. But every one of
// those measurements was taken on a hierarchy whose operator is stored in FLOAT
// (`mac_cutcell_mg.hpp` MReal), and float rounding breaks the singular row-sum identity A*1 = 0 at
// ~eps_f32 per row; under three decades of coefficient contrast that defect is ~1e-4 relative to
// the small couplings. So a float-perturbed near-singular operator is a fully sufficient
// ALTERNATIVE explanation for the same negative pivot, and nothing in the record separates the two.
//
// This probe separates them. It assembles M column by column exactly as CG builds it — one
// mean-removed unit basis vector in, one symmetric V-cycle, the result out — on the SAME code path
// the CG drivers use, and writes the dense matrix to disk. Building this file twice, with and
// without -DPECLET_FLOW_MREAL_DOUBLE, changes ONLY the operator storage precision. If the negative
// pivot disappears in double, the S3 coefficient-coarsening theory loses its evidence; if it
// survives, S3 stands as an independent mechanism.
//
// It also assembles the FINE operator A densely (same unit-vector trick) so the mechanism claim
// itself — the per-row defect of A*1 = 0 — is a measured number rather than an argument.
//
// Analysis (skew, LDL pivots, spectrum of sym(M) on the mean-free subspace) is done by the sibling
// mg_precond_analyze.py: LAPACK gives an unambiguous smallest eigenvalue, where an unpivoted LDL on
// a matrix that is singular BY CONSTRUCTION (the constant mode) can report a sign-noise pivot at
// 1e-12 that means nothing.
//
// Usage:
//   ./mg_dense_precond --n 8 --levels 3 --geom periodic|wallz --ratio 1e3 --out M.bin
//   ./mg_dense_precond --sweep --outdir <dir>        (the whole geom x ratio battery)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "mac_cutcell_mg.hpp"

namespace {

using peclet::flow::C3;
using peclet::flow::CCConst;
using peclet::flow::CCField;
using peclet::flow::CutcellMG;

struct Config {
  int n = 8;
  int levels = 3;
  int geom = 0;      // 0 = fully periodic, 1 = walls +-z
  double ratio = 1;  // density contrast of a sharp mid-height z-slab (1 = uniform coefficient)
};

const char* geomName(int g) {
  return g == 0 ? "periodic" : "wallz";
}

// Write a dense row-major n x n double matrix with a 3-int64 header {magic, rows, cols}.
void writeMatrix(const std::string& path, const std::vector<double>& a, long n) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "mg_dense_precond: cannot open %s for writing\n", path.c_str());
    return;
  }
  const std::int64_t hdr[3] = {0x50434D47444E5331LL, n, n};  // "PCMGDNS1"
  std::fwrite(hdr, sizeof(std::int64_t), 3, f);
  std::fwrite(a.data(), sizeof(double), (size_t)n * (size_t)n, f);
  std::fclose(f);
}

// One configuration: build the hierarchy, assemble A and M densely, report and write.
void run(const Config& cfg, const std::string& outdir) {
  const int nx = cfg.n, ny = cfg.n, nz = cfg.n;
  const int G = CutcellMG::G;
  const C3 ext{nx + 2 * G, ny + 2 * G, nz + 2 * G};
  const long sy = ext.x, sz = (long)ext.x * ext.y;
  const long next = (long)ext.x * ext.y * ext.z;
  const long n = (long)nx * ny * nz;

  CutcellMG mg;
  mg.init(nx, ny, nz, cfg.levels);
  int bc[6] = {0, 0, 0, 0, 0, 0};
  if (cfg.geom == 1) {
    bc[4] = 1;  // -z wall
    bc[5] = 1;  // +z wall
  }
  mg.setBoundaryConditions(bc);

  // Face coefficients c_f = open_f * rho0/rho_f, arithmetic face mean of rho (buildRhoCoeff), from
  // a SHARP mid-height slab: rho = rho0 below, rho0*ratio above. Uniform openness 1 — the geometry
  // is a box, so the ONLY thing varying is the coefficient contrast this probe is about.
  std::vector<double> rho((size_t)next, 1.0);
  for (int z = 0; z < ext.z; ++z) {
    // periodic wrap of the ghost planes onto the inner slab pattern
    int zi = ((z - G) % nz + nz) % nz;
    const double r = (zi >= nz / 2) ? cfg.ratio : 1.0;
    for (int y = 0; y < ext.y; ++y)
      for (int x = 0; x < ext.x; ++x)
        rho[(size_t)(x + (long)y * sy + (long)z * sz)] = r;
  }
  std::vector<double> hcx((size_t)next, 0.0), hcy((size_t)next, 0.0), hcz((size_t)next, 0.0);
  for (int z = G; z < ext.z - G; ++z)
    for (int y = G; y < ext.y - G; ++y)
      for (int x = G; x < ext.x - G; ++x) {
        const long i = x + (long)y * sy + (long)z * sz;
        hcx[(size_t)i] = 1.0 / (0.5 * (rho[(size_t)i] + rho[(size_t)(i - 1)]));
        hcy[(size_t)i] = 1.0 / (0.5 * (rho[(size_t)i] + rho[(size_t)(i - sy)]));
        hcz[(size_t)i] = 1.0 / (0.5 * (rho[(size_t)i] + rho[(size_t)(i - sz)]));
      }
  CCField cx("cx", next), cy("cy", next), cz("cz", next);
  auto hx = Kokkos::create_mirror_view(cx), hy = Kokkos::create_mirror_view(cy),
       hz = Kokkos::create_mirror_view(cz);
  for (long i = 0; i < next; ++i) {
    hx(i) = hcx[(size_t)i];
    hy(i) = hcy[(size_t)i];
    hz(i) = hcz[(size_t)i];
  }
  Kokkos::deep_copy(cx, hx);
  Kokkos::deep_copy(cy, hy);
  Kokkos::deep_copy(cz, hz);
  mg.setOpenness(CCConst(cx), CCConst(cy), CCConst(cz), 1.0, 1.0, 1.0);

  CutcellMG::Level& l0 = mg.level(0);
  CCField v("v", next), y("y", next);
  // Put the V-cycle on the PRODUCTION smoothing schedule. pre_/post_/bottom_ are private and are
  // written by the Krylov entry points, so one zero-iteration solvePCG with the solver's own
  // (2, 2, 12) is the setter — IbmSolver::project calls solvePCG/solveFCG with exactly those.
  {
    CCField s0("s0", next), s1("s1", next), s2("s2", next), s3("s3", next), s4("s4", next);
    mg.solvePCG(s0, s1, s2, s3, s4, y, /*maxit=*/0, 1e-8, 2, 2, 12);
  }
  auto hv = Kokkos::create_mirror_view(v), hy2 = Kokkos::create_mirror_view(y);

  std::vector<long> idx((size_t)n);  // inner ordinal -> extended linear index
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        idx[(size_t)(i + (long)j * nx + (long)k * nx * ny)] =
            (i + G) + (long)(j + G) * sy + (long)(k + G) * sz;

  std::vector<double> A((size_t)n * (size_t)n, 0.0), M((size_t)n * (size_t)n, 0.0);

  // --- the fine operator A, column by column (matvecOverlap is what solvePCG calls) -------------
  for (long c = 0; c < n; ++c) {
    Kokkos::deep_copy(hv, 0.0);
    hv(idx[(size_t)c]) = 1.0;
    Kokkos::deep_copy(v, hv);
    mg.matvecOverlap(l0, y, v);
    Kokkos::deep_copy(hy2, y);
    for (long r = 0; r < n; ++r)
      A[(size_t)r * (size_t)n + (size_t)c] = hy2(idx[(size_t)r]);
  }

  // --- the preconditioner M, column by column ---------------------------------------------------
  // Exactly solvePCG's `precond` lambda, preceded by the mean removal CG applies to the residual
  // before every preconditioner application (removeMean(l0, r) in the CG loop).
  for (long c = 0; c < n; ++c) {
    Kokkos::deep_copy(hv, 0.0);
    hv(idx[(size_t)c]) = 1.0;
    Kokkos::deep_copy(v, hv);
    mg.removeMean(l0, v);
    Kokkos::deep_copy(l0.rhs, v);
    Kokkos::deep_copy(l0.x, 0.0);
    mg.vcycle(0, /*sym=*/true);
    Kokkos::deep_copy(hy2, l0.x);
    for (long r = 0; r < n; ++r)
      M[(size_t)r * (size_t)n + (size_t)c] = hy2(idx[(size_t)r]);
  }

  // --- the mechanism number: the per-row defect of A*1 = 0 ---------------------------------------
  // Every interior row of the singular Poisson operator sums to zero EXACTLY in exact arithmetic
  // (the diagonal is minus the sum of the off-diagonals). Float storage breaks that at ~eps_f32
  // relative to the LARGEST coefficient in the row; on a row that mixes a large and a tiny coupling
  // the same absolute defect is ~eps_f32 * contrast relative to the TINY one. Report both norms.
  double rowsumAbs = 0, rowsumRelMax = 0, rowsumRelMin = 0;
  for (long r = 0; r < n; ++r) {
    double s = 0, mx = 0, mn = 1e300;
    for (long c = 0; c < n; ++c) {
      const double a = A[(size_t)r * (size_t)n + (size_t)c];
      s += a;
      if (c != r && a != 0.0) {
        mx = std::fmax(mx, std::fabs(a));
        mn = std::fmin(mn, std::fabs(a));
      }
    }
    if (mx == 0.0)
      continue;
    rowsumAbs = std::fmax(rowsumAbs, std::fabs(s));
    rowsumRelMax = std::fmax(rowsumRelMax, std::fabs(s) / mx);
    rowsumRelMin = std::fmax(rowsumRelMin, std::fabs(s) / mn);
  }

  // --- skew, printed here for a quick read (the spectrum is Python's job) ------------------------
  double fs = 0, fa = 0;
  for (long r = 0; r < n; ++r)
    for (long c = 0; c < n; ++c) {
      const double m = M[(size_t)r * (size_t)n + (size_t)c],
                   t = M[(size_t)c * (size_t)n + (size_t)r];
      fa += m * m;
      fs += (m - t) * (m - t);
    }
  const double skew = std::sqrt(fs) / std::sqrt(fa + 1e-300);

#ifdef PECLET_FLOW_MREAL_DOUBLE
  const char* prec = "double";
#else
  const char* prec = "float";
#endif
  std::printf(
      "%-8s %-8s ratio %-9.3g  skew %.4e   A*1 defect: abs %.3e  /max|a| %.3e  /min|a| %.3e\n",
      prec, geomName(cfg.geom), cfg.ratio, skew, rowsumAbs, rowsumRelMax, rowsumRelMin);
  std::fflush(stdout);

  if (!outdir.empty()) {
    char tag[256];
    std::snprintf(tag, sizeof(tag), "%s/M_%s_%s_n%d_L%d_r%.0e.bin", outdir.c_str(), prec,
                  geomName(cfg.geom), cfg.n, cfg.levels, cfg.ratio);
    writeMatrix(tag, M, n);
    std::snprintf(tag, sizeof(tag), "%s/A_%s_%s_n%d_L%d_r%.0e.bin", outdir.c_str(), prec,
                  geomName(cfg.geom), cfg.n, cfg.levels, cfg.ratio);
    writeMatrix(tag, A, n);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  std::string outdir;
  bool sweep = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : (char*)"0"; };
    if (a == "--n")
      cfg.n = std::atoi(next());
    else if (a == "--levels")
      cfg.levels = std::atoi(next());
    else if (a == "--geom")
      cfg.geom = std::strcmp(next(), "periodic") == 0 ? 0 : 1;
    else if (a == "--ratio")
      cfg.ratio = std::atof(next());
    else if (a == "--outdir")
      outdir = next();
    else if (a == "--sweep")
      sweep = true;
  }
  Kokkos::initialize(argc, argv);
  {
    if (sweep) {
      const double ratios[] = {1.0, 1e2, 1e3, 1e4, 1e5, 1e6};
      for (int g = 0; g < 2; ++g)
        for (double r : ratios) {
          Config c = cfg;
          c.geom = g;
          c.ratio = r;
          run(c, outdir);
        }
    } else {
      run(cfg, outdir);
    }
  }
  Kokkos::finalize();
  return 0;
}
