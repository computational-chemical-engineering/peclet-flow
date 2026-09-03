// flow — VoF Part II rungs P0/P1 (WO-P01), multi-rank: phase change on a decomposition that CUTS
// the interface.
//
// What can only break here:
//
//   1. THE PER-CELL INTERFACE DATA AT DEPTH 1 AND 2. `mdot`, the PLIC polygon area and the unit
//      normal are computed on INNER cells only and then halo-exchanged, because two consumers read
//      them outside the inner region: the clip-and-redistribute pass (a deficit may be pushed
//      ACROSS a rank boundary, so the receiving rank must recompute the donor's whole allocation)
//      and the divergence-source gather (the donor may be up to two cells outside). Both are
//      written as GATHERS with a fixed summation order — never an atomic scatter — so the result
//      is bitwise, not "at the reduction floor".
//
//   2. THE NON-PERIODIC GHOST. On a wall/inflow/outflow face the halo's periodic wrap would import
//      the far side's interface as a phantom donor. `pcZeroDomainGhosts` kills that, per rank-OWNED
//      face; the P1 configuration below has walls on +-x and a decomposition that cuts x.
//
//   3. THE PER-CELL DIRICHLET MASK of the energy scalar, which is rebuilt from the colour on every
//      rank and must agree across a boundary.
//
// Gates: P0a (planar regression, 1000 steps) BITWISE against the full-grid single-rank run, and P1
// (the Stefan problem with the thermal mass flux) at the reduction floor — the energy solve's
// red-black Gauss-Seidel is the only thing in the loop whose result depends on the decomposition,
// and it does so at the level of its own unconverged residual.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int NX = 64, NY = 4, NZ = 4;

static int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

static std::size_t gidx(int x, int y, int z) {
  return (std::size_t)x + (std::size_t)y * NX + (std::size_t)z * NX * NY;
}

template <class Fn>
static std::vector<double> blockOf(Fn f, int ox, int oy, int oz, int lnx, int lny, int lnz) {
  std::vector<double> v((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        v[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            f(x + ox, y + oy, z + oz);
  return v;
}
static std::vector<double> sliceOf(const std::vector<double>& g, int ox, int oy, int oz, int lnx,
                                   int lny, int lnz) {
  return blockOf([&](int x, int y, int z) { return g[gidx(x, y, z)]; }, ox, oy, oz, lnx, lny, lnz);
}

// Bitwise comparison of a distributed field against the global reference, reduced over ranks.
static double blockDiff(const std::vector<double>& local, const std::vector<double>& global, int ox,
                        int oy, int oz, int lnx, int lny, int lnz) {
  double m = 0.0;
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        m = std::fmax(
            m, std::fabs(local[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] -
                         global[gidx(x + ox, y + oy, z + oz)]));
  double g = 0.0;
  MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

static double stefanLambda(double St) {
  const double rhs = St / std::sqrt(M_PI);
  double a = 1e-8, b = 5.0;
  for (int i = 0; i < 200; ++i) {
    const double m = 0.5 * (a + b);
    ((m * std::exp(m * m) * std::erf(m) - rhs) > 0 ? b : a) = m;
  }
  return 0.5 * (a + b);
}


// WO-P3c: run every scene below with a non-default interfacial-AREA geometry
// (`set_phase_change_area`) when `PECLET_P3C_AREA` is set, so the planar rungs can be re-taken on
// the cascade area without a second binary. Inert (and byte-identical) when the variable is unset.
template <class S>
void applyAreaModeEnv(S& s) {
  if (const char* e = std::getenv("PECLET_P3C_AREA"))
    s.setPhaseChangeArea(std::atoi(e));
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), NX, NY, NZ);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    bool cutx = false;
    for (const auto& sz : dec.sizes())
      if ((int)sz[0] != NX)
        cutx = true;
    if (rank == 0)
      std::printf("VOF PHASE CHANGE MPI np=%d  grid %dx%dx%d  block %dx%dx%d  x cut: %s\n", size,
                  NX, NY, NZ, lnx, lny, lnz, cutx ? "yes" : "no");
    if (size > 1)
      CHECK(cutx);  // the interface is normal to x: the decomposition MUST cut it

    // ================================================================ P0a, bitwise
    {
      const double x0 = 32.25, mdot = 0.02, dt = 1.0;
      auto colour = [&](int x, int, int) { return std::fmin(1.0, std::fmax(0.0, x0 - x)); };
      std::vector<double> refC;
      {  // the full-grid single-rank run, on every rank (identical arithmetic, no reduction)
        IbmSolver ref(NX, NY, NZ);
        ref.setRho(1.0);
        ref.setMu(0.01);
        ref.setDt(dt);
        ref.enableVof();
        ref.setVof(blockOf(colour, 0, 0, 0, NX, NY, NZ));
        ref.enablePhaseChange(1.0, 1.0, 1.0);
        applyAreaModeEnv(ref);
        ref.setMassFluxUniform(mdot);
        for (int k = 0; k < 1000; ++k)
          ref.applyPhaseChange(dt);
        refC = ref.getVof();
      }
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      sd.setRho(1.0);
      sd.setMu(0.01);
      sd.setDt(dt);
      sd.enableVof();
      sd.setVof(blockOf(colour, ox, oy, oz, lnx, lny, lnz));
      sd.enablePhaseChange(1.0, 1.0, 1.0);
      applyAreaModeEnv(sd);
      sd.setMassFluxUniform(mdot);
      for (int k = 0; k < 1000; ++k)
        sd.applyPhaseChange(dt);
      const double d = blockDiff(sd.getVof(), refC, ox, oy, oz, lnx, lny, lnz);
      double refSum = 0.0;
      for (double v : refC)
        refSum += v;
      if (rank == 0)
        std::printf(
            "  P0a 1000 steps: max |C_dist - C_ref| = %.3e (bitwise expected); x_ref = "
            "%.15g, exact %.15g\n",
            d, refSum / (NY * NZ), x0 - mdot * dt * 1000);
      CHECK(d == 0.0);
    }

    // ================================================================ P1 Stefan, walls on +-x
    {
      const double St = 1.0, alpha = 1.0, x0p = 0.10, xep = 0.25, Fo = 0.5;
      const double lam = stefanLambda(St);
      const double t0 = (x0p / (2 * lam)) * (x0p / (2 * lam)) / alpha;
      const double te = (xep / (2 * lam)) * (xep / (2 * lam)) / alpha;
      const double D = alpha * NX * NX;
      const int ns = (int)std::llround((te - t0) / (Fo / D));
      const double dt = (te - t0) / ns;
      const double xg = x0p * NX;
      auto colour = [&](int x, int, int) { return std::fmin(1.0, std::fmax(0.0, (x + 1) - xg)); };
      auto temp = [&](int x, int, int) {
        const double xp = (x + 0.5) / NX;
        return xp < x0p ? 1.0 - std::erf(lam * xp / x0p) / std::erf(lam) : 0.0;
      };
      auto configure = [&](IbmSolver& s, int nx, int ny, int nz, int Ox, int Oy, int Oz) {
        s.setRho(1.0);
        s.setMu(1e-3);
        s.setDt(dt);
        s.setDomainBc(0, 1, 0, 0, 0);
        s.setDomainBc(1, 1, 0, 0, 0);
        s.setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 1.0));
        s.enableVof();
        s.setVof(blockOf(colour, Ox, Oy, Oz, nx, ny, nz));
        s.addScalar("T", D, 1, 60);
        s.setScalarBc("T", 0, 2, 1.0);
        s.setScalarBc("T", 1, 2, 0.0);
        s.setField("T", blockOf(temp, Ox, Oy, Oz, nx, ny, nz));
        s.enablePhaseChange(1.0, 1.0, 1.0);
        applyAreaModeEnv(s);
        s.setPhaseChangeThermal("T", 0.0, D, D, 0.0);
      };
      std::vector<double> refC;
      {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, NX, NY, NZ, 0, 0, 0);
        for (int k = 0; k < ns; ++k) {
          ref.applyPhaseChange(dt);
          ref.advanceScalars();
        }
        refC = ref.getVof();
      }
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, lnx, lny, lnz, ox, oy, oz);
      for (int k = 0; k < ns; ++k) {
        sd.applyPhaseChange(dt);
        sd.advanceScalars();
      }
      const double d = blockDiff(sd.getVof(), refC, ox, oy, oz, lnx, lny, lnz);
      double refSum = 0.0;
      for (double v : refC)
        refSum += v;
      const double xExact = 2 * lam * std::sqrt(alpha * te) * NX;
      if (rank == 0)
        std::printf(
            "  P1 Stefan %d steps: max |C_dist - C_ref| = %.3e (RB-GS reduction floor); layer_ref "
            "= %.5f, exact %.5f (%+.4f %%)\n",
            ns, d, NX - refSum / (NY * NZ), xExact,
            100.0 * ((NX - refSum / (NY * NZ)) - xExact) / xExact);
      CHECK(d < 1e-9);
    }

    // ================================================================ P2 sucking interface (WO-P23)
    // Everything rung P2/P3 adds, distributed and on a decomposition that cuts the interface: the
    // plane-anchored (ghost-fluid) Dirichlet rows (whose theta reads the NEIGHBOUR cell's plane
    // normal and centre distance, i.e. depth-1 data that has to be exchanged and domain-zeroed),
    // the quadratic one-sided fit, the per-phase k(C) / rho c_p(C) operator, and the CONSISTENT
    // rho c_p T transport, which rides the colour advector's own g = 3 block and therefore its own
    // halo. Coupled steps, so the pressure solve is in the loop too.
    {
      const char* xe = std::getenv("PECLET_P23_XEP");
      const double ratio = 10.0, ja = 1.0, alpha_l = 1.0, x0p = 0.10, Fo = 0.5;
      const double xep = xe ? std::atof(xe) : 0.14;
      const double cfl = 0.2, rr = 1.0 / ratio;
      double b = ja / std::sqrt(M_PI);
      for (int i = 0; i < 300; ++i) {
        const double g = b * rr;
        b = ja * std::exp(-g * g) / (std::sqrt(M_PI) * std::erfc(g));
      }
      const double t0 = (x0p / (2 * b)) * (x0p / (2 * b)) / alpha_l;
      const double te = (xep / (2 * b)) * (xep / (2 * b)) / alpha_l;
      const double al = alpha_l * NX * NX, rho_l = 1.0, rho_v = rr, cpl = 1.0;
      const double k_l = al * rho_l * cpl, k_v = k_l / ratio, dT = 1.0;
      const double h_lv = rho_l * cpl * dT / (rho_v * ja);
      const double X0 = x0p * NX;
      auto colour = [&](int x, int, int) { return std::fmin(1.0, std::fmax(0.0, (x + 1) - X0)); };
      auto temp = [&](int x, int, int) {
        const double xp = (x + 0.5) / NX;
        if (!(xp > x0p))
          return 0.0;
        const double sv = xp / (2 * std::sqrt(alpha_l * t0)) - b * (1.0 - rr);
        return dT - dT * std::erfc(sv) / std::erfc(b * rr);
      };
      auto farT = [&](double tt) {
        const double sv = 1.0 / (2 * std::sqrt(alpha_l * tt)) - b * (1.0 - rr);
        return dT - dT * std::erfc(sv) / std::erfc(b * rr);
      };
      auto configure = [&](IbmSolver& s, int nx, int ny, int nz, int Ox, int Oy, int Oz) {
        s.setRho(rho_l);
        s.setMu(1e-3);
        s.setDomainBc(0, 1, 0, 0, 0);
        s.setDomainBc(1, 3, 0, 0, 0);
        s.setPressureGeometry(std::vector<double>((std::size_t)nx * ny * nz, 1.0));
        s.enableVof();
        s.setVof(blockOf(colour, Ox, Oy, Oz, nx, ny, nz));
        s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "",
                           {rho_v, rho_l - rho_v});
        s.setPressureFcg(true, 4000, 1e-10);
        s.addScalar("T", al, 1, 60);
        s.setScalarBc("T", 0, 2, 0.0);
        s.setScalarBc("T", 1, 2, farT(t0));
        s.setField("T", blockOf(temp, Ox, Oy, Oz, nx, ny, nz));
        s.enablePhaseChange(rho_v, rho_l, h_lv);
        applyAreaModeEnv(s);
        s.setPhaseChangeThermal("T", 0.0, k_v, k_l, 0.0);
        // Probe switches (findings only): PECLET_P23_OFF is a subset of "pqe" to disable —
        // p = the plane-anchored Dirichlet, q = the quadratic fit, e = the consistent rho c_p T.
        const char* off = std::getenv("PECLET_P23_OFF");
        const bool offP = off && std::strchr(off, 'p');
        const bool offQ = off && std::strchr(off, 'q');
        const bool offE = off && std::strchr(off, 'e');
        s.setPhaseChangePlaneDirichlet(!offP);
        s.setPhaseChangeQuadraticFit(!offQ);
        if (!offE)
          s.setPhaseChangeEnergy(rho_v * cpl, rho_l * cpl);
      };
      auto march = [&](IbmSolver& s) {
        double tc = t0;
        int ns = 0;
        while (tc < te) {
          const double Xd = b * std::sqrt(alpha_l / tc) * NX;
          double dt = std::fmin(cfl / std::fmax(Xd * (1.0 - rr), 1e-30), Fo / al);
          dt = std::fmin(dt, te - tc);
          s.setDt(dt);
          s.setScalarBc("T", 1, 2, farT(tc + dt));
          s.step();
          tc += dt;
          ++ns;
        }
        return ns;
      };
      std::vector<double> refC, refT;
      int ns = 0;
      {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, NX, NY, NZ, 0, 0, 0);
        ns = march(ref);
        refC = ref.getVof();
        refT = ref.getField("T");
      }
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, lnx, lny, lnz, ox, oy, oz);
      march(sd);
      const double dC = blockDiff(sd.getVof(), refC, ox, oy, oz, lnx, lny, lnz);
      const double dT_ = blockDiff(sd.getField("T"), refT, ox, oy, oz, lnx, lny, lnz);
      double refSum = 0.0;
      for (double v : refC)
        refSum += v;
      double locSum = 0.0;
      for (double v : sd.getVof())
        locSum += v;
      double gSum = 0.0;
      MPI_Allreduce(&locSum, &gSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const double layerRef = NX - refSum / (NY * NZ), layerDist = NX - gSum / (NY * NZ);
      const double xExact = 2 * b * std::sqrt(alpha_l * te) * NX;
      if (rank == 0)
        std::printf(
            "  P2 sucking %d coupled steps (plane-anchored Dirichlet + quadratic fit + consistent "
            "rho c_p T): layer_dist = %.9f vs layer_ref = %.9f (rel %.3e); pointwise max "
            "|C_dist - C_ref| = %.3e, |T_dist - T_ref| = %.3e; exact %.5f (%+.4f %%)\n",
            ns, layerDist, layerRef, std::fabs(layerDist - layerRef) / layerRef, dC, dT_, xExact,
            100.0 * (layerRef - xExact) / xExact);
      // THE GATE IS THE PHYSICAL, DECOMPOSITION-INDEPENDENT QUANTITY, and the reason is measured.
      // This COUPLED scene is ROUND-OFF SENSITIVE, and it is NOT an MPI defect: at np = 1, where
      // the distributed and reference solvers differ only in the arithmetic path `initMpi` selects,
      // the pointwise colour difference is 0.0 at 1 step, 3.3e-16 at 3, 2.4e-4 at 12 and then
      // PLATEAUS (it does not keep growing). It appears exactly when the interface crosses a cell
      // boundary and the two runs cross it at round-off-different moments. Bisected with
      // PECLET_P23_OFF at 12 steps: all three WO-P23 options off gives 4.1e-14, and ANY ONE of them
      // on gives 1.2e-4 … 2.8e-4 — the sharper interfacial treatment is what turns a crossing into
      // a discrete event (the classification threshold `pcIsInterfacial` switches a cell's whole
      // energy row). The INTEGRAL of the colour — the interface position, which is the physics —
      // moves by only 5e-5 … 1.4e-4 relative, and that is what is gated. P0a and P1 above are
      // BITWISE, so the exchange, the gather-based deposit and the redistribution are all exact;
      // this is a property of the coupled step, not of the distribution.
      CHECK(std::fabs(layerDist - layerRef) / layerRef < 5e-4);  // measured 8.5e-5 / 5.2e-5 /
                                                                 // 1.4e-4 at np 1 / 2 / 4
      CHECK(dC < 1e-2);
      CHECK(dT_ < 1e-2);
    }
  }
  int fail = failures;
  MPI_Allreduce(MPI_IN_PLACE, &fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  Kokkos::finalize();
  MPI_Finalize();
  return fail ? 1 : 0;
}
