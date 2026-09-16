// flow — multi-rank validation of the two-phase OPEN BOUNDARIES (VoF rung V-BC, WO-R): the inflow
// colour ghost, the inletOutlet backflow rule, the out-of-domain flux mask and the per-face
// boundary liquid ledger, on a decomposition that CUTS the inflow and the outflow face.
//
// Three things can only break here:
//
//   1. **the mask must be built on GLOBAL indices.** `vof::buildOutsideMask` marks a ghost cell
//      only when its GLOBAL index is outside the global grid, exactly as `clampFill` clamps
//      globally. A per-block test would mark every rank's own block boundary and the interior rank
//      boundaries would take the algebraic flux instead of the geometric one — a silent
//      decomposition-dependent advection.
//   2. **the boundary colour must be applied only by the rank that OWNS the face**
//      (`touchesGlobalFace`, the WO-F rule). Without it every rank would inject the inflow colour
//      into its own block's -z ghost band, i.e. into the middle of the domain.
//   3. **the ledger must be a per-rank partial sum.** Each rank accumulates the boundary fluxes of
//      the faces it owns; the caller MPI_SUMs them. The budget identity then closes globally.
//
// Four configurations on 16x16x32 (the aligned ORB cuts the long z axis at np = 2 and 4). Two are
// KINEMATIC and gated BITWISE — `slug-kin` and `packing-kin`, which is where the decomposition
// exactness of this path is actually asserted; two are COUPLED and gated on conservation and
// cancellation rather than pointwise agreement, because the colour is not a Lipschitz function of
// the velocity (the long comment on case C, and docs/wo_vof_mpi_parity_gates.md):
//
//   * `slug-kin` — the WO-R gate G1 budget, KINEMATIC: uniform inflow at -z, outflow at +z, walls
//     elsewhere, the advecting field the exactly-divergence-free uniform w. A liquid slug is
//     injected and then flushed. No pressure solve and no reduction anywhere in the update, so the
//     colour field and the ledger must be BITWISE identical to the single-rank reference at every
//     np, and the global budget must close to round-off.
//
//   * `jet-coupled` — a liquid inflow into a gas domain at density ratio 100 with the full
//     coupled step (rho(C) closure, variable-density projection). What is gated BITWISE here is
//     that the inflow ghost DENSITY is the inlet fluid's on every rank that owns the face and
//     untouched on every rank that does not; the colour gets the coupled pair of gates (see below).
//
//   * `packing-kin` — V5a x V-BC COMPOSED and kinematic: the sphere packing of the case below,
//     one sphere cutting the outlet plane, driven by a prescribed velocity that varies in all
//     three directions. BITWISE at every np and thread count. This is the gate on the composed
//     cut-cell x open-boundary colour transport, and the one the coupled case cannot be.
//
//   * `packing` — the same scene under the full coupled step. Gated on the CONSERVATIVE content:
//     the WO-R budget identity, the solid-colour rule, and a signed-sum gate saying the colour
//     difference CANCELS. Its pointwise colour gate is a bound on the field going wrong, not a
//     reduction floor — the colour is not a Lipschitz function of the velocity, and the long
//     comment there carries the trace that shows why.
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
using Colo = peclet::flow::Solver<peclet::flow::Colocated>;

static constexpr int NX = 16, NY = 16, NZ = 32;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;
static constexpr double WIN = 1.0;  // inflow speed (cells/s)
static constexpr double DT = 0.2;   // WY Courant 0.2

static std::vector<double> gatherGlobal(const std::vector<double>& local, int ox, int oy, int oz,
                                        int lnx, int lny, int lnz, int rank, int size) {
  std::vector<double> global;
  if (rank == 0)
    global.assign(GCELLS, 0.0);
  for (int r = 0; r < size; ++r) {
    int meta[6] = {ox, oy, oz, lnx, lny, lnz};
    if (r == 0) {
      if (rank == 0)
        for (int z = 0; z < lnz; ++z)
          for (int y = 0; y < lny; ++y)
            std::memcpy(&global[(std::size_t)ox + (std::size_t)(y + oy) * NX +
                                (std::size_t)(z + oz) * NX * NY],
                        &local[(std::size_t)y * lnx + (std::size_t)z * lnx * lny],
                        (std::size_t)lnx * sizeof(double));
      continue;
    }
    if (rank == r) {
      MPI_Send(meta, 6, MPI_INT, 0, 100 + r, MPI_COMM_WORLD);
      MPI_Send(local.data(), (int)local.size(), MPI_DOUBLE, 0, 200 + r, MPI_COMM_WORLD);
    } else if (rank == 0) {
      MPI_Recv(meta, 6, MPI_INT, r, 100 + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<double> buf((std::size_t)meta[3] * meta[4] * meta[5]);
      MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, r, 200 + r, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      for (int z = 0; z < meta[5]; ++z)
        for (int y = 0; y < meta[4]; ++y)
          std::memcpy(&global[(std::size_t)meta[0] + (std::size_t)(y + meta[1]) * NX +
                              (std::size_t)(z + meta[2]) * NX * NY],
                      &buf[(std::size_t)y * meta[3] + (std::size_t)z * meta[3] * meta[4]],
                      (std::size_t)meta[3] * sizeof(double));
    }
  }
  return global;
}

// NaN-AWARE. `std::fmax(m, NaN)` returns m, so the obvious loop silently reports 0.000e+00 for a
// field that has gone NaN — a gate that cannot fail. Propagate instead.
static double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d == d))
      return d;  // NaN
    m = std::fmax(m, d);
  }
  return m;
}
// The SIGNED sum of the difference. `maxAbsDiff` asks whether the two fields agree; this asks
// whether whatever they disagree about was CONSERVED — a difference that cancels is colour moved
// between cells (a flux that landed differently), a difference that does not is colour created or
// destroyed, which is what a decomposition defect does. See the tolerance comment on case C.
static double signedSumDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  for (std::size_t i = 0; i < b.size(); ++i)
    s += a[i] - b[i];
  return s;
}
static long countNonFinite(const std::vector<double>& a) {
  long n = 0;
  for (double v : a)
    if (!(v - v == 0.0))
      ++n;
  return n;
}
static double sumOf(const std::vector<double>& a) {
  double s = 0;
  for (double v : a)
    s += v;
  return s;
}

// ------------------------------------------------------------------- config A: the kinematic slug
static void configureSlug(IbmSolver& s, int lnx, int lny, int lnz) {
  s.setRho(1.0);
  s.setMu(0.0);
  s.setDt(DT);
  for (int f = 0; f < 4; ++f)
    s.setDomainBc(f, 1, 0, 0, 0);
  s.setDomainBc(4, 2, 0.0, 0.0, WIN);
  s.setDomainBc(5, 3, 0, 0, 0);
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.enableVof();
  s.setVof(std::vector<double>((std::size_t)lnx * lny * lnz, 0.0));
  s.setField("w", std::vector<double>((std::size_t)lnx * lny * lnz, WIN));
  s.setVofInflow(4, 1.0);
  s.setVofBackflow(5, 0.0);
}

// ------------------------------------------------------------------- config B: the coupled jet
static void configureJet(IbmSolver& s, int lnx, int lny, int lnz) {
  const double rhoG = 1.0, rhoL = 100.0;
  s.setRho(rhoL);
  s.setMu(0.2);
  s.setDt(0.05);
  for (int f = 0; f < 4; ++f)
    s.setDomainBc(f, 1, 0, 0, 0);
  s.setDomainBc(4, 2, 0.0, 0.0, 0.5);
  s.setDomainBc(5, 3, 0, 0, 0);
  // the validated open-boundary duct settings (scripts/verify_channel_sdflow.py)
  s.setVelocityIterations(60);
  s.setPressureLevels(6);
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 1e30));
  s.enableVof();
  s.setVof(std::vector<double>((std::size_t)lnx * lny * lnz, 0.0));  // a gas domain
  s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "",
                     {rhoG, rhoL - rhoG, 0.0, 0.0});
  // THE DRIVER IS SELECTED LAST: setPropertyModel("rho", ...) fires setDensityMode, which
  // reselects Chebyshev and silently discards a driver chosen earlier (the WO-H "capped at 120"
  // tell). Without this the distributed run falls back to `presIters_` standalone V-cycles.
  s.setPressureFcg(true, 400, 1e-11);
  s.setVofInflow(4, 1.0);  // fed by liquid
  s.setVofBackflow(5, 0.0);
}

// ------------------------------ config D: the COLLOCATED bridge through an open boundary
// All-fluid duct, -z inflow fed by liquid, +z outflow, colour initially filling the upper half.
// Density ratio 1 on purpose: the colour rides as a passive tracer, so the CONSERVATION BUDGET is
// a statement about the transport and the boundary ledger alone, with no two-phase pressure
// coupling in the way.
static void configureColoJet(Colo& s, int oz, int lnx, int lny, int lnz) {
  s.setRho(1.0);
  s.setMu(0.05);
  s.setDt(0.05);
  for (int f = 0; f < 4; ++f)
    s.setDomainBc(f, 1, 0, 0, 0);
  s.setDomainBc(4, 2, 0.0, 0.0, 0.5);  // -z inflow
  s.setDomainBc(5, 3, 0, 0, 0);        // +z outflow
  s.setVelocityIterations(60);
  s.setPressureLevels(4);
  s.setPressureIterations(400);
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 1e30));
  s.enableVof();
  std::vector<double> c0((std::size_t)lnx * lny * lnz, 0.0);
  for (int z = 0; z < lnz; ++z)
    if (z + oz >= NZ / 2)
      for (int y = 0; y < lny; ++y)
        for (int x = 0; x < lnx; ++x)
          c0[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] = 1.0;
  s.setVof(c0);
  s.setVofInflow(4, 1.0);
  s.setVofBackflow(5, 0.0);
}

// ---------------------------------------------- config C: V5a x V-BC composed (WO-R2 item 2)
// A sphere array inside the open-boundary duct, one sphere CUTTING the outlet plane, so the
// cut-cell flux path and the out-of-domain donor rule act on the same faces. The SDF is built from
// GLOBAL cell centres, so every rank derives its own block's geometry from the same scene.
static std::vector<double> packingSdf(int ox, int oy, int oz, int lnx, int lny, int lnz) {
  const double sp[5][4] = {{4.5, 4.5, 11, 3.4},
                           {11.5, 11.5, 11, 3.4},
                           {4.5, 11.5, 20, 3.4},
                           {11.5, 4.5, 20, 3.4},
                           {8.0, 8.0, (double)NZ, 3.6}};  // the last one cuts the +z outlet plane
  std::vector<double> f((std::size_t)lnx * lny * lnz, 1e30);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x) {
        double d = 1e30;
        for (const auto& q : sp) {
          const double dx = x + ox + 0.5 - q[0], dy = y + oy + 0.5 - q[1], dz = z + oz + 0.5 - q[2];
          d = std::fmin(d, std::sqrt(dx * dx + dy * dy + dz * dz) - q[3]);
        }
        f[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] = d;
      }
  return f;
}
// The KINEMATIC driver for the composed scene: a smooth velocity built from GLOBAL cell centres,
// so every decomposition prescribes the SAME field and any decomposition dependence left in the
// colour transport is the transport's own. A UNIFORM field would not do: with every value equal, a
// ghost band filled from the wrong side is indistinguishable from one filled from the right side,
// and the whole halo/bridge path would go untested. It is NOT discretely solenoidal — see the
// budget note on case C0, which is why that case gates reproducibility and not conservation.
static void packingVelocity(IbmSolver& s, int ox, int oy, int oz, int lnx, int lny, int lnz) {
  const double pi = 3.14159265358979323846;
  std::vector<double> fu((std::size_t)lnx * lny * lnz, 0.0), fv(fu), fw(fu);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x) {
        const std::size_t k = (std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny;
        const double gx = x + ox + 0.5, gy = y + oy + 0.5, gz = z + oz + 0.5;
        fu[k] =
            0.10 * std::sin(2 * pi * gx / NX) * std::cos(2 * pi * gy / NY) * std::cos(pi * gz / NZ);
        fv[k] =
            0.10 * std::cos(2 * pi * gx / NX) * std::sin(2 * pi * gy / NY) * std::cos(pi * gz / NZ);
        fw[k] = 0.50 + 0.10 * std::cos(2 * pi * gx / NX) * std::cos(2 * pi * gy / NY) *
                           std::sin(pi * gz / NZ);
      }
  s.setField("u", fu);
  s.setField("v", fv);
  s.setField("w", fw);
}

static void configurePacking(IbmSolver& s, int ox, int oy, int oz, int lnx, int lny, int lnz) {
  s.setRho(1.0);
  s.setMu(0.5);
  s.setDt(0.1);
  for (int f = 0; f < 4; ++f)
    s.setDomainBc(f, 1, 0, 0, 0);
  s.setDomainBc(4, 2, 0.0, 0.0, 0.5);  // inflow at -z
  s.setDomainBc(5, 3, 0, 0, 0);        // outflow at +z
  s.setVelocityIterations(60);
  s.setPressureLevels(4);
  s.setPressureIterations(400);
  s.setSolid(packingSdf(ox, oy, oz, lnx, lny, lnz), true);
  s.enableVof();
  std::vector<double> c0((std::size_t)lnx * lny * lnz, 0.0);
  for (int z = 0; z < lnz; ++z)
    if (z + oz >= NZ / 2)
      for (int y = 0; y < lny; ++y)
        for (int x = 0; x < lnx; ++x)
          c0[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] = 1.0;
  s.setVof(c0);
  s.setVofInflow(4, 1.0);
  s.setVofBackflow(5, 0.0);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), NX, NY, NZ);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    const int gn[3] = {NX, NY, NZ};
    bool cut[3] = {false, false, false};
    for (const auto& sz : dec.sizes())
      for (int a = 0; a < 3; ++a)
        if ((int)sz[a] != gn[a])
          cut[a] = true;
    if (rank == 0)
      std::printf("VOF BC MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n", size, NX,
                  NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "", cut[2] ? "z" : "");
    if (size > 1 && !cut[2]) {
      if (rank == 0)
        std::printf("  FAIL — the decomposition does NOT cut z (the inflow/outflow axis)\n");
      fail = 1;
    }

    // ---------------------------------------------------------------- A: kinematic slug budget
    {
      // 20 slug steps then 480 more: at W = 1 and dt = 0.2 the slug's tail clears the 32-cell
      // domain by step ~182 (so the budget is fully exercised, in == out == 1024) and the run then
      // continues 300 steps INTO THE EMPTIED-DOMAIN REGIME, which is the gate on WO-R2 item 4.
      // WO-R stopped at 185 because it had to: once C is nothing but +-1e-18 round-off,
      // `wyIsMixed` called those cells mixed, the MYC normal of a ~1e-18 stencil is degenerate and
      // `plicAlpha` divided by it — measured on nvidia-cuda at np = 1 (`PECLET_VOF_BC_TRACE=1`,
      // this file): C went -inf at step 186 and NaN at 187. With the wisp guard
      // (`WyAdvector::wispEps`, 1e-8 once `enable_vof` is called) those cells are pure for
      // reconstruction and fluxed algebraically, and the drain runs indefinitely.
      const int nSlug = 20, nAfter = 480;
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configureSlug(sd, lnx, lny, lnz);
      double ledger = 0.0;
      for (int i = 0; i < nSlug + nAfter; ++i) {
        if (i == nSlug)
          sd.setVofInflow(4, 0.0);
        sd.advectVof();
        const auto v = sd.vofBcVolumes();
        if (std::getenv("PECLET_VOF_BC_TRACE") && rank == 0) {
          const auto cc = sd.getVof();
          double sm = 0;
          for (double q : cc)
            sm += q;
          std::printf("    trace step %3d: face %.6g %.6g | C sum %.6g nonfinite %ld\n", i, v[4],
                      v[5], sm, countNonFinite(cc));
        }
        for (int f = 0; f < 6; ++f)
          ledger += v[f];
      }
      const double locSum = sumOf(sd.getVof());
      double gSum = 0.0, gLedger = 0.0;
      MPI_Allreduce(&locSum, &gSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&ledger, &gLedger, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const auto tot = sd.vofBcVolumesTotal();
      double gIn = 0.0, gOut = 0.0;
      double lIn = tot[4], lOut = -tot[5];
      MPI_Allreduce(&lIn, &gIn, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&lOut, &gOut, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const std::vector<double> gc =
          gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configureSlug(ref, NX, NY, NZ);
        double rLedger = 0.0;
        for (int i = 0; i < nSlug + nAfter; ++i) {
          if (i == nSlug)
            ref.setVofInflow(4, 0.0);
          ref.advectVof();
          const auto v = ref.vofBcVolumes();
          for (int f = 0; f < 6; ++f)
            rLedger += v[f];
        }
        const double dc = maxAbsDiff(gc, ref.getVof());
        const long nbad = countNonFinite(gc) + countNonFinite(ref.getVof());
        const double dl = std::fabs(gLedger - rLedger);
        if (nbad != 0) {
          std::printf("  [slug-kin  np=%d] FAIL — %ld non-finite colour cells\n", size, nbad);
          fail = 1;
        }
        if (!(gLedger == gLedger) || !(gIn == gIn) || !(gOut == gOut)) {
          std::printf("  [slug-kin  np=%d] FAIL — the boundary ledger is not finite\n", size);
          fail = 1;
        }
        const double budget = std::fabs(gSum - gLedger);
        std::printf("  [slug-kin  np=%d] colour vs single-rank %.3e (BITWISE required)\n", size,
                    dc);
        std::printf(
            "  [slug-kin  np=%d] ledger vs single-rank %.3e; global budget "
            "|sum(C) - ledger| %.3e; in %.10g out %.10g\n",
            size, dl, budget, gIn, gOut);
        if (!(dc == 0.0)) {
          std::printf("  [slug-kin  np=%d] FAIL — colour is not bitwise\n", size);
          fail = 1;
        }
        if (!(budget < 1e-8)) {
          std::printf("  [slug-kin  np=%d] FAIL — the global budget does not close\n", size);
          fail = 1;
        }
        if (!(std::fabs(gIn - gOut) < 1e-9 * gIn) || !(gIn > 0.0)) {
          std::printf("  [slug-kin  np=%d] FAIL — in/out do not balance\n", size);
          fail = 1;
        }
      }
    }

    // ------------------------------------------------------- B: the coupled jet + the rho ghost
    {
      const int steps = 30;
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configureJet(sd, lnx, lny, lnz);
      long itmax = 0;
      double divmax = 0.0;
      for (int i = 0; i < steps; ++i) {
        sd.step();
        itmax = std::max<long>(itmax, sd.lastPressureIterations());
        divmax = std::fmax(divmax, sd.maxOpenDivergence());
      }
      // the inflow ghost density: rho_liquid on a rank that OWNS the -z face, untouched elsewhere
      auto rv = sd.fieldView("rho");
      auto hr = Kokkos::create_mirror_view(rv);
      Kokkos::deep_copy(hr, rv);
      const int G = 2, ex = lnx + 2 * G, ey = lny + 2 * G;
      const double ghost = hr((std::size_t)(G + lnx / 2) + (std::size_t)(G + lny / 2) * ex +
                              (std::size_t)(G - 1) * ex * ey);
      const bool owns = (oz == 0);
      int bad = (owns && std::fabs(ghost - 100.0) > 1e-9) ? 1 : 0;
      int badAll = 0;
      MPI_Allreduce(&bad, &badAll, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      const std::vector<double> gc =
          gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      long gItmax = 0;
      MPI_Allreduce(&itmax, &gItmax, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configureJet(ref, NX, NY, NZ);
        for (int i = 0; i < steps; ++i)
          ref.step();
        const double dc = maxAbsDiff(gc, ref.getVof());
        const double dsum = signedSumDiff(gc, ref.getVof());
        const double csum = sumOf(ref.getVof());
        std::printf(
            "  [jet-coupl np=%d] colour vs single-rank %.3e (signed sum %.3e); inflow rho ghost "
            "owners wrong on %d rank(s); pressure %ld/400, max|div| %.3e\n",
            size, dc, dsum, badAll, gItmax, divmax);
        // The SAME pair of gates as case C, for the same reason and with the same derivation — see
        // the long comment there. This case happens to sit at 1.3e-15 today, but it is a coupled
        // run through an inflow face whose colour front is axis-aligned, i.e. exactly the
        // configuration in which `mycNormal`'s estimator selection ties; nothing about it earns a
        // tighter pointwise gate than case C, and a tighter one here would only mean this file
        // fails on a different machine instead of the same one.
        const double tol = (size == 1) ? 0.0 : 1e-6;
        if (!(dc <= tol)) {
          std::printf("  [jet-coupl np=%d] FAIL — the colour field is wrong (tol %.1e)\n", size,
                      tol);
          fail = 1;
        }
        const double tolSum = (size == 1) ? 0.0 : 1e-11 * std::fabs(csum);
        if (!(std::fabs(dsum) <= tolSum)) {
          std::printf(
              "  [jet-coupl np=%d] FAIL — the colour difference does not CANCEL: signed sum %.3e "
              "> %.3e\n",
              size, dsum, tolSum);
          fail = 1;
        }
        if (badAll != 0) {
          std::printf("  [jet-coupl np=%d] FAIL — the inflow rho ghost is not the inlet fluid's\n",
                      size);
          fail = 1;
        }
        if (gItmax >= 400) {
          std::printf("  [jet-coupl np=%d] FAIL — the pressure solve CAPPED (run invalid)\n", size);
          fail = 1;
        }
      }
    }
    // ------------------------- C0: V5a x V-BC composed, KINEMATIC (the bitwise gate on this path)
    //
    // The SAME composed scene as case C — the same packing SDF, the same inflow/outflow faces, the
    // same initial colour — driven by a PRESCRIBED velocity instead of the coupled step. Nothing
    // in the update then goes through a reduction, so the colour must be BITWISE identical to the
    // single-rank reference at every np, exactly as `slug-kin` is.
    //
    // WHY THIS CASE EXISTS. Case C below cannot be gated bitwise, and (WO 2026-09-14) it cannot be
    // gated at a reduction floor either — see the long comment there. That left the composed
    // cut-cell x open-boundary path with no exact gate at all, which is the coverage this restores:
    // the out-of-domain mask on GLOBAL indices, the owner-only boundary colour, the openness-
    // weighted domain-face flux and the per-rank ledger are all exercised here, against an
    // absolute standard, with a velocity that VARIES in all three directions so a mis-filled ghost
    // cannot hide behind a uniform field.
    //
    // What it does NOT cover, and case C still must: `setField` clears `outflowCorrValid_`, so the
    // bridge takes its `doOutflow = true` branch here. The outflow-face correction the projection
    // makes lives only on the coupled path.
    {
      const int steps = 20;
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configurePacking(sd, ox, oy, oz, lnx, lny, lnz);
      packingVelocity(sd, ox, oy, oz, lnx, lny, lnz);
      double ledger = 0.0, solidSum = 0.0;
      const double vol0loc = sd.vofDiagnostics().volume;
      for (int i = 0; i < steps; ++i) {
        sd.advectVof();
        const auto v = sd.vofBcVolumes();
        for (int f = 0; f < 6; ++f)
          ledger += v[f];
        solidSum = std::fmax(solidSum, std::fabs(sd.vofDiagnostics().solidSumC));
      }
      const double vol1loc = sd.vofDiagnostics().volume;
      double g0 = 0, g1 = 0, gLed = 0, gSolid = 0;
      MPI_Allreduce(&vol0loc, &g0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&vol1loc, &g1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&ledger, &gLed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&solidSum, &gSolid, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      const std::vector<double> gc =
          gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configurePacking(ref, 0, 0, 0, NX, NY, NZ);
        packingVelocity(ref, 0, 0, 0, NX, NY, NZ);
        for (int i = 0; i < steps; ++i)
          ref.advectVof();
        const double dc = maxAbsDiff(gc, ref.getVof());
        const double budget = std::fabs((g1 - g0) - gLed);
        const long nbad = countNonFinite(gc) + countNonFinite(ref.getVof());
        // The budget is REPORTED, not gated, and this is the one thing the case gives up: the
        // Weymouth-Yue conservation identity rests on the advecting field being discretely
        // divergence-free in the OPENNESS-WEIGHTED sense, and a prescribed analytic field is not
        // (2.6e-03 relative over these 20 steps). The identity is gated where its hypothesis holds
        // — on `slug-kin`, whose uniform w is exactly solenoidal, and on case C below, whose field
        // comes out of the projection. What this case gates is the one thing those two cannot:
        // that the composed cut-cell x open-boundary colour transport is decomposition-EXACT.
        std::printf(
            "  [packing-kin np=%d] colour vs single-rank %.3e (BITWISE required); budget "
            "|d sum(eps_eff C) - ledger| %.3e (rel %.3e, NOT gated — the prescribed field is not "
            "discretely solenoidal); solid colour %.3e\n",
            size, dc, budget, budget / g0, gSolid);
        if (!(dc == 0.0)) {
          std::printf("  [packing-kin np=%d] FAIL — the composed colour is not bitwise\n", size);
          fail = 1;
        }
        if (nbad != 0) {
          std::printf("  [packing-kin np=%d] FAIL — %ld non-finite colour cells\n", size, nbad);
          fail = 1;
        }
        if (!(gSolid == 0.0)) {
          std::printf("  [packing-kin np=%d] FAIL — colour leaked into solid cells\n", size);
          fail = 1;
        }
      }
    }

    // ------------------------------------------- C: V5a x V-BC composed (WO-R2 item 2), coupled
    //
    // WHY THIS ONE IS AT THE REDUCTION FLOOR AND NOT BITWISE. The kinematic pattern the two rungs
    // use separately (`vof_cutcell_mpi`: run the reference on the full grid, slice its velocity
    // into every decomposition with `setField`) cannot be used through an OPEN boundary: the
    // outflow-face correction lives on a ghost face index that `setField` does not carry, and the
    // ghost fill `setField` triggers is exactly the one that erases it (`outflowCorrValid_`,
    // WO-R). So the composed scene is driven by `step()` on both sides and gated at the
    // allreduce-order floor, like the coupled jet above. The BUDGET identity is gated absolutely.
    {
      const int steps = 40;
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configurePacking(sd, ox, oy, oz, lnx, lny, lnz);
      long itmax = 0;
      double ledger = 0.0, solidSum = 0.0;
      const double vol0loc = sd.vofDiagnostics().volume;
      for (int i = 0; i < steps; ++i) {
        sd.step();
        itmax = std::max<long>(itmax, sd.lastPressureIterations());
        const auto v = sd.vofBcVolumes();
        for (int f = 0; f < 6; ++f)
          ledger += v[f];
        solidSum = std::fmax(solidSum, std::fabs(sd.vofDiagnostics().solidSumC));
      }
      const double vol1loc = sd.vofDiagnostics().volume;
      double g0 = 0, g1 = 0, gLed = 0, gSolid = 0;
      long gIt = 0;
      MPI_Allreduce(&vol0loc, &g0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&vol1loc, &g1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&ledger, &gLed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&solidSum, &gSolid, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&itmax, &gIt, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
      const std::vector<double> gc =
          gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configurePacking(ref, 0, 0, 0, NX, NY, NZ);
        for (int i = 0; i < steps; ++i)
          ref.step();
        const double dc = maxAbsDiff(gc, ref.getVof());
        const double budget = std::fabs((g1 - g0) - gLed);
        const double dsum = signedSumDiff(gc, ref.getVof());
        const double csum = sumOf(ref.getVof());
        std::printf(
            "  [packing   np=%d] colour vs single-rank %.3e (signed sum %.3e, rel %.3e); budget "
            "|d sum(eps_eff C) - ledger| %.3e (rel %.3e); solid colour %.3e; "
            "pressure %ld/400\n",
            size, dc, dsum, dsum / csum, budget, budget / g0, gSolid, gIt);
        // THE COLOUR IS NOT A LIPSCHITZ FUNCTION OF THE VELOCITY, so at np > 1 there is no
        // reduction floor to gate the POINTWISE field against. Measured, on this configuration
        // (WO docs/wo_vof_mpi_parity_gates.md, 2026-09-14):
        //
        //   * the composed VoF path itself is decomposition-EXACT — `packing-kin` above drives
        //     this very scene with a prescribed 3-D velocity and is bitwise at np = 2 and 4 and at
        //     1..16 OpenMP threads;
        //   * under `step()` every field the solver carries agrees at the allreduce floor: u, v, w
        //     and their ghosts to 1.4e-14, p to 1.4e-13 on |p| = 1.9e2;
        //   * and yet the colour parts company by 4.795e-09 in exactly TWO adjacent cells, equal
        //     and opposite, i.e. one x-face flux.
        //
        // The mechanism is `mycNormal`'s estimator selection (core/vof/plic.hpp), which is a STRICT
        // comparison — `if (fabs(mm[cn][cn]) > t0) cn = 3;` — between the centred and the Youngs
        // candidate. At a near-axis-aligned interface the two candidates carry the same dominant
        // component, so that comparison is an exact tie: traced here, the centred candidate scored
        // 0.99986893026188128 against Youngs' 0.99986893026188106 at np = 2 (ONE ULP apart, Youngs
        // loses) and 0.99986893026188106 against 0.99986893026188106 single-rank (equal, so `>` is
        // false and the centred one wins). The two winners are the same plane to 1e-6 in their
        // dominant component and differ by 2.9e-06 in the transverse ones; re-tilting the plane by
        // that much moves the liquid volume in the |a| = 3.4e-03 slab of that face by ~a^2 s
        // = 1.5e-09, and the sweep by 4.795e-09. ONE ULP of colour in, 4.8e-09 out — and which way
        // the tie breaks is decided by bits that a reduction order is entitled to change.
        //
        // The 1e-11 this gate used to ask for was therefore never a property of the code: it was a
        // coin toss. It came up heads in CI and on 1 thread, and tails everywhere else — measured
        // over 24 (ranks, threads) pairs, np = 1 is bitwise at 1..16 threads, np = 2 flips at 2..16
        // threads and not at 1, and np = 4 flips at 2, 3, 4, 6 and 12 threads but NOT at 1, 8 and
        // 16, which is the only reason `vof_bc_mpi_np4` passes on the 48-core box at all.
        //
        // So the pointwise gate is no longer a reduction-floor gate and does not pretend to be. The
        // reduction-floor content moves to the SIGNED SUM below, which is Lipschitz and is what a
        // real defect breaks; the pointwise number becomes a bound on the field going WRONG. It is
        // set at 1e-6: three orders above the largest flip this scene can produce (a tie needs the
        // two candidates to agree in the dominant component, which bounds their transverse mass
        // s = 1 - max|m| and hence the flux jump at ~cfl^2 s), and far below what a real defect
        // does here. That second half is MEASURED, not asserted — the WO-F owner rule was deleted
        // (`vofApplyColourBc`'s `touchesGlobalFace` test, so every rank injects the inflow colour
        // into its own block's -z ghost band) and re-run at np = 2: this gate reports 1.000e+00
        // against its 1e-06, the signed sum 5.141e+02 against its 3.886e-08, and `packing-kin`
        // 1.000e+00 against bitwise. Nine orders of margin, not one.
        //
        // The OTHER defect named in this file's header — the out-of-domain mask built on per-block
        // instead of global indices — was injected too, and is worth recording precisely because
        // this case does NOT see it: `packing` stayed at 3.174e-09 and `packing-kin` at exactly
        // zero, while `slug-kin`'s bitwise gate caught it at 2.769e-37. It is caught by the
        // bitwise cases, which is where that coverage lives; the 1e-11 this case used to carry
        // never had it either.
        const double tol = (size == 1) ? 0.0 : 1e-6;
        if (!(dc <= tol)) {
          std::printf("  [packing   np=%d] FAIL — the colour field is wrong (tol %.1e)\n", size,
                      tol);
          fail = 1;
        }
        // The reduction floor, on the quantity that HAS one. An 8192-term naive sum of O(1) values
        // drifts by O(N eps) ~ 2e-12 relative; 1e-11 is five times that, and a flux that a
        // decomposition lost or double-counted does not cancel at all.
        const double tolSum = (size == 1) ? 0.0 : 1e-11 * std::fabs(csum);
        if (!(std::fabs(dsum) <= tolSum)) {
          std::printf(
              "  [packing   np=%d] FAIL — the colour difference does not CANCEL: signed sum %.3e "
              "> %.3e (colour created or destroyed, not moved)\n",
              size, dsum, tolSum);
          fail = 1;
        }
        if (!(budget / g0 < 1e-10)) {
          std::printf("  [packing   np=%d] FAIL — the composed budget does not close\n", size);
          fail = 1;
        }
        if (!(gSolid == 0.0)) {
          std::printf("  [packing   np=%d] FAIL — colour leaked into solid cells\n", size);
          fail = 1;
        }
        if (gIt >= 400) {
          std::printf("  [packing   np=%d] FAIL — the pressure solve CAPPED (run invalid)\n", size);
          fail = 1;
        }
      }
    }

    // ------------------------------------ D: the COLLOCATED bridge through an open boundary
    //
    // The staggered composed case above gates this identity on `IbmSolver`; nothing gated it on
    // `SolverColocated`, whose VoF bridge takes a DIFFERENT path -- it hands the advector the FACE
    // field `uf_/vf_/wf_` and re-fills its ghosts first. `fillGhosts` is BC-unaware and wraps every
    // axis, and the outflow face is a GHOST index, so that fill overwrites the mass-conserving
    // value `bcCorrectOutflow` had just written with the opposite boundary's -- single-rank as
    // well as distributed, since the single-rank fill wraps periodically too.
    //
    // The conserved functional is summed from the CANONICAL colour field, not from the advector
    // census: `volume` (sum eps_eff*C) is populated only with cut-cell geometry attached, and
    // `diagnostics()` returns what the last `advect()` stored, so neither is meaningful BEFORE the
    // first step. This duct is all-fluid, so sum C over the inner block is the right quantity.
    {
      const int steps = 30;
      Colo sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configureColoJet(sd, oz, lnx, lny, lnz);
      double colLedger = 0.0;
      const double colC0 = sumOf(sd.getVof());
      long colIt = 0;
      for (int i = 0; i < steps; ++i) {
        sd.step();
        colIt = std::max<long>(colIt, sd.lastPressureIterations());
        const auto v = sd.vofBcVolumes();
        for (int f = 0; f < 6; ++f)
          colLedger += v[f];
      }
      const double colC1 = sumOf(sd.getVof());
      double gC0 = 0, gC1 = 0, gColLed = 0;
      long gColIt = 0;
      MPI_Allreduce(&colC0, &gC0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&colC1, &gC1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&colLedger, &gColLed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&colIt, &gColIt, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
      const std::vector<double> gcol =
          gatherGlobal(sd.getVof(), ox, oy, oz, lnx, lny, lnz, rank, size);
      if (rank == 0) {
        Colo ref(NX, NY, NZ);
        configureColoJet(ref, 0, NX, NY, NZ);
        for (int i = 0; i < steps; ++i)
          ref.step();
        const double dcol = maxAbsDiff(gcol, ref.getVof());
        const double budget = std::fabs((gC1 - gC0) - gColLed);
        const double csum = sumOf(ref.getVof());
        std::printf(
            "  [colo-jet  np=%d] colour vs single-rank %.3e; budget |d sum(C) - ledger| %.3e "
            "(rel %.3e); pressure %ld/400\n",
            size, dcol, budget, budget / gC0, gColIt);
        const double tolCol = (size == 1) ? 0.0 : 1e-9 * std::fabs(csum);
        if (!(dcol <= tolCol)) {
          std::printf("  [colo-jet  np=%d] FAIL — the colour field is wrong (tol %.1e)\n", size,
                      tolCol);
          fail = 1;
        }
        if (!(budget / gC0 < 1e-10)) {
          std::printf("  [colo-jet  np=%d] FAIL — the boundary budget does not close\n", size);
          fail = 1;
        }
        if (gColIt >= 400) {
          std::printf("  [colo-jet  np=%d] FAIL — the pressure solve CAPPED (run invalid)\n", size);
          fail = 1;
        }
      }
    }
    int gfail = 0;
    MPI_Allreduce(&fail, &gfail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    fail = gfail;
    if (rank == 0)
      std::printf(fail ? "FAILED\n" : "OK\n");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
