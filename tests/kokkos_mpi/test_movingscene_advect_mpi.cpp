// flow — a MOVING analytic scene with ADVECTION ON, distributed (advective cut-wall gate 7).
//
// THE GAP THIS CLOSES. Every other ctest in tests/kokkos_mpi runs `setAdvection(false)`, and none
// of them moves a scene instance. So the suite had no distributed measurement of the two things
// the advective cut-wall campaign (`doc/advective_cutwall_flux_plan.md`) actually changed:
//
//   * rung A0 fills the SOLID-MASKED rows of the advection's scratch inputs with the local
//     rigid-body wall velocity taken from `uBc_`, ANALYTIC-POINTWISE over the extended block —
//     ghosts included. The plan argues that because the fill is pointwise from replicated instance
//     state, no halo exchange is needed for it under MPI. That is an ARGUMENT. This test turns it
//     into a measurement: if a ghost solid row were left holding the masked zero on one rank and
//     the wall velocity on another, the SOU/Koren stencils (which reach 2 cells into the solid)
//     would read different data either side of a rank boundary and the distributed march would
//     drift from single-rank.
//   * the moving-geometry machinery around it — `rebuildGeometry()` per step (fresh cells, the
//     cut-cell overlay, the apertures and the pressure operator all re-derived from the instance
//     transforms), the wall's own volume flux in the projection, and the discrete REACTION force
//     `hydroForceTorqueReaction()`, which is what the campaign's Newton audit reads.
//
// THE CASE. Periodic 48^3 box, ONE analytic sphere (d = 8 cells) TOWED diagonally through fluid
// initially at rest, with explicit SOU advection on. The centre is off-lattice (+0.3 in every
// axis — a grid-plane-aligned moving face produces zero cut cells and goes silently inert, a trap
// recorded in the campaign plan) and the tow path deliberately CROSSES the ORB cut planes in both
// x and y at np = 2 and np = 4: the body, its wall velocity and its cut band are all in flight
// across rank boundaries, which is the only configuration in which the A0 ghost claim can fail.
// Re = U d / nu = 20, dt chosen for CFL = 0.2 (advective, not diffusive, limited), 60 steps, so
// the sphere traverses 12 cells = 1.5 diameters and rebuilds geometry 60 times.
//
// THE GATE. Distributed vs the full-grid SINGLE-RANK reference built in the same executable:
//   * velocity (all 3 components) and pressure, gathered to rank 0 and differenced cell by cell;
//   * the per-instance REACTION force and torque (the coupling force the campaign cares about).
// np = 1 must be BIT-EXACT in the FIELDS (tol 0.0 — initMpi on one rank must not perturb a single
// bit of the moving + advective path). The reaction force is held to a double-precision
// accumulation floor instead, and NOT because of MPI: `hydroForceTorqueReaction` sums with
// `Kokkos::atomic_add` over an unordered device traversal, so it is tolerance-reproducible rather
// than bitwise even between two runs whose fields agree bit for bit. np > 1 lands on the usual
// MG-PCG reduction-order floor — the Krylov inner-product Allreduce reorders with the rank count
// and 60 nonlinear steps amplify it — the established 3e-7-relative class of the other MPI gates.
//
// WHAT IT MEASURED (2026-09-02, first run, RTX 5080 / nvidia-cuda prefix). np = 1 is bit-exact in
// every field (du = dp = 0.000e+00) — but **np = 2 and np = 4 FAIL, and the A0 fill is the cause**.
// The 2x2 ablation, same case, np = 2, max|u_dist - u_ref| over the whole grid (|u|max = 4.03e-02):
//
//     moving + advection ON   (the shipped case)   du = 1.45e-07   <- FAIL (tol 1.21e-08)
//     moving + advection OFF  (GATE7_ADV=0)        du = 5.99e-16
//     static + advection ON   (GATE7_MOVE=0)       du = 3.47e-17
//     moving + advection ON, A0 fill disabled
//       (PECLET_FLOW_ADV_WALLVEL=0)                du = 1.28e-16
//
// So neither the moving-geometry machinery nor the advection is decomposition-dependent on its
// own, and with the A0 fill turned off the moving+advective march is bit-clean across ranks. The
// np-dependence lives entirely in `buildAdvInputs`. It is also LOCAL to the rank boundary: with
// the identical case translated so the body never comes within the ghost ring of a cut plane
// (GATE7_SHIFT=-8) np = 2 reads du = 2.57e-15. The likely mechanism is that `uBc_` is built by
// `buildWallVelocity` over the extended block from a CENTRED difference of `sdf_`
// (`ccSampleExt(sd, e, sx +- 1, ...)`), which at the OUTERMOST ghost plane has no neighbour to
// difference against and clamps — and `uBc_` is never halo-exchanged, so those planes hold a
// different wall velocity than the single-rank run holds at the same global points, and the SOU
// stencil (reach 2) carries it inward. The plan's claim that "the scene is analytic, so ghost
// solid rows are computable pointwise and no extra exchange is needed" is right about the SCENE
// and wrong about this fill, which reads the sampled SDF, not the scene.
//
// An earlier variant of this case whose body ended with its wall band sitting IN rank 0's outer
// ghost planes (centre stopping at x = 27.5 against a cut at x = 32) measured du = 1.385e-03 at
// np = 2 and 1.703e-03 at np = 4 — 3.5 % of max|u| — so the magnitude depends on where the body is
// when the fields are compared, and the worst case is a body parked on a rank boundary.
//
// ABLATION KNOBS (all default to the shipped case): GATE7_ADV=0 turns advection off, GATE7_MOVE=0
// freezes the instance and drives the flow with a body force instead, GATE7_SHIFT=<cells> slides
// the whole tow path, and the solver's own PECLET_FLOW_ADV_WALLVEL=0 disables the A0 fill.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/geom/scene_builder.hpp"

using peclet::flow::IbmSolver;

static constexpr int N = 48, STEPS = 60;
static constexpr std::size_t GCELLS = (std::size_t)N * N * N;
static constexpr double DIAM = 8.0;    // sphere diameter, cells
static constexpr double U = 0.05;      // tow speed, cells per time unit
static constexpr double RE = 20.0;     // U * DIAM / MU  -> finite-Re, advection matters
static constexpr double MU = U * DIAM / RE;
static constexpr double RHO = 1.0;
static constexpr double DT = 4.0;      // CFL = U*DT/h = 0.2; 0.2 cells of body motion per step
static constexpr double OFF = 0.3;     // off-lattice centre (grid-aligned walls go inert)
// Tow direction (unit): diagonal in x-y so the body crosses the ORB cut planes on BOTH cut axes.
static constexpr double DIRX = 0.6, DIRY = 0.8;
// Start so that the 60-step path (7.2 cells in x, 9.6 in y) straddles the np=2/np=4 cut at 24.
static constexpr double X0C = 0.60 * N + OFF, Y0C = 0.57 * N + OFF, Z0 = 0.50 * N + OFF;
// Ablation instruments (see the 2x2 table in the header). GATE7_SHIFT moves the whole tow path by
// a constant so the body stays CLEAR of every rank boundary — the control that localises a failure
// to the boundary rather than to "advection under MPI" in general.
static double gShift() {
  const char* v = std::getenv("GATE7_SHIFT");
  return v ? std::atof(v) : 0.0;
}
static double X0() { return X0C + gShift(); }
static double Y0() { return Y0C + gShift(); }

// The flat scene encoding (peclet/core/geom/scene_builder.hpp): one sphere leaf, one instance.
static void sphereScene(std::vector<int>& ni, std::vector<double>& nr, std::vector<int>& ii,
                        std::vector<double>& ir) {
  namespace g = peclet::core::geom;
  ni.assign(g::kNodeIntStride, 0);
  nr.assign(g::kNodeRealStride, 0.0);
  ni[0] = g::kSphere;
  ni[1] = -1;
  ni[2] = -1;
  nr[0] = 0.5 * DIAM;  // params[0] = radius
  nr[14] = 1.0;        // rotation w
  nr[15] = 1.0;        // scale
  ii.assign(g::kInstanceIntStride, 0);
  ir.assign(g::kInstanceRealStride, 0.0);
  ii[0] = 0;   // shapeRoot
  ii[1] = -1;  // materialId
  ir[0] = X0();
  ir[1] = Y0();
  ir[2] = Z0;
  ir[6] = 1.0;  // rotation w
  ir[7] = 1.0;  // scale
}

static bool envOn(const char* k) {
  const char* v = std::getenv(k);
  return !v || std::atoi(v) != 0;
}

static void configure(IbmSolver& s) {
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setAdvection(envOn("GATE7_ADV"));  // THE point of this test (ablation knob)
  s.setAdvectionScheme(0);    // SOU (the default)
  s.setVelocityIterations(60);
  s.setPressureLevels(4);
  s.setPressurePcg(true, 200, 1e-10);
  std::vector<int> ni, ii;
  std::vector<double> nr, ir;
  sphereScene(ni, nr, ii, ir);
  s.setScene(ni, nr, ii, ir, /*periodic=*/true);
  s.setSolidFromScene(/*cutcell_pressure=*/true);
  const double m = envOn("GATE7_MOVE") ? 1.0 : 0.0;
  if (m == 0.0)
    s.setBodyForce(2e-4, 0.0, 0.0);  // ablation: a static body needs a driver
  s.setInstanceMotion(0, {m * DIRX * U, m * DIRY * U, 0.0}, {0.0, 0.0, 0.0}, nullptr);
}

// One towed step: place the body where it is at the START of the step, keep its velocity, rebuild
// the geometry from the transforms, advance. Identical in the distributed and the reference run.
static void towStep(IbmSolver& s, int it) {
  const double m = envOn("GATE7_MOVE") ? 1.0 : 0.0;
  const double x = X0() + m * DIRX * U * DT * it;
  const double y = Y0() + m * DIRY * U * DT * it;
  s.setInstanceTransform(0, {x, y, Z0}, {0.0, 0.0, 0.0, 1.0});
  s.setInstanceMotion(0, {m * DIRX * U, m * DIRY * U, 0.0}, {0.0, 0.0, 0.0}, nullptr);
  s.rebuildGeometry();
  s.step();
}

// Gather per-rank inner blocks (x-fastest) into the global field on rank 0.
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
            std::memcpy(&global[(std::size_t)ox + (std::size_t)(y + oy) * N +
                                (std::size_t)(z + oz) * N * N],
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
          std::memcpy(&global[(std::size_t)meta[0] + (std::size_t)(y + meta[1]) * N +
                              (std::size_t)(z + meta[2]) * N * N],
                      &buf[(std::size_t)y * meta[3] + (std::size_t)z * meta[3] * meta[4]],
                      (std::size_t)meta[3] * sizeof(double));
    }
  }
  return global;
}

static double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < b.size(); ++i)
    m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}
static double maxAbs(const std::vector<double>& a) {
  double m = 0;
  for (double v : a)
    m = std::fmax(m, std::fabs(v));
  return m;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), N, N, N);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    bool cut[3] = {false, false, false};
    for (const auto& sz : dec.sizes())
      for (int a = 0; a < 3; ++a)
        if ((int)sz[a] != N)
          cut[a] = true;

    // The body must actually FLY ACROSS a rank boundary — otherwise this test silently degrades
    // into "a moving body that happens to sit inside one block", which gates nothing about the A0
    // ghost claim. Report the crossings; demand at least one when the decomposition cuts.
    const double x1 = X0() + DIRX * U * DT * (STEPS - 1), y1 = Y0() + DIRY * U * DT * (STEPS - 1);
    int crossings = 0;
    for (std::size_t r = 0; r < dec.sizes().size(); ++r) {
      const auto b = dec.block(r);
      const double bx = (double)b.origin[0], by = (double)b.origin[1];
      if (bx > 0.0 && ((X0() - bx) * (x1 - bx) < 0.0))
        ++crossings;
      if (by > 0.0 && ((Y0() - by) * (y1 - by) < 0.0))
        ++crossings;
    }
    if (rank == 0)
      std::printf("MOVING-SCENE ADVECT MPI np=%d  grid %d^3  block %dx%dx%d  cut axes: %s%s%s  "
                  "sphere d=%g Re=%g mu=%g dt=%g steps=%d  tow (%.2f,%.2f)->(%.2f,%.2f) "
                  "cut-plane crossings=%d\n",
                  size, N, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "", cut[2] ? "z" : "",
                  DIAM, RE, MU, DT, STEPS, X0(), Y0(), x1, y1, crossings);
    const bool plainRun = gShift() == 0.0 && envOn("GATE7_MOVE") && envOn("GATE7_ADV");
    if (size > 1 && plainRun && crossings == 0) {
      if (rank == 0)
        std::fprintf(stderr,
                     "  FAIL — the towed sphere never crosses an ORB cut plane; this test exists "
                     "to gate the moving wall + its advective inputs ACROSS a rank boundary\n");
      fail = 1;
    }

    // --- distributed run ---
    IbmSolver sd(lnx, lny, lnz);
    sd.initMpi(dec, MPI_COMM_WORLD);
    configure(sd);
    if (!sd.hasMovingInstance()) {
      if (rank == 0)
        std::fprintf(stderr, "  FAIL — the scene instance is not moving (setInstanceMotion)\n");
      fail = 1;
    }
    for (int it = 0; it < STEPS; ++it)
      towStep(sd, it);
    const double divDist = sd.maxOpenDivergence();
    const std::vector<double> frDist = sd.hydroForceTorqueReaction();  // Allreduced in C++

    std::vector<double> gu[3];
    for (int c = 0; c < 3; ++c)
      gu[c] = gatherGlobal(sd.getVelocity(c), ox, oy, oz, lnx, lny, lnz, rank, size);
    const std::vector<double> gp =
        gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);

    // --- full-grid single-rank reference (the validated single-GPU path) on rank 0 ---
    if (rank == 0) {
      IbmSolver ref(N, N, N);
      configure(ref);
      for (int it = 0; it < STEPS; ++it)
        towStep(ref, it);
      const std::vector<double> frRef = ref.hydroForceTorqueReaction();

      double du = 0, umag = 0;
      for (int c = 0; c < 3; ++c) {
        du = std::fmax(du, maxAbsDiff(gu[c], ref.getVelocity(c)));
        umag = std::fmax(umag, maxAbs(ref.getVelocity(c)));
      }
      const double dp = maxAbsDiff(gp, ref.getPressure());
      const double pmag = maxAbs(ref.getPressure());
      double dF = 0, dT = 0;
      for (int k = 0; k < 3; ++k) {
        dF = std::fmax(dF, std::fabs(frDist[k] - frRef[k]));
        dT = std::fmax(dT, std::fabs(frDist[3 + k] - frRef[3 + k]));
      }
      const double fmagv = std::fmax(std::fmax(std::fabs(frRef[0]), std::fabs(frRef[1])),
                                     std::fabs(frRef[2]));
      const double tmagv = std::fmax(std::fmax(std::fabs(frRef[3]), std::fabs(frRef[4])),
                                     std::fabs(frRef[5]));

      // FIELDS: np=1 BIT-EXACT (tol 0.0); np>1 the established 3e-7-relative class of the other
      // scene/field MPI gates (the MG-PCG inner-product Allreduce reorders with the rank count).
      const double rel = (size == 1) ? 0.0 : 3e-7;
      const double utol = rel * std::fmax(umag, 1e-300);
      const double ptol = rel * std::fmax(pmag, 1e-300);
      // FORCE/TORQUE: NOT bitwise even at np=1, and not because of MPI. `hydroForceTorqueReaction`
      // accumulates with `Kokkos::atomic_add` over an unordered device traversal, so the summation
      // order is not reproducible between two runs holding BIT-IDENTICAL fields (the same caveat
      // `mpi_scene_gate.py` states: "tolerance-reproducible, not bitwise"). np=1 therefore gets a
      // double-precision accumulation floor, not zero; np>1 the field class. Torque is scaled by
      // the lever arm the force acts on.
      const double relF = (size == 1) ? 1e-12 : 3e-7;
      const double ftol = relF * std::fmax(fmagv, 1e-300);
      const double ttol = relF * std::fmax(tmagv + fmagv * DIAM, 1e-300);

      // The case must be non-degenerate: a real flow, a real reaction force, a clean projection.
      const bool live = umag > 1e-4 && fmagv > 1e-6;
      const bool ok = du <= utol && dp <= ptol && dF <= ftol && dT <= ttol && live &&
                      divDist < 1e-6;
      std::printf("  np=%d  du=%.3e (|u|=%.4e)  dp=%.3e (|p|=%.4e)  dF=%.3e  dT=%.3e  "
                  "div=%.2e  tol %.1e/%.1e rel\n",
                  size, du, umag, dp, pmag, dF, dT, divDist, rel, relF);
      std::printf("  np=%d  F_dist=(%.17g, %.17g, %.17g)\n", size, frDist[0], frDist[1], frDist[2]);
      std::printf("  np=%d  F_ref =(%.17g, %.17g, %.17g)  T_ref=(%.3e, %.3e, %.3e)\n", size,
                  frRef[0], frRef[1], frRef[2], frRef[3], frRef[4], frRef[5]);
      if (!live)
        std::fprintf(stderr, "  FAIL — degenerate case (|u|=%.3e, |F|=%.3e)\n", umag, fmagv);
      if (!ok)
        fail = 1;
    }
    MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("MOVING-SCENE ADVECT MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
