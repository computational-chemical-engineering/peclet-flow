// flow — multi-rank validation of the VARIABLE-VISCOSITY path (VoF rung V-1, WO-A).
//
// Phase 4 (`doc/variable_viscosity_projection.md`, `tests/study/two_layer_couette.py`) was
// validated host-openmp, single-rank only. The distributed pieces that had never been gated: the
// "mu" ghost ring (`fillPropGhosts`), the HARMONIC face mean assembled across a rank boundary, and
// `minMuInner`'s allreduce feeding the rotational term's chi*mu_min coefficient.
//
// Two configurations on a 16x32x8 grid — the aligned ORB cuts the LONG y axis at np = 2, 4, which
// is deliberately the walled/loaded axis of both configurations (WO-F):
//
//   * `couette-y` — the LITERAL two-layer Couette of tests/study/two_layer_couette.py: plane
//     Couette across y (fixed -y wall, +y lid at U) through a MONOTONE two-layer viscosity stack
//     mu1 | mu2 with a single 10x jump at y = NY/2. The shear stress tau is uniform, so the exact
//     steady Stokes profile is u(y) = tau * int dy/mu — an ANALYTIC, decomposition-independent gate
//     that only the HARMONIC face mean reproduces (the arithmetic mean misses it by O(1%)).
//
//     WO-A had to ship a SYMMETRIC mu2|mu1|mu2 stack here instead, because `fillPropGhosts` /
//     `fillPorousEpsGhosts` applied their domain-face override only `if (!distributed_)`, leaving
//     the mu ghost on a walled face at its PERIODIC WRAP value in every distributed run. With a
//     monotone stack mu(0) != mu(NY-1), so the wrap value is the WRONG layer and the distributed
//     solver disagreed with the single-rank reference by 2.7e-2 relative ALREADY AT np = 1; the
//     symmetric stack had mu(0) == mu(NY-1) and so hid it. WO-F replaced that guard with the
//     per-face `touchesGlobalFace` ownership test (correct at every np, single-rank byte-identical
//     because the test is always true there), which is exactly what the monotone stack gates.
//
//   * `per-y` — fully periodic (no domain BC), with the 10x viscosity jump stacked along the CUT
//     axis, so at np = 2 and 4 it sits exactly on a rank boundary and the harmonic face mean there
//     is assembled from an exchanged ghost. Driven by a zero-mean cos(2 pi y / NY) body force in z
//     (a uniform force in a fully periodic box has no steady state). np-consistency gate only — no
//     closed form.
//
// WALLED AXIS / MPI (WO-A escalation #1, fixed by WO-F): flow's per-face domain BCs used to be
// imposed by EVERY rank on its OWN block faces — `applyVelocityBcCompTo`, the flux-openness
// construction and the pressure MG's per-level boundary openness had no `touchesGlobalFace`
// ownership test, unlike the scalar-transport BCs (`applyScalarBc`), so a cut walled axis silently
// split the domain into independent sub-domains. This test now REQUIRES the walled axis to be cut
// and compares the PRESSURE as well as the velocity — WO-A measured that on a walled column the
// velocity stays clean (4.5e-17) while the pressure is off by 4.0e+02, i.e. only the pressure sees
// the defect.
//
// Comparison protocol (the `tests/kokkos_mpi` pattern): pointwise against a full-grid single-rank
// reference on rank 0. np = 1 bit-exact; np > 1 to the MG-PCG reduction-order floor.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::IVec;
using peclet::flow::IbmSolver;

static constexpr int NX = 16, NY = 32, NZ = 8;
static constexpr double MU1 = 1.0, MU2 = 0.1, ULID = 1.0, RHO = 1.0, DT = 100.0;
static constexpr int STEPS_COUETTE = 80, STEPS_PERY = 40;
static constexpr double FZ = 1e-2;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

struct Config {
  const char* name;
  bool couette;  // walled two-layer Couette (else the periodic cut-axis sandwich)
  int steps;
};

// Both configurations load the CUT y axis: the literal monotone two-layer stack mu1|mu2 (see the
// header), walled for `couette-y` and periodic for `per-y`.
static double muAt(const Config&, int, int y, int) { return (y < NY / 2) ? MU1 : MU2; }
static double srcAt(int, int y, int) { return std::cos(2.0 * M_PI * y / NY); }

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

static void configure(IbmSolver& s, const Config& c, int ox, int oy, int oz, int lnx, int lny,
                      int lnz) {
  s.setRho(RHO);
  s.setMu(MU1);
  s.setDt(DT);
  s.setAdvection(false);  // Stokes: the analytic profile is the steady Stokes solution
  s.setPressurePcg(true, 200, 1e-10);
  s.setVelocityIterations(60);
  if (c.couette) {
    s.setDomainBc(2, 1, 0.0, 0.0, 0.0);   // -y fixed wall
    s.setDomainBc(3, 2, ULID, 0.0, 0.0);  // +y lid at U
  }
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.addField("mu");
  s.setField("mu", blockOf([&](int x, int y, int z) { return muAt(c, x, y, z); }, ox, oy, oz, lnx,
                           lny, lnz));
  s.setPropertyMode(true, /*harmonic=*/true);
  s.exchangeField("mu");
  if (!c.couette) {
    s.addField("src");
    s.setField("src", blockOf(srcAt, ox, oy, oz, lnx, lny, lnz));
    s.exchangeField("src");
    s.setPropertyModel("force_z", peclet::flow::ClosureKind::LinearMix, "src", "", {0.0, FZ});
    // WO-F carried an explicit seed of the SAME field here purely to fill its GHOST ring, because
    // nothing in `step()` exchanged a `force_*` field. WO-G fixed that (`fillCellForceGhosts`), so
    // the seed is gone. It was doubly inert in THIS test anyway: with variable mu but constant
    // density the RHS kernel is `buildRhsForced`, which reads the CELL value `fb(i)` and never
    // `fb(i-strd)` — only `buildRhsVar` (varRho / eps-conservative porous) face-interpolates the
    // force and therefore reads the ghost. The WO-G gate for that is `test_bodyforce_ghost_mpi`.
  }
}

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

// Exact steady Stokes Couette through the monotone two-layer mu1|mu2 stack: tau is uniform, so
// u(y) = tau * integral(dy/mu) and U = tau*(0.5/mu1 + 0.5/mu2).
static double couetteExact(double yc) {
  const double tau = ULID / (0.5 / MU1 + 0.5 / MU2);
  if (yc < 0.5)
    return tau * yc / MU1;
  return tau * 0.5 / MU1 + tau * (yc - 0.5) / MU2;
}
static double couetteError(const std::vector<double>& u) {
  double e = 0;
  for (int y = 0; y < NY; ++y) {
    const std::size_t i =
        (std::size_t)(NX / 2) + (std::size_t)y * NX + (std::size_t)(NZ / 2) * NX * NY;
    e = std::fmax(e, std::fabs(u[i] - couetteExact((y + 0.5) / NY)) / ULID);
  }
  return e;
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
      std::printf("VARMU MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n", size, NX,
                  NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "", cut[2] ? "z" : "");

    const Config configs[] = {{"couette-y", true, STEPS_COUETTE}, {"per-y", false, STEPS_PERY}};

    for (const Config& c : configs) {
      // WO-F: the walled/loaded y axis must be CUT — that is the case the domain-BC and
      // property-ghost ownership tests exist for.
      if (size > 1 && !cut[1]) {
        if (rank == 0)
          std::printf("  [%-9s np=%d] FAIL — the decomposition does NOT cut the walled y axis; "
                      "this test exists to gate the domain-BC / property-ghost rank ownership on "
                      "a CUT walled axis (WO-F)\n",
                      c.name, size);
        fail = 1;
        continue;
      }
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, c, ox, oy, oz, lnx, lny, lnz);
      for (int it = 0; it < c.steps; ++it)
        sd.step();
      std::vector<double> gu[3];
      for (int comp = 0; comp < 3; ++comp)
        gu[comp] = gatherGlobal(sd.getVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
      // WO-F: the PRESSURE is the field that exposes a non-rank-aware domain BC (the velocity of a
      // split domain stays clean); gate it too.
      const std::vector<double> gp =
          gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);

      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, c, 0, 0, 0, NX, NY, NZ);
        for (int it = 0; it < c.steps; ++it)
          ref.step();
        double du = 0, umag = 0;
        for (int comp = 0; comp < 3; ++comp) {
          du = std::fmax(du, maxAbsDiff(gu[comp], ref.getVelocity(comp)));
          umag = std::fmax(umag, maxAbs(ref.getVelocity(comp)));
        }
        const double rel = du / (umag + 1e-300);
        const double dp = maxAbsDiff(gp, ref.getPressure());
        const double pmag = maxAbs(ref.getPressure());
        // np=1 bit-exact; np>1 the MPI reduction-order floor. The pressure needs an ABSOLUTE
        // floor as well as a relative one: this is a pure-shear Stokes flow, so |P| itself is
        // ~1e-8 (numerical zero) and a purely relative gate would be gating round-off. The
        // *velocity* is the strong gate here (the analytic Couette across the cut walled axis);
        // the strong pressure gate lives in test_vardensity_mpi's hydrostatic column, where
        // |P| = O(800) and the pre-WO-F defect showed up as 4.0e+02.
        const double utol = (size == 1) ? 0.0 : std::fmax(1e-15, 1e-11 * umag);
        const double ptol = (size == 1) ? 0.0 : std::fmax(1e-14, 1e-11 * pmag);
        double aerr = -1;
        bool ok = du <= utol && dp <= ptol;
        if (c.couette) {
          aerr = couetteError(gu[0]);
          ok = ok && aerr < 5e-3;  // the harmonic-mean gate (arithmetic misses by O(1%))
        }
        char extra[64] = "";
        if (c.couette)
          std::snprintf(extra, sizeof(extra), "  analytic err=%.4f%%", aerr * 100.0);
        std::printf("  [%-9s np=%d] du=%.3e (rel %.2e, tol %.1e) | dp=%.3e (|P|=%.2e, tol %.1e)"
                    "%s  %s\n",
                    c.name, size, du, rel, utol, dp, pmag, ptol, extra, ok ? "OK" : "FAIL");
        if (!ok)
          fail = 1;
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("VARMU MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
