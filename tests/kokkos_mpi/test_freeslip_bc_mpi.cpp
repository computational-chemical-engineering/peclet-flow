// flow — ISSUES sweep item 7 gate (c): the FREE-SLIP domain BC (type 4) under MPI.
//
// Scene: a half channel driven by a uniform body force, with a NO-SLIP wall on -z and a FREE-SLIP
// (symmetry) face on +z, all-fluid (`setPressureGeometry`), no advection (steady Stokes). Three
// things are asserted, at np = 1/2/4:
//
//   1. DECOMPOSITION-INDEPENDENCE. The distributed run is compared POINTWISE against a full-grid
//      single-rank reference on rank 0 (the tests/kokkos_mpi pattern): BITWISE at np = 1, at the
//      reduction floor above. The decomposition is required to CUT the z axis, so the free-slip
//      face is owned by some ranks and not others -- exactly the `touchesGlobalFace` guard WO-F
//      installed for the other BC types, which a type-4 face has to obey too or the domain splits
//      into independent sub-domains (invisible in u, visible only in P).
//   2. The ANALYTIC profile. The scheme's own discrete solution of `-mu u'' = F` with u = 0 at the
//      z = 0 FACE and du/dz = 0 at the z = NZ face is the continuum parabola plus h^2/4 * F/(2 mu)
//      -- a second difference reproduces a quadratic exactly, and the only inexact ingredient is
//      the NO-SLIP wall's mirror ghost (u(-h/2) = -u(h/2) holds only for an odd function). The
//      free-slip face contributes no error at all, which is the point.
//   3. SYMMETRY. Free slip IS a symmetry plane, so the half channel must equal the lower half of
//      a full 2*NZ channel with no-slip on both sides. That is measured in the single-rank study
//      (tests/study/vof_issues_sweep.py freeslip, 1.1e-12); here only 1 and 2 are run per rank
//      count, to keep the ctest short.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int NX = 16, NY = 4, NZ = 32;
static constexpr double MU = 0.05, RHO = 1.0, DT = 5000.0, GFORCE = 1e-3;
static constexpr int STEPS = 120;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

static void configure(IbmSolver& s, int lnx, int lny, int lnz) {
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setAdvection(false);
  s.setBodyForce(GFORCE, 0.0, 0.0);
  s.setDomainBc(4, 1, 0.0, 0.0, 0.0);  // -z: no-slip wall
  s.setDomainBc(5, 4, 0.0, 0.0, 0.0);  // +z: FREE SLIP
  s.setPressurePcg(true, 400, 1e-12);
  s.setVelocityIterations(400);
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 1e30));
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
  double m = 0;  // NaN-propagating (WO-R2)
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d == d))
      return d;
    m = std::fmax(m, d);
  }
  return m;
}
static double maxAbs(const std::vector<double>& a) {
  double m = 0;
  for (double v : a)
    m = std::fmax(m, std::fabs(v));
  return m;
}

// max |u - u_discrete| over the column, relative to the peak.
static double analyticError(const std::vector<double>& u) {
  const double H = (double)NZ;
  double e = 0, s = 0;
  for (int z = 0; z < NZ; ++z) {
    const double zc = z + 0.5;
    const double ue = (GFORCE / (2.0 * MU)) * (H * H - (H - zc) * (H - zc) + 0.25);
    const std::size_t i =
        (std::size_t)(NX / 2) + (std::size_t)(NY / 2) * NX + (std::size_t)z * NX * NY;
    e = std::fmax(e, std::fabs(u[i] - ue));
    s = std::fmax(s, std::fabs(ue));
  }
  return e / (s + 1e-300);
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
    int ownTop = 0;
    for (std::size_t r = 0; r < (std::size_t)size; ++r) {
      const auto b = dec.block(r);
      for (int a = 0; a < 3; ++a)
        if ((int)b.size[a] != gn[a])
          cut[a] = true;
      if ((int)(b.origin[2] + b.size[2]) == NZ)
        ++ownTop;
    }
    if (rank == 0) {
      std::printf("FREE-SLIP BC MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s  "
                  "ranks owning the +z FREE-SLIP face: %d  z-blocks:",
                  size, NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "",
                  cut[2] ? "z" : "", ownTop);
      for (int r = 0; r < size; ++r)
        std::printf(" [%d,%d)", (int)dec.block((std::size_t)r).origin[2],
                    (int)(dec.block((std::size_t)r).origin[2] + dec.block((std::size_t)r).size[2]));
      std::printf("\n");
    }
    if (size > 1 && !cut[2]) {
      if (rank == 0)
        std::printf("  FAIL — the decomposition does NOT cut the z axis; this test exists to gate "
                    "the free-slip face across a rank boundary\n");
      fail = 1;
    }

    IbmSolver sd(lnx, lny, lnz);
    sd.initMpi(dec, MPI_COMM_WORLD);
    configure(sd, lnx, lny, lnz);
    for (int it = 0; it < STEPS; ++it)
      sd.step();
    std::vector<double> gu[3];
    for (int comp = 0; comp < 3; ++comp)
      gu[comp] = gatherGlobal(sd.getVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
    std::vector<double> gp = gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);
    if (rank == 0) {
      IbmSolver ref(NX, NY, NZ);
      configure(ref, NX, NY, NZ);
      for (int it = 0; it < STEPS; ++it)
        ref.step();
      double du = 0, umag = 0;
      for (int comp = 0; comp < 3; ++comp) {
        du = std::fmax(du, maxAbsDiff(gu[comp], ref.getVelocity(comp)));
        umag = std::fmax(umag, maxAbs(ref.getVelocity(comp)));
      }
      const std::vector<double> pref = ref.getPressure();
      const double dp = maxAbsDiff(gp, pref), pmag = maxAbs(pref);
      const double tol = (size == 1) ? 0.0 : std::fmax(1e-15, 1e-11 * umag);
      const double ptol = (size == 1) ? 0.0 : std::fmax(1e-12, 1e-9 * pmag);
      const double aerr = analyticError(gu[0]);
      const bool ok = du <= tol && dp <= ptol && aerr < 1e-8;
      std::printf("  [np=%d] du=%.3e (|u|=%.3e, tol %.1e)  dP=%.3e (|P|=%.3e, tol %.1e) | "
                  "discrete-parabola rel err %.3e (tol 1.0e-08)  %s\n",
                  size, du, umag, tol, dp, pmag, ptol, aerr, ok ? "OK" : "FAIL");
      if (!ok)
        fail = 1;
    }
    MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("FREE-SLIP BC MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
