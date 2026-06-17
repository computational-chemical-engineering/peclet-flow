// Step 13: gather the distributed field to root and write it as VTI (transport-core I/O).
//
// Runs the DistributedNS solver (Taylor-Green, no solid), gathers the global u-field from all
// rank-owned blocks onto rank 0, and:
//   (a) checks the assembled global field matches the analytic decayed TGV pattern cell-for-cell
//       (gather correctness — a wrong assembly would scramble the pattern), np=1,2,4;
//   (b) writes it to a .vti via tpx::geom::writeVti and reads it back, verifying a bit-exact
//       round-trip (the suite's geometry/field interchange format).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "distributed_ns.cuh"
#include "tpx/geom/grid_sdf.hpp"
#include "tpx/geom/vti_io.hpp"

using dns::DistributedNS;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int N = 32;
  int3 res = make_int3(N, N, 4);
  double k = 2.0 * M_PI / N, nu = 0.05, dt = 0.5;
  const int steps = 10;

  DistributedNS s;
  s.init(res, rank, size, 1.0, nu, dt);  // advection off -> Stokes (TGV decays as a diffusion mode)
  s.set_incremental_pressure(false);  // analytic TGV diffusion-decay check (n_pois=0, classical Chorin)
  int3 e = s.ext(), og = s.origin_incl_ghost();
  int g = s.ghost();
  std::size_t n = s.num_cells();
  std::vector<double> hu(n, 0), hv(n, 0), hw(n, 0);
  for (int lz = g; lz < e.z - g; ++lz)
    for (int ly = g; ly < e.y - g; ++ly)
      for (int lx = g; lx < e.x - g; ++lx) {
        int gx = lx + og.x, gy = ly + og.y;
        size_t i = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        hu[i] = cos(k * gx) * sin(k * (gy + 0.5));
        hv[i] = -sin(k * (gx + 0.5)) * cos(k * gy);
      }
  s.upload_velocity(hu.data(), hv.data(), hw.data());
  for (int it = 0; it < steps; ++it) s.step(/*diff*/ 80, /*pois*/ 0);

  std::vector<double> global = s.gather_to_root(s.u(), 0);

  int fail = 0;
  if (rank == 0) {
    double lam = 2.0 * (2.0 - 2.0 * cos(k));
    double decay = pow(1.0 / (1.0 + nu * dt * lam), steps);
    double maxerr = 0.0;
    for (int z = 0; z < res.z; ++z)
      for (int y = 0; y < res.y; ++y)
        for (int x = 0; x < res.x; ++x) {
          double expect = decay * cos(k * x) * sin(k * (y + 0.5));
          double got = global[(size_t)x + (size_t)y * res.x + (size_t)z * res.x * res.y];
          maxerr = fmax(maxerr, fabs(got - expect));
        }
    if (maxerr > 1e-6) {
      fprintf(stderr, "  gather/pattern mismatch: maxerr=%.2e\n", maxerr);
      ++fail;
    }

    // Write + read back via transport-core VTI.
    tpx::geom::GridSdf grid;
    grid.dims = {res.x, res.y, res.z};
    grid.origin = {0, 0, 0};
    grid.spacing = {1, 1, 1};
    grid.values.resize(global.size());
    for (size_t i = 0; i < global.size(); ++i) grid.values[i] = (float)global[i];
    std::string path = "distributed_u.vti";
    tpx::geom::writeVti(path, grid, "u");
    tpx::geom::GridSdf rb = tpx::geom::readVti(path);
    std::remove(path.c_str());
    if (rb.dims[0] != res.x || rb.dims[1] != res.y || rb.dims[2] != res.z) ++fail;
    int rterr = 0;
    for (size_t i = 0; i < grid.values.size(); ++i)
      if (rb.values[i] != grid.values[i]) ++rterr;
    if (rterr) {
      fprintf(stderr, "  VTI round-trip mismatches: %d\n", rterr);
      ++fail;
    }
    printf("# gather+VTI: pattern maxerr=%.2e  vti_roundtrip_mismatches=%d\n", maxerr, rterr);
  }

  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0)
      printf("OK (np=%d): gathered global field matches analytic TGV and round-trips through VTI\n",
             size);
    else
      fprintf(stderr, "FAILED (np=%d): %d\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
