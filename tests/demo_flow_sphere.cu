// Demo: distributed Navier-Stokes flow around a sphere, writing a velocity-magnitude VTI.
//
// A runnable example of the DistributedNS solver end-to-end: decompose a periodic box across MPI
// ranks, place an SDF sphere (no-slip), drive flow past it with a body force, integrate the full
// Navier-Stokes equations, gather the global field to rank 0 and write it as a ParaView .vti.
//
//   mpirun -np 4 ./demo_flow_sphere [N] [steps] [out.vti]
//
// Defaults: N=48, steps=60, out=flow_sphere.vti. (CI runs a tiny smoke case.)
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "distributed_ns.cuh"
#include "tpx/geom/vti_io.hpp"

using dns::DistributedNS;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int N = (argc > 1) ? std::atoi(argv[1]) : 48;
  int steps = (argc > 2) ? std::atoi(argv[2]) : 60;
  std::string out = (argc > 3) ? argv[3] : "flow_sphere.vti";

  int3 res = make_int3(N, N, std::max(8, N / 4));
  double nu = 0.05, dt = 0.2, fx = 0.02;
  double cx = N * 0.5, cy = N * 0.5, cz = res.z * 0.5, R = N * 0.16;

  DistributedNS sol;
  sol.init(res, rank, size, 1.0, nu, dt);
  sol.set_advection(true);
  sol.set_body_force(fx, 0, 0);

  int3 e = sol.ext(), og = sol.origin_incl_ghost();
  int g = sol.ghost();
  std::size_t n = sol.num_cells();
  std::vector<double> hu(n, 0), hv(n, 0), hw(n, 0), solid(n, 0.0);
  for (int lz = 0; lz < e.z; ++lz)
    for (int ly = 0; ly < e.y; ++ly)
      for (int lx = 0; lx < e.x; ++lx) {
        int gx = ((lx + og.x) % N + N) % N, gy = ((ly + og.y) % N + N) % N,
            gz = ((lz + og.z) % res.z + res.z) % res.z;
        double dx = (gx + 0.5) - cx, dy = (gy + 0.5) - cy, dz = (gz + 0.5) - cz;
        size_t i = (size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y;
        if (dx * dx + dy * dy + dz * dz < R * R) solid[i] = 1.0;
      }
  for (int lz = g; lz < e.z - g; ++lz)
    for (int ly = g; ly < e.y - g; ++ly)
      for (int lx = g; lx < e.x - g; ++lx)
        hu[(size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y] = 1.0;  // uniform inflow
  sol.upload_velocity(hu.data(), hv.data(), hw.data());
  sol.set_solid(solid);

  if (rank == 0)
    std::printf("# demo: %dx%dx%d on %d ranks, sphere R=%.1f, %d steps\n", res.x, res.y, res.z, size,
                R, steps);
  double t0 = MPI_Wtime();
  for (int s = 0; s < steps; ++s) sol.step(/*diff*/ 30, /*pois*/ 40);
  double t = MPI_Wtime() - t0;

  std::vector<double> gu = sol.gather_to_root(sol.u(), 0);
  std::vector<double> gv = sol.gather_to_root(sol.v(), 0);
  std::vector<double> gw = sol.gather_to_root(sol.w(), 0);

  if (rank == 0) {
    tpx::geom::VtiVector vel;
    vel.dims = {res.x, res.y, res.z};
    vel.origin = {0, 0, 0};
    vel.spacing = {1, 1, 1};
    vel.values.resize(3 * gu.size());
    double umax = 0;
    for (size_t i = 0; i < gu.size(); ++i) {
      vel.values[3 * i] = (float)gu[i];
      vel.values[3 * i + 1] = (float)gv[i];
      vel.values[3 * i + 2] = (float)gw[i];
      umax = std::fmax(umax, std::sqrt(gu[i] * gu[i] + gv[i] * gv[i] + gw[i] * gw[i]));
    }
    tpx::geom::writeVtiVector(out, vel, "velocity");  // ParaView glyphs/streamlines
    std::printf("# wrote %s  (peak speed %.4f, %.1f ms/step over %d ranks)\n", out.c_str(), umax,
                1e3 * t / steps, size);
  }
  MPI_Finalize();
  return 0;
}
