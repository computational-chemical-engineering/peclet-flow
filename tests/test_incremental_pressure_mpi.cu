// Multi-rank regression test for the incremental-rotational pressure correction
// (DistributedNS::set_incremental_pressure). Stokes flow through a periodic sphere packing with the
// cut-cell IBM + cut-cell pressure operator (PCG). Exercises every new code path -- the predictor's
// -grad(Phi) term (sub_gradpot_k), the rotational potential update (pot_update_k), the Phi halo
// exchange, and the b_[0]-scratch convergence change -- and checks the physical invariants that must
// hold at any rank count: incompressible (small open flux divergence), EXACT no-slip in the deep solid,
// finite, and actual flow. Deterministic -> identical across np=1,2,4.
#include "tpx/common/mpi.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include "distributed_ns.cuh"

using dns::DistributedNS;

// 2x2x2 sphere packing, SDF < 0 inside (min-image), as a host array on the extended block.
__host__ inline double packing_sdf(double x, double y, double z, int N) {
  double R = N * 0.18, best = 1e30;
  for (int cz = 0; cz < 2; ++cz)
    for (int cy = 0; cy < 2; ++cy)
      for (int cx = 0; cx < 2; ++cx) {
        double sx = (cx + 0.5) * N / 2.0, sy = (cy + 0.5) * N / 2.0, sz = (cz + 0.5) * N / 2.0;
        double dx = x - sx, dy = y - sy, dz = z - sz;
        dx -= N * round(dx / N); dy -= N * round(dy / N); dz -= N * round(dz / N);
        best = fmin(best, sqrt(dx * dx + dy * dy + dz * dz) - R);
      }
  return best;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  const int N = 32;
  int3 res = make_int3(N, N, N);
  const double nu = 0.1, dt = 60.0, fx = 1e-3;

  DistributedNS s;
  s.init(res, rank, size, 1.0, nu, dt);
  s.set_body_force(fx, 0.0, 0.0);
  s.set_advection(false);                 // creeping (Stokes)
  s.set_incremental_pressure(true);       // <-- the scheme under test
  s.set_velocity_multigrid(true, 4, 12);
  s.set_pressure_pcg(true, 120, 1e-9);

  int3 e = s.ext(), og = s.origin_incl_ghost();
  std::size_t n = s.num_cells();
  std::vector<double> sdf(n);
  for (int lz = 0; lz < e.z; ++lz)
    for (int ly = 0; ly < e.y; ++ly)
      for (int lx = 0; lx < e.x; ++lx)
        sdf[(std::size_t)lx + (std::size_t)ly * e.x + (std::size_t)lz * e.x * e.y] =
            packing_sdf(lx + og.x, ly + og.y, lz + og.z, N);
  s.set_ibm_solid(sdf, make_float3(0, 0, 0));     // no-slip on the spheres
  s.set_cutcell_pressure_operator(sdf, /*galerkin=*/true);

  int g = s.ghost();
  for (int it = 0; it < 60; ++it) s.step(/*n_diff=*/0, /*n_pois=*/0);

  std::vector<double> u(n), v(n), w(n), p(n);
  s.download_velocity(u.data(), v.data(), w.data());
  cudaMemcpy(p.data(), s.pressure_potential(), n * sizeof(double), cudaMemcpyDeviceToHost);

  // local checks over inner cells
  double leak = 0.0, umax = 0.0, umean = 0.0, pmax = 0.0;
  long nfl = 0, nbad = 0;
  for (int lz = g; lz < e.z - g; ++lz)
    for (int ly = g; ly < e.y - g; ++ly)
      for (int lx = g; lx < e.x - g; ++lx) {
        std::size_t i = (std::size_t)lx + (std::size_t)ly * e.x + (std::size_t)lz * e.x * e.y;
        double sd = packing_sdf(lx + og.x, ly + og.y, lz + og.z, N);
        if (!std::isfinite(u[i]) || !std::isfinite(p[i])) ++nbad;
        if (sd < -2.0) leak = fmax(leak, fabs(u[i]));       // deep solid: must be exactly 0
        if (sd > 0.0) { umax = fmax(umax, fabs(u[i])); umean += u[i]; ++nfl; }
        pmax = fmax(pmax, fabs(p[i]));
      }
  double div = s.max_open_divergence();  // collective

  double g_leak, g_umax, g_pmax, g_umean; long g_nfl, g_nbad;
  MPI_Allreduce(&leak, &g_leak, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&umax, &g_umax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&pmax, &g_pmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&umean, &g_umean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&nfl, &g_nfl, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&nbad, &g_nbad, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  int fail = 0;
  if (rank == 0) {
    double mean = g_umean / (double)g_nfl;
    bool finite = (g_nbad == 0);
    bool incompressible = div < 1e-6 * g_umax;
    bool no_slip = (g_leak == 0.0);
    bool flowing = mean > 0.0 && g_pmax > 0.0;  // incremental scheme built a non-trivial pressure
    fail = (finite && incompressible && no_slip && flowing) ? 0 : 1;
    printf("np=%d  incremental-rotational pressure, sphere packing Stokes flow\n", size);
    printf("  u_max=%.4e  <u>=%.4e  max|p_pot|=%.4e  open-div=%.3e  solid-leak=%.3e\n", g_umax, mean,
           g_pmax, div, g_leak);
    printf("  finite=%d incompressible=%d no-slip=%d flowing=%d   %s\n", finite, incompressible,
           no_slip, flowing, fail ? "FAIL" : "OK");
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return fail;
}
