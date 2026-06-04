// Physics validation of the distributed Robust-Scaled velocity IBM: plane Poiseuille flow driven by a
// body force through an SDF-defined channel (walls at non-grid positions -> cut cells). Steady Stokes
// u(y) must match the analytic parabola u(y) = (fx/2nu)(y-ylo)(yhi-y) with no-slip (u=0) at the SDF
// walls. The flow is x-independent so it is divergence-free (no projection); this isolates the
// IBM-modified diffusion solve. The steady solution is deterministic -> identical across np=1,2,4.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "distributed_stokes.cuh"

using dstokes::DistributedStokes;

// non-integer wall positions so the SDF interface falls between u-sample points (-> real cut cells)
static const double kYlo = 10.5, kYhi = 21.5;  // channel walls (index units)
__host__ inline double channel_sdf(double gy) { return fmin(gy - kYlo, kYhi - gy); }

template <typename F>
static void for_inner(const DistributedStokes& s, F&& f) {
  int3 e = s.ext();
  int3 og = s.origin_incl_ghost();
  int g = s.ghost();
  for (int lz = g; lz < e.z - g; ++lz)
    for (int ly = g; ly < e.y - g; ++ly)
      for (int lx = g; lx < e.x - g; ++lx)
        f((size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y, lx + og.x, ly + og.y, lz + og.z);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(16, 32, 8);  // channel along y
  const double nu = 0.1, dt = 50.0, fx = 0.01;  // large dt -> near-steady backward-Euler solve
  DistributedStokes s;
  s.init(res, rank, size, nu, dt);
  s.set_body_force(fx, 0.0, 0.0);

  // SDF channel (all extended cells, from global y); negative inside the solid walls
  int3 e = s.ext(), og = s.origin_incl_ghost();
  size_t n = s.num_cells();
  std::vector<double> sdf(n);
  for (int lz = 0; lz < e.z; ++lz)
    for (int ly = 0; ly < e.y; ++ly)
      for (int lx = 0; lx < e.x; ++lx)
        sdf[(size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y] = channel_sdf(ly + og.y);
  s.set_ibm_solid(sdf, make_float3(0, 0, 0));  // no-slip walls

  // drive to steady state (diffusion only; x-independent flow is divergence-free)
  for (int it = 0; it < 25; ++it) s.step(/*n_diff=*/200, /*n_pois=*/0);

  std::vector<double> u(n), v(n), w(n);
  s.download_velocity(u.data(), v.data(), w.data());

  // compare to the analytic parabola at fluid cells; track no-slip at the walls
  double amp = fx / (2.0 * nu);
  double max_rel = 0.0, max_wall = 0.0, lcl_umax = 0.0, ana_umax = amp * 0.25 * (kYhi - kYlo) * (kYhi - kYlo);
  long nfluid = 0;
  for_inner(s, [&](size_t i, int, int gy, int) {
    double sd = channel_sdf(gy);
    double ana = amp * (gy - kYlo) * (kYhi - gy);
    if (sd > 1.0) {  // interior fluid (away from the cut cells): check the parabola
      double rel = fabs(u[i] - ana) / (ana_umax + 1e-30);
      max_rel = fmax(max_rel, rel);
      lcl_umax = fmax(lcl_umax, u[i]);
      ++nfluid;
    }
    if (sd < -1.0) max_wall = fmax(max_wall, fabs(u[i]));  // deep solid: must stay ~0
  });

  double g_rel = 0, g_wall = 0, g_umax = 0;
  long g_nf = 0;
  MPI_Reduce(&max_rel, &g_rel, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&max_wall, &g_wall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&lcl_umax, &g_umax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&nfluid, &g_nf, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  int fail = 0;
  if (rank == 0) {
    bool prof_ok = g_rel < 0.03 && !std::isnan(g_rel);     // parabola within 3%
    bool wall_ok = g_wall < 0.02 * g_umax;                 // velocity ~0 in the solid
    fail = (prof_ok && wall_ok && g_nf > 0) ? 0 : 1;
    printf("np=%d  res=%dx%dx%d  IBM Poiseuille (SDF channel y in [%.0f,%.0f])\n", size, res.x, res.y,
           res.z, kYlo, kYhi);
    printf("  centreline u: numeric %.4f  analytic %.4f\n", g_umax, ana_umax);
    printf("  max rel err vs parabola = %.3e   solid leak max = %.3e   %s\n", g_rel, g_wall,
           fail ? "FAIL" : "OK");
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return fail;
}
