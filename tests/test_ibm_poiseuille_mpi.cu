// Physics validation of the distributed Robust-Scaled velocity IBM: plane Poiseuille flow driven by a
// body force through an SDF-defined channel (walls at non-grid positions -> cut cells). Steady Stokes
// u(y) must match the analytic parabola u(y) = (fx/2nu)(y-ylo)(yhi-y) with no-slip (u=0) at the SDF
// walls. The flow is x-independent so it is divergence-free (no projection); this isolates the
// IBM-modified diffusion solve. The steady solution is deterministic -> identical across np=1,2,4.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "distributed_ns.cuh"

using dns::DistributedNS;

// non-integer wall positions so the SDF interface falls between u-sample points (-> real cut cells)
static const double kYlo = 10.5, kYhi = 21.5;  // channel walls (index units)
__host__ inline double channel_sdf(double gy) { return fmin(gy - kYlo, kYhi - gy); }

template <typename F>
static void for_inner(const DistributedNS& s, F&& f) {
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
  const double nu = 0.1, dt = 50.0, fx = 0.01;  // large dt -> near-steady backward-Euler (stiff diffusion)
  double amp = fx / (2.0 * nu);
  double ana_umax = amp * 0.25 * (kYhi - kYlo) * (kYhi - kYlo);

  // run the channel with RB-GS (vmg=false) or velocity multigrid (vmg=true); return parabola error +
  // solid leak + centreline velocity.
  auto run = [&](bool vmg, double& g_rel, double& g_wall, double& g_umax) {
    DistributedNS s;
    s.init(res, rank, size, 1.0, nu, dt);
    s.set_incremental_pressure(false);  // analytic Poiseuille (n_pois=0, no projection -> classical Chorin)
    s.set_body_force(fx, 0.0, 0.0);
    int3 e = s.ext(), og = s.origin_incl_ghost();
    size_t n = s.num_cells();
    std::vector<double> sdf(n);
    for (int lz = 0; lz < e.z; ++lz)
      for (int ly = 0; ly < e.y; ++ly)
        for (int lx = 0; lx < e.x; ++lx)
          sdf[(size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y] = channel_sdf(ly + og.y);
    s.set_ibm_solid(sdf, make_float3(0, 0, 0));  // no-slip walls
    if (vmg) s.set_velocity_multigrid(true, /*levels=*/2, /*v_cycles=*/8);

    int steps = vmg ? 25 : 25, ndiff = vmg ? 0 : 200;  // vmg uses V-cycles (n_diff ignored)
    for (int it = 0; it < steps; ++it) s.step(/*n_diff=*/ndiff, /*n_pois=*/0);

    std::vector<double> u(n), v(n), w(n);
    s.download_velocity(u.data(), v.data(), w.data());
    double max_rel = 0.0, max_wall = 0.0, lcl_umax = 0.0;
    long nfluid = 0;
    for_inner(s, [&](size_t i, int, int gy, int) {
      double sd = channel_sdf(gy);
      double ana = amp * (gy - kYlo) * (kYhi - gy);
      if (sd > 1.0) {
        max_rel = fmax(max_rel, fabs(u[i] - ana) / (ana_umax + 1e-30));
        lcl_umax = fmax(lcl_umax, u[i]);
        ++nfluid;
      }
      if (sd < -1.0) max_wall = fmax(max_wall, fabs(u[i]));
    });
    MPI_Allreduce(&max_rel, &g_rel, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_wall, &g_wall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&lcl_umax, &g_umax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  };

  double rg_rel, rg_wall, rg_umax, vg_rel, vg_wall, vg_umax;
  run(/*vmg=*/false, rg_rel, rg_wall, rg_umax);
  run(/*vmg=*/true, vg_rel, vg_wall, vg_umax);

  int fail = 0;
  if (rank == 0) {
    bool rb_ok = rg_rel < 0.03 && rg_wall < 0.02 * rg_umax && !std::isnan(rg_rel);
    bool vmg_ok = vg_rel < 0.03 && vg_wall < 0.02 * vg_umax && !std::isnan(vg_rel);
    fail = (rb_ok && vmg_ok) ? 0 : 1;
    printf("np=%d  res=%dx%dx%d  IBM Poiseuille (SDF channel y in [%.0f,%.0f]), analytic u_max %.4f\n",
           size, res.x, res.y, res.z, kYlo, kYhi, ana_umax);
    printf("  RB-GS  (200 sweeps/step): u_max %.4f  rel err %.3e  solid leak %.3e   %s\n", rg_umax,
           rg_rel, rg_wall, rb_ok ? "OK" : "FAIL");
    printf("  vel-MG (8 V-cycles/step): u_max %.4f  rel err %.3e  solid leak %.3e   %s\n", vg_umax,
           vg_rel, vg_wall, vmg_ok ? "OK" : "FAIL");
    printf("  %s\n", fail ? "FAIL" : "OK");
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return fail;
}
