// Step 9: validate the reusable DistributedNS solver (src/distributed_ns.cuh) by reproducing
// two analytic results through its public API, at np=1,2,4:
//   - Taylor-Green vortex decay (no solid)              -> backward-Euler decay rate to <0.1%;
//   - Poiseuille channel flow (slab walls + body force) -> parabolic profile + momentum balance.
// Each rank builds the inner field / solid mask for its block from global coordinates, drives the
// solver, and the analytic checks are reduced across ranks. (Per-operator cell-for-cell consistency
// is already covered by the other tests; here we confirm the assembled, reusable solver is correct.)
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "distributed_ns.cuh"

using dns::DistributedNS;

// Visit each inner cell, calling f(local_index, gx, gy, gz).
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

static int tgv_case(int rank, int size) {
  int N = 32;
  int3 res = make_int3(N, N, 4);
  double k = 2.0 * M_PI / N, nu = 0.05, dt = 0.5;
  DistributedNS s;
  s.init(res, rank, size, nu, dt);
  std::size_t n = s.num_cells();
  std::vector<double> hu(n, 0), hv(n, 0), hw(n, 0);
  for_inner(s, [&](size_t i, int gx, int gy, int) {
    hu[i] = cos(k * gx) * sin(k * (gy + 0.5));
    hv[i] = -sin(k * (gx + 0.5)) * cos(k * gy);
  });
  s.upload_velocity(hu.data(), hv.data(), hw.data());

  double max_u0_l = 0;
  for_inner(s, [&](size_t i, int, int, int) { max_u0_l = fmax(max_u0_l, fabs(hu[i])); });

  const int steps = 20;
  for (int it = 0; it < steps; ++it) s.step(/*diff*/ 60, /*pois*/ 0);

  std::vector<double> ou(n), ov(n), ow(n);
  s.download_velocity(ou.data(), ov.data(), ow.data());
  double max_uf_l = 0;
  for_inner(s, [&](size_t i, int, int, int) { max_uf_l = fmax(max_uf_l, fabs(ou[i])); });

  double max_u0 = 0, max_uf = 0;
  MPI_Allreduce(&max_u0_l, &max_u0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&max_uf_l, &max_uf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double lam = 2.0 * (2.0 - 2.0 * cos(k));
  double expect = pow(1.0 / (1.0 + nu * dt * lam), steps);
  double rel = fabs(max_uf / max_u0 - expect) / expect;
  if (rank == 0) printf("# TGV: decay=%.6f analytic=%.6f rel_err=%.2e\n", max_uf / max_u0, expect, rel);
  return (rel < 1e-3) ? 0 : 1;
}

static int poiseuille_case(int rank, int size) {
  int N = 32;
  int3 res = make_int3(N, N, 4);
  int wall = 4;
  double nu = 0.1, dt = 20.0, g = 1e-3;
  DistributedNS s;
  s.init(res, rank, size, nu, dt);
  s.set_body_force(g, 0, 0);

  // solid mask over ALL local cells (inner+ghost) by wrapped global y.
  std::size_t n = s.num_cells();
  int3 e = s.ext();
  int3 og = s.origin_incl_ghost();
  std::vector<double> solid(n, 0.0);
  for (int lz = 0; lz < e.z; ++lz)
    for (int ly = 0; ly < e.y; ++ly)
      for (int lx = 0; lx < e.x; ++lx) {
        int gy = ((ly + og.y) % N + N) % N;
        if (gy < wall || gy >= N - wall)
          solid[(size_t)lx + (size_t)ly * e.x + (size_t)lz * e.x * e.y] = 1.0;
      }
  s.set_solid(solid);

  for (int it = 0; it < 400; ++it) s.step(/*diff*/ 80, /*pois*/ 0);
  s.exchange_all();  // refresh ghosts so the residual stencil is valid at block boundaries

  std::vector<double> ou(n), ov(n), ow(n);
  s.download_velocity(ou.data(), ov.data(), ow.data());

  long sx = 1, sy = e.x;
  double peak_l = 0, res_l = 0;
  for_inner(s, [&](size_t i, int, int gy, int) {
    if (gy <= wall || gy >= N - wall - 1) return;  // interior fluid only
    peak_l = fmax(peak_l, ou[i]);
    double lap_y = ou[i + sy] - 2.0 * ou[i] + ou[i - sy];
    res_l = fmax(res_l, fabs(nu * lap_y + g));
    (void)sx;
  });
  double peak = 0, resid = 0;
  MPI_Allreduce(&peak_l, &peak, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&res_l, &resid, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double W = N - 2.0 * wall + 1.0, peak_analytic = g * W * W / (8.0 * nu);
  double perr = fabs(peak - peak_analytic) / peak_analytic;
  if (rank == 0)
    printf("# Poiseuille: peak=%.5f analytic=%.5f peak_err=%.2e max|nu*Lap_y u + g|=%.2e\n", peak,
           peak_analytic, perr, resid);
  return (resid < 1e-6 && perr < 0.02) ? 0 : 1;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int f1 = tgv_case(rank, size);
  int f2 = poiseuille_case(rank, size);
  int total = 0, loc = f1 + f2;
  MPI_Allreduce(&loc, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0)
      printf("OK (np=%d): DistributedNS reproduces TGV decay and Poiseuille profile\n", size);
    else
      fprintf(stderr, "FAILED (np=%d): tgv=%d poiseuille=%d\n", size, f1, f2);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
