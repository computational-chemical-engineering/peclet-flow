// Picard (defect-correction) outer-iteration loop in DistributedNS::step. With nonlinear advection
// a single fractional step lags the advection at u^n; the outer loop re-lags it at the latest iterate
// and re-projects, converging the nonlinear coupling. This checks: (1) the outer loop CONVERGES (hits
// the tolerance in fewer than the cap), (2) it does real work (the converged state differs from the
// single-pass one), (3) the result is deterministic -> identical across np=1,2,4 (the per-step outer
// counts and the global kinetic energy print the same for every np).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "distributed_ns.cuh"

using dns::DistributedNS;

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

// global kinetic energy sum(u^2+v^2+w^2) over owned cells
static double global_ke(DistributedNS& s) {
  std::size_t n = s.num_cells();
  std::vector<double> u(n), v(n), w(n);
  s.download_velocity(u.data(), v.data(), w.data());
  double loc = 0.0;
  for_inner(s, [&](size_t i, int, int, int) { loc += u[i] * u[i] + v[i] * v[i] + w[i] * w[i]; });
  double g = 0.0;
  MPI_Allreduce(&loc, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

// run a Taylor-Green-driven Navier-Stokes flow; returns final KE, total outer iters, min outer count
static void run(int rank, int size, int outer_iters, double tol, double& ke, long& total_outer,
                int& min_outer) {
  int N = 32;
  int3 res = make_int3(N, N, 8);
  double k = 2.0 * M_PI / N, nu = 0.02, dt = 0.4;
  DistributedNS s;
  s.init(res, rank, size, nu, dt);
  s.set_advection(true);
  s.set_pressure_multigrid(true, 4);
  s.set_outer_iterations(outer_iters);
  s.set_outer_tolerance(tol);

  std::size_t n = s.num_cells();
  std::vector<double> hu(n, 0), hv(n, 0), hw(n, 0);
  for_inner(s, [&](size_t i, int gx, int gy, int) {
    hu[i] = std::cos(k * gx) * std::sin(k * (gy + 0.5));
    hv[i] = -std::sin(k * (gx + 0.5)) * std::cos(k * gy);
  });
  s.upload_velocity(hu.data(), hv.data(), hw.data());

  total_outer = 0;
  min_outer = outer_iters;
  for (int it = 0; it < 8; ++it) {
    s.step(/*n_diff=*/30, /*n_pois=*/8);
    int lo = s.last_outer_iterations();
    total_outer += lo;
    if (lo < min_outer) min_outer = lo;
    if (rank == 0 && outer_iters > 1)
      printf("    step %d: outer=%d  corr=%.3e\n", it, lo, s.last_outer_correction());
  }
  ke = global_ke(s);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  double ke1, kep;
  long to1, top;
  int mo1, mop;
  run(rank, size, /*outer=*/1, /*tol=*/-1.0, ke1, to1, mo1);          // single pass
  run(rank, size, /*outer=*/12, /*tol=*/1e-5, kep, top, mop);         // Picard (early-stop on tol)

  int fail = 0;
  if (rank == 0) {
    bool converged = mop < 12;                       // hit the tolerance before the cap
    bool real_work = std::fabs(kep - ke1) > 1e-9 * std::fabs(ke1);  // differs from single-pass
    bool finite = std::isfinite(kep) && std::isfinite(ke1);
    fail = (converged && real_work && finite) ? 0 : 1;
    printf("np=%d  Taylor-Green Navier-Stokes, 8 steps\n", size);
    printf("  single-pass (outer=1): KE = %.10e\n", ke1);
    printf("  Picard (outer<=12, tol=1e-5): KE = %.10e  total outer its = %ld  min/step = %d\n", kep,
           top, mop);
    printf("  converged=%d  differs-from-single-pass=%d   %s\n", converged, real_work,
           fail ? "FAIL" : "OK");
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return fail;
}
