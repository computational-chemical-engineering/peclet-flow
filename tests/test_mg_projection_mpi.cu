// Wire-up test: DistributedStokes::step with the multigrid pressure solve (set_pressure_multigrid).
// Project a divergent random velocity field and measure the residual divergence. The multigrid V-cycle
// must drive div(u) far lower than the single-level Red-Black GS for the same iteration budget (the
// V-cycle kills the low-frequency error that GS leaves), confirming the V-cycle is correctly wired into
// the projection. Distribution exactness is already covered cell-for-cell by test_multigrid_mpi; here
// the global divergence is, by construction, rank-count invariant. np=1,2,4.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "distributed_stokes.cuh"

using dstokes::DistributedStokes;

__host__ inline double hash01(int x, int y, int z, int seed) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791) ^ (unsigned long long)(seed * 2654435761u);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL - 0.5;
}

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

// global max |div(u)| over inner cells (velocity ghosts refreshed first so block-boundary stencils
// read neighbour values).
static double global_max_div(DistributedStokes& s) {
  s.exchange_all();
  std::size_t n = s.num_cells();
  std::vector<double> u(n), v(n), w(n);
  s.download_velocity(u.data(), v.data(), w.data());
  int3 e = s.ext();
  long sx = 1, sy = e.x, sz = (long)e.x * e.y;
  double loc = 0.0;
  for_inner(s, [&](size_t i, int, int, int) {
    double d = (u[i + sx] - u[i]) + (v[i + sy] - v[i]) + (w[i + sz] - w[i]);
    loc = fmax(loc, fabs(d));
  });
  double gmax = 0.0;
  MPI_Allreduce(&loc, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return gmax;
}

// one pure projection of a fixed divergent random field; returns residual max|div|. mg=true uses the
// multigrid V-cycle (n_pois V-cycles), mg=false the single-level GS (n_pois sweeps).
static double project_field(int rank, int size, int3 res, int n_pois, bool mg) {
  DistributedStokes s;
  s.init(res, rank, size, /*nu=*/0.0, /*dt=*/0.1);
  if (mg) s.set_pressure_multigrid(true, /*levels=*/4);
  std::size_t n = s.num_cells();
  std::vector<double> hu(n, 0), hv(n, 0), hw(n, 0);
  for_inner(s, [&](size_t i, int gx, int gy, int gz) {
    hu[i] = hash01(gx, gy, gz, 11);
    hv[i] = hash01(gx, gy, gz, 22);
    hw[i] = hash01(gx, gy, gz, 33);
  });
  s.upload_velocity(hu.data(), hv.data(), hw.data());
  s.step(/*n_diff=*/0, /*n_pois=*/n_pois);  // nu=0, no diffusion -> pure projection of the field
  return global_max_div(s);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(64, 64, 64);
  const int n_pois = 8;

  // initial divergence of the raw field (for context)
  double div0 = project_field(rank, size, res, 0, false);  // n_pois=0 -> no projection
  double div_gs = project_field(rank, size, res, n_pois, false);
  double div_mg = project_field(rank, size, res, n_pois, true);

  int fail = 0;
  if (rank == 0) {
    bool small = (div_mg < 1e-3) && !std::isnan(div_mg);
    bool better = (div_mg < 0.1 * div_gs);  // V-cycle beats GS for the same budget
    fail = (small && better) ? 0 : 1;
    printf("np=%d  res=%dx%dx%d  pure projection, %d iters\n", size, res.x, res.y, res.z, n_pois);
    printf("  initial max|div|              = %.3e\n", div0);
    printf("  single-level GS  (%d sweeps)   = %.3e\n", n_pois, div_gs);
    printf("  multigrid V-cycle (%d cycles)  = %.3e   (%.1fx lower)   %s\n", n_pois, div_mg,
           div_gs / div_mg, small ? "small-OK" : "TOO-LARGE");
    printf("  %s\n", fail ? "FAIL" : "OK");
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return fail;
}
