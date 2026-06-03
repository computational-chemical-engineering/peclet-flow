// Distributed global reductions over a MacGridHalo's owned (inner) cells: max(|.|) (CFL / convergence)
// and remove_mean (the pure-Neumann pressure null-space removal). Both are needed by the distributed
// multigrid pressure solve and any real distributed step().
//
// Reference: the global field is a deterministic hash over global coords, so the host can compute the
// EXACT global sum / max|.| / mean by sweeping all global cells -- no serial GPU run needed. Each rank
// fills its inner block from the same hash; mac_reduce / mac_remove_mean must reproduce the host
// reference. max is order-independent (compared exactly); the atomic sum differs only by float rounding
// (compared to a tight relative tolerance).
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "mac_halo.cuh"
#include "mac_reductions.cuh"

__host__ __device__ inline double hash01(int x, int y, int z, int seed) {
  unsigned long long h = (unsigned long long)(x * 73856093) ^ (unsigned long long)(y * 19349663) ^
                         (unsigned long long)(z * 83492791) ^ (unsigned long long)(seed * 2654435761u);
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return (double)(h & 0xFFFFFFULL) / (double)0x1000000ULL - 0.5;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  cudaSetDevice(0);

  int3 res = make_int3(40, 28, 24);
  const int seed = 5;

  // --- Host reference over the full global grid (exact) ---
  double ref_sum = 0.0, ref_maxabs = 0.0;
  for (int z = 0; z < res.z; ++z)
    for (int y = 0; y < res.y; ++y)
      for (int x = 0; x < res.x; ++x) {
        double f = hash01(x, y, z, seed);
        ref_sum += f;
        ref_maxabs = std::fmax(ref_maxabs, std::fabs(f));
      }
  long long gcount = (long long)res.x * res.y * res.z;
  double ref_mean = ref_sum / (double)gcount;
  // max|f - mean| (for the remove_mean check)
  double ref_maxabs_centered = 0.0;
  for (int z = 0; z < res.z; ++z)
    for (int y = 0; y < res.y; ++y)
      for (int x = 0; x < res.x; ++x)
        ref_maxabs_centered =
            std::fmax(ref_maxabs_centered, std::fabs(hash01(x, y, z, seed) - ref_mean));

  // --- Distributed: fill inner block from the same hash ---
  MacGridHalo mac;
  mac.init(res, rank, size, {true, true, true}, /*ghost_width=*/2, MPI_COMM_WORLD);
  int3 ext = mac.local_ext, og = mac.origin_incl_ghost;
  size_t nl = mac.num_local_cells();
  std::vector<double> hf(nl, 0.0);
  for (int lz = mac.ghost; lz < ext.z - mac.ghost; ++lz)
    for (int ly = mac.ghost; ly < ext.y - mac.ghost; ++ly)
      for (int lx = mac.ghost; lx < ext.x - mac.ghost; ++lx) {
        size_t i = (size_t)lx + (size_t)ly * ext.x + (size_t)lz * ext.x * ext.y;
        hf[i] = hash01(lx + og.x, ly + og.y, lz + og.z, seed);
      }
  double* ef = nullptr;
  cudaMalloc(&ef, nl * 8);
  cudaMemcpy(ef, hf.data(), nl * 8, cudaMemcpyHostToDevice);

  // reduce: sum + max|.|
  double gsum = 0.0, gmax = 0.0;
  cfdmpi::mac_reduce(ef, mac, MPI_COMM_WORLD, &gsum, &gmax);

  // remove_mean: subtract the global mean over the whole extended block
  cfdmpi::mac_remove_mean(ef, mac, MPI_COMM_WORLD);
  // after removal: global sum ~0 and max|.| == ref_maxabs_centered
  double gsum2 = 0.0, gmax2 = 0.0;
  cfdmpi::mac_reduce(ef, mac, MPI_COMM_WORLD, &gsum2, &gmax2);
  cudaFree(ef);

  int fail = 0;
  if (rank == 0) {
    double err_sum = std::fabs(gsum - ref_sum) / (std::fabs(ref_sum) + 1e-300);
    double err_max = std::fabs(gmax - ref_maxabs);
    double err_centered = std::fabs(gmax2 - ref_maxabs_centered);
    // gsum2 should be ~0; scale tolerance by the field magnitude * cell count
    double sum2_scale = ref_maxabs * (double)gcount;
    double err_sum2 = std::fabs(gsum2) / sum2_scale;

    if (err_sum > 1e-11) fail = 1;
    if (err_max != 0.0) fail = 1;     // max is order-independent -> exact
    if (err_centered != 0.0) fail = 1;
    if (err_sum2 > 1e-11) fail = 1;

    printf("np=%d  res=%dx%dx%d\n", size, res.x, res.y, res.z);
    printf("  sum:        dist=%.15e  ref=%.15e  rel.err=%.2e\n", gsum, ref_sum, err_sum);
    printf("  max|.|:     dist=%.15e  ref=%.15e  abs.err=%.2e\n", gmax, ref_maxabs, err_max);
    printf("  remove_mean: residual sum=%.3e (rel %.2e)  max|f-mean| dist=%.15e ref=%.15e (err %.2e)\n",
           gsum2, err_sum2, gmax2, ref_maxabs_centered, err_centered);
    printf("  %s\n", fail ? "FAIL" : "OK");
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return fail;
}
