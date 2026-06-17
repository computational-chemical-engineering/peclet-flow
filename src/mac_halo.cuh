/// @file
/// @brief MPI distributed-halo adapter for the MAC grid, built on transport-core.
// cfd-gpu — MPI distributed-halo adapter for the MAC grid, built on transport-core.
//
// The MAC fields (u, v, w staggered; p, sdf centered) are all stored as res.x*res.y*res.z device
// arrays, x-fastest (I = x + y*nx + z*nx*ny), with periodicity currently handled by in-kernel
// wrapping (get_idx in cut_cell_ibm.cuh). This adapter is the first step toward distributing the solver
// across MPI ranks: it decomposes that global cell grid into rank-owned blocks (transport-core ORB),
// and exchanges a width-1 ghost layer for any double cell-field laid out as the *extended local
// block* (inner cells + ghost). A distributed solver replaces global wrapping with: fill ghosts via
// exchange(), then run the existing 7-point stencils on the local extended array.
//
// Ghost width is fixed at 1 (cfd's stencils are 7-point / nearest-neighbour). This adapter is
// independent of the pybind module and the iterative solvers; threading it through step()'s sweeps is
// the next (larger) task — see cfd-gpu/CLAUDE.md.
#pragma once

#include "tpx/common/mpi.hpp"

#include <array>
#include <cstddef>

#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/grid_halo.hpp"
#include "tpx/halo/grid_halo_cuda.cuh"

struct MacGridHalo {
  tpx::decomp::BlockDecomposer<3> dec;
  tpx::halo::GridHalo<3> halo;
  tpx::halo::DeviceGridExchange<double> exch;

  int3 global_res{};       // full grid resolution
  int3 local_ext{};        // this rank's extended-block dims (inner + 2*ghost)
  int3 origin_incl_ghost{};  // global coord of the extended block's (0,0,0) corner
  int rank = 0;
  int size = 1;
  int ghost = 1;

  /// Build the decomposition and halo for `global_res`, assigning block `rank` to this MPI rank.
  /// ghost_width must cover the widest stencil reach (cfd: 1 for the Laplacian/diffusion, 2 for the
  /// Koren TVD advection flux which reads phi_LL..phi_RR).
  void init(int3 global_resolution, int rank_, int size_, std::array<bool, 3> periodic,
            int ghost_width = 1, MPI_Comm comm = MPI_COMM_WORLD) {
    global_res = global_resolution;
    rank = rank_;
    size = size_;
    ghost = ghost_width;
    dec.init(static_cast<std::size_t>(size),
             {global_res.x, global_res.y, global_res.z});
    halo.buildTopology(dec, rank, ghost, periodic, comm);
    exch.init(halo);

    auto e = halo.indexer().sizeInclGhost();
    auto o = halo.indexer().originInclGhost();
    local_ext = make_int3(static_cast<int>(e[0]), static_cast<int>(e[1]), static_cast<int>(e[2]));
    origin_incl_ghost = make_int3(static_cast<int>(o[0]), static_cast<int>(o[1]),
                                  static_cast<int>(o[2]));
  }

  /// Number of cells in the extended local block (allocation size for a local field).
  std::size_t num_local_cells() const {
    return static_cast<std::size_t>(local_ext.x) * local_ext.y * local_ext.z;
  }

  /// Inner-block resolution owned by this rank (excludes ghost).
  int3 inner_res() const {
    return make_int3(local_ext.x - 2 * ghost, local_ext.y - 2 * ghost, local_ext.z - 2 * ghost);
  }

  /// Fill the ghost layer of a device field (extended-block layout) from neighbours. Host-staged.
  void exchange(double* d_field) { exch.exchange(d_field); }
};
