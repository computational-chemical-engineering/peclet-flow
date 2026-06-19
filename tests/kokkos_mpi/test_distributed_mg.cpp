// cfd-gpu — distributed (MPI) cut-cell pressure MULTIGRID V-cycle.
//
// The third MPI milestone: the geometric V-cycle itself, decomposed over ranks. A level hierarchy is built
// by coarsening the global grid 2:1 per level; each level gets its OWN transport-core halo
// (DeviceGridExchangeKokkos over a BlockDecomposer of that level's grid). The transport-core ORB decomposition
// coarsens cleanly -- each rank's coarse block is exactly its fine block halved -- so the restriction /
// prolongation stay LOCAL (a coarse cell's 2^3 fine children all live on the same rank); only the smoother /
// residual / prolong ghosts cross ranks, via the per-level halo. The singular periodic null space is removed
// with the distributed mean (MPI_Allreduce). Running the same code with MPI_COMM_SELF (a size-1 block = the
// full grid) is the serial reference: the distributed V-cycle must match it to ~1e-11. Validates the per-level-
// halo + local-transfer pattern that turns the single-GPU CutcellMG into the multi-rank pressure multigrid.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "mac_cutcell_mg_kokkos.hpp"  // restrictAvg, prolongAdd, residualCutcell, buildCutcellOp, cutcellSmoothColor

#include "tpx/common/types.hpp"
#include "tpx/common/view.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/grid_halo.hpp"
#include "tpx/halo/grid_halo_kokkos.hpp"

using tpx::Index; using tpx::IVec;
using tpx::decomp::BlockDecomposer;
using tpx::halo::DeviceGridExchangeKokkos; using tpx::halo::GridHalo;
using cfdk::CCField; using cfdk::CCConst; using cfdk::CCExec; using cfdk::C3; using cfdk::FPV; using cfdk::FPC;

static constexpr int kDim = 3, G = 1, NLEV = 4, NVCYC = 6, PRE = 2, POST = 2, BOTTOM = 8;

static double source(int gx, int gy, int gz, IVec<kDim> gs) {
  return std::sin(2.0 * M_PI * gx / gs[0]) + std::cos(4.0 * M_PI * gy / gs[1]) * std::sin(2.0 * M_PI * gz / gs[2]);
}

struct Level {
  std::unique_ptr<GridHalo<kDim>> halo;
  std::unique_ptr<DeviceGridExchangeKokkos<double>> dev;
  C3 e, inner, og, ratio{2, 2, 2}, cfac{1, 1, 1};
  Index n = 0; long gCells = 0;
  CCField x, rhs, res, ox, oy, oz;
  FPV AC, AW, AE, AS, AN, AB, AT;
};

// Distributed multigrid for the periodic all-fluid Laplacian; same code for MPI_COMM_WORLD (decomposed) and
// MPI_COMM_SELF (size-1 full-grid serial reference).
struct DistMG {
  MPI_Comm comm; int rank, size;
  std::vector<Level> lv;

  void init(MPI_Comm c, IVec<kDim> gs) {
    comm = c; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    std::array<bool, kDim> per{true, true, true};
    IVec<kDim> g = gs, cf{1, 1, 1};
    for (int L = 0; L < NLEV; ++L) {
      Level v;
      v.halo = std::make_unique<GridHalo<kDim>>();
      BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), g);
      v.halo->buildTopology(dec, rank, G, per, comm);
      v.dev = std::make_unique<DeviceGridExchangeKokkos<double>>(); v.dev->init(*v.halo);
      const auto& idx = v.halo->indexer();
      const auto eg = idx.sizeInclGhost(), ino = idx.sizeInner(), o = idx.originInclGhost();
      v.e = {(int)eg[0], (int)eg[1], (int)eg[2]}; v.inner = {(int)ino[0], (int)ino[1], (int)ino[2]};
      v.og = {(int)o[0], (int)o[1], (int)o[2]}; v.cfac = {(int)cf[0], (int)cf[1], (int)cf[2]};
      v.n = idx.numCellsInclGhost();
      v.gCells = (long)g[0] * g[1] * g[2];
      v.x = CCField("x", v.n); v.rhs = CCField("rhs", v.n); v.res = CCField("res", v.n);
      v.ox = CCField("ox", v.n); v.oy = CCField("oy", v.n); v.oz = CCField("oz", v.n);
      Kokkos::deep_copy(v.ox, 1.0); Kokkos::deep_copy(v.oy, 1.0); Kokkos::deep_copy(v.oz, 1.0);  // all-fluid
      for (FPV* p : {&v.AC, &v.AW, &v.AE, &v.AS, &v.AN, &v.AB, &v.AT}) *p = FPV("A", v.n);
      const double s = 1.0 / (double)(cf[0]);  // uniform: cfac isotropic here
      cfdk::buildCutcellOp(v.AC, v.AW, v.AE, v.AS, v.AN, v.AB, v.AT, CCConst(v.ox), CCConst(v.oy), CCConst(v.oz),
                           v.e, G, 1.0 / (cf[0] * cf[0]), 1.0 / (cf[1] * cf[1]), 1.0 / (cf[2] * cf[2]));
      (void)s;
      lv.push_back(std::move(v));
      for (int d = 0; d < 3; ++d) { g[d] /= 2; cf[d] *= 2; }
    }
  }

  double meanSum(Level& v, CCField f) {
    CCExec sp; double s = 0; C3 e = v.e;
    Kokkos::parallel_reduce("ms", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(sp, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z, double& a) { a += f((long)x + (long)y * e.x + (long)z * (long)e.x * e.y); }, s);
    double g = 0; MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm); return g;
  }
  void removeMean(Level& v, CCField f) {
    const double mean = meanSum(v, f) / (double)v.gCells; CCExec sp; C3 e = v.e; CCField ff = f;
    Kokkos::parallel_for("rm", Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(sp, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
      KOKKOS_LAMBDA(int x, int y, int z) { ff((long)x + (long)y * e.x + (long)z * (long)e.x * e.y) -= mean; }); sp.fence();
  }
  void smooth(Level& v, int sweeps) {
    for (int k = 0; k < sweeps; ++k)
      for (int color = 0; color < 2; ++color) {
        v.dev->exchange(v.x);
        cfdk::cutcellSmoothColor(v.x, CCConst(v.rhs), FPC(v.AC), FPC(v.AW), FPC(v.AE), FPC(v.AS), FPC(v.AN),
                                 FPC(v.AB), FPC(v.AT), v.e, v.og, G, color);
      }
  }
  void vcycle(int L) {
    Level& v = lv[L];
    if (L + 1 == (int)lv.size()) { smooth(v, BOTTOM); removeMean(v, v.x); return; }
    smooth(v, PRE);
    v.dev->exchange(v.x);
    cfdk::residualCutcell(v.res, CCConst(v.x), CCConst(v.rhs), FPC(v.AC), FPC(v.AW), FPC(v.AE), FPC(v.AS),
                          FPC(v.AN), FPC(v.AB), FPC(v.AT), v.e, G);
    Level& c = lv[L + 1];
    cfdk::restrictAvg(c.rhs, CCConst(v.res), c.e, v.e, G, c.inner, v.ratio);  // LOCAL (coarse block = fine/2)
    Kokkos::deep_copy(c.x, 0.0);
    vcycle(L + 1);
    c.dev->exchange(c.x);
    cfdk::prolongAdd(v.x, CCConst(c.x), v.e, c.e, G, v.inner, v.ratio);       // LOCAL
    smooth(v, POST);
    removeMean(v, v.x);
  }
  // solve A x = b by NVCYC V-cycles; b set into level-0 rhs (mean removed). Solution left in level-0 x.
  void solve(CCField b) {
    Level& l0 = lv[0];
    Kokkos::deep_copy(l0.rhs, b); removeMean(l0, l0.rhs); Kokkos::deep_copy(l0.x, 0.0);
    for (int v = 0; v < NVCYC; ++v) vcycle(0);
  }
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    IVec<kDim> gs{32, 32, 32};

    // distributed V-cycle solve
    DistMG mg; mg.init(MPI_COMM_WORLD, gs);
    Level& l0 = mg.lv[0];
    CCField b("b", l0.n);
    { auto hb = Kokkos::create_mirror_view(b); for (Index c = 0; c < l0.n; ++c) hb(c) = 0.0;
      mg.lv[0].halo->indexer().forEachInner([&](const IVec<kDim>& l) {
        hb(mg.lv[0].halo->indexer().localMdToLocal(l)) = source(l[0] + l0.og.x, l[1] + l0.og.y, l[2] + l0.og.z, gs); });
      Kokkos::deep_copy(b, hb); }
    mg.solve(b);
    auto hx = Kokkos::create_mirror_view(l0.x); Kokkos::deep_copy(hx, l0.x);

    // serial reference: same code, MPI_COMM_SELF (size-1 block = full grid)
    DistMG ref; ref.init(MPI_COMM_SELF, gs);
    Level& r0 = ref.lv[0];
    CCField rb("rb", r0.n);
    { auto hrb = Kokkos::create_mirror_view(rb); for (Index c = 0; c < r0.n; ++c) hrb(c) = 0.0;
      ref.lv[0].halo->indexer().forEachInner([&](const IVec<kDim>& l) {
        hrb(ref.lv[0].halo->indexer().localMdToLocal(l)) = source(l[0] + r0.og.x, l[1] + r0.og.y, l[2] + r0.og.z, gs); });
      Kokkos::deep_copy(rb, hrb); }
    ref.solve(rb);
    auto hr = Kokkos::create_mirror_view(r0.x); Kokkos::deep_copy(hr, r0.x);

    const C3 re = r0.e; double maxdiff = 0.0;
    mg.lv[0].halo->indexer().forEachInner([&](const IVec<kDim>& l) {
      const int gx = l[0] + l0.og.x, gy = l[1] + l0.og.y, gz = l[2] + l0.og.z;
      const double a = hx(mg.lv[0].halo->indexer().localMdToLocal(l));
      const double rr = hr((long)(gx + G) + (long)(gy + G) * re.x + (long)(gz + G) * (long)re.x * re.y);
      const double d = std::fabs(a - rr); if (d > maxdiff) maxdiff = d;
    });
    if (maxdiff > 1e-11) { ++fail; std::fprintf(stderr, "[rank %d] max|distributed - serial V-cycle| = %.3e\n", rank, maxdiff); }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0) std::printf("OK (np=%d): distributed cut-cell MG V-cycle == serial reference\n", size);
    else std::fprintf(stderr, "FAILED (np=%d): %d ranks differ\n", size, totalFail);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
