// cfd-gpu — distributed (MPI) cut-cell pressure Poisson + the distributed reductions.
//
// Solves the (singular, periodic) cut-cell pressure system A phi = b, A = -div(open grad), by
// Conjugate Gradients with the field ORB-decomposed over MPI ranks. This exercises the two pieces
// the multi-rank sdflow pressure solve needs beyond the grid halo: (1) the halo'd matvec
// (GridHalo.exchange before applyCutcellOp) and (2) the GLOBAL reductions -- every dot product, the
// residual max, and the constant-null-space mean removal are a local Kokkos parallel_reduce
// followed by MPI_Allreduce. The CG with distributed reductions is algebraically identical to
// single-rank CG (same operator, same global inner products), so a fixed iteration count must match
// a serial reference to ~1e-12. Validates the distributed Krylov + reduction pattern the CutcellMG
// / MG-PCG / Chebyshev pressure drivers will use. Any backend.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>

#include "mac_pressure.hpp"  // buildCutcellOp, applyCutcellOp, CCField, C3
#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/grid_halo.hpp"
#include "peclet/core/halo/grid_halo_topology.hpp"

using peclet::core::Index;
using peclet::core::IVec;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::GridHalo;
using peclet::core::halo::GridHaloTopology;
using peclet::flow::C3;
using peclet::flow::CCConst;
using peclet::flow::CCExec;
using peclet::flow::CCField;
using OpV = Kokkos::View<double*, peclet::flow::CCMem>;

static constexpr int kDim = 3, G = 1, CGIT = 40;

static double source(int gx, int gy, int gz, IVec<kDim> gs) {
  return std::sin(2.0 * M_PI * gx / gs[0]) +
         std::cos(4.0 * M_PI * gy / gs[1]) * std::sin(2.0 * M_PI * gz / gs[2]);
}

// ---- distributed reductions over inner cells (local Kokkos reduce + MPI_Allreduce) ----
struct Reduce {
  C3 e;
  MPI_Comm comm;
  double dot(CCConst a, CCConst b) const {
    CCExec sp;
    double s = 0;
    C3 ee = e;
    Kokkos::parallel_reduce(
        "dot",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(sp, {G, G, G},
                                                       {ee.x - G, ee.y - G, ee.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * ee.x + (long)z * (long)ee.x * ee.y;
          acc += a(i) * b(i);
        },
        s);
    double g = 0;
    MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm);
    return g;
  }
  double maxabs(CCConst a) const {
    CCExec sp;
    double m = 0;
    C3 ee = e;
    Kokkos::parallel_reduce(
        "max",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(sp, {G, G, G},
                                                       {ee.x - G, ee.y - G, ee.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * ee.x + (long)z * (long)ee.x * ee.y;
          const double v = Kokkos::fabs(a(i));
          if (v > acc)
            acc = v;
        },
        Kokkos::Max<double>(m));
    double g = 0;
    MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MAX, comm);
    return g;
  }
  void removeMean(CCField f, long globalCells) const {
    CCExec sp;
    double s = 0;
    C3 ee = e;
    Kokkos::parallel_reduce(
        "meansum",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(sp, {G, G, G},
                                                       {ee.x - G, ee.y - G, ee.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * ee.x + (long)z * (long)ee.x * ee.y;
          acc += f(i);
        },
        s);
    double g = 0;
    MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, comm);
    const double mean = g / (double)globalCells;
    CCField ff = f;
    Kokkos::parallel_for(
        "meansub",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(sp, {G, G, G},
                                                       {ee.x - G, ee.y - G, ee.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * ee.x + (long)z * (long)ee.x * ee.y;
          ff(i) -= mean;
        });
    sp.fence();
  }
};

static void axpy(CCField y, double a, CCConst x) {  // y += a x
  CCExec sp;
  CCField yy = y;
  std::size_t n = y.extent(0);
  Kokkos::parallel_for(
      "axpy", Kokkos::RangePolicy<CCExec>(sp, 0, n),
      KOKKOS_LAMBDA(std::size_t i) { yy(i) += a * x(i); });
  sp.fence();
}
static void aypx(CCField y, double a, CCConst x) {  // y = x + a y
  CCExec sp;
  CCField yy = y;
  std::size_t n = y.extent(0);
  Kokkos::parallel_for(
      "aypx", Kokkos::RangePolicy<CCExec>(sp, 0, n),
      KOKKOS_LAMBDA(std::size_t i) { yy(i) = x(i) + a * yy(i); });
  sp.fence();
}

// Conjugate Gradients (mean-removed, singular periodic operator). exchange() supplies the matvec
// ghosts.
template <class Exch>
static void cg(CCField x, CCConst b, OpV AC, OpV AW, OpV AE, OpV AS, OpV AN, OpV AB, OpV AT, C3 e,
               Exch&& exchange, const Reduce& red, long gCells, int iters) {
  const std::size_t n = x.extent(0);
  CCField r("r", n), p("p", n), Ap("Ap", n);
  auto matvec = [&](CCField y, CCField v) {
    exchange(v);
    applyCutcellOp(y, CCConst(v), AC, AW, AE, AS, AN, AB, AT, e, G);
    red.removeMean(y, gCells);
  };
  Kokkos::deep_copy(x, 0.0);
  Kokkos::deep_copy(r, b);
  red.removeMean(r, gCells);
  Kokkos::deep_copy(p, r);
  double rr = red.dot(CCConst(r), CCConst(r));
  for (int it = 0; it < iters; ++it) {
    matvec(Ap, p);
    const double pAp = red.dot(CCConst(p), CCConst(Ap));
    if (pAp <= 1e-300)
      break;
    const double alpha = rr / pAp;
    axpy(x, alpha, CCConst(p));
    axpy(r, -alpha, CCConst(Ap));
    red.removeMean(r, gCells);
    const double rrn = red.dot(CCConst(r), CCConst(r));
    aypx(p, rrn / rr, CCConst(r));
    rr = rrn;
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    IVec<kDim> gs{24, 20, 16};
    std::array<bool, kDim> periodic{true, true, true};
    const long gCells = (long)gs[0] * gs[1] * gs[2];

    // ---- distributed CG ----
    BlockDecomposer<kDim> dec(static_cast<std::size_t>(size), gs);
    GridHaloTopology<kDim> halo;
    halo.buildTopology(dec, rank, G, periodic, MPI_COMM_WORLD);
    const auto& idx = halo.indexer();
    const Index n = idx.numCellsInclGhost();
    const auto eg = idx.sizeInclGhost(), og = idx.originInclGhost();
    const C3 e{(int)eg[0], (int)eg[1], (int)eg[2]};
    GridHalo<double> dev;
    dev.init(halo);
    Reduce red{e, MPI_COMM_WORLD};

    CCField ox("ox", n), oy("oy", n),
        oz("oz", n);  // all-fluid openness (periodic Laplacian operator)
    Kokkos::deep_copy(ox, 1.0);
    Kokkos::deep_copy(oy, 1.0);
    Kokkos::deep_copy(oz, 1.0);
    OpV AC("AC", n), AW("AW", n), AE("AE", n), AS("AS", n), AN("AN", n), AB("AB", n), AT("AT", n);
    peclet::flow::buildCutcellOp(AC, AW, AE, AS, AN, AB, AT, CCConst(ox), CCConst(oy), CCConst(oz),
                                 e, G, 1.0, 1.0, 1.0);

    CCField b("b", n), x("x", n);
    auto hb = Kokkos::create_mirror_view(b);
    for (Index c = 0; c < n; ++c)
      hb(c) = 0.0;
    idx.forEachInner([&](const IVec<kDim>& l) {
      hb(idx.localMdToLocal(l)) = source(l[0] + og[0], l[1] + og[1], l[2] + og[2], gs);
    });
    Kokkos::deep_copy(b, hb);
    cg(
        x, CCConst(b), AC, AW, AE, AS, AN, AB, AT, e, [&](CCField v) { dev.exchange(v); }, red,
        gCells, CGIT);
    auto hx = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(hx, x);

    // ---- serial reference CG on the full global grid (every rank, redundant; compare its own
    // block) ----
    const C3 ge{(int)gs[0] + 2 * G, (int)gs[1] + 2 * G, (int)gs[2] + 2 * G};
    const Index gn = (Index)ge.x * ge.y * ge.z;
    CCField gox("gox", gn), goy("goy", gn), goz("goz", gn);
    Kokkos::deep_copy(gox, 1.0);
    Kokkos::deep_copy(goy, 1.0);
    Kokkos::deep_copy(goz, 1.0);
    OpV gAC("gAC", gn), gAW("gAW", gn), gAE("gAE", gn), gAS("gAS", gn), gAN("gAN", gn),
        gAB("gAB", gn), gAT("gAT", gn);
    peclet::flow::buildCutcellOp(gAC, gAW, gAE, gAS, gAN, gAB, gAT, CCConst(gox), CCConst(goy),
                                 CCConst(goz), ge, G, 1.0, 1.0, 1.0);
    CCField gb("gb", gn), gx("gx", gn);
    auto hgb = Kokkos::create_mirror_view(gb);
    for (Index c = 0; c < gn; ++c)
      hgb(c) = 0.0;
    for (int z = 0; z < gs[2]; ++z)
      for (int y = 0; y < gs[1]; ++y)
        for (int xx = 0; xx < gs[0]; ++xx)
          hgb((long)(xx + G) + (long)(y + G) * ge.x + (long)(z + G) * (long)ge.x * ge.y) =
              source(xx, y, z, gs);
    Kokkos::deep_copy(gb, hgb);
    Reduce gred{ge, MPI_COMM_SELF};  // serial: SELF comm -> Allreduce is a no-op
    // serial periodic ghost exchange for the matvec
    auto serialFill = [&](CCField f) {
      CCExec sp;
      long st[3] = {1, ge.x, (long)ge.x * ge.y};
      int dims[3] = {ge.x, ge.y, ge.z};
      int N3[3] = {(int)gs[0], (int)gs[1], (int)gs[2]};
      for (int a = 0; a < 3; ++a) {
        const int bb = (a + 1) % 3, cc = (a + 2) % 3;
        const long sa = st[a], sb = st[bb], sc = st[cc];
        const int N = N3[a];
        CCField ff = f;
        Kokkos::parallel_for(
            "sfill",
            Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(sp, {0, 0}, {dims[bb], dims[cc]}),
            KOKKOS_LAMBDA(int p0, int p1) {
              const long base = (long)p0 * sb + (long)p1 * sc;
              for (int gl = 0; gl < G; ++gl) {
                ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
                ff(base + (long)(G + N + gl) * sa) = ff(base + (long)(G + gl) * sa);
              }
            });
        sp.fence();
      }
    };
    cg(gx, CCConst(gb), gAC, gAW, gAE, gAS, gAN, gAB, gAT, ge, serialFill, gred, gCells, CGIT);
    auto hgx = Kokkos::create_mirror_view(gx);
    Kokkos::deep_copy(hgx, gx);

    double maxdiff = 0.0;
    idx.forEachInner([&](const IVec<kDim>& l) {
      const int gxk = l[0] + og[0], gyk = l[1] + og[1], gzk = l[2] + og[2];
      const double a = hx(idx.localMdToLocal(l));
      const double r =
          hgx((long)(gxk + G) + (long)(gyk + G) * ge.x + (long)(gzk + G) * (long)ge.x * ge.y);
      const double d = std::fabs(a - r);
      if (d > maxdiff)
        maxdiff = d;
    });
    if (maxdiff > 1e-11) {
      ++fail;
      std::fprintf(stderr, "[rank %d] max|distributed - serial CG| = %.3e\n", rank, maxdiff);
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): distributed cut-cell pressure CG == serial reference\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d ranks differ\n", size, totalFail);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
