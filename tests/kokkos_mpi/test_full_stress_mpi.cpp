// flow — the transpose term of the full viscous stress is decomposition-invariant BITWISE
// (doc/full_viscous_stress.md WO-4, gate G4 "T bitwise equal at np 1, 2, 4").
//
// T_c (src/full_stress.hpp) is a pointwise stencil on mu and the time-base velocity u^n; what a
// decomposition can change is only the ghost ring it reads -- the mu fill incl. edges
// (fillPropGhosts(edges=true), the fillMuGhosts path) and the u^n fill (fillVelGhostsTo(c, 0,
// false), the fullStressPrelude path). So: a random periodic field (mu log-uniform over a ratio of
// 1e3, u uniform in [-1, 1], a hash of the GLOBAL index, so every decomposition sees the same
// field) is written on the inner cells of every rank with NaN-poisoned ghosts, filled by the
// solver's own fills, and T_c is evaluated by the production kernel with the face props
// makeFaceProps builds for a variable mu (arithmetic mean, staggered placement). Rank 0 evaluates
// the same field on the whole box with a single-rank solver (np 1, periodic local fill) and
// compares the gathered T_c bit for bit, every inner cell, every component.
//
// Two passes: (1) no solid (the `kSolid = false` kernel); (2) a random regular-node mask (about a
// quarter of the nodes irregular) through the `kSolid = true` kernel, its ghosts filled the way
// buildFullStressRegMasks does (double carrier, fillPropGhosts(edges=true), packed to uint8).
// np 4 must cut two axes (a 2x2 split), np 2 at least one.
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <vector>

#include "flow_ibm.hpp"
#include "full_stress.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::C3;
using peclet::flow::CCConst;
using peclet::flow::CCField;
using peclet::flow::IbmSolver;
using peclet::flow::RegConst;
using peclet::flow::RegMask;
using peclet::flow::VarFaceProps;

static constexpr int NX = 16, NY = 12, NZ = 8;
static constexpr int G = IbmSolver::G;

// splitmix64 of (field id, periodic global index) -> uniform [0, 1).
static double urand(int id, int gx, int gy, int gz) {
  gx = ((gx % NX) + NX) % NX;
  gy = ((gy % NY) + NY) % NY;
  gz = ((gz % NZ) + NZ) % NZ;
  std::uint64_t z =
      0x9E3779B97F4A7C15ULL * (std::uint64_t)(1 + id) + (std::uint64_t)(gx + NX * (gy + NY * gz));
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return (double)(z >> 11) * 0x1.0p-53;
}

// Field ids: 0 = mu, 1..3 = u_0..u_2, 4..6 = solid flag of the reg mask of component 0..2.
static double fieldValue(int id, int gx, int gy, int gz) {
  const double r = urand(id, gx, gy, gz);
  if (id == 0)
    return std::pow(10.0, 3.0 * r);  // mu in [1, 1e3)
  if (id <= 3)
    return 2.0 * r - 1.0;         // u in [-1, 1)
  return (r < 0.25) ? 0.0 : 1.0;  // regular-node flag
}

// T_c(u^n) for c = 0..2 on the inner cells of a block (origin o, size ln) owned by solver s,
// flattened x fastest over the inner cells. withReg: the kSolid kernel with the random mask.
static std::vector<double> evalT(IbmSolver& s, const int o[3], const int ln[3], bool withReg) {
  const int e[3] = {ln[0] + 2 * G, ln[1] + 2 * G, ln[2] + 2 * G};
  const long n = (long)e[0] * e[1] * e[2];
  const long st[3] = {1L, (long)e[0], (long)e[0] * e[1]};
  auto fill = [&](CCField f, int id) {
    auto h = Kokkos::create_mirror_view(f);
    for (int z = 0; z < e[2]; ++z)
      for (int y = 0; y < e[1]; ++y)
        for (int x = 0; x < e[0]; ++x) {
          const bool inner =
              x >= G && x < e[0] - G && y >= G && y < e[1] - G && z >= G && z < e[2] - G;
          h((long)x + (long)y * e[0] + (long)z * e[0] * e[1]) =
              inner ? fieldValue(id, o[0] + x - G, o[1] + y - G, o[2] + z - G)
                    : std::numeric_limits<double>::quiet_NaN();
        }
    Kokkos::deep_copy(f, h);
  };
  CCField mu("fs_mpi_mu", n),
      v[3] = {CCField("fs_mpi_v0", n), CCField("fs_mpi_v1", n), CCField("fs_mpi_v2", n)};
  fill(mu, 0);
  s.fillPropGhosts(mu, /*edges=*/true);  // = fillMuGhosts
  for (int c = 0; c < 3; ++c) {
    fill(v[c], 1 + c);
    s.fillVelGhostsTo(v[c], c, 0, /*doOutflow=*/false);  // = fullStressPrelude step 3
  }
  RegMask reg[3];
  if (withReg) {
    CCField tmp("fs_mpi_regtmp", n);
    for (int c = 0; c < 3; ++c) {
      fill(tmp, 4 + c);
      s.fillPropGhosts(tmp, /*edges=*/true);  // = buildFullStressRegMasks
      reg[c] = RegMask("fs_mpi_reg", n);
      RegMask r = reg[c];
      CCField t = tmp;
      Kokkos::parallel_for(
          "fs_mpi_reg_pack", Kokkos::RangePolicy<peclet::flow::CCExec>(0, n),
          KOKKOS_LAMBDA(long i) { r(i) = (t(i) != 0.0) ? 1 : 0; });
    }
  }
  Kokkos::fence();
  const CCConst vc[3] = {CCConst(v[0]), CCConst(v[1]), CCConst(v[2])};
  CCField rs("fs_mpi_rs", n);
  Kokkos::deep_copy(rs, 1.0);
  std::vector<double> out;
  out.reserve(3L * ln[0] * ln[1] * ln[2]);
  for (int c = 0; c < 3; ++c) {
    VarFaceProps fp;  // what makeFaceProps(c) builds for a variable, arithmetic mu (staggered)
    fp.haveMu = true;
    fp.mu = CCConst(mu);
    fp.msc = st[c];
    fp.harmMu = false;
    fp.haveRho = false;
    fp.rhoIdtC = 0.0;
    CCField bb("fs_mpi_bb", n);
    if (withReg) {
      const RegConst rr[3] = {reg[0], reg[1], reg[2]};
      peclet::flow::addFullStressRhs<VarFaceProps, true>(c, bb, CCConst(rs), vc, fp,
                                                         C3{e[0], e[1], e[2]}, G, 1.0, false, rr);
    } else {
      peclet::flow::addFullStressRhs<VarFaceProps, false>(c, bb, CCConst(rs), vc, fp,
                                                          C3{e[0], e[1], e[2]}, G, 1.0, false);
    }
    Kokkos::fence();
    auto h = Kokkos::create_mirror_view(bb);
    Kokkos::deep_copy(h, bb);
    for (int z = G; z < e[2] - G; ++z)
      for (int y = G; y < e[1] - G; ++y)
        for (int x = G; x < e[0] - G; ++x)
          out.push_back(h((long)x + (long)y * e[0] + (long)z * e[0] * e[1]));
  }
  return out;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), NX, NY, NZ);
    const int gn[3] = {NX, NY, NZ};
    int ncut = 0;
    for (int a = 0; a < 3; ++a) {
      bool cut = false;
      for (const auto& sz : dec.sizes())
        cut = cut || ((int)sz[a] != gn[a]);
      ncut += cut ? 1 : 0;
    }
    const int needCut = (size >= 4) ? 2 : (size >= 2 ? 1 : 0);
    if (rank == 0)
      std::printf("FULL_STRESS_MPI np=%d grid %dx%dx%d, %d cut axes (need >= %d)\n", size, NX, NY,
                  NZ, ncut, needCut);
    if (ncut < needCut)
      fail = 1;

    auto blk = dec.block(rank);
    const int o[3] = {(int)blk.origin[0], (int)blk.origin[1], (int)blk.origin[2]};
    const int ln[3] = {(int)blk.size[0], (int)blk.size[1], (int)blk.size[2]};
    IbmSolver s(ln[0], ln[1], ln[2]);
    s.initMpi(dec, MPI_COMM_WORLD);

    for (int pass = 0; pass < 2; ++pass) {
      const bool withReg = (pass == 1);
      const std::vector<double> mine = evalT(s, o, ln, withReg);
      // Gather every rank's block (3 components, inner cells x fastest) on rank 0.
      std::vector<int> counts(size), displs(size);
      long total = 0;
      for (int r = 0; r < size; ++r) {
        const auto b = dec.block(r);
        counts[r] = 3 * (int)(b.size[0] * b.size[1] * b.size[2]);
        displs[r] = (int)total;
        total += counts[r];
      }
      std::vector<double> all(rank == 0 ? total : 0);
      MPI_Gatherv(mine.data(), (int)mine.size(), MPI_DOUBLE, all.data(), counts.data(),
                  displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        const int zo[3] = {0, 0, 0}, gl[3] = {NX, NY, NZ};
        IbmSolver ref(NX, NY, NZ);  // np 1: single-rank, periodic local fill
        const std::vector<double> want = evalT(ref, zo, gl, withReg);
        long bad = 0, nonzero = 0, nonfinite = 0;
        double maxAbs = 0.0;
        for (int r = 0; r < size; ++r) {
          const auto b = dec.block(r);
          const int bs[3] = {(int)b.size[0], (int)b.size[1], (int)b.size[2]};
          const long bn = (long)bs[0] * bs[1] * bs[2];
          for (int c = 0; c < 3; ++c)
            for (int z = 0; z < bs[2]; ++z)
              for (int y = 0; y < bs[1]; ++y)
                for (int x = 0; x < bs[0]; ++x) {
                  const double got =
                      all[displs[r] + c * bn + x + (long)bs[0] * (y + (long)bs[1] * z)];
                  const int gx = (int)b.origin[0] + x, gy = (int)b.origin[1] + y,
                            gz = (int)b.origin[2] + z;
                  const double ref1 =
                      want[(long)c * NX * NY * NZ + gx + (long)NX * (gy + (long)NY * gz)];
                  if (std::memcmp(&got, &ref1, sizeof(double)) != 0)
                    ++bad;
                  if (!std::isfinite(got))
                    ++nonfinite;
                  if (got != 0.0)
                    ++nonzero;
                  maxAbs = std::fmax(maxAbs, std::fabs(ref1));
                }
        }
        const bool ok = (bad == 0) && (nonfinite == 0) && (nonzero > total / 2);
        std::printf(
            "  %-12s T_c vs np 1: %ld/%ld bitwise mismatches, %ld non-finite, %ld nonzero, "
            "max|T| %.3e  %s\n",
            withReg ? "reg mask" : "no solid", bad, total, nonfinite, nonzero, maxAbs,
            ok ? "PASS" : "FAIL");
        if (!ok)
          fail = 1;
      }
    }
  }
  int gfail = 0;
  MPI_Allreduce(&fail, &gfail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  Kokkos::finalize();
  MPI_Finalize();
  return gfail;
}
