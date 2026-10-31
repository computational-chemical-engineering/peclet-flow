// flow — the sharp viscous interface is decomposition-invariant BITWISE
// (doc/vof_sharp_viscous_stress.md §2 "np 1/2/4 bitwise", §7, WO-5 / G-SV6 at the operator level).
//
// Everything the sharp interface adds is a local stencil with a fixed summation order: the band
// classes (7-point colour stencil inside the depth-3 colour ghosts), the edge owners and fractions
// (planes at distance <= 2), the tau_hat / D_nn fit of the owned interfacial cells on u^n (velocity
// ghosts at depth 2 incl. edges), ONE exchange of tau_hat / D_nn, the implicit band and the
// explicit band terms. So: a colour (a sphere, cell fractions from 4^3 samples of the GLOBAL
// coordinates) and a u^n (a hash of the global index) identical for every decomposition are
// written on the inner cells of every rank (NaN-poisoned ghosts), the production pieces run in the
// order step() runs them (updateProperties, resolveSharpViscous, sharpViscousGeometry,
// fullStressPrelude, sharpViscousFit, the bands of makeSharpFaceProps, addFullStressRhs with the
// sharp face props + addSharpViscousRhs), and every inner value -- class, fallback flag, tau_hat,
// D_nn, the 7 band coefficients and the explicit RHS of the 3 components -- is gathered and
// compared bit for bit with a single-rank solver on the whole box. Two passes: triply periodic,
// and no-slip walls on both y faces (the domain-face rules). np 4 must cut two axes.
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::C3;
using peclet::flow::CCConst;
using peclet::flow::CCField;
using peclet::flow::ClosureKind;
using peclet::flow::IbmSolver;
namespace sv = peclet::flow::vof;

static constexpr int NX = 16, NY = 12, NZ = 10;
static constexpr int G = IbmSolver::G;
static constexpr int kVals = 30;  // per inner cell: cls, fb, tau(3), dnn, 3 x 7 bands, 3 rhs

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

// The colour of global cell (gx, gy, gz): a sphere, 4^3 samples (a plane crossing the walls too).
static double colourValue(int gx, int gy, int gz) {
  const double c[3] = {7.3, 5.9, 4.6}, r = 3.7;
  int in = 0;
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i) {
        const double x = gx + (i + 0.5) / 4 - c[0], y = gy + (j + 0.5) / 4 - c[1],
                     z = gz + (k + 0.5) / 4 - c[2];
        in += (x * x + y * y + z * z < r * r) ? 1 : 0;
      }
  return in / 64.0;
}

static std::vector<double> evalSharp(IbmSolver& s, const int o[3], const int ln[3]) {
  const int e[3] = {ln[0] + 2 * G, ln[1] + 2 * G, ln[2] + 2 * G};
  const long n = (long)e[0] * e[1] * e[2];
  std::vector<double> col((long)ln[0] * ln[1] * ln[2]);
  for (int z = 0; z < ln[2]; ++z)
    for (int y = 0; y < ln[1]; ++y)
      for (int x = 0; x < ln[0]; ++x)
        col[x + (long)ln[0] * (y + (long)ln[1] * z)] = colourValue(o[0] + x, o[1] + y, o[2] + z);
  s.setVof(col);
  s.setPropertyModel("mu", ClosureKind::LinearMix, "C", "", {1.0, 1e-2 - 1.0});
  s.setViscousStress("full");
  s.setVofViscousInterface("sharp");
  s.updateProperties();
  s.resolveSharpViscous();
  s.sharpViscousGeometry();
  const auto st = s.sharpViscousState();
  for (int c = 0; c < 3; ++c) {
    auto h = Kokkos::create_mirror_view(st.old[c]);
    for (int z = 0; z < e[2]; ++z)
      for (int y = 0; y < e[1]; ++y)
        for (int x = 0; x < e[0]; ++x) {
          const bool inner =
              x >= G && x < e[0] - G && y >= G && y < e[1] - G && z >= G && z < e[2] - G;
          h((long)x + (long)y * e[0] + (long)z * e[0] * e[1]) =
              inner ? 2.0 * urand(1 + c, o[0] + x - G, o[1] + y - G, o[2] + z - G) - 1.0
                    : std::numeric_limits<double>::quiet_NaN();
        }
    Kokkos::deep_copy(st.old[c], h);
  }
  s.fullStressPrelude();  // mu uniformity + the u^n ghosts incl. edges
  s.sharpViscousFit();
  s.fillMuGhosts();
  Kokkos::fence();
  const auto sv2 = s.sharpViscousState();  // after the fit: the geometry copy is final
  std::vector<std::vector<double>> cols;
  {
    auto cl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sv2.cls);
    auto fb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sv2.fb);
    std::vector<double> a(n), b(n);
    for (long i = 0; i < n; ++i) {
      a[i] = cl(i);
      b[i] = fb(i);
    }
    cols.push_back(a);
    cols.push_back(b);
  }
  auto pull = [&](CCField f) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
    cols.emplace_back(h.data(), h.data() + n);
  };
  for (int a = 0; a < 3; ++a)
    pull(sv2.tau[a]);
  pull(sv2.dnn);
  CCField rs("sv_mpi_rs", n);
  Kokkos::deep_copy(rs, 1.0);
  const CCConst v[3] = {CCConst(sv2.old[0]), CCConst(sv2.old[1]), CCConst(sv2.old[2])};
  const CCConst t[3] = {CCConst(sv2.tau[0]), CCConst(sv2.tau[1]), CCConst(sv2.tau[2])};
  for (int c = 0; c < 3; ++c) {
    CCField B[7];
    for (auto& x : B)
      x = CCField("sv_mpi_band", n);
    peclet::flow::ibmBuildDiffusionVar(B[0], B[1], B[2], B[3], B[4], B[5], B[6], e[0], e[1], e[2],
                                       G, s.makeSharpFaceProps(c), 1.0, 1.0, 1.0);
    Kokkos::fence();
    for (auto& x : B)
      pull(x);
  }
  for (int c = 0; c < 3; ++c) {
    CCField bb("sv_mpi_bb", n);
    peclet::flow::addFullStressRhs<sv::SharpFaceProps, false>(
        c, bb, CCConst(rs), v, s.makeSharpFaceProps(c), C3{e[0], e[1], e[2]}, G, 1.0, false);
    sv::addSharpViscousRhs(c, bb, CCConst(rs), v, sv2.geo, t, CCConst(sv2.dnn), G, 1.0, false);
    Kokkos::fence();
    pull(bb);
  }
  std::vector<double> out;
  out.reserve((long)kVals * ln[0] * ln[1] * ln[2]);
  for (const auto& f : cols)
    for (int z = G; z < e[2] - G; ++z)
      for (int y = G; y < e[1] - G; ++y)
        for (int x = G; x < e[0] - G; ++x)
          out.push_back(f[(long)x + (long)y * e[0] + (long)z * e[0] * e[1]]);
  return out;
}

static void configure(IbmSolver& s, bool walls) {
  if (walls) {
    s.setDomainBc(2, 1, 0.0, 0.0, 0.0);  // -y wall
    s.setDomainBc(3, 1, 0.0, 0.0, 0.0);  // +y wall
  }
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
      std::printf("SHARP_VISCOUS_MPI np=%d grid %dx%dx%d, %d cut axes (need >= %d)\n", size, NX, NY,
                  NZ, ncut, needCut);
    if (ncut < needCut)
      fail = 1;
    auto blk = dec.block(rank);
    const int o[3] = {(int)blk.origin[0], (int)blk.origin[1], (int)blk.origin[2]};
    const int ln[3] = {(int)blk.size[0], (int)blk.size[1], (int)blk.size[2]};
    for (int pass = 0; pass < 2; ++pass) {
      const bool walls = (pass == 1);
      std::vector<double> mine;
      {
        IbmSolver s(ln[0], ln[1], ln[2]);
        configure(s, walls);
        s.initMpi(dec, MPI_COMM_WORLD);
        mine = evalSharp(s, o, ln);
      }
      std::vector<int> counts(size), displs(size);
      long total = 0;
      for (int r = 0; r < size; ++r) {
        const auto b = dec.block(r);
        counts[r] = kVals * (int)(b.size[0] * b.size[1] * b.size[2]);
        displs[r] = (int)total;
        total += counts[r];
      }
      std::vector<double> all(rank == 0 ? total : 0);
      MPI_Gatherv(mine.data(), (int)mine.size(), MPI_DOUBLE, all.data(), counts.data(),
                  displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        const int zo[3] = {0, 0, 0}, gl[3] = {NX, NY, NZ};
        IbmSolver ref(NX, NY, NZ);
        configure(ref, walls);
        const std::vector<double> want = evalSharp(ref, zo, gl);
        long bad = 0, nonfinite = 0, band = 0, fallbacks = 0, rhsNz = 0;
        const long NN = (long)NX * NY * NZ;
        for (int r = 0; r < size; ++r) {
          const auto b = dec.block(r);
          const int bs[3] = {(int)b.size[0], (int)b.size[1], (int)b.size[2]};
          const long bn = (long)bs[0] * bs[1] * bs[2];
          for (int f = 0; f < kVals; ++f)
            for (int z = 0; z < bs[2]; ++z)
              for (int y = 0; y < bs[1]; ++y)
                for (int x = 0; x < bs[0]; ++x) {
                  const double got =
                      all[displs[r] + f * bn + x + (long)bs[0] * (y + (long)bs[1] * z)];
                  const long gi =
                      ((int)b.origin[0] + x) +
                      (long)NX * (((int)b.origin[1] + y) + (long)NY * ((int)b.origin[2] + z));
                  const double w = want[(long)f * NN + gi];
                  if (std::memcmp(&got, &w, sizeof(double)) != 0)
                    ++bad;
                  if (!std::isfinite(got))
                    ++nonfinite;
                  if (f == 0 && got != 0.0)
                    ++band;
                  if (f == 1 && got != 0.0)
                    ++fallbacks;
                  if (f >= kVals - 3 && got != 0.0)
                    ++rhsNz;
                }
        }
        const bool ok = bad == 0 && nonfinite == 0 && band > 100 && fallbacks == 0 && rhsNz > 0;
        std::printf(
            "  %-9s %ld/%ld bitwise mismatches vs np 1, %ld non-finite, %ld band cells, "
            "%ld fallbacks, %ld nonzero RHS  %s\n",
            walls ? "y walls" : "periodic", bad, total, nonfinite, band, fallbacks, rhsNz,
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
