// flow — the width-1 ghost ring INCLUDING EDGES (and corners) of the two fills the full viscous
// stress reads (doc/full_viscous_stress.md §10, WO-0, gate G0).
//
// The transpose term T_c (src/full_stress.hpp) reads mu at i - s_c +- s_a and the time-base
// velocity v_a at i - s_c + s_a: EDGE ghosts of the (a, c) planes. The existing Laplacian
// cross-face viscosity already reads the mu edge ghost (VarFaceProps::beta), so a missing edge here
// would be a latent bug of the shipped operator too (Q7 -> WO-0b).
//
// Fully periodic box, a field f = gx + 1e3*gy + 1e6*gz written on the INNER cells of every rank
// (global coordinates), the ghosts poisoned with NaN, then
//   * fillPropGhosts(f, /*edges=*/true)          -- the mu path (fillMuGhosts)
//   * fillVelGhostsTo(f, comp, 0, false), comp 0-2 -- the u^n path of the §4.3 prelude
// and every cell of the width-1 shell [G-1, e-G+1)^3 that is not inner is compared EXACTLY with
// the periodic-wrapped global value. Mismatches are counted per class (face / edge / corner); the
// gate is 0 in every class. np 4 must cut two axes (a 2x2 split), np 2 at least one.
//
// Second case (review S1, validation/REVIEW_FULLSTRESS.md): inflow -z / outflow +z, x and y
// periodic. The u^n fill (fillVelGhostsTo(f, comp, 0, false)) keeps the projected outflow face of
// w, the first ghost index along z. The distributed fill (np 1 included) must reproduce the
// single-rank fill BITWISE on the whole width-1 shell: every rank also fills the same global field
// on a single-rank solver of the whole box and memcmps its own block's shell against it. The w
// outflow plane carries a distinct value (the "projected" face) so a wrapped inlet value shows.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::CCField;
using peclet::flow::IbmSolver;

static constexpr int NX = 16, NY = 12, NZ = 8;

static double value(int gx, int gy, int gz) {
  gx = ((gx % NX) + NX) % NX;
  gy = ((gy % NY) + NY) % NY;
  gz = ((gz % NZ) + NZ) % NZ;
  return gx + 1e3 * gy + 1e6 * gz;
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
    auto blk = dec.block(rank);
    const int o[3] = {(int)blk.origin[0], (int)blk.origin[1], (int)blk.origin[2]};
    const int ln[3] = {(int)blk.size[0], (int)blk.size[1], (int)blk.size[2]};
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
      std::printf("GHOST_EDGES np=%d grid %dx%dx%d, %d cut axes (need >= %d)\n", size, NX, NY, NZ,
                  ncut, needCut);
    if (ncut < needCut)
      fail = 1;

    IbmSolver s(ln[0], ln[1], ln[2]);
    s.initMpi(dec, MPI_COMM_WORLD);
    constexpr int G = IbmSolver::G;
    const int e[3] = {ln[0] + 2 * G, ln[1] + 2 * G, ln[2] + 2 * G};
    const long n = (long)e[0] * e[1] * e[2];
    CCField f("ghost_edges_f", n);
    auto h = Kokkos::create_mirror_view(f);
    const char* names[4] = {"fillPropGhosts(edges)", "fillVelGhostsTo(c=0)", "fillVelGhostsTo(c=1)",
                            "fillVelGhostsTo(c=2)"};
    for (int path = 0; path < 4; ++path) {
      for (int z = 0; z < e[2]; ++z)
        for (int y = 0; y < e[1]; ++y)
          for (int x = 0; x < e[0]; ++x) {
            const bool inner =
                x >= G && x < e[0] - G && y >= G && y < e[1] - G && z >= G && z < e[2] - G;
            h((long)x + (long)y * e[0] + (long)z * e[0] * e[1]) =
                inner ? value(o[0] + x - G, o[1] + y - G, o[2] + z - G)
                      : std::numeric_limits<double>::quiet_NaN();
          }
      Kokkos::deep_copy(f, h);
      if (path == 0)
        s.fillPropGhosts(f, /*edges=*/true);
      else
        s.fillVelGhostsTo(f, path - 1, 0, /*doOutflow=*/false);
      Kokkos::fence();
      Kokkos::deep_copy(h, f);
      long bad[4] = {0, 0, 0, 0},
           seen[4] = {0, 0, 0, 0};  // by #ghost axes: 1 face, 2 edge, 3 corner
      for (int z = G - 1; z < e[2] - G + 1; ++z)
        for (int y = G - 1; y < e[1] - G + 1; ++y)
          for (int x = G - 1; x < e[0] - G + 1; ++x) {
            const int k =
                (x < G || x >= e[0] - G) + (y < G || y >= e[1] - G) + (z < G || z >= e[2] - G);
            if (k == 0)
              continue;
            ++seen[k];
            const double got = h((long)x + (long)y * e[0] + (long)z * e[0] * e[1]);
            if (!(got == value(o[0] + x - G, o[1] + y - G, o[2] + z - G)))
              ++bad[k];
          }
      long all[8] = {bad[1], bad[2], bad[3], seen[1], seen[2], seen[3], 0, 0};
      long red[8];
      MPI_Allreduce(all, red, 8, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (rank == 0)
        std::printf("  %-22s mismatches: face %ld/%ld  edge %ld/%ld  corner %ld/%ld  %s\n",
                    names[path], red[0], red[3], red[1], red[4], red[2], red[5],
                    (red[0] + red[1] + red[2]) == 0 ? "PASS" : "FAIL");
      if (red[0] + red[1] + red[2] != 0)
        fail = 1;
    }

    // ---- S1: outflow +z (inflow -z), distributed vs single-rank, memcmp on the shell ----
    // Variant 0: x and y periodic. Variant 1: no-slip walls at -x/+x (the domain BCs are applied
    // after the restored face plane). The fills are the prelude's own: fillPropGhosts(edges) for
    // mu and fillVelGhostsTo(f, c, 0, false, /*faceEdges=*/true) for u^n. A shell cell the
    // single-rank fill leaves unwritten (NaN: the tangential components' outflow ghost plane,
    // which doOutflow = false does not touch and T never reads) has no single-rank value to match
    // and is counted apart, not compared.
    const int eg[3] = {NX + 2 * G, NY + 2 * G, NZ + 2 * G};
    CCField fg("ghost_edges_fg", (long)eg[0] * eg[1] * eg[2]);
    auto hg = Kokkos::create_mirror_view(fg);
    // inner cells = value(); the w outflow face (global z index NZ + G, inner x/y) = value + 0.25
    // (the projected face); every other ghost NaN.
    const auto fillField = [&](auto& hh, const int* ee, const int* off, int comp) {
      for (int z = 0; z < ee[2]; ++z)
        for (int y = 0; y < ee[1]; ++y)
          for (int x = 0; x < ee[0]; ++x) {
            const bool inXY = x >= G && x < ee[0] - G && y >= G && y < ee[1] - G;
            const bool inner = inXY && z >= G && z < ee[2] - G;
            const int gz = off[2] + z - G;
            const bool face = comp == 2 && inXY && gz == NZ;
            hh((long)x + (long)y * ee[0] + (long)z * ee[0] * ee[1]) =
                inner  ? value(off[0] + x - G, off[1] + y - G, gz)
                : face ? value(off[0] + x - G, off[1] + y - G, 0) + 0.25
                       : std::numeric_limits<double>::quiet_NaN();
          }
    };
    const int o0[3] = {0, 0, 0};
    const char* vname[2] = {"x,y periodic", "x walls"};
    for (int variant = 0; variant < 2; ++variant) {
      const auto setBcs = [variant](IbmSolver& q) {
        q.setDomainBc(4, 2, 0.0, 0.0, 1e-3);  // -z inflow
        q.setDomainBc(5, 3, 0.0, 0.0, 0.0);   // +z outflow
        if (variant == 1) {
          q.setDomainBc(0, 1, 0.0, 0.0, 0.0);  // -x wall
          q.setDomainBc(1, 1, 0.0, 0.0, 0.0);  // +x wall
        }
      };
      IbmSolver so(ln[0], ln[1], ln[2]);
      setBcs(so);
      so.initMpi(dec, MPI_COMM_WORLD);
      IbmSolver sg(NX, NY, NZ);  // the single-rank reference on the whole box (every rank, small)
      setBcs(sg);
      for (int path = 0; path < 4; ++path) {
        const int comp = path - 1;  // -1 = the mu path
        fillField(h, e, o, comp);
        Kokkos::deep_copy(f, h);
        fillField(hg, eg, o0, comp);
        Kokkos::deep_copy(fg, hg);
        if (comp < 0) {
          so.fillPropGhosts(f, /*edges=*/true);
          sg.fillPropGhosts(fg, /*edges=*/true);
        } else {
          so.fillVelGhostsTo(f, comp, 0, /*doOutflow=*/false, /*faceEdges=*/true);
          sg.fillVelGhostsTo(fg, comp, 0, /*doOutflow=*/false, /*faceEdges=*/true);
        }
        Kokkos::fence();
        Kokkos::deep_copy(h, f);
        Kokkos::deep_copy(hg, fg);
        long bad[4] = {0, 0, 0, 0}, seen[4] = {0, 0, 0, 0}, undef = 0;
        for (int z = G - 1; z < e[2] - G + 1; ++z)
          for (int y = G - 1; y < e[1] - G + 1; ++y)
            for (int x = G - 1; x < e[0] - G + 1; ++x) {
              const int k =
                  (x < G || x >= e[0] - G) + (y < G || y >= e[1] - G) + (z < G || z >= e[2] - G);
              if (k == 0)
                continue;
              const double got = h((long)x + (long)y * e[0] + (long)z * e[0] * e[1]);
              const double ref = hg((long)(o[0] + x) + (long)(o[1] + y) * eg[0] +
                                    (long)(o[2] + z) * eg[0] * eg[1]);
              if (std::isnan(ref)) {
                ++undef;
                continue;
              }
              ++seen[k];
              if (std::memcmp(&got, &ref, sizeof(double)) != 0)
                ++bad[k];
            }
        long all[7] = {bad[1], bad[2], bad[3], seen[1], seen[2], seen[3], undef}, red[7];
        MPI_Allreduce(all, red, 7, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0)
          std::printf(
              "  outflow+z (%s) %-22s vs single rank (memcmp): face %ld/%ld  edge "
              "%ld/%ld  corner %ld/%ld  (unwritten %ld)  %s\n",
              vname[variant], names[path], red[0], red[3], red[1], red[4], red[2], red[5], red[6],
              (red[0] + red[1] + red[2]) == 0 ? "PASS" : "FAIL");
        if (red[0] + red[1] + red[2] != 0)
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
