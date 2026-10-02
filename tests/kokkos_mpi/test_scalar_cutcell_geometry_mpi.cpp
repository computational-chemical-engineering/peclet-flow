// Cut-cell scalar geometry record under MPI (doc/scalar_ibm_design.md §2.4-§2.5, §3; WO-2).
//
// np = 1, 2, 4 on the production ORB, two configurations — an off-centre sphere in a periodic box,
// and a sphere cut by a -x WALL with +-x walls and +-y 'slip' faces (constant extension and mirror
// of the SDF ghost band):
//   * the §2.5 prerequisite on the MPI path: sdf_ carries BOTH ghost layers after the GridHalo
//     exchange + extendSdfDomainGhosts (every face-slab ghost sample of layers 1 and 2 equals the
//     global SDF at the expected index — the neighbour's, the periodic image, the extension or the
//     mirror);
//   * the record, by global index, is BITWISE equal to a single-rank build of the same global
//     problem: kappa, the three snapped apertures (inner faces and the high face at ext - G), the
//     unknown flags, and every facet (alpha, normal, centroid, body, probe distance, rung,
//     weights, lattice offsets);
//   * the census summed over ranks equals the single-rank census.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <map>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using IbmSolver = peclet::flow::IbmSolver;
namespace scg = peclet::flow::scg;
static constexpr int G = IbmSolver::G;
static constexpr int N = 32;

static int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

static bool same(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

static int expectIndex(int i, int n, int bcLo, int bcHi) {
  if (i >= 0 && i < n)
    return i;
  const int t = i < 0 ? bcLo : bcHi;
  if (t == 0)
    return ((i % n) + n) % n;
  if (t == 4)
    return i < 0 ? -1 - i : 2 * n - 1 - i;
  return i < 0 ? 0 : n - 1;
}

static std::vector<double> sphereSdf(const double c[3], double R) {
  std::vector<double> s((std::size_t)N * N * N);
  for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
      for (int x = 0; x < N; ++x) {
        const double dx = x - c[0], dy = y - c[1], dz = z - c[2];
        s[(std::size_t)x + (std::size_t)y * N + (std::size_t)z * N * N] =
            std::sqrt(dx * dx + dy * dy + dz * dz) - R;
      }
  return s;
}

struct Host {  // host copies of one record
  int ex, ey, ez;
  Kokkos::View<double*, Kokkos::HostSpace> sdf, kappa, sax, say, saz, unknown;
  std::vector<long> fcell;
  std::vector<double> fdat;  // per facet: alpha, n[3], c[3], sF, w[8]  (16)
  std::vector<int> fint;     // per facet: body, rung, off[8] decoded to dx,dy,dz (26)
  std::map<long, std::vector<long>> byCell;  // extended cell index -> facets
};

static Host hostRecord(IbmSolver& s, int nx, int ny, int nz) {
  Host h;
  h.ex = nx + 2 * G;
  h.ey = ny + 2 * G;
  h.ez = nz + 2 * G;
  const auto& r = s.scalarCutGeometry();
  auto H = Kokkos::HostSpace();
  h.sdf = Kokkos::create_mirror_view_and_copy(H, s.fieldView("sdf"));
  h.kappa = Kokkos::create_mirror_view_and_copy(H, r.kappa);
  h.sax = Kokkos::create_mirror_view_and_copy(H, r.sax);
  h.say = Kokkos::create_mirror_view_and_copy(H, r.say);
  h.saz = Kokkos::create_mirror_view_and_copy(H, r.saz);
  h.unknown = Kokkos::create_mirror_view_and_copy(H, r.unknown);
  auto cell = Kokkos::create_mirror_view_and_copy(H, r.fac.cell);
  auto alpha = Kokkos::create_mirror_view_and_copy(H, r.fac.alpha);
  auto nrm = Kokkos::create_mirror_view_and_copy(H, r.fac.normal);
  auto cen = Kokkos::create_mirror_view_and_copy(H, r.fac.centroid);
  auto body = Kokkos::create_mirror_view_and_copy(H, r.fac.body);
  auto sF = Kokkos::create_mirror_view_and_copy(H, r.fac.sF);
  auto off = Kokkos::create_mirror_view_and_copy(H, r.fac.offF);
  auto w = Kokkos::create_mirror_view_and_copy(H, r.fac.wF);
  auto rung = Kokkos::create_mirror_view_and_copy(H, r.fac.rungF);
  const long sxy = (long)h.ex * h.ey;
  for (long f = 0; f < r.fac.n; ++f) {
    h.fcell.push_back(cell(f));
    h.byCell[cell(f)].push_back(f);
    h.fdat.push_back(alpha(f));
    for (int d = 0; d < 3; ++d)
      h.fdat.push_back(nrm(f, d));
    for (int d = 0; d < 3; ++d)
      h.fdat.push_back(cen(f, d));
    h.fdat.push_back(sF(f));
    for (int k = 0; k < 8; ++k)
      h.fdat.push_back(w(f, k));
    h.fint.push_back(body(f));
    h.fint.push_back((int)rung(f));
    for (int k = 0; k < 8; ++k) {  // decode the linear offset (|d| <= 2 < ex/2)
      const long o = off(f, k);
      const long dz = std::lround((double)o / (double)sxy);
      const long rem = o - dz * sxy;
      const long dy = std::lround((double)rem / (double)h.ex);
      const long dx = rem - dy * h.ex;
      h.fint.push_back((int)dx);
      h.fint.push_back((int)dy);
      h.fint.push_back((int)dz);
    }
  }
  return h;
}

static void runCase(const char* label, const double c[3], double R, const int bc[6], int rank,
                    int size) {
  const std::vector<double> gsdf = sphereSdf(c, R);
  peclet::core::decomp::BlockDecomposer<3> dec =
      peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), N, N, N);
  const auto blk = dec.block(rank);
  const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
  const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
  std::vector<double> lsdf((std::size_t)lnx * lny * lnz);
  for (int z = 0; z < lnz; ++z)
    for (int y = 0; y < lny; ++y)
      for (int x = 0; x < lnx; ++x)
        lsdf[(std::size_t)x + (std::size_t)y * lnx + (std::size_t)z * lnx * lny] =
            gsdf[(std::size_t)(x + ox) + (std::size_t)(y + oy) * N + (std::size_t)(z + oz) * N * N];

  IbmSolver sd(lnx, lny, lnz);
  for (int f = 0; f < 6; ++f)
    if (bc[f] != 0)
      sd.setDomainBc(f, bc[f], 0.0, 0.0, 0.0);
  sd.initMpi(N, N, N, MPI_COMM_WORLD);
  sd.setSolid(lsdf, false);
  IbmSolver sr(N, N, N);  // the single-rank reference of the same global problem
  for (int f = 0; f < 6; ++f)
    if (bc[f] != 0)
      sr.setDomainBc(f, bc[f], 0.0, 0.0, 0.0);
  sr.setSolid(gsdf, false);

  const Host D = hostRecord(sd, lnx, lny, lnz);
  const Host Rf = hostRecord(sr, N, N, N);
  const long dsy = D.ex, dsz = (long)D.ex * D.ey, rsy = Rf.ex, rsz = (long)Rf.ex * Rf.ey;
  auto gIndex = [&](int lx, int ly, int lz) {  // local extended -> reference extended
    return (long)(lx - G + ox + G) + (long)(ly - G + oy + G) * rsy + (long)(lz - G + oz + G) * rsz;
  };

  // (1) sdf_ ghost layers 1-2 on every face slab of this rank's block
  long ghostBad = 0, ghostChecked = 0;
  const int ln[3] = {lnx, lny, lnz}, org[3] = {ox, oy, oz};
  for (int a = 0; a < 3; ++a)
    for (int side = 0; side < 2; ++side)
      for (int layer = 1; layer <= G; ++layer) {
        const int b = (a + 1) % 3, cc = (a + 2) % 3;
        for (int jb = 0; jb < ln[b]; ++jb)
          for (int jc = 0; jc < ln[cc]; ++jc) {
            int l3[3];
            l3[a] = side ? ln[a] - 1 + layer : -layer;
            l3[b] = jb;
            l3[cc] = jc;
            int g3[3];
            for (int d = 0; d < 3; ++d)
              g3[d] = expectIndex(l3[d] + org[d], N, bc[2 * d], bc[2 * d + 1]);
            const double got =
                D.sdf((long)(l3[0] + G) + (long)(l3[1] + G) * dsy + (long)(l3[2] + G) * dsz);
            ++ghostChecked;
            if (!same(
                    got,
                    gsdf[(std::size_t)g3[0] + (std::size_t)g3[1] * N + (std::size_t)g3[2] * N * N]))
              ++ghostBad;
          }
      }

  // (2) per-cell fields, inner cells, plus the high face at ext - G on each axis
  long cellBad = 0, faceBad = 0, facetBad = 0, facetCount = 0;
  for (int z = G; z < lnz + G; ++z)
    for (int y = G; y < lny + G; ++y)
      for (int x = G; x < lnx + G; ++x) {
        const long i = x + y * dsy + z * dsz, j = gIndex(x, y, z);
        if (!same(D.kappa(i), Rf.kappa(j)) || !same(D.unknown(i), Rf.unknown(j)) ||
            !same(D.sax(i), Rf.sax(j)) || !same(D.say(i), Rf.say(j)) || !same(D.saz(i), Rf.saz(j)))
          ++cellBad;
        if (x == lnx + G - 1 && (!same(D.sax(i + 1), Rf.sax(gIndex(x + 1, y, z)))))
          ++faceBad;
        if (y == lny + G - 1 && (!same(D.say(i + dsy), Rf.say(gIndex(x, y + 1, z)))))
          ++faceBad;
        if (z == lnz + G - 1 && (!same(D.saz(i + dsz), Rf.saz(gIndex(x, y, z + 1)))))
          ++faceBad;
        // (3) facets of this cell
        auto itD = D.byCell.find(i);
        auto itR = Rf.byCell.find(j);
        const std::size_t nD = itD == D.byCell.end() ? 0 : itD->second.size();
        const std::size_t nR = itR == Rf.byCell.end() ? 0 : itR->second.size();
        if (nD != nR) {
          ++facetBad;
          continue;
        }
        for (std::size_t k = 0; k < nD; ++k) {
          const long fd = itD->second[k], fr = itR->second[k];
          ++facetCount;
          bool ok = true;
          for (int q = 0; q < 16; ++q)
            ok = ok && same(D.fdat[fd * 16 + q], Rf.fdat[fr * 16 + q]);
          for (int q = 0; q < 26; ++q)
            ok = ok && D.fint[fd * 26 + q] == Rf.fint[fr * 26 + q];
          if (!ok)
            ++facetBad;
        }
      }

  long loc[6] = {ghostBad, ghostChecked, cellBad, faceBad, facetBad, facetCount}, glob[6];
  MPI_Allreduce(loc, glob, 6, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  const auto cD = sd.scalarCutCensus();
  const auto cR = sr.scalarCutCensus();
  const bool censusOk = cD.numUnknowns == cR.numUnknowns && cD.numCutCells == cR.numCutCells &&
                        cD.numFacets == cR.numFacets && cD.numTwoSided == cR.numTwoSided &&
                        cD.numThinSolid == cR.numThinSolid && cD.numSealed == cR.numSealed &&
                        cD.rungs[0] == cR.rungs[0] && cD.rungs[1] == cR.rungs[1] &&
                        cD.rungs[2] == cR.rungs[2] && cD.rungs[3] == cR.rungs[3];
  if (rank == 0)
    std::printf(
        "  np=%d %-22s: ghosts %ld/%ld stale, cells %ld, high faces %ld, facets %ld/%ld "
        "differ; census facets %ld (R0 %ld) unknowns %ld sealed %ld vs single-rank %ld "
        "%ld %ld -> %s\n",
        size, label, glob[0], glob[1], glob[2], glob[3], glob[4], glob[5], cD.numFacets,
        cD.rungs[0], cD.numUnknowns, cD.numSealed, cR.numFacets, cR.numUnknowns, cR.numSealed,
        censusOk ? "equal" : "DIFFER");
  CHECK(glob[0] == 0 && glob[1] > 0);
  CHECK(glob[2] == 0 && glob[3] == 0 && glob[4] == 0);
  CHECK(glob[5] == cR.numFacets);
  CHECK(censusOk);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Kokkos::initialize(argc, argv);
  {
    const int periodic[6] = {0, 0, 0, 0, 0, 0};
    const int walls[6] = {1, 1, 4, 4, 0, 0};
    const double c1[3] = {15.3, 16.1, 14.7}, c2[3] = {2.2, 16.4, 15.6};
    runCase("periodic sphere", c1, 9.3, periodic, rank, size);
    runCase("x walls + y slip", c2, 7.7, walls, rank, size);
  }
  Kokkos::finalize();
  int all = 0;
  MPI_Allreduce(&failures, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(all ? "%d failure(s)\n" : "OK\n", all);
  MPI_Finalize();
  return all ? 1 : 0;
}
