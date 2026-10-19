/// @file
/// @brief ctest `decomposition_policy`: the census of the default ("auto") level-0 decomposition
/// against the legacy aligned ORB (doc/vof_distributed_partition_design.md §5.2, §7). Pure host
/// function (CutcellMG::decomposition), no MPI launch, no Kokkos initialisation.
///
/// Census: the design's grids plus every grid flow's MPI ctests and coupling's MPI tests
/// decompose, at np 1..8, 12, 16, 24, 32, 48, 96 (and 192 on 128x96x64). For each configuration,
/// auto = levels 0, legacy = levels 1:
///   1. legacy exact => auto == legacy (origins, sizes, tree split positions);
///   2. maxCells(auto) <= maxCells(legacy);
///   3. nestDepth(auto) >= min(nestDepth(legacy), 3);
///   4. 128x96x64 at np 3/12/24/32/96 (and 192): max == min, >= 4 in-place halvings (evenness);
///   5. 48^3 at np 4: max == min;
///   6. two calls identical;
///   7. splitAxes(auto) is a subset of splitAxes(legacy);
///   8. 768x128x256 and 480x80x160 at np 2..32: a legacy partition that keeps y whole keeps it;
/// plus the §9 block shapes of 128x96x64 at np 3, 12, 24, and np 192 = one 16^3 block per rank.
/// Prints the CHANGED list (auto != legacy): the state-hash re-baseline list.
#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

#include "mac_cutcell_mg.hpp"

using peclet::flow::CutcellMG;
using Dec = peclet::core::decomp::BlockDecomposer<3>;
using peclet::core::Index;
using peclet::core::IVec;

namespace {

int fails = 0;
#define CHECK(cond, ...)        \
  do {                          \
    if (!(cond)) {              \
      std::printf("FAIL ");     \
      std::printf(__VA_ARGS__); \
      std::printf("\n");        \
      ++fails;                  \
    }                           \
  } while (0)

Index cells(const IVec<3>& s) {
  return s[0] * s[1] * s[2];
}
Index maxCells(const Dec& d) {
  Index m = 0;
  for (const auto& s : d.sizes())
    m = std::max(m, cells(s));
  return m;
}
Index minCells(const Dec& d) {
  Index m = cells(d.sizes()[0]);
  for (const auto& s : d.sizes())
    m = std::min(m, cells(s));
  return m;
}
bool same(const Dec& a, const Dec& b) {
  if (a.numBlocks() != b.numBlocks() || a.origins() != b.origins() || a.sizes() != b.sizes())
    return false;
  std::vector<int> da, db;
  std::vector<Index> va, vb;
  a.flattenTree(da, va);
  b.flattenTree(db, vb);
  return da == db && va == vb;
}
// in-place halvings by block evenness alone (no minimum extent), up to 8: the design's "halvings"
int evenHalvings(const Dec& d0) {
  auto can = [](Index e) { return e % 2 == 0 && e / 2 >= 2; };
  Dec d = d0;
  int n = 0;
  for (; n < 8; ++n) {
    IVec<3> r{1, 1, 1};
    for (int k = 0; k < 3; ++k)
      if (can(d.globalSize()[k]))
        r[k] = 2;
    if (r == IVec<3>{1, 1, 1})
      break;
    bool even = true;
    for (std::size_t b = 0; b < d.numBlocks(); ++b)
      for (int k = 0; k < 3; ++k)
        if (r[k] == 2 && (d.origins()[b][k] % 2 || d.sizes()[b][k] % 2))
          even = false;
    if (!even)
      break;
    d = d.coarsened(r);
  }
  return n;
}
// block shape -> count
std::map<std::array<Index, 3>, int> shapes(const Dec& d) {
  std::map<std::array<Index, 3>, int> m;
  for (const auto& s : d.sizes())
    ++m[{s[0], s[1], s[2]}];
  return m;
}
std::string shapeStr(const Dec& d) {
  std::string out;
  char buf[64];
  for (const auto& [s, n] : shapes(d)) {
    std::snprintf(buf, sizeof buf, "%s%dx %lldx%lldx%lld", out.empty() ? "" : " + ", n,
                  (long long)s[0], (long long)s[1], (long long)s[2]);
    out += buf;
  }
  return out;
}

}  // namespace

int main() {
  const std::vector<IVec<3>> grids = {
      // the design's census set
      {128, 96, 64},
      {48, 48, 48},
      {64, 64, 64},
      {96, 96, 96},
      {128, 128, 128},
      {192, 192, 192},
      {256, 128, 128},
      {384, 288, 192},
      {40, 40, 40},
      {100, 50, 50},
      // the grids flow's (tests/kokkos_mpi, tests/regression) and coupling's MPI tests decompose
      {8, 8, 40},
      {16, 8, 32},
      {16, 16, 16},
      {16, 16, 32},
      {16, 32, 8},
      {24, 24, 48},
      {32, 32, 32},
      {32, 64, 16},
      {36, 36, 36},
      {48, 32, 32},
      {48, 32, 40},
      {48, 40, 30},
      {64, 4, 4},
      {64, 48, 32},
      // limiting: no factors of two
      {30, 30, 30}};
  const std::vector<std::size_t> nps = {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 96};
  int nConf = 0, nChanged = 0;
  for (const auto& G : grids) {
    std::vector<std::size_t> list = nps;
    if (G == IVec<3>{128, 96, 64})
      list.push_back(192);
    for (std::size_t np : list) {
      if ((Index)np > cells(G))
        continue;
      ++nConf;
      const Dec L = CutcellMG::decomposition(np, (int)G[0], (int)G[1], (int)G[2], 1);
      const Dec A = CutcellMG::decomposition(np, (int)G[0], (int)G[1], (int)G[2]);
      const Dec A2 = CutcellMG::decomposition(np, (int)G[0], (int)G[1], (int)G[2], 0, 1.05);
      const bool exL = maxCells(L) == minCells(L);
      const int dL = CutcellMG::nestDepth(L), dA = CutcellMG::nestDepth(A);
      char name[64];
      std::snprintf(name, sizeof name, "%lldx%lldx%lld np %zu", (long long)G[0], (long long)G[1],
                    (long long)G[2], np);
      CHECK(!exL || same(A, L), "%s: (1) legacy exact but auto differs", name);
      CHECK(maxCells(A) <= maxCells(L), "%s: (2) maxCells auto %lld > legacy %lld", name,
            (long long)maxCells(A), (long long)maxCells(L));
      CHECK(dA >= std::min(dL, 3), "%s: (3) nestDepth auto %d < min(legacy %d, 3)", name, dA, dL);
      CHECK(same(A, A2), "%s: (6) two calls differ", name);
      CHECK((CutcellMG::splitAxes(A) & ~CutcellMG::splitAxes(L)) == 0u,
            "%s: (7) auto cuts an axis legacy leaves whole (auto %u legacy %u)", name,
            CutcellMG::splitAxes(A), CutcellMG::splitAxes(L));
      Index tot = 0;
      for (const auto& s : A.sizes())
        tot += cells(s);
      CHECK(A.numBlocks() == np && tot == cells(G), "%s: auto does not tile", name);
      if (!same(A, L)) {
        ++nChanged;
        std::printf(
            "CHANGED %s: legacy max/min %.3f depth %d [%s] -> auto max/min %.3f depth %d "
            "[%s]\n",
            name, (double)maxCells(L) / (double)minCells(L), dL, shapeStr(L).c_str(),
            (double)maxCells(A) / (double)minCells(A), dA, shapeStr(A).c_str());
      }
    }
  }
  std::printf("census: %d configurations, %d changed\n", nConf, nChanged);

  // (4), (5), §9 shapes and the np = 192 limiting case
  for (std::size_t np : {3, 12, 24, 32, 96, 192}) {
    const Dec A = CutcellMG::decomposition(np, 128, 96, 64);
    CHECK(maxCells(A) == minCells(A), "128x96x64 np %zu: (4) not exact (%s)", np,
          shapeStr(A).c_str());
    CHECK(evenHalvings(A) >= 4, "128x96x64 np %zu: %d in-place halvings < 4", np, evenHalvings(A));
  }
  {
    const Dec A = CutcellMG::decomposition(4, 48, 48, 48);
    CHECK(maxCells(A) == minCells(A), "48^3 np 4: (5) not exact (%s)", shapeStr(A).c_str());
  }
  using S = std::map<std::array<Index, 3>, int>;
  const std::pair<std::size_t, S> table[] = {
      {3, S{{{128, 32, 64}, 1}, {{64, 64, 64}, 2}}},
      {12, S{{{32, 32, 64}, 0}, {{64, 16, 64}, 0}}},  // shapes only (the design gives no counts)
      {24, S{{{32, 16, 64}, 8}, {{32, 32, 32}, 16}}},
      {192, S{{{16, 16, 16}, 192}}},
  };
  for (const auto& [np, want] : table) {
    const Dec A = CutcellMG::decomposition(np, 128, 96, 64);
    std::printf("128x96x64 np %zu: auto %s\n", np, shapeStr(A).c_str());
    S got = shapes(A);
    if (np == 12)
      for (auto& kv : got)
        kv.second = 0;
    CHECK(got == want, "128x96x64 np %zu: block shapes differ from the design's §9 table", np);
  }
  // (8) channel-shaped grids: a legacy partition that keeps y whole keeps it whole
  for (const auto& G : {IVec<3>{768, 128, 256}, IVec<3>{480, 80, 160}})
    for (std::size_t np = 2; np <= 32; ++np) {
      const Dec L = CutcellMG::decomposition(np, (int)G[0], (int)G[1], (int)G[2], 1);
      const Dec A = CutcellMG::decomposition(np, (int)G[0], (int)G[1], (int)G[2]);
      const bool yWholeL = (CutcellMG::splitAxes(L) & 2u) == 0u;
      CHECK(!yWholeL || (CutcellMG::splitAxes(A) & 2u) == 0u,
            "%lldx%lldx%lld np %zu: (8) auto splits y, legacy does not", (long long)G[0],
            (long long)G[1], (long long)G[2], np);
      CHECK(maxCells(A) <= maxCells(L), "%lldx%lldx%lld np %zu: (2) maxCells", (long long)G[0],
            (long long)G[1], (long long)G[2], np);
      CHECK(CutcellMG::nestDepth(A) >= std::min(CutcellMG::nestDepth(L), 3),
            "%lldx%lldx%lld np %zu: (3) nestDepth auto %d legacy %d", (long long)G[0],
            (long long)G[1], (long long)G[2], np, CutcellMG::nestDepth(A), CutcellMG::nestDepth(L));
      if (!same(A, L))
        std::printf(
            "CHANGED %lldx%lldx%lld np %zu (channel): legacy max/min %.3f depth %d -> auto "
            "max/min %.3f depth %d\n",
            (long long)G[0], (long long)G[1], (long long)G[2], np,
            (double)maxCells(L) / (double)minCells(L), CutcellMG::nestDepth(L),
            (double)maxCells(A) / (double)minCells(A), CutcellMG::nestDepth(A));
    }
  std::printf(fails ? "decomposition_policy: FAIL (%d)\n" : "decomposition_policy: PASS\n", fails);
  return fails ? 1 : 0;
}
