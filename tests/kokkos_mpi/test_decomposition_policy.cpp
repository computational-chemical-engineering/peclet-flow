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
///   3. nestDepth(auto) >= min(nestDepth(legacy), 3), and nestDepth(auto) >= nestDepth(legacy)
///      whenever auto took a coarse-first partition (neither legacy nor the balance-aware aligned
///      ORB A2, recomputed here; review 2026-10-20 #3);
///   4. 128x96x64 at np 3/12/24/32/96 (and 192): max == min, >= 4 in-place halvings (evenness);
///   5. 48^3 at np 4: max == min;
///   6. two calls identical;
///   7. splitAxes(auto) is a subset of splitAxes(legacy);
///   8. 768x128x256 and 480x80x160 at np 2..32: a legacy partition that keeps y whole keeps it;
///   9. levels 2..7 (the legacy coarse-first search) on the published configurations -- foxberry
///      384^3 and 400^3, channel_dns 377/754/1508/3016 x 240 x 503 and 384x128x128 -- reproduce the
///      pre-partition partitions bit for bit (CutcellMG::fingerprint recorded from core fcb3b06 +
///      flow 91964b7; review 2026-10-20 #1);
///  10. at np > 1, nestDepth(d) == the leading in-place coarsenings (rows neither telescoped nor
///      final) of CutcellMG::predict(telescope = true, min_extent = 4), capped at 4, for d = auto
///      and legacy: nestDepth re-implements initMpi's telescope trigger (review 2026-10-20 #4);
/// plus the §9 block shapes of 128x96x64 at np 3, 12, 24, and np 192 = one 16^3 block per rank.
/// Prints the CHANGED list (auto != legacy): the state-hash re-baseline list.
#include <algorithm>
#include <cstdint>
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
// The balance-aware aligned ORB the auto policy's A2 branch builds (mac_cutcell_mg.hpp).
Dec alignedBalanced(std::size_t np, const IVec<3>& G) {
  Dec d;
  d.init(np, G, CutcellMG::coarsenAlignment((int)G[0], (int)G[1], (int)G[2]), IVec<3>{1, 1, 1},
         peclet::core::decomp::SplitAxis::Balanced);
  return d;
}
// auto took a coarse-first candidate (C5/C4): it is neither legacy nor A2
bool tookCoarseFirst(const Dec& A, const Dec& L, std::size_t np, const IVec<3>& G) {
  return !same(A, L) && !same(A, alignedBalanced(np, G));
}
// In-place coarsenings before the first telescope stage, as CutcellMG::predict forecasts them
// (initMpi's trigger at min extent 4), capped at 4 as nestDepth is.
int predictedNestDepth(const IVec<3>& G, std::size_t np, int decompLevels) {
  const auto rows =
      CutcellMG::predict((int)G[0], (int)G[1], (int)G[2], (int)np, 8, true, 4, decompLevels);
  int n = 0;
  for (const auto& r : rows) {
    if (r.tele || (r.ratio.x == 1 && r.ratio.y == 1 && r.ratio.z == 1))
      break;
    ++n;
  }
  return std::min(n, 4);
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
  int nConf = 0, nChanged = 0, nCoarseFirst = 0;
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
      const bool isC = tookCoarseFirst(A, L, np, G);
      nCoarseFirst += isC ? 1 : 0;
      CHECK(!isC || dA >= dL, "%s: (3) coarse-first auto nests %d < legacy %d", name, dA, dL);
      if (np > 1) {
        const int pA = predictedNestDepth(G, np, 0), pL = predictedNestDepth(G, np, 1);
        CHECK(dA == pA && dL == pL,
              "%s: (10) nestDepth auto %d legacy %d != predict's in-place depth auto %d legacy %d",
              name, dA, dL, pA, pL);
      }
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
  std::printf("census: %d configurations, %d changed, %d coarse-first\n", nConf, nChanged,
              nCoarseFirst);

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
      CHECK(!tookCoarseFirst(A, L, np, G) || CutcellMG::nestDepth(A) >= CutcellMG::nestDepth(L),
            "%lldx%lldx%lld np %zu: (3) coarse-first auto nests %d < legacy %d", (long long)G[0],
            (long long)G[1], (long long)G[2], np, CutcellMG::nestDepth(A), CutcellMG::nestDepth(L));
      CHECK(CutcellMG::nestDepth(A) == predictedNestDepth(G, np, 0) &&
                CutcellMG::nestDepth(L) == predictedNestDepth(G, np, 1),
            "%lldx%lldx%lld np %zu: (10) nestDepth != predict's in-place depth", (long long)G[0],
            (long long)G[1], (long long)G[2], np);
      if (!same(A, L))
        std::printf(
            "CHANGED %lldx%lldx%lld np %zu (channel): legacy max/min %.3f depth %d -> auto "
            "max/min %.3f depth %d\n",
            (long long)G[0], (long long)G[1], (long long)G[2], np,
            (double)maxCells(L) / (double)minCells(L), CutcellMG::nestDepth(L),
            (double)maxCells(A) / (double)minCells(A), CutcellMG::nestDepth(A));
    }
  // (9) levels >= 2: the legacy coarse-first search (SplitAxis::Longest), bit for bit. The values
  // are CutcellMG::fingerprint of CutcellMG::decomposition(np, G, L) built against core fcb3b06 +
  // flow 91964b7 (the pre-partition trees).
  struct Golden {
    IVec<3> g;
    std::size_t np;
    int levels;
    std::uint64_t fp;
  };
  const Golden golden[] = {
      {{384, 384, 384}, 24, 2, 0xc76cada7a9cf6f6full},
      {{384, 384, 384}, 24, 3, 0xc76cada7a9cf6f6full},
      {{384, 384, 384}, 24, 4, 0xc76cada7a9cf6f6full},
      {{384, 384, 384}, 24, 5, 0xc76cada7a9cf6f6full},
      {{384, 384, 384}, 24, 6, 0xc76cada7a9cf6f6full},
      {{384, 384, 384}, 24, 7, 0xc76cada7a9cf6f6full},
      {{384, 384, 384}, 48, 2, 0x56973b1c6f4aa7b7ull},
      {{384, 384, 384}, 48, 3, 0x56973b1c6f4aa7b7ull},
      {{384, 384, 384}, 48, 4, 0x56973b1c6f4aa7b7ull},
      {{384, 384, 384}, 48, 5, 0x56973b1c6f4aa7b7ull},
      {{384, 384, 384}, 48, 6, 0x56973b1c6f4aa7b7ull},
      {{384, 384, 384}, 48, 7, 0x56973b1c6f4aa7b7ull},
      {{384, 384, 384}, 96, 2, 0x8c951d34c8581337ull},
      {{384, 384, 384}, 96, 3, 0x8c951d34c8581337ull},
      {{384, 384, 384}, 96, 4, 0x8c951d34c8581337ull},
      {{384, 384, 384}, 96, 5, 0x8c951d34c8581337ull},
      {{384, 384, 384}, 96, 6, 0x8c951d34c8581337ull},
      {{384, 384, 384}, 96, 7, 0x8c951d34c8581337ull},
      {{384, 384, 384}, 192, 2, 0xe4e414c09bbcdb6cull},
      {{384, 384, 384}, 192, 3, 0xe4e414c09bbcdb6cull},
      {{384, 384, 384}, 192, 4, 0xe4e414c09bbcdb6cull},
      {{384, 384, 384}, 192, 5, 0xe4e414c09bbcdb6cull},
      {{384, 384, 384}, 192, 6, 0xe4e414c09bbcdb6cull},
      {{384, 384, 384}, 192, 7, 0xe4e414c09bbcdb6cull},
      {{384, 384, 384}, 384, 2, 0x0db64c506de5f5c9ull},
      {{384, 384, 384}, 384, 3, 0x0db64c506de5f5c9ull},
      {{384, 384, 384}, 384, 4, 0x0db64c506de5f5c9ull},
      {{384, 384, 384}, 384, 5, 0x0db64c506de5f5c9ull},
      {{384, 384, 384}, 384, 6, 0x0db64c506de5f5c9ull},
      {{384, 384, 384}, 384, 7, 0x0db64c506de5f5c9ull},
      {{384, 384, 384}, 768, 2, 0x2cba956cc53f6297ull},
      {{384, 384, 384}, 768, 3, 0x2cba956cc53f6297ull},
      {{384, 384, 384}, 768, 4, 0x2cba956cc53f6297ull},
      {{384, 384, 384}, 768, 5, 0x2cba956cc53f6297ull},
      {{384, 384, 384}, 768, 6, 0x2cba956cc53f6297ull},
      {{384, 384, 384}, 768, 7, 0x2cba956cc53f6297ull},
      {{384, 384, 384}, 1536, 2, 0x6b2c44fefb80e50cull},
      {{384, 384, 384}, 1536, 3, 0x6b2c44fefb80e50cull},
      {{384, 384, 384}, 1536, 4, 0x6b2c44fefb80e50cull},
      {{384, 384, 384}, 1536, 5, 0x6b2c44fefb80e50cull},
      {{384, 384, 384}, 1536, 6, 0x6b2c44fefb80e50cull},
      {{384, 384, 384}, 1536, 7, 0x6b2c44fefb80e50cull},
      {{400, 400, 400}, 48, 2, 0x54b7dda96e2dd65full},
      {{400, 400, 400}, 48, 3, 0xf56231788a323f4full},
      {{400, 400, 400}, 48, 4, 0xf56231788a323f4full},
      {{400, 400, 400}, 48, 5, 0xf56231788a323f4full},
      {{400, 400, 400}, 48, 6, 0xf56231788a323f4full},
      {{400, 400, 400}, 48, 7, 0xf56231788a323f4full},
      {{400, 400, 400}, 96, 2, 0x352e55a97a9a28efull},
      {{400, 400, 400}, 96, 3, 0x352e55a97a9a28efull},
      {{400, 400, 400}, 96, 4, 0x352e55a97a9a28efull},
      {{400, 400, 400}, 96, 5, 0x352e55a97a9a28efull},
      {{400, 400, 400}, 96, 6, 0x352e55a97a9a28efull},
      {{400, 400, 400}, 96, 7, 0x352e55a97a9a28efull},
      {{400, 400, 400}, 192, 2, 0x0965e39854d7dcb4ull},
      {{400, 400, 400}, 192, 3, 0x0965e39854d7dcb4ull},
      {{400, 400, 400}, 192, 4, 0x0965e39854d7dcb4ull},
      {{400, 400, 400}, 192, 5, 0x0965e39854d7dcb4ull},
      {{400, 400, 400}, 192, 6, 0x0965e39854d7dcb4ull},
      {{400, 400, 400}, 192, 7, 0x0965e39854d7dcb4ull},
      {{400, 400, 400}, 384, 2, 0x27a98983bbceb84dull},
      {{400, 400, 400}, 384, 3, 0x27a98983bbceb84dull},
      {{400, 400, 400}, 384, 4, 0x27a98983bbceb84dull},
      {{400, 400, 400}, 384, 5, 0x27a98983bbceb84dull},
      {{400, 400, 400}, 384, 6, 0x27a98983bbceb84dull},
      {{400, 400, 400}, 384, 7, 0x27a98983bbceb84dull},
      {{377, 240, 503}, 1, 2, 0xe42b457e073921dbull},
      {{377, 240, 503}, 1, 3, 0xe42b457e073921dbull},
      {{377, 240, 503}, 1, 4, 0xe42b457e073921dbull},
      {{377, 240, 503}, 1, 5, 0xe42b457e073921dbull},
      {{377, 240, 503}, 1, 6, 0xe42b457e073921dbull},
      {{377, 240, 503}, 1, 7, 0xe42b457e073921dbull},
      {{754, 240, 503}, 2, 2, 0xe51d0609f3f2fa21ull},
      {{754, 240, 503}, 2, 3, 0xe51d0609f3f2fa21ull},
      {{754, 240, 503}, 2, 4, 0xe51d0609f3f2fa21ull},
      {{754, 240, 503}, 2, 5, 0xe51d0609f3f2fa21ull},
      {{754, 240, 503}, 2, 6, 0xe51d0609f3f2fa21ull},
      {{754, 240, 503}, 2, 7, 0xe51d0609f3f2fa21ull},
      {{1508, 240, 503}, 4, 2, 0xdb90dd30d00501d3ull},
      {{1508, 240, 503}, 4, 3, 0x6483cea14f6e9f6full},
      {{1508, 240, 503}, 4, 4, 0x6483cea14f6e9f6full},
      {{1508, 240, 503}, 4, 5, 0x6483cea14f6e9f6full},
      {{1508, 240, 503}, 4, 6, 0x6483cea14f6e9f6full},
      {{1508, 240, 503}, 4, 7, 0x6483cea14f6e9f6full},
      {{3016, 240, 503}, 8, 2, 0xbb2164ae7892e7f9ull},
      {{3016, 240, 503}, 8, 3, 0x4c4a3e149bcd28fdull},
      {{3016, 240, 503}, 8, 4, 0xcfadf268c155a0bdull},
      {{3016, 240, 503}, 8, 5, 0xcfadf268c155a0bdull},
      {{3016, 240, 503}, 8, 6, 0xcfadf268c155a0bdull},
      {{3016, 240, 503}, 8, 7, 0xcfadf268c155a0bdull},
      {{384, 128, 128}, 2, 2, 0x95d56daae4b3edf8ull},
      {{384, 128, 128}, 2, 3, 0x95d56daae4b3edf8ull},
      {{384, 128, 128}, 2, 4, 0x95d56daae4b3edf8ull},
      {{384, 128, 128}, 2, 5, 0x95d56daae4b3edf8ull},
      {{384, 128, 128}, 2, 6, 0x95d56daae4b3edf8ull},
      {{384, 128, 128}, 2, 7, 0x95d56daae4b3edf8ull},
      {{384, 128, 128}, 4, 2, 0x796d44fff0fc7cbbull},
      {{384, 128, 128}, 4, 3, 0x796d44fff0fc7cbbull},
      {{384, 128, 128}, 4, 4, 0x796d44fff0fc7cbbull},
      {{384, 128, 128}, 4, 5, 0x796d44fff0fc7cbbull},
      {{384, 128, 128}, 4, 6, 0x796d44fff0fc7cbbull},
      {{384, 128, 128}, 4, 7, 0x796d44fff0fc7cbbull},
      {{384, 128, 128}, 8, 2, 0xd76f88eae7c4e594ull},
      {{384, 128, 128}, 8, 3, 0xd76f88eae7c4e594ull},
      {{384, 128, 128}, 8, 4, 0xd76f88eae7c4e594ull},
      {{384, 128, 128}, 8, 5, 0xd76f88eae7c4e594ull},
      {{384, 128, 128}, 8, 6, 0xd76f88eae7c4e594ull},
      {{384, 128, 128}, 8, 7, 0xd76f88eae7c4e594ull},
      {{384, 128, 128}, 16, 2, 0x6cd590d87514e30cull},
      {{384, 128, 128}, 16, 3, 0x6cd590d87514e30cull},
      {{384, 128, 128}, 16, 4, 0x6cd590d87514e30cull},
      {{384, 128, 128}, 16, 5, 0x6cd590d87514e30cull},
      {{384, 128, 128}, 16, 6, 0x6cd590d87514e30cull},
      {{384, 128, 128}, 16, 7, 0x6cd590d87514e30cull},
      {{480, 80, 160}, 16, 2, 0x689bdd6c028b5f1cull},
      {{480, 80, 160}, 16, 3, 0x689bdd6c028b5f1cull},
      {{480, 80, 160}, 16, 4, 0x689bdd6c028b5f1cull},
      {{480, 80, 160}, 16, 5, 0x689bdd6c028b5f1cull},
      {{480, 80, 160}, 16, 6, 0x689bdd6c028b5f1cull},
      {{480, 80, 160}, 16, 7, 0x689bdd6c028b5f1cull},
  };
  int nGolden = 0;
  for (const auto& r : golden) {
    const Dec d = CutcellMG::decomposition(r.np, (int)r.g[0], (int)r.g[1], (int)r.g[2], r.levels);
    ++nGolden;
    CHECK(CutcellMG::fingerprint(d) == r.fp,
          "%lldx%lldx%lld np %zu levels %d: (9) partition differs from the pre-partition one "
          "(fingerprint %016llx, recorded %016llx)",
          (long long)r.g[0], (long long)r.g[1], (long long)r.g[2], r.np, r.levels,
          (unsigned long long)CutcellMG::fingerprint(d), (unsigned long long)r.fp);
  }
  std::printf(
      "levels >= 2: %d (published configuration, levels) rows checked against the pre-partition "
      "trees\n",
      nGolden);
  std::printf(fails ? "decomposition_policy: FAIL (%d)\n" : "decomposition_policy: PASS\n", fails);
  return fails ? 1 : 0;
}
