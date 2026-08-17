/// @file
/// @brief flow — OPT-IN forensics for the ghost-projection overlay (`PECLET_FLOW_GP_DEBUG`).
///
/// Analysis-only instrumentation, in the style of `PECLET_FLOW_AGMG_DEBUG`: nothing here runs
/// unless the environment variable is set, and nothing here feeds the solve. It answers phase-A2
/// of `doc/ghost_hardening_plan.md` — "is the divergence seeded by a handful of identifiable
/// pathological rows, and which branch of the face cascade produces them?".
///
///   PECLET_FLOW_GP_DEBUG=1   print the row/face census: per-state face counts, theta histogram
///                            (incl. the EXTENDED sliver band theta in (1,2)), the row rescale
///                            rho histogram, max |closure weight| per row, and the neighbour
///                            rho-MISMATCH histogram (max_nb |log10(rho_r / rho_nb)|, the row
///                            scaling's contribution to operator asymmetry), plus the 20 worst
///                            rows by rho with their full face anatomy.
///   PECLET_FLOW_GP_DEBUG=2   additionally dump every overlay row to
///                            `PECLET_FLOW_GP_DEBUG_FILE` (default `gp_rows_rank<r>.bin`) as
///                            fixed-width records, for offline correlation against the packing
///                            (see tests/study/ghost_projection_apriori.py --forensics).
///
/// Record layout (little-endian, one per overlay row), matching the numpy dtype in the reader:
///   int32   x, y, z            inner-grid coordinates of the row's cell
///   float32 rho                row rescale = min(1, min_f D_f) over the MATRIX weights
///   int8    coupled            1 if the row has any phi coupling
///   int8    state[6]           face cascade state, slot k = 2*axis + (0 = plus, 1 = minus)
///   float32 th[6]              theta per face
///   float32 wm1[6], wm2[6]     matrix (implicit phi) closure weights
///   float32 w1[6],  w2[6]      rhs/diagnostic closure weights
#ifndef PECLET_FLOW_GHOST_PROJECTION_DEBUG_HPP
#define PECLET_FLOW_GHOST_PROJECTION_DEBUG_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ghost_projection.hpp"

namespace peclet::flow {

/// 0 = off (default). Read once per call; cheap enough, and keeps the flag hot-swappable in tests.
inline int gpDebugLevel() {
  const char* s = std::getenv("PECLET_FLOW_GP_DEBUG");
  return s ? std::atoi(s) : 0;
}

/// Census + optional per-row dump of the built overlay. `nn` is the inner grid, `nRows` the row
/// count returned by buildGpOverlay, `idMap` the inner-cell -> row map (-1 = no row). `rank` only
/// labels the output. Host-side; copies the overlay out of device memory once.
inline void gpDebugReport(const GpOverlay& ov, int nRows, C3 nn,
                          Kokkos::View<int*, CCMem> idMap, int rank = 0) {
  const int level = gpDebugLevel();
  if (level <= 0 || nRows <= 0)
    return;
  const auto h_cell = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), Kokkos::subview(ov.cell, Kokkos::make_pair(0, nRows)));
  const auto h_rho = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), Kokkos::subview(ov.rescale, Kokkos::make_pair(0, nRows)));
  const auto h_cpl = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), Kokkos::subview(ov.coupled, Kokkos::make_pair(0, nRows)));
  const auto sub6 = Kokkos::make_pair(0, 6 * nRows);
  const auto h_st = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        Kokkos::subview(ov.state, sub6));
  const auto h_th =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(ov.th, sub6));
  const auto h_wm1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                         Kokkos::subview(ov.wm_n1, sub6));
  const auto h_wm2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                         Kokkos::subview(ov.wm_n2, sub6));
  const auto h_w1 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(ov.w_n1, sub6));
  const auto h_w2 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(ov.w_n2, sub6));
  const auto h_id = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), idMap);

  const char* stName[5] = {"COUPLED", "QUAD", "LIN", "BC_ONLY", "EXPLICIT"};
  long stCount[5] = {0, 0, 0, 0, 0};
  long extendedTh = 0, clampedThLo = 0, clampedThHi = 0;
  // rho decades: [1e-5,1e-4) ... [1e-1,1) and == 1
  long rhoDec[7] = {0, 0, 0, 0, 0, 0, 0};
  long wDec[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // max |w| decades 1e0..1e7
  long decoupled = 0;
  double rhoMin = 1.0, wMax = 0.0;

  std::vector<float> rowMaxW((std::size_t)nRows, 0.0f);
  for (int s = 0; s < nRows; ++s) {
    if (!h_cpl(s))
      ++decoupled;
    const float rho = h_rho(s);
    rhoMin = rho < rhoMin ? rho : rhoMin;
    int d = rho >= 1.0f ? 6 : (int)std::floor(std::log10((double)rho)) + 6;
    rhoDec[d < 0 ? 0 : (d > 6 ? 6 : d)]++;
    float mw = 0.0f;
    for (int k = 0; k < 6; ++k) {
      const int8_t st = h_st(s * 6 + k);
      stCount[st < 0 || st > 4 ? 0 : st]++;
      const float th = h_th(s * 6 + k);
      if (st == GP_QUAD || st == GP_LIN) {
        if (th > 1.0f)
          ++extendedTh;
        if (th <= (float)(1e-4 * 1.0000001))
          ++clampedThLo;
        if (th >= 2.0f)
          ++clampedThHi;
        const float a1 = std::fabs(h_wm1(s * 6 + k)), a2 = std::fabs(h_wm2(s * 6 + k));
        mw = mw > a1 ? mw : a1;
        mw = mw > a2 ? mw : a2;
        const float b1 = std::fabs(h_w1(s * 6 + k)), b2 = std::fabs(h_w2(s * 6 + k));
        mw = mw > b1 ? mw : b1;
        mw = mw > b2 ? mw : b2;
      }
    }
    rowMaxW[(std::size_t)s] = mw;
    wMax = mw > wMax ? mw : wMax;
    int wd = mw <= 1.0f ? 0 : (int)std::floor(std::log10((double)mw));
    wDec[wd < 0 ? 0 : (wd > 7 ? 7 : wd)]++;
  }

  // Neighbour rho mismatch: the row scaling is a LEFT diagonal scaling of a nonsymmetric operator,
  // so a row whose rho differs sharply from its neighbours' has coefficients a_ij / a_ji differing
  // by that ratio. Cells with no overlay row scale by 1.
  const long nInner = (long)nn.x * nn.y * nn.z;
  long mmDec[6] = {0, 0, 0, 0, 0, 0};  // |log10 ratio| in [0,1),[1,2),...,[5,inf)
  double mmMax = 0.0;
  auto rhoAt = [&](int x, int y, int z) -> float {
    const int ix = ((x % nn.x) + nn.x) % nn.x, iy = ((y % nn.y) + nn.y) % nn.y,
              iz = ((z % nn.z) + nn.z) % nn.z;
    const long c = (long)ix + (long)iy * nn.x + (long)iz * (long)nn.x * nn.y;
    if (c < 0 || c >= nInner)
      return 1.0f;
    const int s = h_id(c);
    return s >= 0 ? h_rho(s) : 1.0f;
  };
  std::vector<float> rowMismatch((std::size_t)nRows, 0.0f);
  for (int s = 0; s < nRows; ++s) {
    const int c = h_cell(s);
    const int x = c % nn.x, y = (c / nn.x) % nn.y, z = c / (nn.x * nn.y);
    const double lr = std::log10((double)h_rho(s));
    double worst = 0.0;
    const int off[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (auto& o : off) {
      const double d = std::fabs(lr - std::log10((double)rhoAt(x + o[0], y + o[1], z + o[2])));
      worst = d > worst ? d : worst;
    }
    rowMismatch[(std::size_t)s] = (float)worst;
    mmMax = worst > mmMax ? worst : mmMax;
    int b = (int)worst;
    mmDec[b < 0 ? 0 : (b > 5 ? 5 : b)]++;
  }

  std::printf("[gp] rank %d overlay census: %d rows of %ld inner cells (%.2f %%), %ld decoupled\n",
              rank, nRows, nInner, 100.0 * nRows / (double)nInner, decoupled);
  std::printf("[gp]   face states:");
  for (int i = 0; i < 5; ++i)
    std::printf("  %s=%ld", stName[i], stCount[i]);
  std::printf("\n[gp]   theta: EXTENDED(1,2)=%ld  clamped-low(1e-4)=%ld  clamped-high(2)=%ld\n",
              extendedTh, clampedThLo, clampedThHi);
  std::printf("[gp]   rho decades  [1e-5,1e-4)=%ld [1e-4,1e-3)=%ld [1e-3,1e-2)=%ld "
              "[1e-2,1e-1)=%ld [1e-1,1)=%ld ==1:%ld   min=%.3e\n",
              rhoDec[0] + rhoDec[1], rhoDec[2], rhoDec[3], rhoDec[4], rhoDec[5], rhoDec[6], rhoMin);
  std::printf("[gp]   max|w| decades");
  for (int i = 0; i < 8; ++i)
    std::printf(" 1e%d:%ld", i, wDec[i]);
  std::printf("   max=%.3e\n", wMax);
  std::printf("[gp]   neighbour rho mismatch |log10| bins");
  for (int i = 0; i < 6; ++i)
    std::printf(" %d:%ld", i, mmDec[i]);
  std::printf("   max=%.2f decades\n", mmMax);

  // the 20 worst rows by rho, with their full face anatomy
  std::vector<int> ord((std::size_t)nRows);
  for (int i = 0; i < nRows; ++i)
    ord[(std::size_t)i] = i;
  const int show = nRows < 20 ? nRows : 20;
  std::partial_sort(ord.begin(), ord.begin() + show, ord.end(),
                    [&](int a, int b) { return h_rho(a) < h_rho(b); });
  std::printf("[gp]   worst %d rows by rho:\n", show);
  for (int i = 0; i < show; ++i) {
    const int s = ord[(std::size_t)i];
    const int c = h_cell(s);
    std::printf("[gp]     (%4d,%4d,%4d) rho=%.4e maxw=%.3e mism=%.2f |", c % nn.x,
                (c / nn.x) % nn.y, c / (nn.x * nn.y), h_rho(s), rowMaxW[(std::size_t)s],
                rowMismatch[(std::size_t)s]);
    for (int k = 0; k < 6; ++k)
      std::printf(" %s(th=%.3e,wm=%.3e/%.3e)", stName[h_st(s * 6 + k)], h_th(s * 6 + k),
                  h_wm1(s * 6 + k), h_wm2(s * 6 + k));
    std::printf("\n");
  }
  std::fflush(stdout);

  if (level < 2)
    return;
  const char* fenv = std::getenv("PECLET_FLOW_GP_DEBUG_FILE");
  std::string path = fenv ? std::string(fenv) : std::string("gp_rows");
  path += "_rank" + std::to_string(rank) + ".bin";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::printf("[gp]   COULD NOT OPEN %s for the per-row dump\n", path.c_str());
    return;
  }
  for (int s = 0; s < nRows; ++s) {
    const int c = h_cell(s);
    const int32_t xyz[3] = {c % nn.x, (c / nn.x) % nn.y, c / (nn.x * nn.y)};
    const float rho = h_rho(s);
    const int8_t cpl = h_cpl(s);
    int8_t st[6];
    float th[6], wm1[6], wm2[6], w1[6], w2[6];
    for (int k = 0; k < 6; ++k) {
      st[k] = h_st(s * 6 + k);
      th[k] = h_th(s * 6 + k);
      wm1[k] = h_wm1(s * 6 + k);
      wm2[k] = h_wm2(s * 6 + k);
      w1[k] = h_w1(s * 6 + k);
      w2[k] = h_w2(s * 6 + k);
    }
    std::fwrite(xyz, sizeof(int32_t), 3, f);
    std::fwrite(&rho, sizeof(float), 1, f);
    std::fwrite(&cpl, sizeof(int8_t), 1, f);
    std::fwrite(st, sizeof(int8_t), 6, f);
    std::fwrite(th, sizeof(float), 6, f);
    std::fwrite(wm1, sizeof(float), 6, f);
    std::fwrite(wm2, sizeof(float), 6, f);
    std::fwrite(w1, sizeof(float), 6, f);
    std::fwrite(w2, sizeof(float), 6, f);
  }
  std::fclose(f);
  std::printf("[gp]   per-row dump -> %s (%d records)\n", path.c_str(), nRows);
  std::fflush(stdout);
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_GHOST_PROJECTION_DEBUG_HPP
