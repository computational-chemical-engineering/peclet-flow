// G-SYNC (doc/vof_gpu_coarse_tail_design.md §5.5, WO-S4): the device step's host-synchronization
// budget. A small single-rank block-VoF bubble in a walled box (the bubble column's configuration:
// variable density by a linear closure, CSF, MG-PCG, superficial-velocity shift) runs 10 steps
// while an in-process Kokkos Tools fence callback counts, per step,
//   - GLOBAL fences on the device (Kokkos::fence(), exec-less deep_copy: each one drains every
//     stream, and the GPU then idles until the host relaunches), and
//   - combined-reducer fences ("fence after copying values back": a parallel_reduce with >= 2
//     results reduces through host memory and fences twice, even into device Views).
// It asserts both against the measured post-S4 budget. Host backends have no device to drain:
// the test skips there (exit 77).
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"

namespace {
long gGlobal = 0, gCombined = 0;
bool gCount = false;
void onBeginFence(const char* name, const uint32_t devid, uint64_t* handle) {
  *handle = 0;
  if (!gCount)
    return;
  const auto id = Kokkos::Tools::Experimental::identifier_from_devid(devid);
  const bool device = id.type != Kokkos::Tools::Experimental::DeviceType::Serial &&
                      id.type != Kokkos::Tools::Experimental::DeviceType::OpenMP &&
                      id.type != Kokkos::Tools::Experimental::DeviceType::Threads &&
                      id.type != Kokkos::Tools::Experimental::DeviceType::HPX;
  // A global fence reaches the callback once per enabled space with instance bits 0 (and device
  // bits 0): count the device space's only.
  if (device && id.instance_id == 0 && id.device_id == 0)
    ++gGlobal;
  if (std::strstr(name, "fence after copying values back") != nullptr)
    ++gCombined;
}
void onEndFence(uint64_t) {}

// The measured post-S4 budget. Global: 4/step here and on the bubble column (fencebt census), all
// in the two block-container reads the batch_stats/bbox package owns (vof/block_container.hpp read
// #1 and read #2). Combined reducers: 12/step here (13 on the bubble column): vmg_resid_max (the
// velocity MG's stop test), rho_minmax and overlap_census, all outside the MG files; §5.5 asks for
// 0, which is an open ruling (see the WO-S4 commit) -- until then this pins "no new one".
constexpr double kGlobalBudget = 4.0;
constexpr double kCombinedBudget = 12.0;
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    if (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
      std::printf("device_sync_budget: host backend, nothing to measure -- SKIP\n");
      rc = 77;
    } else {
      using Stag = peclet::flow::Solver<peclet::flow::Staggered>;
      const int NX = 32, NY = 24, NZ = 16;
      const double rl = 1.0, rg = 0.02, ml = 0.01, mg = 2e-4, g = 2e-3, R = 4.0;
      Stag s(NX, NY, NZ);
      s.setRho(rl);
      s.setMu(ml);
      s.setDomainBc(2, 1, 0.0, 0.0, 0.0);  // -y wall
      s.setDomainBc(3, 1, 0.0, 0.0, 0.0);  // +y wall
      s.setPressureGeometry(std::vector<double>((std::size_t)NX * NY * NZ, 10.0));
      s.enableVof();
      s.setVof(std::vector<double>((std::size_t)NX * NY * NZ, 0.0));
      s.setPropertyModel("rho", peclet::flow::ClosureKind::LinearMix, "C", "", {rl, rg - rl});
      s.setPropertyModel("mu", peclet::flow::ClosureKind::LinearMix, "C", "", {ml, mg - ml});
      s.setSurfaceTension(0.05);
      s.setPressurePcg(true, 800, 1e-8);
      s.enableVofBlocks({{{NX / 2.0, NY / 2.0, NZ / 2.0, R}}});
      s.enableVofBlockCsf();
      const double alpha = (4.0 / 3.0) * M_PI * R * R * R / ((double)NX * NY * NZ);
      const double rav = rl + (rg - rl) * alpha;
      s.setPropertyModel("force_x", peclet::flow::ClosureKind::LinearMix, "C", "",
                         {(rl - rav) * g, (rg - rl) * g});
      s.setSuperficialVelocity(true, 0, 0.0);
      Kokkos::Tools::Experimental::set_begin_fence_callback(onBeginFence);
      Kokkos::Tools::Experimental::set_end_fence_callback(onEndFence);
      const int warm = 2, steps = 8;
      for (int n = 0; n < warm + steps; ++n) {
        const auto L = s.vofStepLimits();
        s.setDt(0.25 * std::fmin(L.cflDt, L.capillaryDt));
        gCount = n >= warm;
        s.step();
        gCount = false;
      }
      Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
      Kokkos::Tools::Experimental::set_end_fence_callback(nullptr);
      const double perG = (double)gGlobal / steps, perC = (double)gCombined / steps;
      std::printf(
          "device_sync_budget: global fences/step %.2f (budget %.1f), combined-reducer "
          "fences/step %.2f (budget %.1f)\n",
          perG, kGlobalBudget, perC, kCombinedBudget);
      if (perG > kGlobalBudget || perC > kCombinedBudget) {
        std::printf("device_sync_budget: FAIL\n");
        rc = 1;
      } else {
        std::printf("device_sync_budget: PASS\n");
      }
    }
  }
  Kokkos::finalize();
  return rc;
}
