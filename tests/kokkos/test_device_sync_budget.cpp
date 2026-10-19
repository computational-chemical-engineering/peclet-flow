// G-SYNC (doc/vof_gpu_coarse_tail_design.md §5.5, WO-S4): the device step's host-synchronization
// budget. A small single-rank block-VoF bubble in a walled box (the bubble column's configuration:
// variable density by a linear closure, CSF, MG-PCG, superficial-velocity shift) runs 10 steps
// while in-process Kokkos Tools callbacks count, per step,
//   - GLOBAL fences on the device (Kokkos::fence(), exec-less deep_copy: each one drains every
//     stream, and the GPU then idles until the host relaunches),
//   - combined-reducer fences ("fence after copying values back": a parallel_reduce with >= 2
//     results reduces through host memory and fences twice, even into device Views), each
//     attributed to the parallel_reduce label it belongs to, and
//   - device-space allocations / deallocations (with cudaMallocAsync off a cudaMalloc/cudaFree
//     raises no fence event, so S4(e) "no per-step View" is only visible here).
// Two configurations: 'fp64' V-cycles + zero guess, and 'auto' V-cycles + the projected guess (the
// WO-S2 sites: the FP32 exit's single-result mean, the wBad read, the guess packet). Both must
// launch the side-instance bottom prefactor of WO-X (diagnostics counter > 0).
// `test_device_sync_budget hoist` instead runs the 'auto' + projected configuration twice, the
// WO-X hoist off (the lazy factor in the bottom solve, coefficients in project()) and on, and
// compares the state byte for byte (§5.6: the hoisted factor is bitwise the lazy one).
// Host backends have no device to drain: the test skips there (exit 77).
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <map>
#include <string>
#include <vector>

#include "flow_ibm.hpp"

namespace {
long gGlobal = 0, gAlloc = 0, gDealloc = 0;
std::map<std::string, long> gCombined;  // combined-reducer fences by parallel_reduce label
std::string gReduce;                    // the label of the last parallel_reduce begun
std::map<std::string, long> gAllocBy;   // device allocations by View label
bool gCount = false;

bool isDevice(uint32_t devid) {
  const auto id = Kokkos::Tools::Experimental::identifier_from_devid(devid);
  return id.type != Kokkos::Tools::Experimental::DeviceType::Serial &&
         id.type != Kokkos::Tools::Experimental::DeviceType::OpenMP &&
         id.type != Kokkos::Tools::Experimental::DeviceType::Threads &&
         id.type != Kokkos::Tools::Experimental::DeviceType::HPX;
}
void onBeginFence(const char* name, const uint32_t devid, uint64_t* handle) {
  *handle = 0;
  if (!gCount)
    return;
  // A tool with a parallel_reduce callback makes Kokkos fence globally around each kernel callback
  // (ToolRequirements::requires_global_fencing): those fences are the probe's, not the step's.
  if (std::strstr(name, "Profile Tool Fence") != nullptr)
    return;
  const auto id = Kokkos::Tools::Experimental::identifier_from_devid(devid);
  // A global fence reaches the callback once per enabled space with instance bits 0 (and device
  // bits 0): count the device space's only.
  if (isDevice(devid) && id.instance_id == 0 && id.device_id == 0)
    ++gGlobal;
  if (std::strstr(name, "fence after copying values back") != nullptr)
    ++gCombined[gReduce.empty() ? std::string("(no parallel_reduce)") : gReduce];
}
void onEndFence(uint64_t) {}
void onBeginReduce(const char* name, const uint32_t, uint64_t* kid) {
  *kid = 0;
  gReduce = name;
}
// Not cleared at the end: Kokkos fences "after copying values back" once the end callback ran.
void onEndReduce(uint64_t) {}
bool deviceSpace(const Kokkos::Tools::SpaceHandle h) {
  return std::strcmp(h.name, "Host") != 0;
}
void onAlloc(const Kokkos::Tools::SpaceHandle h, const char* label, const void*, const uint64_t) {
  if (gCount && deviceSpace(h)) {
    // Kokkos 5.1.1 allocates a 0-byte team scratch per TeamPolicy launch (Cuda_Instance.cpp:426):
    // Kokkos-internal and iteration-dependent, reported but not pinned.
    if (std::strstr(label, "TeamScratchMemory") == nullptr)
      ++gAlloc;
    ++gAllocBy[std::string(h.name) + ":" + label];
  }
}
void onDealloc(const Kokkos::Tools::SpaceHandle h, const char*, const void*, const uint64_t) {
  if (gCount && deviceSpace(h))
    ++gDealloc;
}

// The measured post-S4b budget (both configurations; the bubble column's fencebt census agrees):
//   global fences 0/step -- the block container's reads #1 and #2 are instance-scoped (WO-S4b);
//   combined-reducer fences: none from any label but the allowlist below, all outside the MG files
//     (vmg_resid_max, the velocity MG's stop test, once per sweep check, so iteration-dependent;
//     rho_minmax; overlap_census). §5.5 asks for 0; these three are a separate package (Q-6, the
//     2026-10-19 coordinator ruling). A label 'peclet::flow::mg*' is never allowed.
//   device allocations / deallocations per step: 0 -- no per-step View (S4(e)); Kokkos' own
//     per-launch team scratch (below) is listed but not counted (measured 15 and 14.4/step).
constexpr double kGlobalBudget = 0.0;
const char* const kCombinedAllow[] = {"peclet::flow::vmg_resid_max", "rho_minmax",
                                      "vof::block::overlap_census"};
constexpr double kAllocBudget = 0.0;

using Stag = peclet::flow::Solver<peclet::flow::Staggered>;
constexpr int NX = 32, NY = 24, NZ = 16;

// The bubble: `autoGuess` selects 'auto' V-cycles + the projected guess (else 'fp64' + zero).
void setup(Stag& s, bool autoGuess) {
  const double rl = 1.0, rg = 0.02, ml = 0.01, mg = 2e-4, g = 2e-3, R = 4.0;
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
  s.setPressureBottomMode(1);  // agglomerated: the direct bottom (the bubble column's), so X fires
  if (autoGuess) {
    s.setPressureVcyclePrecision(0);  // 'auto'
    s.setPressureInitialGuess(2, 2);  // 'projected', history 2
  }
  s.enableVofBlocks({{{NX / 2.0, NY / 2.0, NZ / 2.0, R}}});
  s.enableVofBlockCsf();
  const double alpha = (4.0 / 3.0) * M_PI * R * R * R / ((double)NX * NY * NZ);
  const double rav = rl + (rg - rl) * alpha;
  s.setPropertyModel("force_x", peclet::flow::ClosureKind::LinearMix, "C", "",
                     {(rl - rav) * g, (rg - rl) * g});
  s.setSuperficialVelocity(true, 0, 0.0);
}

void advance(Stag& s, int n0, int n1, int countFrom) {
  for (int n = n0; n < n1; ++n) {
    const auto L = s.vofStepLimits();
    s.setDt(0.25 * std::fmin(L.cflDt, L.capillaryDt));
    gCount = n >= countFrom;
    s.step();
    gCount = false;
  }
}

// One configuration's budget. Returns 0 (pass) or 1.
int budget(bool autoGuess) {
  const char* tag = autoGuess ? "auto+projected" : "fp64+zero";
  gGlobal = gAlloc = gDealloc = 0;
  gCombined.clear();
  gAllocBy.clear();
  Stag s(NX, NY, NZ);
  setup(s, autoGuess);
  namespace KT = Kokkos::Tools::Experimental;
  KT::set_begin_fence_callback(onBeginFence);
  KT::set_end_fence_callback(onEndFence);
  KT::set_begin_parallel_reduce_callback(onBeginReduce);
  KT::set_end_parallel_reduce_callback(onEndReduce);
  KT::set_allocate_data_callback(onAlloc);
  KT::set_deallocate_data_callback(onDealloc);
  const int warm = 2, steps = 8;
  advance(s, 0, warm + steps, warm);
  KT::set_begin_fence_callback(nullptr);
  KT::set_end_fence_callback(nullptr);
  KT::set_begin_parallel_reduce_callback(nullptr);
  KT::set_end_parallel_reduce_callback(nullptr);
  KT::set_allocate_data_callback(nullptr);
  KT::set_deallocate_data_callback(nullptr);
  int rc = 0;
  const double perG = (double)gGlobal / steps;
  const double perA = (double)gAlloc / steps, perD = (double)gDealloc / steps;
  long comb = 0;
  for (const auto& [label, n] : gCombined) {
    bool allowed = false;
    for (const char* a : kCombinedAllow)
      allowed = allowed || label == a;
    if (label.rfind("peclet::flow::mg", 0) == 0)
      allowed = false;
    comb += n;
    std::printf("device_sync_budget [%s]: combined-reducer fences/step %.2f from '%s'%s\n", tag,
                (double)n / steps, label.c_str(), allowed ? "" : "  <-- NOT ALLOWED");
    if (!allowed)
      rc = 1;
  }
  for (const auto& [label, n] : gAllocBy)
    std::printf("device_sync_budget [%s]: device allocations/step %.2f of '%s'\n", tag,
                (double)n / steps, label.c_str());
  const long pre = s.pressurePrefactorCount();
  std::printf(
      "device_sync_budget [%s]: global fences/step %.2f (budget %.1f), combined %.2f (allowlist "
      "only), device View alloc/dealloc per step %.2f/%.2f (budget %.1f), bottom prefactors %ld (> "
      "0)\n",
      tag, perG, kGlobalBudget, (double)comb / steps, perA, perD, kAllocBudget, pre);
  if (perG > kGlobalBudget || perA > kAllocBudget || perD > kAllocBudget || pre <= 0)
    rc = 1;
  std::printf("device_sync_budget [%s]: %s\n", tag, rc ? "FAIL" : "PASS");
  return rc;
}

// WO-X hoist off vs on, 'auto' + projected: the state after 10 steps byte for byte.
int hoistBitwise() {
  const int steps = 10;
  std::vector<std::vector<double>> st[2];
  long pre[2];
  for (int on = 0; on < 2; ++on) {
    Stag s(NX, NY, NZ);
    setup(s, true);
    s.setPressurePrefactorHoist(on == 1);
    advance(s, 0, steps, steps);
    for (int c = 0; c < 3; ++c)
      st[on].push_back(s.getVelocity(c));
    st[on].push_back(s.getPressure());
    st[on].push_back(s.getVof());
    pre[on] = s.pressurePrefactorCount();
  }
  int rc = (pre[0] == 0 && pre[1] > 0) ? 0 : 1;
  const char* names[] = {"u", "v", "w", "p", "C"};
  for (std::size_t k = 0; k < st[0].size(); ++k) {
    const bool same =
        st[0][k].size() == st[1][k].size() &&
        std::memcmp(st[0][k].data(), st[1][k].data(), st[0][k].size() * sizeof(double)) == 0;
    std::printf("prefactor_hoist_bitwise: %s %s\n", names[k], same ? "identical" : "DIFFERS");
    rc |= same ? 0 : 1;
  }
  std::printf("prefactor_hoist_bitwise: prefactors off %ld (== 0), on %ld (> 0): %s\n", pre[0],
              pre[1], rc ? "FAIL" : "PASS");
  return rc;
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    const bool hoist = argc > 1 && std::strcmp(argv[argc - 1], "hoist") == 0;
    if (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
      std::printf("device_sync_budget: host backend, nothing to measure -- SKIP\n");
      rc = 77;
    } else if (hoist) {
      rc = hoistBitwise();
    } else {
      rc = budget(false) | budget(true);
    }
  }
  Kokkos::finalize();
  return rc;
}
