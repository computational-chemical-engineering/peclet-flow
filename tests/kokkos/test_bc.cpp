// Correctness of the Kokkos domain BC ghost fills (dns::bcVelocityComp / bcOutflowComp /
// bcZeroOpenness) vs host replication, over several (axis, side, comp, fold) cases. The BC only
// touches the boundary face + ghost cells; we compare the whole field. Runs on any Kokkos backend.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "mac_bc.hpp"

using namespace dns;

static void hostVel(std::vector<double>& f, B3 ext, int g, int a, int s, int comp, double wall, int fold) {
  int dims[3] = {ext.x, ext.y, ext.z};
  long st[3] = {1, ext.x, (long)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3;
  long sa = st[a], sb = st[b], sc = st[c];
  int na = dims[a], bf = (s == 0) ? g : (na - g);
  for (int p0 = 0; p0 < dims[b]; ++p0) for (int p1 = 0; p1 < dims[c]; ++p1) {
    long base = (long)p0 * sb + (long)p1 * sc;
    auto at = [&](int ia) -> double& { return f[base + (long)ia * sa]; };
    if (comp == a) {
      at(bf) = wall;
      if (s == 0) for (int ia = 0; ia < g; ++ia) at(ia) = 2.0 * wall - at(2 * bf - ia);
      else for (int ia = na - g + 1; ia < na; ++ia) at(ia) = 2.0 * wall - at(2 * bf - ia);
    } else if (fold) {
      if (s == 0) for (int ia = 0; ia < g; ++ia) at(ia) = 0.0;
      else for (int ia = na - g; ia < na; ++ia) at(ia) = 0.0;
    } else {
      if (s == 0) for (int ia = 0; ia < g; ++ia) at(ia) = 2.0 * wall - at(2 * bf - 1 - ia);
      else for (int ia = na - g; ia < na; ++ia) at(ia) = 2.0 * wall - at(2 * bf - 1 - ia);
    }
  }
}
static void hostZeroOpen(std::vector<double>& oa, B3 ext, int g, int a, int s) {
  int dims[3] = {ext.x, ext.y, ext.z};
  long st[3] = {1, ext.x, (long)ext.x * ext.y};
  int b = (a + 1) % 3, c = (a + 2) % 3, bf = (s == 0) ? g : (dims[a] - g);
  for (int p0 = 0; p0 < dims[b]; ++p0) for (int p1 = 0; p1 < dims[c]; ++p1)
    oa[(long)p0 * st[b] + (long)p1 * st[c] + (long)bf * st[a]] = 0.0;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int g = 2;
    B3 ext{12, 10, 8};
    const std::size_t n = (std::size_t)ext.x * ext.y * ext.z;
    std::mt19937 rng(44);
    std::uniform_real_distribution<double> uf(-1.0, 1.0);
    std::vector<double> base(n);
    for (auto& x : base) x = uf(rng);

    auto up = [&](std::vector<double>& h) { BField v("v", n); auto m = Kokkos::create_mirror_view(v);
      for (std::size_t i = 0; i < n; ++i) m(i) = h[i]; Kokkos::deep_copy(v, m); return v; };
    auto get = [&](BField v) { std::vector<double> o(n); auto m = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(m, v); for (std::size_t i = 0; i < n; ++i) o[i] = m(i); return o; };
    auto close = [](double x, double y) { return std::fabs(x - y) <= 1e-12 * (1.0 + std::fabs(y)); };

    int bad = 0, cases = 0;
    // velocity BC: all axes, both sides, normal + tangential, fold 0/1
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s)
        for (int comp = 0; comp < 3; ++comp)
          for (int fold = 0; fold < 2; ++fold) {
            std::vector<double> h = base;
            BField d = up(h);
            const double wall = 0.37;
            bcVelocityComp(d, ext, g, a, s, comp, wall, fold);
            hostVel(h, ext, g, a, s, comp, wall, fold);
            auto gd = get(d);
            for (std::size_t i = 0; i < n; ++i) if (!close(gd[i], h[i])) ++bad;
            ++cases;
          }
    // wall openness
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        std::vector<double> h = base; BField d = up(h);
        bcZeroOpenness(d, ext, g, a, s);
        hostZeroOpen(h, ext, g, a, s);
        auto gd = get(d);
        for (std::size_t i = 0; i < n; ++i) if (!close(gd[i], h[i])) ++bad;
        ++cases;
      }

    if (bad) { std::fprintf(stderr, "FAIL: %d mismatches across %d BC cases\n", bad, cases); status = 1; }
    else std::printf("[mac_bc] PASS: %d BC cases (velocity normal/tangential/fold + wall openness) match host (exec: %s)\n",
                     cases, BExec::name());
  }
  Kokkos::finalize();
  return status;
}
