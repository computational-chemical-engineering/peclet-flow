// Field registry on the flow Solver — the named directory of cell fields (multiphysics container).
// Verifies the built-in fields are adopted, a user field allocates + round-trips through the inner
// region (scatterInner/gatherInner), and fieldView aliases the live buffer.
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"

namespace {
int failures = 0;
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      ++failures;                                                                        \
    }                                                                                    \
  } while (0)

void run() {
  using peclet::flow::IbmSolver;
  const int nx = 6, ny = 5, nz = 4;
  IbmSolver s(nx, ny, nz);

  // Built-in members are registered.
  CHECK(s.hasField("u") && s.hasField("v") && s.hasField("w"));
  CHECK(s.hasField("p") && s.hasField("sdf"));
  CHECK(!s.hasField("temperature"));
  auto names = s.fieldNames();
  CHECK(names.size() == 5);  // u,v,w,p,sdf; sorted
  CHECK(names.front() == "p");

  // Add a user field: zero-initialised, then x-fastest inner round-trip.
  s.addField("temperature");
  CHECK(s.hasField("temperature"));
  std::vector<double> in((std::size_t)nx * ny * nz);
  for (std::size_t i = 0; i < in.size(); ++i)
    in[i] = 1.0 + (double)i;
  s.setField("temperature", in);
  auto out = s.getField("temperature");
  CHECK(out.size() == in.size());
  bool exact = true;
  for (std::size_t i = 0; i < in.size(); ++i)
    exact = exact && (out[i] == in[i]);
  CHECK(exact);

  // fieldView aliases the live buffer: mutate a ghost-included element, gatherInner unaffected at
  // ghosts but inner cell (g,g,g) maps to inner index 0.
  auto fv = s.fieldView("temperature");
  const auto bs = s.blockShape();
  const int g = s.ghostWidth();
  const long ex = bs[0], ey = bs[1];
  // inner (0,0,0) lives at flat (g + g*ex + g*ex*ey)
  const long inner0 = g + (long)g * ex + (long)g * ex * ey;
  auto h = Kokkos::create_mirror_view(fv);
  Kokkos::deep_copy(h, fv);
  CHECK(h(inner0) == in[0]);
  h(inner0) = 123.0;
  Kokkos::deep_copy(fv, h);
  CHECK(s.getField("temperature")[0] == 123.0);

  // Idempotent add: existing data preserved.
  s.addField("temperature");
  CHECK(s.getField("temperature")[0] == 123.0);
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  run();
  Kokkos::finalize();
  if (failures == 0) {
    std::printf("OK\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", failures);
  return 1;
}
