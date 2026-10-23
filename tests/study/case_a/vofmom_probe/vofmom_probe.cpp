// Q6 diagnosis (case A, bubble_column_perf/validation/Q6_DIAG.md): replay ONE momentum-consistent
// VoF advection (vof::MomentumConsistentAdvector, rung V2b) on a solver state dumped by
// probe_vofmom_rho.py and write the half-shifted colour C^e — the density rho^e of the control
// volume that carried the momentum — which the Python API does not expose. The advected velocity
// and the new colour are written too, so the replay is checked against the solver's own
// (diagnostics.vof_advected_velocity, get_vof) before C^e is trusted.
//
// usage: vofmom_probe DIR
//   in : DIR/meta.txt  "n dth step rho_g rho_l wisp_eps cfl_limit muscl"
//        DIR/{C,u,v,w}.bin  n^3 doubles, x fastest; u, v, w in the SOLVER's low-face convention
//        (u[i] is the -x face of cell i), velocities and dth = dt/h in consistent units
//   out: DIR/{Ce0,Ce1,Ce2}_seed.bin  C^e at the START of the advection (clipped from C^n planes)
//        DIR/{Ce0,Ce1,Ce2}.bin       C^e after the advection, solver indexing (CV_c(i) = u_c(i))
//        DIR/{ua0,ua1,ua2}.bin, DIR/Cnew.bin
#include <cstdio>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "vof/colour_field.hpp"
#include "vof/momentum_advect.hpp"

namespace {
using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SField;
namespace vf = peclet::flow::vof;

std::vector<double> readBin(const std::string& p, long len) {
  std::vector<double> v(len);
  std::ifstream f(p, std::ios::binary);
  if (!f.read(reinterpret_cast<char*>(v.data()), len * static_cast<long>(sizeof(double))))
    throw std::runtime_error("short read: " + p);
  return v;
}

void writeBin(const std::string& p, const std::vector<double>& v) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<long>(v.size() * sizeof(double)));
}

// Inner cell (x,y,z) of the extended block <- src[(x+sx)%n, (y+sy)%n, (z+sz)%n].
void putInner(SField dst, I3 e, int g, int n, const std::vector<double>& src, int sx, int sy,
              int sz) {
  auto h = Kokkos::create_mirror_view(dst);
  Kokkos::deep_copy(h, dst);
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        h(L3(g + x, g + y, g + z, e)) =
            src[((x + sx) % n) + n * (((y + sy) % n) + n * ((z + sz) % n))];
  Kokkos::deep_copy(dst, h);
}

std::vector<double> getInner(SField src, I3 e, int g, int n) {
  auto h = Kokkos::create_mirror_view(src);
  Kokkos::deep_copy(h, src);
  std::vector<double> v(static_cast<long>(n) * n * n);
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        v[x + n * (y + n * z)] = h(L3(g + x, g + y, g + z, e));
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: vofmom_probe DIR\n");
    return 2;
  }
  const std::string dir = argv[1];
  Kokkos::initialize(argc, argv);
  {
    int n = 0, muscl = 0;
    long step = 0;
    double dth = 0, rg = 0, rl = 0, weps = 0, cfl = 0;
    {
      std::ifstream m(dir + "/meta.txt");
      if (!(m >> n >> dth >> step >> rg >> rl >> weps >> cfl >> muscl))
        throw std::runtime_error("bad meta.txt");
    }
    const long len = static_cast<long>(n) * n * n;
    vf::WyAdvector w;
    w.init(n, n, n, 1.0, 3);  // h = 1, as the solver (cell units); dth carries dt/h
    w.cflLimit = cfl;
    w.interfaceLocalCfl = true;
    w.wispEps = weps;
    const I3 e = w.extent();
    const int g = w.ghost();
    w.exchange = [e, g](SField f) { vf::periodicFill(f, e, g, true, true, true); };

    putInner(w.colour(), e, g, n, readBin(dir + "/C.bin", len), 0, 0, 0);
    const char* vn[3] = {"/u.bin", "/v.bin", "/w.bin"};
    for (int d = 0; d < 3; ++d)  // u_advector(i) = u_solver(i + s_d) (vof/colour_field.hpp)
      putInner(w.faceVel(d), e, g, n, readBin(dir + vn[d], len), d == 0, d == 1, d == 2);
    w.exchange(w.colour());
    for (int d = 0; d < 3; ++d)
      w.exchange(w.faceVel(d));

    vf::MomentumConsistentAdvector mom;
    mom.init(w, rg, rl);
    mom.momentumMuscl = muscl != 0;
    // The seed C^e, exactly as advect() builds it (planes of C^n).
    w.reconstruct();
    mom.buildShiftedColour(w);
    for (int c = 0; c < 3; ++c)
      writeBin(dir + "/Ce" + std::to_string(c) + "_seed.bin", getInner(mom.shiftedColour(c), e, g, n));

    mom.advect(w, dth, step);  // dt in cell units = dt/h, as the solver passes dt_ with h = 1
    for (int c = 0; c < 3; ++c) {
      writeBin(dir + "/Ce" + std::to_string(c) + ".bin", getInner(mom.shiftedColour(c), e, g, n));
      writeBin(dir + "/ua" + std::to_string(c) + ".bin", getInner(mom.advectedVelocity(c), e, g, n));
    }
    writeBin(dir + "/Cnew.bin", getInner(w.colour(), e, g, n));
    const auto dg = mom.diagnostics();
    std::printf("VOFMOM_PROBE n=%d step=%ld floored=%ld clamped=%ld minRhoC=%.6e sumM0=%.17g "
                "sumM=%.17g\n",
                n, step, dg.floored, mom.clampedFluxes(), dg.minRhoC, dg.sumM0[0], dg.sumM[0]);
  }
  Kokkos::finalize();
  return 0;
}
