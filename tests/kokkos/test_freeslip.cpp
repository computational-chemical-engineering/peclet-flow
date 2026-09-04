// FREE-SLIP / SYMMETRY domain BC (set_domain_bc type 4) -- analytic gates on BOTH grids.
//
// Type 4 = zero normal velocity + zero normal derivative of the tangential components + pressure
// Neumann (a wall for the pressure, a mirror for the tangential velocity). A symmetric problem cut
// in half by a type-4 face must reproduce the full symmetric problem POINTWISE, which is what every
// gate below is built on:
//   A. Half Poiseuille channel. Cut-cell SDF wall at a half-integer (pointwise EXACT on the
//      quadratic profile, the fact scripts/verify_poiseuille_flow.py rests on) at one side, a
//      type-4 face at the other. Gate: max_node |u - parabola| at solver tolerance, and
//      max_node |u_half - u_full| against the full channel (both cut-cell walls, periodic) at
//      solver tolerance. Both sides of the axis (s = 0 and s = 1), staggered AND collocated.
//   B. Uniform flow along free-slip faces (four type-4 faces, body-force driven, advection ON):
//      u stays uniform and equals F t / rho to round-off; v, w stay 0.
//   C. The pressure solve: Stokes flow through a sphere in a box with type-4 faces at +-y against
//      the periodic box of twice the height holding the sphere AND its mirror image (whose
//      symmetry planes are exactly the type-4 faces). Pointwise agreement at the pressure
//      solver's tolerance -- the projection, the cut-cell MG's Neumann rows on a type-4 face and
//      the momentum stencil's ghosts all have to be right for this to hold.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <string>
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

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d <= m))
      m = d;  // NaN-propagating
  }
  return m;
}
double maxAbs(const std::vector<double>& a) {
  double m = 0.0;
  for (double v : a)
    if (!(std::fabs(v) <= m))
      m = std::fabs(v);
  return m;
}

// ---------------------------------------------------------------------------------------------
// A. Poiseuille: body-force-driven flow along x between a cut-cell wall and a symmetry face.
//    Node y = j (the u/collocated cell centre), the +y domain face at y = ny - 0.5, the -y face
//    at y = -0.5. ylo/yhi are the SDF wall positions (half-integers -> cut cells).
// ---------------------------------------------------------------------------------------------
constexpr double RHO = 1.0, MU = 0.1, DT = 50.0, F = 0.01;

template <class Grid>
std::vector<double> runChannel(int nx, int ny, int nz, double ylo, double yhi, int slipFace,
                               bool cutcellPressure, int& stepsTaken) {
  // slipFace = -1: both walls are SDF walls (the full channel); 2 / 3: that face is type 4 and
  // only the OTHER wall is an SDF wall.
  peclet::flow::Solver<Grid> s(nx, ny, nz);
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(DT);
  s.setBodyForce(F, 0.0, 0.0);
  s.setVelocityIterations(400);
  s.setVelocityResidualTolerance(1e-12);
  s.setPressurePcg(true, 50, 1e-11);
  if (slipFace >= 0)
    s.setDomainBc(slipFace, 4, 0.0, 0.0, 0.0);
  std::vector<double> sdf((std::size_t)nx * ny * nz);
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        double d;
        if (slipFace < 0)
          d = std::min(y - ylo, yhi - y);
        else if (slipFace == 3)
          d = y - ylo;  // wall below, symmetry face above
        else
          d = yhi - y;  // wall above, symmetry face below
        sdf[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny] = d;
      }
  s.setSolid(sdf, cutcellPressure);
  double prev = 0.0;
  stepsTaken = 0;
  for (int it = 0; it < 800; ++it) {
    s.step();
    ++stepsTaken;
    const double now = maxAbs(s.getVelocity(0));
    if (it > 5 && std::fabs(now - prev) < 1e-12 * (now + 1e-300))
      break;
    prev = now;
  }
  std::vector<double> out = s.getVelocity(0);
  std::vector<double> v = s.getVelocity(1);
  // the wall-normal component must vanish identically (x-independent flow)
  CHECK(maxAbs(v) < 1e-12);
  return out;
}

template <class Grid>
void gateChannel(const char* grid, bool cutcellPressure) {
  const int nx = 8, nz = 8, nyH = 16, nyF = 32;
  const double ylo = 3.5, yhi = 2.0 * (nyH - 0.5) - ylo;  // 27.5: the mirror of ylo about 15.5
  int stF = 0, stT = 0, stB = 0;
  // full channel (periodic y, two cut-cell walls) -- the reference
  const std::vector<double> uF = runChannel<Grid>(nx, nyF, nz, ylo, yhi, -1, cutcellPressure, stF);
  // top-slip half channel: wall at ylo, type-4 face at +y (= the full channel's centre 15.5)
  const std::vector<double> uT = runChannel<Grid>(nx, nyH, nz, ylo, yhi, 3, cutcellPressure, stT);
  // bottom-slip half channel: type-4 face at -y (y' = -0.5 <-> y = 15.5), wall at yhi - 16 = 11.5
  const std::vector<double> uB =
      runChannel<Grid>(nx, nyH, nz, ylo, yhi - 16.0, 2, cutcellPressure, stB);
  auto at = [&](const std::vector<double>& u, int ny, int x, int y, int z) {
    return u[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny];
  };
  const double c = F / (2.0 * MU);
  double errF = 0, errT = 0, errB = 0, dT = 0, dB = 0, umax = 0;
  for (int z = 0; z < nz; ++z)
    for (int x = 0; x < nx; ++x) {
      for (int y = 0; y < nyF; ++y) {
        if (y <= ylo || y >= yhi)
          continue;  // fluid nodes only
        const double ana = c * (y - ylo) * (yhi - y);
        errF = std::max(errF, std::fabs(at(uF, nyF, x, y, z) - ana));
        umax = std::max(umax, ana);
        if (y < nyH) {  // lower half <-> top-slip half channel, node for node
          errT = std::max(errT, std::fabs(at(uT, nyH, x, y, z) - ana));
          dT = std::max(dT, std::fabs(at(uT, nyH, x, y, z) - at(uF, nyF, x, y, z)));
        } else {  // upper half <-> bottom-slip half channel, y' = y - 16
          errB = std::max(errB, std::fabs(at(uB, nyH, x, y - 16, z) - ana));
          dB = std::max(dB, std::fabs(at(uB, nyH, x, y - 16, z) - at(uF, nyF, x, y, z)));
        }
      }
    }
  std::printf(
      "[A %s] u_max %.4f  node error vs parabola: full %.2e (steps %d)  top-slip half %.2e (%d)  "
      "bottom-slip half %.2e (%d);  half-vs-full pointwise: top %.2e  bottom %.2e\n",
      grid, umax, errF, stF, errT, stT, errB, stB, dT, dB);
  // Exact on the quadratic -> the closure's float floor (the cut-cell closure computes in float,
  // so the node error scales with u_max: verify_poiseuille_flow.py reads 6.5e-8 / 1.1e-6 /
  // 2.5e-5 at u_max 0.45 / 1.8 / 8.5). Gate RELATIVE to u_max with a 10x margin. A first-order
  // treatment of the symmetry face (a Neumann copy, or a wall) would read O(1e-2 .. 1e-1).
  const double tol = 3e-5 * umax;
  CHECK(errF < tol);
  CHECK(errT < tol);
  CHECK(errB < tol);
  CHECK(dT < tol);
  CHECK(dB < tol);
}

// ---------------------------------------------------------------------------------------------
// B. Uniform tangential flow: free-slip on all four y/z faces, periodic x, body force along x,
//    advection on. The exact solution is u = F t / rho everywhere, v = w = 0.
// ---------------------------------------------------------------------------------------------
template <class Grid>
void gateUniform(const char* grid, bool advection) {
  const int n = 8, steps = 20;
  const double dt = 1.0, f = 0.01;
  peclet::flow::Solver<Grid> s(n, n, n);
  s.setRho(RHO);
  s.setMu(MU);
  s.setDt(dt);
  s.setBodyForce(f, 0.0, 0.0);
  s.setAdvection(advection);
  s.setVelocityIterations(200);
  s.setPressurePcg(true, 50, 1e-11);
  for (int face = 2; face < 6; ++face)
    s.setDomainBc(face, 4, 0.0, 0.0, 0.0);
  s.setPressureGeometry(std::vector<double>((std::size_t)n * n * n, 1e30));  // all fluid
  for (int it = 0; it < steps; ++it)
    s.step();
  const std::vector<double> u = s.getVelocity(0), v = s.getVelocity(1), w = s.getVelocity(2);
  const double uex = steps * dt * f / RHO;
  double du = 0.0;
  for (double x : u)
    if (!(std::fabs(x - uex) <= du))
      du = std::fabs(x - uex);
  const double div = s.maxOpenDivergence();
  std::printf("[B %s advection=%d] max|u - F t/rho| = %.2e (u = %.3f)  max|v| = %.2e  max|w| = %.2e  "
              "div %.1e\n",
              grid, (int)advection, du, uex, maxAbs(v), maxAbs(w), div);
  // Staggered, advection OFF: the all-fluid domain-BC momentum solve is the DOUBLE
  // const-coefficient fold smoother -> round-off over 20 steps (measured 4.6e-12). Advection ON
  // (the domain-BC path solves advection implicitly, i.e. the FLOAT-stored FOU stencil,
  // Solver::FV) and the collocated grid (its domain-BC smoother is the float all-fluid IBM stencil
  // either way): the per-row rounding of A.1 is ~1e-7 relative (the WO-M storage-precision axis,
  // not the BC; measured 9.8e-8 and 3.1e-8) -> a relative float gate.
  const bool dbl = !Grid::collocated && !advection;
  CHECK(du < (dbl ? 1e-10 : 1e-6 * uex));
  CHECK(maxAbs(v) < 1e-12);
  CHECK(maxAbs(w) < 1e-12);
}

// ---------------------------------------------------------------------------------------------
// C. Stokes flow through a sphere: type-4 faces at +-y vs the mirror-periodic twin.
//    Half box nx x ny x nz with the sphere at (cx, cy, cz); the twin is nx x 2ny x nz periodic
//    with the sphere and its image at y = 2ny - 1 - cy (mirror about the +y face y = ny - 0.5;
//    the -y face y = -0.5 is then a symmetry plane too, through the periodic image).
// ---------------------------------------------------------------------------------------------
void gateSphere() {
  const int nx = 16, ny = 16, nz = 16, steps = 8;
  const double R = 4.3, cx = 8.0, cy = 8.0, cz = 8.0;
  auto sphereSdf = [&](int NY, bool twin) {
    std::vector<double> sdf((std::size_t)nx * NY * nz);
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < NY; ++y)
        for (int x = 0; x < nx; ++x) {
          const double dx = x - cx, dz = z - cz;
          double d = std::sqrt(dx * dx + (y - cy) * (y - cy) + dz * dz) - R;
          if (twin) {
            const double yi = 2.0 * ny - 1.0 - cy;
            d = std::min(d, std::sqrt(dx * dx + (y - yi) * (y - yi) + dz * dz) - R);
          }
          sdf[(std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * NY] = d;
        }
    return sdf;
  };
  auto configure = [&](peclet::flow::IbmSolver& s) {
    s.setRho(1.0);
    s.setMu(1.0);
    s.setDt(1.0);
    s.setBodyForce(1.0, 0.0, 0.0);
    s.setAdvection(false);
    s.setVelocityIterations(2000);
    s.setVelocityResidualTolerance(1e-12);
    s.setPressureLevels(3);
    s.setPressurePcg(true, 200, 1e-12);
  };
  peclet::flow::IbmSolver sh(nx, ny, nz);
  configure(sh);
  sh.setDomainBc(2, 4, 0.0, 0.0, 0.0);
  sh.setDomainBc(3, 4, 0.0, 0.0, 0.0);
  sh.setSolid(sphereSdf(ny, false), true);
  peclet::flow::IbmSolver sf(nx, 2 * ny, nz);
  configure(sf);
  sf.setSolid(sphereSdf(2 * ny, true), true);
  for (int it = 0; it < steps; ++it) {
    sh.step();
    sf.step();
  }
  double d[3] = {0, 0, 0}, umax = 0;
  for (int c = 0; c < 3; ++c) {
    const std::vector<double> uh = sh.getVelocity(c), uf = sf.getVelocity(c);
    for (int z = 0; z < nz; ++z)
      for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
          const std::size_t ih = (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * ny;
          const std::size_t iF =
              (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * nx * (2 * ny);
          const double dd = std::fabs(uh[ih] - uf[iF]);
          if (!(dd <= d[c]))
            d[c] = dd;
          umax = std::max(umax, std::fabs(uf[iF]));
        }
  }
  const double divh = sh.maxOpenDivergence(), divf = sf.maxOpenDivergence();
  std::printf("[C] sphere in a slip box vs mirror-periodic twin: max|du| %.2e  |dv| %.2e  |dw| %.2e  "
              "(u_max %.3e, rel %.2e)  div %.1e / %.1e\n",
              d[0], d[1], d[2], umax, std::max({d[0], d[1], d[2]}) / umax, divh, divf);
  CHECK(std::max({d[0], d[1], d[2]}) / umax < 1e-7);
  // the divergence floor is the float-stored pressure operator's (A.1 rounding), ~5e-9 relative
  CHECK(divh / umax < 1e-6 && divf / umax < 1e-6);
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("[freeslip] exec %s\n", Kokkos::DefaultExecutionSpace::name());
    gateChannel<peclet::flow::Staggered>("staggered", /*cutcellPressure=*/true);
    gateChannel<peclet::flow::Colocated>("collocated", /*cutcellPressure=*/false);
    gateUniform<peclet::flow::Staggered>("staggered", false);
    gateUniform<peclet::flow::Staggered>("staggered", true);
    gateUniform<peclet::flow::Colocated>("collocated", false);
    gateUniform<peclet::flow::Colocated>("collocated", true);
    gateSphere();
    if (failures == 0)
      std::printf("[freeslip] PASS: type-4 free-slip faces reproduce the symmetric problems\n");
    else
      std::fprintf(stderr, "[freeslip] FAILED (%d checks)\n", failures);
  }
  Kokkos::finalize();
  return failures == 0 ? 0 : 1;
}
