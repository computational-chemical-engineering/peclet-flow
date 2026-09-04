/// @file
/// @brief flow — device property closures: material properties / body forces as functions of
/// fields.
///
/// A closure writes one target cell field (a material property like rho/mu, or a momentum
/// body-force component) as a pointwise function of one or two input fields (a transported scalar,
/// pressure, a phase fraction). Dispatch is a host-side enum switch launching ONE dedicated Kokkos
/// kernel per closure kind — no per-cell Python, no device virtual dispatch. The Solver applies its
/// closure list in registration order at the top of step() (properties frozen over the step;
/// segregated coupling).
///
/// This is the seam for field–field coupling: Boussinesq buoyancy (force from temperature),
/// temperature-dependent viscosity (Arrhenius), composition-dependent density (linear mixture), and
/// tabulated properties. A user escape hatch (set_field on a property) bypasses closures entirely.
#ifndef PECLET_FLOW_PROPERTY_CLOSURES_HPP
#define PECLET_FLOW_PROPERTY_CLOSURES_HPP

#include <array>
#include <string>

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"

namespace peclet::flow {

enum class ClosureKind {
  LinearMix,        // out = p0 + p1*in0 + p2*in1
  BoussinesqForce,  // out = p0*p1*p2*(in0 - p3)   [rho0, g, beta, T0] -> buoyancy body force
  ArrheniusMu,      // out = p0*exp(p1*(1/in0 - 1/p2))   [mu_ref, B, Tref]
  Table1D           // out = piecewise-linear interp of (tabX, tabY) at in0
};

// A registered closure. `out`/`in0`/`in1` are resolved from the field registry at registration; a
// later redistribution that reallocates fields must re-resolve them — which is why the three
// registry NAMES are kept alongside the resolved handles. Without them a rebalance leaves a
// closure writing into (and reading from) the previous block's allocation; see
// `Solver::rebindFieldAliases`.
struct Closure {
  ClosureKind kind;
  CCField out;
  CCConst in0, in1;
  std::string outName, in0Name, in1Name;  // registry keys, for re-resolution after a redistribute
  std::array<double, 4> p{{0, 0, 0, 0}};
  CCField tabX, tabY;  // Table1D nodes (ascending tabX)
  int nTab = 0;
};

// Apply one closure over the inner cells (ghosts untouched — refilled by the field's own exchange).
inline void applyClosure(const Closure& cl, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  const ClosureKind kind = cl.kind;
  CCField out = cl.out;
  CCConst in0 = cl.in0, in1 = cl.in1;
  const double p0 = cl.p[0], p1 = cl.p[1], p2 = cl.p[2], p3 = cl.p[3];
  const bool haveIn1 = (in1.data() != nullptr) && (in1.extent(0) == out.extent(0));
  CCConst tx = cl.tabX, ty = cl.tabY;
  const int nTab = cl.nTab;
  Kokkos::parallel_for(
      "peclet::flow::apply_closure", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        double v;
        switch (kind) {
          case ClosureKind::LinearMix:
            v = p0 + p1 * in0(i) + (haveIn1 ? p2 * in1(i) : 0.0);
            break;
          case ClosureKind::BoussinesqForce:
            v = p0 * p1 * p2 * (in0(i) - p3);
            break;
          case ClosureKind::ArrheniusMu:
            v = p0 * Kokkos::exp(p1 * (1.0 / in0(i) - 1.0 / p2));
            break;
          default: {  // Table1D: clamped piecewise-linear interpolation
            const double s = in0(i);
            if (nTab <= 0) {
              v = 0.0;
            } else if (s <= tx(0)) {
              v = ty(0);
            } else if (s >= tx(nTab - 1)) {
              v = ty(nTab - 1);
            } else {
              int lo = 0, hi = nTab - 1;  // binary search for the bracketing node
              while (hi - lo > 1) {
                const int mid = (lo + hi) / 2;
                if (tx(mid) <= s)
                  lo = mid;
                else
                  hi = mid;
              }
              const double t = (s - tx(lo)) / (tx(hi) - tx(lo));
              v = ty(lo) + t * (ty(hi) - ty(lo));
            }
          }
        }
        out(i) = v;
      });
}


// Apply one closure on the GHOST BAND of a single domain face (axis `a`, side `side`), i.e. on the
// `g` ghost layers outside the boundary — the cells `applyClosure` above deliberately leaves alone.
//
// WHY THIS EXISTS (WO-R item 5). The property ghost policy is a Neumann copy of the inner cell
// (`IbmSolver::fillPropGhosts`). At a wall that is right: the ghost's job is to make the
// wall-normal derivative vanish. At an INFLOW carrying a different phase it is wrong by up to the
// density ratio: the face density in the momentum time term and in the projection coefficient at
// the inlet face is the arithmetic mean of the inner cell and the ghost, so a liquid inlet next to
// a gas interior would be given a face density half-way between the two phases *of the interior*,
// not of the incoming fluid. Since rho and mu are CLOSURES of the colour field and the colour
// ghost now carries the prescribed inflow value, the consistent fix is not a second BC rule but
// the same closure evaluated at the ghost: rho_ghost = rho(C_inflow), by construction.
//
// This is a sibling kernel, not a widened `applyClosure`: the interior kernel is validated and its
// range is what makes VoF/varRho/porous bit-identical when no VoF inflow colour is set. The caller
// runs this only on faces that have one (`IbmSolver::vofBcPropGhosts`).
inline void applyClosureFaceGhost(const Closure& cl, C3 e, int g, int a, int side) {
  CCExec space;
  const ClosureKind kind = cl.kind;
  CCField out = cl.out;
  CCConst in0 = cl.in0, in1 = cl.in1;
  const double p0c = cl.p[0], p1c = cl.p[1], p2c = cl.p[2], p3c = cl.p[3];
  const bool haveIn1 = (in1.data() != nullptr) && (in1.extent(0) == out.extent(0));
  CCConst tx = cl.tabX, ty = cl.tabY;
  const int nTab = cl.nTab;
  const int dims[3] = {e.x, e.y, e.z};
  const long st[3] = {1, (long)e.x, (long)e.x * e.y};
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = st[a], sb = st[b], sc = st[c];
  const int lo = (side == 0) ? 0 : (dims[a] - g);
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>;
  Kokkos::parallel_for(
      "peclet::flow::apply_closure_face_ghost", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int j0, int j1) {
        const long base = (long)j0 * sb + (long)j1 * sc;
        for (int k = 0; k < g; ++k) {
          const long i = base + (long)(lo + k) * sa;
          double v;
          switch (kind) {
            case ClosureKind::LinearMix:
              v = p0c + p1c * in0(i) + (haveIn1 ? p2c * in1(i) : 0.0);
              break;
            case ClosureKind::BoussinesqForce:
              v = p0c * p1c * p2c * (in0(i) - p3c);
              break;
            case ClosureKind::ArrheniusMu:
              v = p0c * Kokkos::exp(p1c * (1.0 / in0(i) - 1.0 / p2c));
              break;
            default: {  // Table1D, same clamped piecewise-linear rule as applyClosure
              const double sv = in0(i);
              if (nTab <= 0) {
                v = 0.0;
              } else if (sv <= tx(0)) {
                v = ty(0);
              } else if (sv >= tx(nTab - 1)) {
                v = ty(nTab - 1);
              } else {
                int l = 0, h = nTab - 1;
                while (h - l > 1) {
                  const int mid = (l + h) / 2;
                  if (tx(mid) <= sv)
                    l = mid;
                  else
                    h = mid;
                }
                const double t = (sv - tx(l)) / (tx(h) - tx(l));
                v = ty(l) + t * (ty(h) - ty(l));
              }
            }
          }
          out(i) = v;
        }
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_PROPERTY_CLOSURES_HPP
