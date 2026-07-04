/// @file
/// @brief flow — cell-centred scalar transport (advection–diffusion) on the cut-cell grid.
///
/// A transported scalar c (temperature, concentration, phase fraction) obeys, in the solver's
/// divided-by-dt convention (dx = 1 grid units; physical diffusivity converted by the Python layer):
///
///   (1/dt)(c^{n+1} - c^n) + div(open u c) = div(open D grad c) + S
///
/// Diffusion is backward-Euler implicit (an openness-weighted 7-band operator, solved by the same
/// red-black Gauss–Seidel as the pressure Poisson — cutcellSmoothColor); advection is explicit,
/// conservative flux-form, reusing the momentum limiter helpers (sadv::tvd / sou / fou_flux) with the
/// MAC face-normal velocities (staggered: C[fd].u is the -fd face velocity, co-located with the face
/// openness). Closed faces (openness 0) carry no flux and no diffusion, so an immersed solid is
/// adiabatic (zero-flux) for free and solid cells stay frozen (A_C = 1/dt, all off-diagonals 0).
///
/// This header holds the field-agnostic kernels + the per-scalar state; the Solver owns the scalars
/// and calls advanceScalars() at the end of step() with the just-projected divergence-free velocity.
#ifndef PECLET_FLOW_SCALAR_TRANSPORT_HPP
#define PECLET_FLOW_SCALAR_TRANSPORT_HPP

#include <Kokkos_Core.hpp>
#include <string>

#include "mac_cutcell.hpp"
#include "staggered_advection.hpp"

namespace peclet::flow {

// Per-scalar BC on a domain face: 0 periodic (via the halo/periodic fill), 1 Neumann zero-flux
// (ghost = inner, i.e. adiabatic wall), 2 Dirichlet value (ghost = 2*value - inner reflection).
enum class ScalarBc { Periodic = 0, Neumann = 1, Dirichlet = 2 };

// One transported scalar. `c` aliases the Solver's registered field (fields_); the rest is private
// scratch on the same G=2 block. Bands are double (a scalar is cheap; no float-quantization concern).
struct ScalarField {
  std::string name;
  CCField c, cOld, b;                  // solution (registered), time base c^n, rhs
  CCField AC, AW, AE, AS, AN, AB, AT;  // implicit diffusion+time 7-band operator
  double D = 0.0;                      // constant diffusivity (grid units)
  int scheme = 1;                      // explicit advection flux: 0 FOU, 1 Koren TVD, 2 SOU
  int iters = 50;                      // RB-GS sweeps for the implicit diffusion solve
  int bc[6] = {0, 0, 0, 0, 0, 0};      // -x,+x,-y,+y,-z,+z (ScalarBc)
  double bcVal[6] = {0, 0, 0, 0, 0, 0};
  bool stencilBuilt = false;
};

// Build the implicit diffusion+time 7-band operator over inner cells:
//   A_C = idt + D*(ox(i)+ox(i+sx)+oy(i)+oy(i+sy)+oz(i)+oz(i+sz)),  A_off = -D*open_face.
// Openness-weighted (closed faces drop out) — the scalar analog of buildCutcellOp with the 1/dt
// diagonal. ox(i) is the -x face openness of cell i (== +x face of cell i-1), matching divergOpen.
inline void scalarBuildDiffusionOpen(CCField AC, CCField AW, CCField AE, CCField AS, CCField AN,
                                     CCField AB, CCField AT, CCConst ox, CCConst oy, CCConst oz,
                                     double D, double idt, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_build_diff", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        const double tw = D * ox(i), te = D * ox(i + sx);
        const double ts = D * oy(i), tn = D * oy(i + sy);
        const double tb = D * oz(i), tt = D * oz(i + sz);
        AW(i) = -tw;
        AE(i) = -te;
        AS(i) = -ts;
        AN(i) = -tn;
        AB(i) = -tb;
        AT(i) = -tt;
        AC(i) = idt + te + tw + tn + ts + tt + tb;
      });
}

// b = idt*c^n - div(open u c^n): explicit conservative openness-weighted advection into the RHS.
// U/V/W are the MAC face-normal velocities (staggered: C[fd].u; collocated: uf_/vf_/wf_) — U(i) is
// the -x face velocity of cell i, co-located with ox(i). scheme: 0 FOU, 1 Koren TVD, 2 SOU.
inline void scalarBuildRhs(CCField b, CCConst cOld, CCConst U, CCConst V, CCConst W, CCConst ox,
                           CCConst oy, CCConst oz, double idt, int scheme, C3 e, int g) {
  CCExec space;
  using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
  Kokkos::parallel_for(
      "peclet::flow::scalar_build_rhs", MD(space, {g, g, g}, {e.x - g, e.y - g, e.z - g}),
      KOKKOS_LAMBDA(int lx, int ly, int lz) {
        const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
        const long i = (long)lx + (long)ly * sy + (long)lz * sz;
        double adv = 0.0;
        // per axis fd: flux through the +fd face (open(i+s), vel U(i+s)) minus the -fd face
        // (open(i), vel U(i)). Face value from the upwind limiter on the 4-cell stencil.
        for (int fd = 0; fd < 3; ++fd) {
          const long s = (fd == 0) ? sx : (fd == 1) ? sy : sz;
          CCConst Uf = (fd == 0) ? U : (fd == 1) ? V : W;
          CCConst Of = (fd == 0) ? ox : (fd == 1) ? oy : oz;
          const double velm = Uf(i), velp = Uf(i + s);
          const double om = Of(i), op = Of(i + s);
          const double cLL = cOld(i - 2 * s), cL = cOld(i - s), cR = cOld(i), cRR = cOld(i + s),
                       cRRR = cOld(i + 2 * s);
          double Fp, Fm;
          if (scheme == 0) {  // first-order upwind
            Fp = op * sadv::fou_flux(cR, cRR, velp);
            Fm = om * sadv::fou_flux(cL, cR, velm);
          } else if (scheme == 2) {  // second-order upwind (unlimited)
            Fp = op * sadv::sou(cL, cR, cRR, cRRR, velp);
            Fm = om * sadv::sou(cLL, cL, cR, cRR, velm);
          } else {  // Koren TVD (default)
            Fp = op * sadv::tvd(cL, cR, cRR, cRRR, velp);
            Fm = om * sadv::tvd(cLL, cL, cR, cRR, velm);
          }
          adv += Fp - Fm;
        }
        b(i) = idt * cOld(i) - adv;
      });
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_SCALAR_TRANSPORT_HPP
