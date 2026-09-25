/// @file
/// @brief flow — face/cell material-property accessors for the variable-coefficient momentum
/// operator.
///
/// The variable-viscosity diffusion stencil (ibmBuildDiffusionVar) is templated on a small accessor
/// supplying the per-cell time diagonal idiag(i) = rho/dt and the per-face viscosity beta between
/// two adjacent cells. Two models:
///   - UniformFaceProps: two constants — reproduces the constant-mu operator (used to cross-check
///   the
///     Var kernel against the scalar ibmBuildDiffusion at the solution level).
///   - FieldFaceProps: idiag from a (constant here) rho, beta from a per-cell mu field, averaged at
///     the face either arithmetically (default) or harmonically (correct for a viscosity jump — the
///     shear stress mu*du/dy is continuous across a material interface).
///
/// Averaging is done in double; the caller casts the assembled band to float once (matching the
/// constant path's `(float)(idiag + 6*beta)`), so no per-cell float rounding creeps into the sum.
#ifndef PECLET_FLOW_FACE_PROPS_HPP
#define PECLET_FLOW_FACE_PROPS_HPP

#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"

namespace peclet::flow {

// Constant properties — the Var kernel with this accessor is the constant-mu operator (equivalence
// check for the variable path).
struct UniformFaceProps {
  double idiag_, beta_;
  KOKKOS_INLINE_FUNCTION double idiag(long) const { return idiag_; }
  KOKKOS_INLINE_FUNCTION double beta(long, long) const { return beta_; }
};

// Per-cell viscosity field (+ constant density for the time diagonal). harmonic=true uses the
// harmonic face mean (continuous shear stress across a viscosity jump); false = arithmetic.
struct FieldFaceProps {
  CCConst mu;
  double rhoIdt;  // rho / dt (constant-density momentum time term)
  bool harmonic;
  KOKKOS_INLINE_FUNCTION double idiag(long) const { return rhoIdt; }
  KOKKOS_INLINE_FUNCTION double beta(long i, long j) const {
    const double a = mu(i), b = mu(j);
    if (harmonic) {
      const double s = a + b;
      return (s > 0.0) ? (2.0 * a * b / s) : 0.0;
    }
    return 0.5 * (a + b);
  }
};

// General accessor: viscosity constant-or-field AND density constant-or-field. The time diagonal
// for a variable density is the FACE density of the staggered velocity unknown (component stride
// sc): idiag(i) = 0.5*(rho(i)+rho(i-sc))/dt — the arithmetic mean (mass is volume-additive), and
// the SAME face density the variable-density projection uses, which is what makes discrete
// hydrostatic balance exact. Flags select the constant fallbacks so a single-variable case (only mu
// or only rho) composes.
struct VarFaceProps {
  CCConst mu;
  bool haveMu = false;
  double muC = 0.0;
  bool harmMu = false;
  CCConst rho;
  bool haveRho = false;
  double rhoIdtC = 0.0;  // rho_/dt fallback
  double idt = 0.0;      // 1/dt (variable-rho path)
  long sc = 0;           // component face stride (staggered velocity placement)
  // Viscosity placement stride: the staggered unknown u_c(i) sits on the -c face of cell i, so
  // its control volume is centred there, not on cell i. 0 = collocated (the unknown IS the cell).
  long msc = 0;
  KOKKOS_INLINE_FUNCTION double idiag(long i) const {
    return haveRho ? 0.5 * (rho(i) + rho(i - sc)) * idt : rhoIdtC;
  }
  // The two-point mean of mu across the cell face between i and j (arithmetic or harmonic).
  KOKKOS_INLINE_FUNCTION double mean2(long i, long j) const {
    const double a = mu(i), b = mu(j);
    if (harmMu) {
      const double s = a + b;
      return (s > 0.0) ? (2.0 * a * b / s) : 0.0;
    }
    return 0.5 * (a + b);
  }
  // The viscosity on the control-volume face between the unknowns at i and j = i +- stride.
  // Collocated (msc == 0): the cell face between cells i and j. Staggered: along the component
  // axis the face between the unknowns i and i+msc is the CENTRE of cell i (and between i and
  // i-msc the centre of cell i-msc), so mu is read there directly; across it (j = i +- s_a,
  // a != c) the face is the cell EDGE shared by cells i, j, i-msc, j-msc -- the configured mean
  // across the a-face (series), averaged over the two cells along c (parallel). The former
  // mean2(i, j) placed mu half a cell towards +c for every component: an O(h) error, and in a
  // walled direction a mirror asymmetry (bubble columns collected at the low wall).
  KOKKOS_INLINE_FUNCTION double beta(long i, long j) const {
    if (!haveMu)
      return muC;
    if (msc == 0)
      return mean2(i, j);
    if (j - i == msc)
      return mu(i);
    if (i - j == msc)
      return mu(j);
    return 0.5 * (mean2(i, j) + mean2(i - msc, j - msc));
  }
};

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FACE_PROPS_HPP
