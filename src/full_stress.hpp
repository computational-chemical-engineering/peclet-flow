/// @file
/// @brief flow — the transpose half of the full deviatoric viscous stress on the staggered grid.
///
/// Design: `doc/full_viscous_stress.md` (§4–§5). The Laplacian band kernel
/// (`ibmBuildDiffusionVar`) discretizes ∇·(μ∇u); with variable μ the momentum equation needs
/// ∇·(μ(∇u + ∇uᵀ)) = ∇·(2μD). This header adds the missing term
///
///   T_c = Σ_a ∂_a(μ ∂_c u_a)
///
/// as the divergence of conservative MAC stress elements, EXPLICIT on the time base uⁿ:
///
///   normal element at cell k:      N_c(k)    = β_c(k, k+s_c) · (v_c(k+s_c) − v_c(k))
///   shear element, edge (+a):      S_c(i,+a) = β_c(i, i+s_a) · (v_a(i+s_a) − v_a(i+s_a−s_c))
///   shear element, edge (−a):      S_c(i,−a) = β_c(i, i−s_a) · (v_a(i)     − v_a(i−s_c))
///   T_c(i) = w_c · (((N_c(i) − N_c(i−s_c)) + (S_c(i,+a1) − S_c(i,−a1))) + (S_c(i,+a2) −
///   S_c(i,−a2))) bb_c(i) += rs_c(i) · T_c(i)
///
/// with a1 < a2 the two other axes (this association order is fixed), β_c the SAME face viscosity
/// the Laplacian bands use (`VarFaceProps::beta` of component c, before the ×w multiplication),
/// and w_c the metric weight of the COMPONENT axis (§3). Because β is bitwise symmetric under
/// (i, j) swap, S_c(i,+a) ≡ S_c(i+s_a,−a) and the sum telescopes exactly. Never the product form
/// (∇μ)·(∇u)ᵀ, never on the Picard iterate: register, flow "Variable viscosity uses the full
/// deviatoric stress".
#ifndef PECLET_FLOW_FULL_STRESS_HPP
#define PECLET_FLOW_FULL_STRESS_HPP

#include <cstdint>
#include <Kokkos_Core.hpp>

#include "mac_cutcell.hpp"

namespace peclet::flow {

/// Per-component regular-node mask (§5.4): 1 = a velocity node with neither a solid nor a cut
/// row. Read only on the solid path (`kSolid`).
using RegMask = Kokkos::View<std::uint8_t*, CCMem>;
using RegConst = Kokkos::View<const std::uint8_t*, CCMem>;

/// Add rs_c · T_c(uⁿ) to the momentum RHS `bb` of component `c` over the inner cells [g, e−g)³.
///
///   v[3]      the time-base velocity uⁿ (index units), ghosts filled incl. the (a, c) edges;
///   fp        the component-c face props (`makeFaceProps(c)`), μ ghosts filled incl. edges;
///   wc        the metric weight of axis c (1.0 exactly on the isotropic path);
///   skipLoC   rows at coordinate g along c are BC-imposed (low face of a non-periodic global
///             c-face on this rank): they receive nothing (§5.5);
///   reg[3]    (kSolid only) the regular-node masks, ghosts filled: an element is included iff
///             every velocity node it touches is regular; a non-regular row receives nothing.
template <class FaceProps, bool kSolid>
inline void addFullStressRhs(int c, CCField bb, CCConst rs, const CCConst (&v)[3], FaceProps fp,
                             C3 e, int g, double wc, bool skipLoC,
                             const RegConst (&reg)[3] = {RegConst(), RegConst(), RegConst()}) {
  const long st[3] = {1L, (long)e.x, (long)e.x * e.y};
  const int a1 = (c == 0) ? 1 : 0, a2 = (c == 2) ? 1 : 2;
  const long sc = st[c], s1 = st[a1], s2 = st[a2];
  const CCConst vc = v[c], va1 = v[a1], va2 = v[a2];
  const RegConst rc = reg[c], ra1 = reg[a1], ra2 = reg[a2];
  const int lo[3] = {g + ((skipLoC && c == 0) ? 1 : 0), g + ((skipLoC && c == 1) ? 1 : 0),
                     g + ((skipLoC && c == 2) ? 1 : 0)};
  ccFor3(
      "peclet::flow::full_stress_rhs", C3{lo[0], lo[1], lo[2]}, C3{e.x - g, e.y - g, e.z - g},
      KOKKOS_LAMBDA(int x, int y, int z) {
        const long i = (long)x + (long)y * st[1] + (long)z * st[2];
        if constexpr (kSolid) {
          if (!rc(i))
            return;  // every element of this row touches u_c(i): nothing is included
        }
        // Normal elements at the cell centres i and i - s_c.
        double nP = fp.beta(i, i + sc) * (vc(i + sc) - vc(i));
        double nM = fp.beta(i - sc, i) * (vc(i) - vc(i - sc));
        // Shear elements on the four (a, c) edges of the node.
        double p1 = fp.beta(i, i + s1) * (va1(i + s1) - va1(i + s1 - sc));
        double m1 = fp.beta(i, i - s1) * (va1(i) - va1(i - sc));
        double p2 = fp.beta(i, i + s2) * (va2(i + s2) - va2(i + s2 - sc));
        double m2 = fp.beta(i, i - s2) * (va2(i) - va2(i - sc));
        if constexpr (kSolid) {
          if (!rc(i + sc))
            nP = 0.0;
          if (!rc(i - sc))
            nM = 0.0;
          if (!(rc(i + s1) && ra1(i + s1) && ra1(i + s1 - sc)))
            p1 = 0.0;
          if (!(rc(i - s1) && ra1(i) && ra1(i - sc)))
            m1 = 0.0;
          if (!(rc(i + s2) && ra2(i + s2) && ra2(i + s2 - sc)))
            p2 = 0.0;
          if (!(rc(i - s2) && ra2(i) && ra2(i - sc)))
            m2 = 0.0;
        }
        const double t = wc * (((nP - nM) + (p1 - m1)) + (p2 - m2));
        bb(i) += rs(i) * t;
      });
}

/// The local min and max of `mu` over [g−1, e−g+1)³ — the inner cells plus the one ghost ring the
/// transpose term reads (§4.3 step 2). One instance-scoped reduction; the caller reduces across
/// ranks and compares exactly.
inline void muRangeInner(CCConst mu, C3 e, int g, double& lo, double& hi) {
  CCExec space;
  Kokkos::MinMaxScalar<double> r;
  Kokkos::parallel_reduce(
      "peclet::flow::full_stress_mu_range",
      MDRange3<CCExec>(space, {g - 1, g - 1, g - 1}, {e.x - g + 1, e.y - g + 1, e.z - g + 1}),
      KOKKOS_LAMBDA(int x, int y, int z, Kokkos::MinMaxScalar<double>& acc) {
        const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
        const double m = mu(i);
        if (m < acc.min_val)
          acc.min_val = m;
        if (m > acc.max_val)
          acc.max_val = m;
      },
      Kokkos::MinMax<double>(r));
  lo = r.min_val;
  hi = r.max_val;
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FULL_STRESS_HPP
