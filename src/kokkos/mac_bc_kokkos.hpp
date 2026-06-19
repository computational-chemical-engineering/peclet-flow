// cfd-gpu — portable (Kokkos) native per-face domain boundary conditions for the MAC grid.
//
// Kokkos port of mac_bc.cuh: fill the NON-periodic boundary face/ghosts the halo leaves untouched.
// Each kernel runs over the boundary face's perpendicular (b,c) plane; one thread owns a column and
// writes its own ghosts (disjoint, no races). MAC staggered convention: component a is stored at the
// -a face of its cell. Faithful copies of the reflection / fold / outflow logic. Scalar wall velocity
// (the per-position inlet-profile variant is a later specialization). Runs on any Kokkos backend.
#ifndef CFD_MAC_BC_KOKKOS_HPP
#define CFD_MAC_BC_KOKKOS_HPP

#include <Kokkos_Core.hpp>

namespace cfdk {

using BExec = Kokkos::DefaultExecutionSpace;
using BMem = BExec::memory_space;
using BField = Kokkos::View<double*, BMem>;

struct B3 {
  int x, y, z;
};

namespace bcdetail {
KOKKOS_INLINE_FUNCTION void axisDims(B3 ext, int (&dims)[3], long (&strides)[3]) {
  dims[0] = ext.x; dims[1] = ext.y; dims[2] = ext.z;
  strides[0] = 1; strides[1] = ext.x; strides[2] = static_cast<long>(ext.x) * ext.y;
}
}  // namespace bcdetail

// Fill component comp (0=u,1=v,2=w) ghosts for one domain face (axis a, side s=0 low/1 high) with a
// scalar wall velocity. fold=1 drops the tangential wall face (ghost=0, implicit diffusion).
inline void bcVelocityComp(BField f, B3 ext, int g, int a, int s, int comp, double wall, int fold,
                           BField prof = BField(), int prof_nc = 0) {
  BExec space;
  int dims[3]; long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const int bf = (s == 0) ? g : (na - g);
  const bool hasProf = prof.extent(0) > 0;   // per-position inlet profile (resampled to the face grid)
  using MD = Kokkos::MDRangePolicy<BExec, Kokkos::Rank<2>>;
  Kokkos::parallel_for(
      "cfdk::bc_vel", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        const double wc = hasProf ? prof((static_cast<long>(p0) * prof_nc + p1) * 3 + comp) : wall;
        auto at = [&](int ia) -> double& { return f(base + static_cast<long>(ia) * sa); };
        if (comp == a) {  // normal: Dirichlet face + odd reflection
          at(bf) = wc;
          if (s == 0)
            for (int ia = 0; ia < g; ++ia) at(ia) = 2.0 * wc - at(2 * bf - ia);
          else
            for (int ia = na - g + 1; ia < na; ++ia) at(ia) = 2.0 * wc - at(2 * bf - ia);
        } else if (fold) {  // tangential implicit: drop wall face
          if (s == 0)
            for (int ia = 0; ia < g; ++ia) at(ia) = 0.0;
          else
            for (int ia = na - g; ia < na; ++ia) at(ia) = 0.0;
        } else {  // tangential explicit: cell-centred reflection about bf-0.5
          if (s == 0)
            for (int ia = 0; ia < g; ++ia) at(ia) = 2.0 * wc - at(2 * bf - 1 - ia);
          else
            for (int ia = na - g; ia < na; ++ia) at(ia) = 2.0 * wc - at(2 * bf - 1 - ia);
        }
      });

}

// Zero-gradient (Neumann) outflow velocity ghost for component comp on one face.
inline void bcOutflowComp(BField f, B3 ext, int g, int a, int s, int comp, int fold) {
  BExec space;
  int dims[3]; long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  using MD = Kokkos::MDRangePolicy<BExec, Kokkos::Rank<2>>;
  Kokkos::parallel_for(
      "cfdk::bc_outflow", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        auto at = [&](int ia) -> double& { return f(base + static_cast<long>(ia) * sa); };
        if (s == 0) {
          const int src = (comp == a) ? g + 1 : g;
          const int last = (comp == a) ? g : g - 1;
          const double v = fold ? 0.0 : at(src);
          for (int ia = 0; ia <= last; ++ia) at(ia) = v;
        } else {
          const int src = na - g - 1;
          const double v = fold ? 0.0 : at(src);
          for (int ia = na - g; ia < na; ++ia) at(ia) = v;
        }
      });

}

// Implicit-diffusion face-fold accumulation at the boundary-adjacent inner cell.
inline void bcDiffusionFold(BField dcorr, BField brhs, B3 ext, int g, int a, int s, double dval, double bval) {
  BExec space;
  int dims[3]; long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int bic = (s == 0) ? g : (dims[a] - g - 1);
  using MD = Kokkos::MDRangePolicy<BExec, Kokkos::Rank<2>>;
  Kokkos::parallel_for(
      "cfdk::bc_fold", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
        const long i = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc + static_cast<long>(bic) * sa;
        dcorr(i) += dval;
        brhs(i) += bval;
      });

}

// Hold the pressure ghost at 0 on an outflow face (Dirichlet p=0; the open face couples to it).
inline void bcZeroPressureGhost(BField phi, B3 ext, int g, int a, int s) {
  BExec space;
  int dims[3]; long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const int lo = (s == 0) ? 0 : (na - g), hi = (s == 0) ? (g - 1) : (na - 1);
  using MD = Kokkos::MDRangePolicy<BExec, Kokkos::Rank<2>>;
  Kokkos::parallel_for(
      "cfdk::bc_zero_p_ghost", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
        const long base = (long)p0 * sb + (long)p1 * sc;
        for (int ia = lo; ia <= hi; ++ia) phi(base + (long)ia * sa) = 0.0;
      });

}

// Projection correction of the high-side outflow normal face (index na-g) that correct_k misses:
// f -= phi[bf] - phi[bf-sa] (with the Dirichlet ghost phi[bf]=0 -> += phi_inner). (correct_outflow_k.)
inline void bcCorrectOutflow(BField f, BField phi, B3 ext, int g, int a) {
  BExec space;
  int dims[3]; long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  using MD = Kokkos::MDRangePolicy<BExec, Kokkos::Rank<2>>;
  Kokkos::parallel_for(
      "cfdk::bc_correct_outflow", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
        const long bf = (long)p0 * sb + (long)p1 * sc + (long)(dims[a] - g) * sa;
        f(bf) -= phi(bf) - phi(bf - sa);
      });

}

// Set the a-component face openness on a domain face to `val` (Neumann wall/inflow -> 0; the periodic fill
// would otherwise wrap the wrong value into an outflow face from the opposite boundary -> set it open = 1).
inline void bcSetOpenness(BField oa, B3 ext, int g, int a, int s, double val) {
  BExec space;
  int dims[3]; long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int bf = (s == 0) ? g : (dims[a] - g);
  using MD = Kokkos::MDRangePolicy<BExec, Kokkos::Rank<2>>;
  Kokkos::parallel_for(
      "cfdk::bc_setopen", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
        oa(static_cast<long>(p0) * sb + static_cast<long>(p1) * sc + static_cast<long>(bf) * sa) = val;
      });

}
inline void bcZeroOpenness(BField oa, B3 ext, int g, int a, int s) { bcSetOpenness(oa, ext, g, a, s, 0.0); }

}  // namespace cfdk

#endif  // CFD_MAC_BC_KOKKOS_HPP
