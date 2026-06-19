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
inline void bcVelocityComp(BField f, B3 ext, int g, int a, int s, int comp, double wall, int fold) {
  BExec space;
  int dims[3]; long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const int bf = (s == 0) ? g : (na - g);
  using MD = Kokkos::MDRangePolicy<BExec, Kokkos::Rank<2>>;
  Kokkos::parallel_for(
      "cfdk::bc_vel", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        auto at = [&](int ia) -> double& { return f(base + static_cast<long>(ia) * sa); };
        if (comp == a) {  // normal: Dirichlet face + odd reflection
          at(bf) = wall;
          if (s == 0)
            for (int ia = 0; ia < g; ++ia) at(ia) = 2.0 * wall - at(2 * bf - ia);
          else
            for (int ia = na - g + 1; ia < na; ++ia) at(ia) = 2.0 * wall - at(2 * bf - ia);
        } else if (fold) {  // tangential implicit: drop wall face
          if (s == 0)
            for (int ia = 0; ia < g; ++ia) at(ia) = 0.0;
          else
            for (int ia = na - g; ia < na; ++ia) at(ia) = 0.0;
        } else {  // tangential explicit: cell-centred reflection about bf-0.5
          if (s == 0)
            for (int ia = 0; ia < g; ++ia) at(ia) = 2.0 * wall - at(2 * bf - 1 - ia);
          else
            for (int ia = na - g; ia < na; ++ia) at(ia) = 2.0 * wall - at(2 * bf - 1 - ia);
        }
      });
  space.fence();
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
  space.fence();
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
  space.fence();
}

// Zero the a-component face openness on a wall face -> homogeneous Neumann pressure.
inline void bcZeroOpenness(BField oa, B3 ext, int g, int a, int s) {
  BExec space;
  int dims[3]; long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int bf = (s == 0) ? g : (dims[a] - g);
  using MD = Kokkos::MDRangePolicy<BExec, Kokkos::Rank<2>>;
  Kokkos::parallel_for(
      "cfdk::bc_zopen", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
        oa(static_cast<long>(p0) * sb + static_cast<long>(p1) * sc + static_cast<long>(bf) * sa) = 0.0;
      });
  space.fence();
}

}  // namespace cfdk

#endif  // CFD_MAC_BC_KOKKOS_HPP
