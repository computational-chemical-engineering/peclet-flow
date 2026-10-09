/// @file
/// @brief flow — portable (Kokkos) native per-face domain boundary conditions for the MAC grid.
///
/// Kokkos port of mac_bc.cuh: fill the NON-periodic boundary face/ghosts the halo leaves untouched.
/// Each kernel runs over the boundary face's perpendicular (b,c) plane; one thread owns a column
/// and writes its own ghosts (disjoint, no races). MAC staggered convention: component a is stored
/// at the -a face of its cell. Faithful copies of the reflection / fold / outflow logic. Scalar
/// wall velocity (the per-position inlet-profile variant is a later specialization). Runs on any
/// Kokkos backend.
#ifndef PECLET_FLOW_MAC_BC_HPP
#define PECLET_FLOW_MAC_BC_HPP

#include <Kokkos_Core.hpp>

#include "policy.hpp"

namespace peclet::flow {

using BExec = Kokkos::DefaultExecutionSpace;
using BMem = BExec::memory_space;
using BField = Kokkos::View<double*, BMem>;

struct B3 {
  int x, y, z;
};

namespace bcdetail {
KOKKOS_INLINE_FUNCTION void axisDims(B3 ext, int (&dims)[3], long (&strides)[3]) {
  dims[0] = ext.x;
  dims[1] = ext.y;
  dims[2] = ext.z;
  strides[0] = 1;
  strides[1] = ext.x;
  strides[2] = static_cast<long>(ext.x) * ext.y;
}
// One column of bcVelocityComp: the ghosts of component `comp` on face (axis a, side s) along the
// column `at(ia)` (ia = the index along a, na = the extent along a), wall velocity `wc`.
template <class At>
KOKKOS_INLINE_FUNCTION void velFaceColumn(const At& at, int na, int g, int a, int s, int comp,
                                          double wc, int fold) {
  const int bf = (s == 0) ? g : (na - g);
  if (comp == a) {  // normal: Dirichlet face + odd reflection
    at(bf) = wc;
    if (s == 0)
      for (int ia = 0; ia < g; ++ia)
        at(ia) = 2.0 * wc - at(2 * bf - ia);
    else
      for (int ia = na - g + 1; ia < na; ++ia)
        at(ia) = 2.0 * wc - at(2 * bf - ia);
  } else if (fold) {  // tangential implicit: drop wall face
    if (s == 0)
      for (int ia = 0; ia < g; ++ia)
        at(ia) = 0.0;
    else
      for (int ia = na - g; ia < na; ++ia)
        at(ia) = 0.0;
  } else {  // tangential explicit: cell-centred reflection about bf-0.5
    if (s == 0)
      for (int ia = 0; ia < g; ++ia)
        at(ia) = 2.0 * wc - at(2 * bf - 1 - ia);
    else
      for (int ia = na - g; ia < na; ++ia)
        at(ia) = 2.0 * wc - at(2 * bf - 1 - ia);
  }
}
}  // namespace bcdetail

// Fill component comp (0=u,1=v,2=w) ghosts for one domain face (axis a, side s=0 low/1 high) with a
// scalar wall velocity. fold=1 drops the tangential wall face (ghost=0, implicit diffusion).
inline void bcVelocityComp(BField f, B3 ext, int g, int a, int s, int comp, double wall, int fold,
                           BField prof = BField(), int prof_nc = 0) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const bool hasProf =
      prof.extent(0) > 0;  // per-position inlet profile (resampled to the face grid)
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_vel", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        const double wc = hasProf ? prof((static_cast<long>(p0) * prof_nc + p1) * 3 + comp) : wall;
        auto at = [&](int ia) -> double& { return f(base + static_cast<long>(ia) * sa); };
        bcdetail::velFaceColumn(at, na, g, a, s, comp, wc, fold);
      });
}

// §14 H-3(e): up to six bcVelocityComp applications on ONE axis a -- (field, component, side, wall
// velocity or inlet profile) -- in one launch over the face's perpendicular plane. Each column runs
// the applications in table order, so for every field the faces keep their sequential order
// (s = 0 before s = 1) and different fields are independent: bit-identical to the separate launches
// in that order. The caller merges only faces whose ghost planes are disjoint (ext_a > 2g).
struct BcVelFace {
  BField f, prof;  // the field; the resampled inlet profile (empty: the scalar `wall`)
  double wall = 0.0;
  int comp = 0, s = 0, profNc = 0;
};
struct BcVelFaces {
  static constexpr int kMax = 6;
  BcVelFace j[kMax];
  int n = 0;
};
inline void bcVelocityFaces(const BcVelFaces& J, B3 ext, int g, int a, int fold) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_vel", MD(space, {0, 0}, {dims[b], dims[c]}), KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        for (int k = 0; k < J.n; ++k) {
          const BcVelFace& q = J.j[k];
          const double wc = (q.prof.extent(0) > 0)
                                ? q.prof((static_cast<long>(p0) * q.profNc + p1) * 3 + q.comp)
                                : q.wall;
          auto at = [&](int ia) -> double& { return q.f(base + static_cast<long>(ia) * sa); };
          bcdetail::velFaceColumn(at, na, g, a, q.s, q.comp, wc, fold);
        }
      });
}

// FREE-SLIP / SYMMETRY face (BC type 4) for component comp on one domain face (staggered).
//   * normal component (comp == a): impermeable -- the boundary face is held at 0 and the ghosts
//     are the odd reflection about it, exactly the no-slip wall with wall velocity 0;
//   * tangential components: zero normal derivative. Explicit (fold=0): the EVEN reflection about
//     the face, ghost(ia) = inner(mirror) -- exact for any profile symmetric about the face, so a
//     half-channel with this face reproduces the full channel pointwise. Implicit (fold=1): the
//     face is dropped (ghost = 0) and setupBcDiffusion folds -beta onto the diagonal (the mirror
//     neighbour equals the cell itself, so the face's beta*(ghost - inner) vanishes) -- the same
//     fold the outflow's zero-gradient tangential treatment uses.
inline void bcSlipComp(BField f, B3 ext, int g, int a, int s, int comp, int fold) {
  if (comp == a) {
    bcVelocityComp(f, ext, g, a, s, comp, 0.0, fold);
    return;
  }
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const int bf = (s == 0) ? g : (na - g);
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_slip", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        auto at = [&](int ia) -> double& { return f(base + static_cast<long>(ia) * sa); };
        if (s == 0)
          for (int ia = 0; ia < g; ++ia)
            at(ia) = fold ? 0.0 : at(2 * bf - 1 - ia);  // even reflection about bf-0.5
        else
          for (int ia = na - g; ia < na; ++ia)
            at(ia) = fold ? 0.0 : at(2 * bf - 1 - ia);
      });
}

// Even (mirror) reflection of a cell-centered field about one domain face: ghost(ia) =
// inner(mirror). The collocated tangential free-slip ghost (a symmetric profile is reproduced
// exactly, which bcNeumannGhost's copy of the boundary cell into every layer is not).
inline void bcMirrorGhost(BField f, B3 ext, int g, int a, int s) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_mirror_ghost", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        auto at = [&](int ia) -> double& { return f(base + static_cast<long>(ia) * sa); };
        if (s == 0)
          for (int ia = 0; ia < g; ++ia)
            at(ia) = at(2 * g - 1 - ia);  // mirror about g-1/2
        else
          for (int ia = na - g; ia < na; ++ia)
            at(ia) = at(2 * (na - g) - 1 - ia);  // about na-g-1/2
      });
}

// Collocated (cell-centered) velocity Dirichlet / no-slip ghost on one domain face. The wall sits
// at the boundary FACE (between the last inner cell and the first ghost), so EVERY component is
// reflected about it
// -- ghost = 2*wc - mirror(interior) makes the face-interpolated value equal `wc`. No
// normal/tangential split and no implicit fold: explicit reflection, re-imposed each smoother sweep
// (converges to no-slip). `wc` is the scalar `wall`, or a per-position value from `prof` (the
// resampled inlet profile, indexed (p0*prof_nc+p1)*3+comp over the face's perpendicular plane) when
// one is supplied (e.g. the BFS step inlet).
inline void bcVelocityColocated(BField f, B3 ext, int g, int a, int s, double wall, int comp = 0,
                                BField prof = BField(), int prof_nc = 0) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const bool hasProf = prof.extent(0) > 0;
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_vel_coloc", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        const double wc = hasProf ? prof((static_cast<long>(p0) * prof_nc + p1) * 3 + comp) : wall;
        auto at = [&](int ia) -> double& { return f(base + static_cast<long>(ia) * sa); };
        if (s == 0)
          for (int ia = 0; ia < g; ++ia)
            at(ia) = 2.0 * wc - at(2 * g - 1 - ia);  // mirror about g-1/2
        else
          for (int ia = na - g; ia < na; ++ia)
            at(ia) = 2.0 * wc - at(2 * (na - g) - 1 - ia);  // about na-g-1/2
      });
}

// A0 cell bodies (doc/vof_step_performance_design.md §5.2) of the two pressure ghost policies
// below: one transverse position `base` of the face, its ghost layers [lo, hi] along the stride sa.
template <class FV>
KOKKOS_INLINE_FUNCTION void bcNeumannGhostCell(const FV& f, long base, long sa, int bic, int lo,
                                               int hi) {
  const double pin = f(base + static_cast<long>(bic) * sa);
  for (int ia = lo; ia <= hi; ++ia)
    f(base + static_cast<long>(ia) * sa) = pin;
}
template <class FV>
KOKKOS_INLINE_FUNCTION void bcZeroPressureGhostCell(const FV& phi, long base, long sa, int lo,
                                                    int hi) {
  for (int ia = lo; ia <= hi; ++ia)
    phi(base + (long)ia * sa) = 0.0;
}

// Zero-gradient (Neumann) ghost on one domain face: every ghost layer = the boundary-adjacent inner
// cell (cell-centered). Used for the collocated pressure increment phi at walls so the
// cell-centered correction carries no spurious normal acceleration (the same role pressureBcGhost
// plays for P in the predictor).
template <class FV>
inline void bcNeumannGhostT(FV f, B3 ext, int g, int a, int s) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const int bic = (s == 0) ? g : (na - g - 1);
  const int lo = (s == 0) ? 0 : (na - g), hi = (s == 0) ? (g - 1) : (na - 1);
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_neumann_ghost", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        bcNeumannGhostCell(f, static_cast<long>(p0) * sb + static_cast<long>(p1) * sc, sa, bic, lo,
                           hi);
      });
}
inline void bcNeumannGhost(BField f, B3 ext, int g, int a, int s) {
  bcNeumannGhostT(f, ext, g, a, s);
}

// Zero-gradient (Neumann) outflow velocity ghost for component comp on one face.
inline void bcOutflowComp(BField f, B3 ext, int g, int a, int s, int comp, int fold) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_outflow", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        auto at = [&](int ia) -> double& { return f(base + static_cast<long>(ia) * sa); };
        if (s == 0) {
          const int src = (comp == a) ? g + 1 : g;
          const int last = (comp == a) ? g : g - 1;
          const double v = fold ? 0.0 : at(src);
          for (int ia = 0; ia <= last; ++ia)
            at(ia) = v;
        } else {
          const int src = na - g - 1;
          const double v = fold ? 0.0 : at(src);
          for (int ia = na - g; ia < na; ++ia)
            at(ia) = v;
        }
      });
}

// Implicit-diffusion face-fold accumulation at the boundary-adjacent inner cell.
inline void bcDiffusionFold(BField dcorr, BField brhs, B3 ext, int g, int a, int s, double dval,
                            double bval) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int bic = (s == 0) ? g : (dims[a] - g - 1);
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_fold", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long i =
            static_cast<long>(p0) * sb + static_cast<long>(p1) * sc + static_cast<long>(bic) * sa;
        dcorr(i) += dval;
        brhs(i) += bval;
      });
}

// Hold the pressure ghost at 0 on an outflow face (Dirichlet p=0; the open face couples to it).
inline void bcZeroPressureGhost(BField phi, B3 ext, int g, int a, int s) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const int lo = (s == 0) ? 0 : (na - g), hi = (s == 0) ? (g - 1) : (na - 1);
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_zero_p_ghost", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        bcZeroPressureGhostCell(phi, (long)p0 * sb + (long)p1 * sc, sa, lo, hi);
      });
}

// Projection correction of the high-side outflow normal face (index na-g) that correct_k misses:
// f -= phi[bf] - phi[bf-sa] (with the Dirichlet ghost phi[bf]=0 -> += phi_inner).
// (correct_outflow_k.)
//
// PHASE 2 (doc/anisotropic_metric.md §3): `wa` is the per-axis pressure weight w_a = 1/h_a'^2 of
// the face's own axis, applied OUTSIDE the existing expression exactly as in the interior
// kernels (projectCorrect and friends) this face continues. 1.0 on the isotropic path (exact).
inline void bcCorrectOutflow(BField f, BField phi, B3 ext, int g, int a, double wa = 1.0) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_correct_outflow", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long bf = (long)p0 * sb + (long)p1 * sc + (long)(dims[a] - g) * sa;
        f(bf) -= wa * (phi(bf) - phi(bf - sa));
      });
}

// VARIABLE-DENSITY sibling of bcCorrectOutflow (WO-R item 4; the gap
// doc/variable_density_projection.md section 4 listed as "revisit with a two-phase outflow case").
//
// Every other face of the staggered correction carries the mobility factor rho0/rho_f
// (projectCorrectVar); the outflow face did not, so at a density ratio r the outflow face was
// corrected by a factor r too much or too little relative to its neighbours. Constant density
// hides it exactly (rho_f == rho0 makes the factor 1), which is why the channel/BFS validations
// never saw it.
//
// rho_f is the SAME arithmetic face mean the operator coefficient and every other corrected face
// use (buildRhoCoeff / projectCorrectVar). At the outflow face the outer cell is a ghost whose rho
// the Neumann property policy (fillPropGhosts) set to the inner cell's rho, so this evaluates to
// the inner cell's rho — the statement WO-R makes — while remaining literally the same expression
// as the interior kernel, which is what keeps the two consistent if the ghost policy ever changes.
inline void bcCorrectOutflowVar(BField f, BField phi, BField rho, double rho0, B3 ext, int g, int a,
                                bool harmonic, double wa = 1.0) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_correct_outflow_var", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long bf = (long)p0 * sb + (long)p1 * sc + (long)(dims[a] - g) * sa;
        const double ra = rho(bf), rb = rho(bf - sa);
        // matched to projectCorrectVar / projectCorrectVarHarm (the harmonic knob is a measured
        // ablation, set_rho_face_harmonic; it must switch BOTH or the outflow face disagrees with
        // the interior it is continuing)
        const double rf = harmonic ? (2.0 * ra * rb / (ra + rb)) : (0.5 * (ra + rb));
        f(bf) -= wa * (rho0 / rf * (phi(bf) - phi(bf - sa)));
      });
}

// Save / restore the HIGH-side domain-face plane of a field over the block's INNER transverse
// range. SCALING_ISSUES #3/#8: that face is the first GHOST index along axis `a`, and the
// distributed halo wraps periodically on every axis, so after an exchange it carries the OPPOSITE
// boundary's value -- which destroys the mass-conserving outflow correction the projection wrote
// there and that `fillVelGhostsTo(..., doOutflow = false)` exists to preserve.
//
// INNER transverse range ONLY, and that is the point: the plane's transverse ghost rows are
// legitimately the NEIGHBOUR's inner values, which the exchange has just delivered correctly.
// Putting stale local values back over them would trade one wrong plane for another.
inline void bcSaveHighFacePlaneInner(BField dst, BField src, B3 ext, int g, int a) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int bf = dims[a] - g, nb = dims[b] - 2 * g, nc = dims[c] - 2 * g;
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_save_high_face", MD(space, {0, 0}, {nb, nc}),
      KOKKOS_LAMBDA(int p0, int p1) {
        dst((long)p0 + (long)p1 * nb) =
            src((long)(p0 + g) * sb + (long)(p1 + g) * sc + (long)bf * sa);
      });
}
inline void bcRestoreHighFacePlaneInner(BField dst, BField src, B3 ext, int g, int a) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int bf = dims[a] - g, nb = dims[b] - 2 * g, nc = dims[c] - 2 * g;
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_restore_high_face", MD(space, {0, 0}, {nb, nc}),
      KOKKOS_LAMBDA(int p0, int p1) {
        dst((long)(p0 + g) * sb + (long)(p1 + g) * sc + (long)bf * sa) =
            src((long)p0 + (long)p1 * nb);
      });
}

// Set the a-component face openness on a domain face to `val` (Neumann wall/inflow -> 0; the
// periodic fill would otherwise wrap the wrong value into an outflow face from the opposite
// boundary -> set it open = 1).
inline void bcSetOpenness(BField oa, B3 ext, int g, int a, int s, double val) {
  BExec space;
  int dims[3];
  long strides[3];
  bcdetail::axisDims(ext, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int bf = (s == 0) ? g : (dims[a] - g);
  using MD = MDRange2<BExec>;
  Kokkos::parallel_for(
      "peclet::flow::bc_setopen", MD(space, {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        oa(static_cast<long>(p0) * sb + static_cast<long>(p1) * sc + static_cast<long>(bf) * sa) =
            val;
      });
}
inline void bcZeroOpenness(BField oa, B3 ext, int g, int a, int s) {
  bcSetOpenness(oa, ext, g, a, s, 0.0);
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_MAC_BC_HPP
