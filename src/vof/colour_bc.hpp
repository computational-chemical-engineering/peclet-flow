/// @file
/// @brief flow — VoF rung V-BC (WO-R): the colour field's OPEN-BOUNDARY ghost rules and the
/// out-of-domain mask that tells the Weymouth–Yue flux which donors are boundary DATA.
///
/// Rung V2a gave the colour field exactly one non-periodic ghost policy: `clampFill`, the
/// globally-clamped zero-gradient copy. That is the right rule for a WALL (a wall neither creates
/// nor destroys colour, and the 90° neutral continuation is what the MYC/height stencils want),
/// and it is the wrong rule for an INFLOW, where the colour of the incoming fluid is a prescribed
/// boundary datum, not an extrapolation of the interior.
///
/// Three rules live here, one per domain-BC type (`IbmSolver::setDomainBc`):
///
///  * **inflow (type 2)** — `bcColourConst` / `bcColourProfile` overwrite ALL `g` ghost layers of
///    the face with the prescribed colour, after `clampFill`. All `g` layers, not one: the MYC
///    stencil of the first inner cell reads ±1, the donor-ring reconstruction reaches one cell
///    further out, and the V3 height columns reach ±3, so a partially-filled band would let the
///    clamped interior value back into the stencil through the outer layers.
///  * **outflow (type 3)** — zero-gradient (`clampFill`, unchanged) where the fluid LEAVES, and
///    `bcColourBackflow` where the face velocity points back INTO the domain: there the ghost
///    carries the prescribed backflow colour instead. This is OpenFOAM's `inletOutlet` /
///    `inletOutletFvPatchField` (Rusche, PhD thesis 2002, §4), the standard VoF outlet: an outlet
///    is a place where you know what leaves (whatever is inside) but must state what comes back.
///    Zero-gradient on a backflow face re-injects whatever happens to be sitting at the outlet,
///    which for a draining film means the film is fed back into the domain.
///  * **wall (type 1)** — nothing here. `clampFill` IS the 90° neutral continuation
///    (Afkhami & Bussmann, IJNMF 57:453, 2008, expressed as fractions); WO-S replaces it with the
///    θ-consistent fill and a domain wall is just a flat SDF wall at the face.
///
/// ## The `outside` mask and why the flux needs it
///
/// `wyFaceFlux` fluxes the donor cell's PLIC SLAB: it reconstructs a plane in the donor from the
/// donor's own 3³ colour stencil and integrates the liquid volume in the |a|-thick layer next to
/// the face. For an interior donor that is the whole point of a geometric scheme. For a GHOST
/// donor on an inflow face it is meaningless and actively wrong:
///
///  * the inflow band is uniform by construction (`bcColourConst`), so its MYC normal is the
///    degenerate normal of a constant field — there is no interface there to reconstruct;
///  * a FRACTIONAL inflow colour (`set_vof_inflow(face, 0.3)`) is a statement about the incoming
///    FLUX ("30 % of what enters is liquid"), not about a sub-cell interface position. The slab
///    integral of a reconstructed plane would put all 0.3 of it on one side of the cell and let
///    the flux jump between 0 and `a` as the slab thickness crosses the plane.
///
/// So the flux on a face whose donor lies outside the global domain is the ALGEBRAIC `C_donor·a`.
/// `wyFaceFluxBc` (in `advect_wy.hpp`) is that rule; the mask built here is what selects it.
/// Everything else — the dilation term, the sweep structure, the frozen flag — is untouched, so
/// the conservation identity still telescopes: a boundary face's flux appears in exactly one
/// inner cell's budget and nowhere else, which is what makes the colour budget of gate G1
/// (`Σ eps C − ∫in + ∫out = const`) close to round-off.
///
/// The mask marks EVERY ghost cell outside the global domain on a non-periodic axis, not just the
/// inflow ones. It is installed on the advector only when a VoF boundary colour has actually been
/// set (`IbmSolver::vofBcActive_`), so with no VoF BC the advector takes its validated V1 path bit
/// for bit (WO-R gate G5). Where it IS installed, the change at a wall is nil (the wall's normal
/// face velocity is zero, so the flux is zero on both branches) and the change at an outflow face
/// is confined to backflow (an outgoing face's donor is the inner cell).
#ifndef PECLET_FLOW_VOF_COLOUR_BC_HPP
#define PECLET_FLOW_VOF_COLOUR_BC_HPP

#include <Kokkos_Core.hpp>

#include "mac_stencils.hpp"  // peclet::flow::SExec, SField, I3, L3
#include "vof/advect_wy.hpp"  // UCField

namespace peclet::flow::vof {

namespace bcdetail {
KOKKOS_INLINE_FUNCTION void axisDims(I3 e, int (&dims)[3], long (&strides)[3]) {
  dims[0] = e.x;
  dims[1] = e.y;
  dims[2] = e.z;
  strides[0] = 1;
  strides[1] = e.x;
  strides[2] = static_cast<long>(e.x) * e.y;
}
}  // namespace bcdetail

/// 1 for every ghost cell of the block that lies OUTSIDE the global domain on a non-periodic axis,
/// 0 everywhere else (inner cells, and ghosts that are interior halo on a periodic axis or across
/// a rank boundary). `o` is the GLOBAL index of the block's inner cell (0,0,0) and `gs` the global
/// grid size — the same pair `clampFill` uses, and for the same reason: the test must be a GLOBAL
/// one, so that a rank in the middle of a decomposition marks nothing and the mask is
/// decomposition-independent by construction.
inline void buildOutsideMask(UCField m, I3 e, int g, I3 o, I3 gs, bool px, bool py, bool pz) {
  const bool p0 = px, p1 = py, p2 = pz;
  const int q0 = gs.x, q1 = gs.y, q2 = gs.z;
  Kokkos::parallel_for(
      "vof::buildOutsideMask",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const int gx = x - g + o.x, gy = y - g + o.y, gz = z - g + o.z;
        const bool out = (!p0 && (gx < 0 || gx >= q0)) || (!p1 && (gy < 0 || gy >= q1)) ||
                         (!p2 && (gz < 0 || gz >= q2));
        m(L3(x, y, z, e)) = out ? 1u : 0u;
      });
  Kokkos::fence();
}

/// Overwrite ALL `g` ghost layers of domain face (axis `a`, side `s`) with a constant colour.
/// Call AFTER the exchange and `clampFill`; the caller is responsible for the ownership test
/// (`touchesGlobalFace`) so a rank whose block does not touch that global face writes nothing.
inline void bcColourConst(SField f, I3 e, int g, int a, int s, double value) {
  int dims[3];
  long strides[3];
  bcdetail::axisDims(e, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const int lo = (s == 0) ? 0 : (na - g);
  Kokkos::parallel_for(
      "vof::bcColourConst",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<2>>(SExec(), {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        for (int k = 0; k < g; ++k)
          f(base + static_cast<long>(lo + k) * sa) = value;
      });
  Kokkos::fence();
}

/// Same, with a per-position colour on the face's (b, c) plane. `prof` is indexed
/// `prof(p0 * nc + p1)` over the block's FULL ghost-inclusive transverse extents — i.e. it has
/// already been resampled with the clamp rule `IbmSolver::setDomainBcProfile` uses, so the kernel
/// indexes it directly by face position and needs no bounds arithmetic.
inline void bcColourProfile(SField f, I3 e, int g, int a, int s, SField prof, int nc) {
  int dims[3];
  long strides[3];
  bcdetail::axisDims(e, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const int lo = (s == 0) ? 0 : (na - g);
  Kokkos::parallel_for(
      "vof::bcColourProfile",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<2>>(SExec(), {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        const double v = prof(static_cast<long>(p0) * nc + p1);
        for (int k = 0; k < g; ++k)
          f(base + static_cast<long>(lo + k) * sa) = v;
      });
  Kokkos::fence();
}

/// `inletOutlet` on an outflow face: where the boundary face velocity points INTO the domain, the
/// ghost band takes `backflow`; where it points out, the band is left as `clampFill` wrote it
/// (zero-gradient, so what leaves is what is inside).
///
/// `un` is the advector's own face-velocity component `a` (HIGH-face convention, `un(i)` is the
/// `+a` face of cell `i`), so the boundary face of side 0 is the `+a` face of the last ghost cell
/// (`a`-index `g-1`) and the boundary face of side 1 is the `+a` face of the last inner cell
/// (`a`-index `na - g - 1`). Inward is `+` on side 0 and `−` on side 1. The test is on the
/// velocity the advector is about to flux with, which is why this must run AFTER
/// `bridgeVelocityToVof` — see `IbmSolver::vofApplyColourBc`.
inline void bcColourBackflow(SField f, I3 e, int g, int a, int s, SField un, double backflow) {
  int dims[3];
  long strides[3];
  bcdetail::axisDims(e, dims, strides);
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  const long sa = strides[a], sb = strides[b], sc = strides[c];
  const int na = dims[a];
  const int lo = (s == 0) ? 0 : (na - g);
  const int fa = (s == 0) ? (g - 1) : (na - g - 1);  // a-index of the cell owning the boundary face
  const double sgn = (s == 0) ? 1.0 : -1.0;          // sign of an INWARD normal velocity
  Kokkos::parallel_for(
      "vof::bcColourBackflow",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<2>>(SExec(), {0, 0}, {dims[b], dims[c]}),
      KOKKOS_LAMBDA(int p0, int p1) {
        const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
        if (sgn * un(base + static_cast<long>(fa) * sa) <= 0.0)
          return;  // leaving (or exactly zero): keep the zero-gradient band
        for (int k = 0; k < g; ++k)
          f(base + static_cast<long>(lo + k) * sa) = backflow;
      });
  Kokkos::fence();
}

}  // namespace peclet::flow::vof

#endif  // PECLET_FLOW_VOF_COLOUR_BC_HPP
