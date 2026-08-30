// Test-only scene builders shared by tests/kokkos/test_vof_advect.cpp and
// tests/kokkos_mpi/test_vof_advect_mpi.cpp (VoF rung V1, WO-E).
//
// Everything here is a pure function of GLOBAL cell/face indices, so a block that owns a sub-range
// of the global grid gets bit-identical data to the corresponding part of a single-block run. That
// is what makes the np 1/2/4 bitwise gate meaningful rather than a tautology.
//
// The one piece with real content is `fillLeVeque`: the 3D deformation field is sampled as the
// DISCRETE CURL of an edge-based vector potential rather than pointwise, so the discrete face
// divergence is round-off rather than O(h^2). Weymouth-Yue conserves volume to exactly the accuracy
// with which that divergence vanishes -- the dilation term contributes `H(C-1/2) * div * dt/h` to
// the volume budget of EVERY cell, interior full cells included -- so pointwise sampling would put
// the conservation floor at O(h^2), not at 1e-15.
#ifndef PECLET_FLOW_TESTS_VOF_ADVECT_SCENES_HPP
#define PECLET_FLOW_TESTS_VOF_ADVECT_SCENES_HPP

#include <cmath>
#include <Kokkos_Core.hpp>

#include "vof/advect_wy.hpp"
#include "vof/plic.hpp"

namespace vofscene {

using peclet::flow::I3;
using peclet::flow::L3;
using peclet::flow::SExec;
using peclet::flow::SField;
using peclet::flow::vof::WyAdvector;

/// Where this block sits in the global grid: `o` is the global index of inner cell (0,0,0).
struct Block {
  I3 n{0, 0, 0};  ///< inner cell counts
  I3 e{0, 0, 0};  ///< extended (inner + 2g) counts
  int g = 3;      ///< ghost width
  I3 o{0, 0, 0};  ///< global index of the first inner cell
};

inline Block blockOf(const WyAdvector& a, I3 origin) {
  return Block{a.inner(), a.extent(), a.ghost(), origin};
}

// ------------------------------------------------------------------------------ ghost fills

/// Periodic wrap of a single-block extended field (serial reference path; the MPI path uses
/// core's GridHalo instead). Sequential axis passes, so edges/corners come out right.
inline void periodicFill(SField f, I3 e, int g, bool px, bool py, bool pz) {
  const long st[3] = {1, e.x, static_cast<long>(e.x) * e.y};
  const int dims[3] = {e.x, e.y, e.z};
  const bool per[3] = {px, py, pz};
  for (int a = 0; a < 3; ++a) {
    if (!per[a])
      continue;
    const int b = (a + 1) % 3, c = (a + 2) % 3;
    const long sa = st[a], sb = st[b], sc = st[c];
    const int N = dims[a] - 2 * g;
    Kokkos::parallel_for(
        "vofscene::periodicFill",
        Kokkos::MDRangePolicy<SExec, Kokkos::Rank<2>>(SExec(), {0, 0}, {dims[b], dims[c]}),
        KOKKOS_LAMBDA(int p0, int p1) {
          const long base = static_cast<long>(p0) * sb + static_cast<long>(p1) * sc;
          for (int gl = 0; gl < g; ++gl) {
            f(base + static_cast<long>(gl) * sa) = f(base + static_cast<long>(gl + N) * sa);
            f(base + static_cast<long>(g + N + gl) * sa) = f(base + static_cast<long>(g + gl) * sa);
          }
        });
    Kokkos::fence();
  }
}

/// Zero-gradient (Neumann) fill of the ghost cells that lie OUTSIDE the global domain on a
/// non-periodic axis: each such cell takes the value at its globally CLAMPED index.
///
/// The clamp is per axis and global, which is what makes it decomposition-independent. The obvious
/// alternative -- sequential zero-gradient axis passes over the block -- is NOT: a corner cell that
/// is outside the domain in y but whose x-neighbourhood is an interior halo gets the x-extension of
/// the true row on one decomposition and the x-extension of the y = 0 row on another, and that
/// difference reaches the colour field through the MYC stencil of the inner corner cell. (Which is
/// exactly the "MPI differs at the last bit only in ghost-adjacent cells" symptom the work order
/// lists; it was found by reasoning about the fill, before it could be blamed on the halo.)
///
/// The clamped source is always inside the domain, hence either an inner cell of this block or a
/// ghost the exchange has already filled: a ghost at global index `< 0` on axis a only exists when
/// `origin[a] < g`, so index 0 on that axis is inside the block's extended range. Call AFTER the
/// halo exchange / periodic fill.
inline void clampFill(SField f, Block b, I3 gs, bool px, bool py, bool pz) {
  if (px && py && pz)
    return;
  const I3 e = b.e, o = b.o;
  const int g = b.g;
  const bool per[3] = {px, py, pz};
  const int gsz[3] = {gs.x, gs.y, gs.z};
  const bool p0 = per[0], p1 = per[1], p2 = per[2];
  const int q0 = gsz[0], q1 = gsz[1], q2 = gsz[2];
  Kokkos::parallel_for(
      "vofscene::clampFill",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        const int gx = x - g + o.x, gy = y - g + o.y, gz = z - g + o.z;
        const int cx = p0 ? gx : (gx < 0 ? 0 : (gx >= q0 ? q0 - 1 : gx));
        const int cy = p1 ? gy : (gy < 0 ? 0 : (gy >= q1 ? q1 - 1 : gy));
        const int cz = p2 ? gz : (gz < 0 ? 0 : (gz >= q2 ? q2 - 1 : gz));
        if (cx != gx || cy != gy || cz != gz)
          f(L3(x, y, z, e)) = f(L3(cx - o.x + g, cy - o.y + g, cz - o.z + g, e));
      });
  Kokkos::fence();
}

// ------------------------------------------------------------------------------ colour fields

/// Visit every cell of the extended block with (local linear index, global cell index).
template <typename F>
inline void forEachExtended(Block b, F fn) {
  const I3 e = b.e;
  const int g = b.g;
  const I3 o = b.o;
  Kokkos::parallel_for(
      "vofscene::forEachExtended",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {0, 0, 0}, {e.x, e.y, e.z}),
      KOKKOS_LAMBDA(int x, int y, int z) {
        fn(L3(x, y, z, e), x - g + o.x, y - g + o.y, z - g + o.z);
      });
  Kokkos::fence();
}

/// Sphere of fluid (C = 1 inside), exact to ~5e-7 relative volume at R/h = 16 via the octree
/// helper in plic.hpp. Domain is [0,1]^3 with h = 1/gn.
inline void initSphere(SField c, Block b, double h, double cx, double cy, double cz, double r,
                       int levels = 4) {
  forEachExtended(
      b, KOKKOS_LAMBDA(long i, int gx, int gy, int gz) {
        c(i) =
            peclet::flow::vof::sphereCellFraction(cx, cy, cz, r, gx * h, gy * h, gz * h, h, levels);
      });
}

/// Slab `zlo < x < zhi` along axis `axis`, exact fractions (axis-aligned planes).
inline void initSlab(SField c, Block b, double h, int axis, double lo, double hi) {
  forEachExtended(
      b, KOKKOS_LAMBDA(long i, int gx, int gy, int gz) {
        const int gi[3] = {gx, gy, gz};
        const double a = gi[axis] * h, bb = a + h;
        const double cover = Kokkos::fmax(0.0, Kokkos::fmin(bb, hi) - Kokkos::fmax(a, lo));
        c(i) = cover / h;
      });
}

/// Zalesak's slotted disk in the (x,y) plane, uniform in z. Domain [0,1]^3; the disk is a circle of
/// radius `r` about (cx,cy) with a rectangular slot of half-width `sw` reaching up to y = `stop`.
/// Fractions by n x n midpoint subsampling (the metric compares the transported field against THIS
/// field, so the sampling error is common to both and does not enter the error).
inline void initZalesak(SField c, Block b, double h, double cx, double cy, double r, double sw,
                        double stop, int nsub = 64) {
  const double w = 1.0 / nsub;
  forEachExtended(
      b, KOKKOS_LAMBDA(long i, int gx, int gy, int) {
        const double x0 = gx * h, y0 = gy * h;
        int in = 0;
        for (int q = 0; q < nsub; ++q) {
          const double y = y0 + (q + 0.5) * w * h;
          for (int p = 0; p < nsub; ++p) {
            const double x = x0 + (p + 0.5) * w * h;
            const double dx = x - cx, dy = y - cy;
            const bool inCircle = dx * dx + dy * dy < r * r;
            const bool inSlot = Kokkos::fabs(dx) < sw && y < stop;
            if (inCircle && !inSlot)
              ++in;
          }
        }
        c(i) = static_cast<double>(in) / (static_cast<double>(nsub) * nsub);
      });
}

// ------------------------------------------------------------------------------ velocity fields

/// Uniform translation. Discrete divergence is EXACTLY zero (identical values differenced).
inline void fillUniform(WyAdvector& a, Block b, double ux, double uy, double uz) {
  SField u = a.faceU(), v = a.faceV(), w = a.faceW();
  forEachExtended(
      b, KOKKOS_LAMBDA(long i, int, int, int) {
        u(i) = ux;
        v(i) = uy;
        w(i) = uz;
      });
}

/// Solid-body rotation about (cx,cy) in the (x,y) plane, w = 0. Discrete divergence is EXACTLY
/// zero: u does not depend on x and v does not depend on y, so each 1D difference is 0 bitwise.
inline void fillRotation(WyAdvector& a, Block b, double h, double cx, double cy, double omega) {
  SField u = a.faceU(), v = a.faceV(), w = a.faceW();
  forEachExtended(
      b, KOKKOS_LAMBDA(long i, int gx, int gy, int) {
        u(i) = -omega * ((gy + 0.5) * h - cy);
        v(i) = omega * ((gx + 1.0) * h - cx);
        w(i) = 0.0;
      });
}

/// The LeVeque 3D deformation field
///     u =  2 sin^2(pi x) sin(2 pi y) sin(2 pi z) cos(pi t / T)
///     v = -  sin(2 pi x) sin^2(pi y) sin(2 pi z) cos(pi t / T)
///     w = -  sin(2 pi x) sin(2 pi y) sin^2(pi z) cos(pi t / T)
/// sampled as the discrete curl of the edge vector potential
///     A = ( 0 , -psi2(x,z) sin(2 pi y) , psi1(x,y) sin(2 pi z) ) cos(pi t / T),
///     psi1 = sin^2(pi x) sin^2(pi y) / pi ,  psi2 = sin^2(pi x) sin^2(pi z) / pi .
/// (curl A reproduces the field exactly; the discrete curl reproduces it to O(h^2) while making
/// the discrete face divergence identically zero in exact arithmetic.)
KOKKOS_INLINE_FUNCTION double lvSin2(double t) {
  const double s = Kokkos::sin(3.14159265358979323846 * t);
  return s * s;
}
/// A_y of the LeVeque vector potential (lives on y-edges: node x, centre y, node z).
KOKKOS_INLINE_FUNCTION double lvAy(double x, double y, double z, double phase) {
  const double PI = 3.14159265358979323846;
  return -(lvSin2(x) * lvSin2(z) / PI) * Kokkos::sin(2.0 * PI * y) * phase;
}
/// A_z of the LeVeque vector potential (lives on z-edges: node x, node y, centre z).
KOKKOS_INLINE_FUNCTION double lvAz(double x, double y, double z, double phase) {
  const double PI = 3.14159265358979323846;
  return (lvSin2(x) * lvSin2(y) / PI) * Kokkos::sin(2.0 * PI * z) * phase;
}

inline void fillLeVeque(WyAdvector& a, Block b, double h, double phase) {
  SField u = a.faceU(), v = a.faceV(), w = a.faceW();
  const double invh = 1.0 / h;
  forEachExtended(
      b, KOKKOS_LAMBDA(long i, int gx, int gy, int gz) {
        const double xn = gx * h, yn = gy * h, zn = gz * h;  // node
        const double yc = (gy + 0.5) * h, zc = (gz + 0.5) * h;
        const double xp = (gx + 1) * h, yp = (gy + 1) * h, zp = (gz + 1) * h;  // next node
        // u = dAz/dy - dAy/dz  at (xp, yc, zc)
        u(i) = (lvAz(xp, yp, zc, phase) - lvAz(xp, yn, zc, phase)) * invh -
               (lvAy(xp, yc, zp, phase) - lvAy(xp, yc, zn, phase)) * invh;
        // v = -dAz/dx  at (xc, yp, zc)      (A_x = 0)
        v(i) = -(lvAz(xp, yp, zc, phase) - lvAz(xn, yp, zc, phase)) * invh;
        // w =  dAy/dx  at (xc, yc, zp)
        w(i) = (lvAy(xp, yc, zp, phase) - lvAy(xn, yc, zp, phase)) * invh;
      });
}

// ------------------------------------------------------------------------------ metrics

/// sum |a - b| over the inner region (cell-fraction units).
inline double l1Diff(SField a, SField b, I3 e, I3 n, int g) {
  double s = 0.0;
  Kokkos::parallel_reduce(
      "vofscene::l1",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                    {g + n.x, g + n.y, g + n.z}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = L3(x, y, z, e);
        acc += Kokkos::fabs(a(i) - b(i));
      },
      s);
  Kokkos::fence();
  return s;
}

/// max |a - b| over the inner region.
inline double lInfDiff(SField a, SField b, I3 e, I3 n, int g) {
  double s = 0.0;
  Kokkos::parallel_reduce(
      "vofscene::linf",
      Kokkos::MDRangePolicy<SExec, Kokkos::Rank<3>>(SExec(), {g, g, g},
                                                    {g + n.x, g + n.y, g + n.z}),
      KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
        const long i = L3(x, y, z, e);
        acc = Kokkos::fmax(acc, Kokkos::fabs(a(i) - b(i)));
      },
      Kokkos::Max<double>(s));
  Kokkos::fence();
  return s;
}

inline SField copyOf(SField src, const char* name) {
  SField dst(name, src.extent(0));
  Kokkos::deep_copy(dst, src);
  return dst;
}

}  // namespace vofscene

#endif  // PECLET_FLOW_TESTS_VOF_ADVECT_SCENES_HPP
