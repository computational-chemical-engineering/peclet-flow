// flow — the per-cell DRAG-COEFFICIENT ghost ring (VoF blocker WO-I).
//
// The defect this test exists for. Under `porous_` (the volume-averaged CFD-DEM gas phase),
// `addDragDiagonal` builds the staggered momentum diagonal of component c from the FACE drag
//
//     beta_f(i) = 0.5*(beta(i) + beta(i - stride_c))        [flow_ibm.hpp, addDragDiagonal]
//
// so that it matches the projection's SIMPLE-like coefficient and correction,
// w_f = idt/(idt + beta_f) (`buildPorousCoeffDrag`/`Cons`, `projectCorrectPorous*`) — the
// (diagonal <-> operator <-> correction) consistency of `doc/porous_drag_scheme.md` §2, the same
// three-way face-mean argument `doc/variable_density_projection.md` §1/§3 makes for rho.
//
// But NO writer of "drag_beta" filled its ghost ring, and the only `fillPropGhosts(dragBeta_)` sat
// inside `project()` — i.e. AFTER the momentum stencil builds, which all run at (or after) the top
// of `step()`. `setField` and `applyClosure` write the inner cells only; the external CFD-DEM
// writer's driver folds its ghost-band deposit onto the owners and then ZEROES that band
// (single rank) or leaves the reverse-halo residue in it (MPI). So on the FIRST INNER PLANE of
// every block the momentum diagonal read beta_f = 0.5*beta — a factor-2 error in the drag — while
// the projection on that same face used the full, freshly exchanged value. That is precisely the
// mismatch `addDragDiagonal`'s own comment records: the incremental pressure loop gains
// (idt+beta_f)/(idt+beta_f^momentum) instead of 1 and "the accumulated pressure diverges
// exponentially". The fix (`fillDragBetaGhosts()`, called next to WO-G's `fillCellForceGhosts()`
// right after `updateProperties()`) routes the field through the same rank-aware `fillPropGhosts`
// the projection uses, so both phases see the identical beta_f.
//
// Every configuration is a UNIFORM beta in a fully periodic box with a
// uniform body force, inviscid and with advection off, so the momentum solve is diagonal and
//
//     u^{n+1} = (idt*u^n + F) / (idt + beta_f),        idt = rho/dt,
//
// i.e. the velocity IS a read-out of the momentum diagonal. A uniform u is divergence-free (and
// eps == 1 is uniform, so div(eps u) = 0 with d(eps)/dt = 0), so the porous projection is an exact
// no-op and the whole step is a pure diagonal recursion. With rho/dt = 1 and beta = 3 the diagonal
// is 4.0f (exact in float) and every iterate is a dyadic rational, so the healthy answer is
// bit-exact; the defective plane's diagonal is 2.5f.
//
// TWO gates, and the pair is the point:
//
//   1. THE DIAGONAL, where the defect lives. `getMomentumDiagonal(c)` reads the assembled float
//      stencil AC back, and every cell must equal 4.0 EXACTLY — including the first inner plane of
//      every block. This is the WO's literal requirement ("the momentum diagonal asserted uniform
//      on the first inner plane") and it localizes the offending plane. It is sampled after the
//      FIRST step, deliberately: in this pure-flow harness nothing re-dirties the ghost, so
//      `project()`'s (late) fill happens to leave a valid ring behind for step 2 onward and the
//      defect is a step-1-only effect here. In the real CFD-DEM loop the driver rewrites
//      "drag_beta" inner-only and FOLDS+ZEROES its ghost band before every `step()`, so every step
//      is a step 1 — which is why the velocity error below, seeded once, is the conservative
//      lower bound on what the coupled path pays.
//   2. THE VELOCITY, i.e. what the defect costs. Note it does NOT show up as a non-uniform u: the
//      periodic projection removes the non-uniform part of u* and what survives is the (wrong)
//      MEAN — the same mechanism WO-G measured for the halved body force. So the velocity gate that
//      fires is the analytic VALUE, not the spread: measured before the fix, u is uniform to
//      5.6e-17 but sits at 0.33210449 (per-z, 32 cells) / 0.33217773 (per-x, 16 cells) instead of
//      0.33203125 — a 2.2e-4 / 4.4e-4 relative error scaling as 1/N_axis, exactly one bad plane's
//      worth. Both are gated; the spread gate is kept because it is what a WALL-bounded or
//      non-uniform-beta variant would trip.
//
// The gates are ABSOLUTE physics, not a distributed-vs-reference comparison — deliberately, exactly
// as in `test_bodyforce_ghost_mpi.cpp`: the single-rank reference carries the identical defect, so
// `du = 0` proves nothing and np = 1 must (and does) fail on its own.
//
// Configurations (NX x NY x NZ = 16 x 16 x 32: np=2 cuts z, np=4 cuts x and z):
//
//   * `per-z`  — porous continuity, non-conservative momentum, beta written by an EXTERNAL writer
//                (`setField`, no exchange — the CFD-DEM path), force_z. Stencil site:
//                `rebuildStencils`. Requires z to be CUT at np > 1.
//   * `per-x`  — the same on the x axis (stride 1 instead of ex*ey). At np = 2 its "rank boundary"
//                is the PERIODIC WRAP plane in x, which is the single-rank half of the same defect
//                and an equally hard gate; at np = 4 x is genuinely cut. Runs at every np.
//   * `cons-z` — the eps-CONSERVATIVE porous momentum (`setPorousConservative(true)`): the diagonal
//                is eps_f*rho/dt + beta_f and the RHS goes through `buildRhsVar`, so this covers the
//                variable-density sibling of the stencil build. Requires z to be CUT.
//
// np = 1 is additionally compared bitwise against a full-grid single-rank reference; np > 1 lands on
// the usual MPI reduction-order floor (see `test_vardensity_mpi.cpp`) — though on these
// configurations the projection is a no-op, so in practice it comes out bitwise there too.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <vector>

#include "flow_ibm.hpp"
#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::flow::IbmSolver;

static constexpr int NX = 16, NY = 16, NZ = 32, STEPS = 4;
static constexpr double RHO0 = 1.0, DT = 1.0, FORCE = 1.0, BETA = 3.0;
static constexpr std::size_t GCELLS = (std::size_t)NX * NY * NZ;

struct Config {
  const char* name;
  int comp;     // forced velocity component (0=x, 1=y, 2=z) == the axis whose face beta is read
  bool cons;    // eps-conservative porous momentum (buildRhsVar path) vs the plain porous momentum
  bool needCut;  // the forced axis MUST be cut at np > 1 (loud coverage loss if it stops being)
};

static void configure(IbmSolver& s, const Config& c, int lnx, int lny, int lnz) {
  s.setRho(RHO0);
  s.setMu(0.0);  // inviscid: the momentum operator is the pure diagonal idt + beta_f
  s.setDt(DT);
  s.setAdvection(false);
  double f[3] = {0.0, 0.0, 0.0};
  f[c.comp] = FORCE;
  s.setBodyForce(f[0], f[1], f[2]);  // the UNIFORM scalar body force, not a force FIELD: this test
                                     // must not depend on WO-G's cell-force ghost path
  s.setPressureGeometry(std::vector<double>((std::size_t)lnx * lny * lnz, 10.0));
  s.setPorousContinuity(true);  // binds "eps" (seeded to 1 over the whole padded block)
  s.setPorousConservative(c.cons);
  s.enableDrag();  // allocates "drag_beta" + routes it onto the momentum diagonal
  // The external CFD-DEM writer path: inner cells only, ghosts left as registered (zero) — exactly
  // what the coupling driver's ghost fold leaves behind at single rank.
  s.setField("drag_beta", std::vector<double>((std::size_t)lnx * lny * lnz, BETA));
}

// The analytic diagonal recursion u^{n+1} = (idt*u^n + F)/(idt + beta), evaluated in double with the
// same float diagonal the solver builds.
static double analyticVelocity() {
  const double ac = (double)(float)(RHO0 / DT + BETA);
  double u = 0.0;
  for (int n = 0; n < STEPS; ++n)
    u = (RHO0 / DT * u + FORCE) / ac;
  return u;
}

// Gather per-rank inner blocks (x-fastest) into the global field on rank 0.
static std::vector<double> gatherGlobal(const std::vector<double>& local, int ox, int oy, int oz,
                                        int lnx, int lny, int lnz, int rank, int size) {
  std::vector<double> global;
  if (rank == 0)
    global.assign(GCELLS, 0.0);
  for (int r = 0; r < size; ++r) {
    int meta[6] = {ox, oy, oz, lnx, lny, lnz};
    if (r == 0) {
      if (rank == 0)
        for (int z = 0; z < lnz; ++z)
          for (int y = 0; y < lny; ++y)
            std::memcpy(&global[(std::size_t)ox + (std::size_t)(y + oy) * NX +
                                (std::size_t)(z + oz) * NX * NY],
                        &local[(std::size_t)y * lnx + (std::size_t)z * lnx * lny],
                        (std::size_t)lnx * sizeof(double));
      continue;
    }
    if (rank == r) {
      MPI_Send(meta, 6, MPI_INT, 0, 100 + r, MPI_COMM_WORLD);
      MPI_Send(local.data(), (int)local.size(), MPI_DOUBLE, 0, 200 + r, MPI_COMM_WORLD);
    } else if (rank == 0) {
      MPI_Recv(meta, 6, MPI_INT, r, 100 + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<double> buf((std::size_t)meta[3] * meta[4] * meta[5]);
      MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, r, 200 + r, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      for (int z = 0; z < meta[5]; ++z)
        for (int y = 0; y < meta[4]; ++y)
          std::memcpy(&global[(std::size_t)meta[0] + (std::size_t)(y + meta[1]) * NX +
                              (std::size_t)(z + meta[2]) * NX * NY],
                      &buf[(std::size_t)y * meta[3] + (std::size_t)z * meta[3] * meta[4]],
                      (std::size_t)meta[3] * sizeof(double));
    }
  }
  return global;
}

static double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  // WO-R2: NaN-PROPAGATING. `std::fmax(m, NaN) == m`, so the obvious loop returns 0.000e+00 for a
  // field that has gone entirely NaN and every bitwise gate built on it passes (WO-R found this on
  // a drained open-boundary run). A non-finite difference must fail, so return it.
  double m = 0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = std::fabs(a[i] - b[i]);
    if (!(d == d))
      return d;  // NaN
    m = std::fmax(m, d);
  }
  return m;
}
static double maxAbs(const std::vector<double>& a) {
  double m = 0;
  for (double v : a)
    m = std::fmax(m, std::fabs(v));
  return m;
}

// Index of the plane (along `axis`) whose velocity deviates most from the global mean — with the
// defect this is the first inner plane of a rank block (or the periodic wrap plane), never an
// interior one, so the report names the offender.
static int worstPlane(const std::vector<double>& u, int axis) {
  const int n[3] = {NX, NY, NZ};
  double mean = 0;
  for (double v : u)
    mean += v;
  mean /= (double)u.size();
  int worst = -1;
  double worstDev = -1;
  for (int p = 0; p < n[axis]; ++p) {
    double dev = 0;
    for (int b = 0; b < n[(axis + 2) % 3]; ++b)
      for (int a = 0; a < n[(axis + 1) % 3]; ++a) {
        int idx[3];
        idx[axis] = p;
        idx[(axis + 1) % 3] = a;
        idx[(axis + 2) % 3] = b;
        dev = std::fmax(dev, std::fabs(u[(std::size_t)idx[0] + (std::size_t)idx[1] * NX +
                                         (std::size_t)idx[2] * NX * NY] -
                                       mean));
      }
    if (dev > worstDev) {
      worstDev = dev;
      worst = p;
    }
  }
  return worst;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto dec = peclet::flow::CutcellMG::decomposition(static_cast<std::size_t>(size), NX, NY, NZ);
    auto blk = dec.block(rank);
    const int ox = (int)blk.origin[0], oy = (int)blk.origin[1], oz = (int)blk.origin[2];
    const int lnx = (int)blk.size[0], lny = (int)blk.size[1], lnz = (int)blk.size[2];
    const int gn[3] = {NX, NY, NZ};
    bool cut[3] = {false, false, false};
    for (const auto& sz : dec.sizes())
      for (int a = 0; a < 3; ++a)
        if ((int)sz[a] != gn[a])
          cut[a] = true;
    if (rank == 0)
      std::printf("DRAGBETA-GHOST MPI np=%d  grid %dx%dx%d  block %dx%dx%d  cut axes: %s%s%s\n",
                  size, NX, NY, NZ, lnx, lny, lnz, cut[0] ? "x" : "", cut[1] ? "y" : "",
                  cut[2] ? "z" : "");

    const Config configs[] = {{"per-z", 2, false, true},
                              {"per-x", 0, false, false},
                              {"cons-z", 2, true, true}};
    const double want = analyticVelocity();

    for (const Config& c : configs) {
      if (c.needCut && size > 1 && !cut[c.comp]) {
        if (rank == 0)
          std::printf("  [%-6s np=%d] FAIL — the decomposition does NOT cut the forced axis %d; "
                      "this test exists to gate the drag_beta ghost ACROSS a rank boundary (WO-I)\n",
                      c.name, size, c.comp);
        fail = 1;
        continue;
      }
      IbmSolver sd(lnx, lny, lnz);
      sd.initMpi(dec, MPI_COMM_WORLD);
      configure(sd, c, lnx, lny, lnz);
      std::vector<double> diagLocal;
      for (int it = 0; it < STEPS; ++it) {
        sd.step();
        // The diagonal is sampled after the FIRST step — see the "one step" note in the header:
        // in this pure-flow harness `project()`'s late fill repairs the ghost from step 2 onward,
        // whereas the real CFD-DEM writer re-zeroes it before every step.
        if (it == 0)
          diagLocal = sd.getMomentumDiagonal(c.comp);
      }

      std::vector<double> gu[3];
      for (int comp = 0; comp < 3; ++comp)
        gu[comp] = gatherGlobal(sd.getVelocity(comp), ox, oy, oz, lnx, lny, lnz, rank, size);
      const std::vector<double> gp =
          gatherGlobal(sd.getPressure(), ox, oy, oz, lnx, lny, lnz, rank, size);
      // THE primary gate: the assembled momentum diagonal itself, read where the face drag is
      // formed. (The velocity gate below sees the same defect only through the projection, which on
      // a periodic box turns one bad plane into a uniform mean shift.)
      const std::vector<double> gd =
          gatherGlobal(diagLocal, ox, oy, oz, lnx, lny, lnz, rank, size);

      if (rank == 0) {
        IbmSolver ref(NX, NY, NZ);
        configure(ref, c, NX, NY, NZ);
        for (int it = 0; it < STEPS; ++it)
          ref.step();
        double du = 0, umag = 0;
        for (int comp = 0; comp < 3; ++comp) {
          du = std::fmax(du, maxAbsDiff(gu[comp], ref.getVelocity(comp)));
          umag = std::fmax(umag, maxAbs(ref.getVelocity(comp)));
        }
        const double dp = maxAbsDiff(gp, ref.getPressure());
        const double pmag = maxAbs(ref.getPressure());
        const double utol = (size == 1) ? 0.0 : std::fmax(1e-15, 1e-11 * umag);
        const double ptol = (size == 1) ? 0.0 : std::fmax(1e-12, 1e-11 * pmag);
        bool ok = du <= utol && dp <= ptol;

        // THE gate: the FACE drag on the momentum diagonal is uniform <=> the velocity is uniform
        // and equal to the analytic diagonal recursion.
        const std::vector<double>& uf = gu[c.comp];
        double lo = uf[0], hi = uf[0];
        for (double v : uf) {
          lo = std::fmin(lo, v);
          hi = std::fmax(hi, v);
        }
        const double spread = hi - lo, verr = std::fabs(0.5 * (hi + lo) - want) / want;
        // The other two components must be identically zero (no force, no coupling), and the
        // accumulated pressure must stay at zero (the projection sees a divergence-free field).
        double cross = 0;
        for (int comp = 0; comp < 3; ++comp)
          if (comp != c.comp)
            cross = std::fmax(cross, maxAbs(gu[comp]));
        const double pabs = maxAbs(gp);
        // The momentum diagonal must be EXACTLY idt + beta on every cell — including the first
        // inner plane of every block, which is the one that read the ghost.
        double dlo = gd[0], dhi = gd[0];
        for (double v : gd) {
          dlo = std::fmin(dlo, v);
          dhi = std::fmax(dhi, v);
        }
        const double wantDiag = (double)(float)(RHO0 / DT + BETA);
        const bool diagOk = (dlo == wantDiag) && (dhi == wantDiag);
        ok = ok && diagOk && spread <= 1e-13 * want && verr <= 1e-13 && cross <= 1e-13 * want &&
             pabs <= 1e-11;
        std::printf("  [%-6s np=%d] du=%.3e dp=%.3e (tol %.1e/%.1e) | diag [%.17g, %.17g] "
                    "(want %.17g, worst plane %d) | u spread=%.3e (want 0) value=%.17g "
                    "(want %.17g) cross=%.2e max|P|=%.2e  %s\n",
                    c.name, size, du, dp, utol, ptol, dlo, dhi, wantDiag, worstPlane(gd, c.comp),
                    spread, 0.5 * (hi + lo), want, cross, pabs, ok ? "OK" : "FAIL");
        if (!ok)
          fail = 1;
      }
      MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
      std::printf("DRAGBETA-GHOST MPI (np=%d): %s\n", size, fail ? "FAIL" : "PASS");
  }
  Kokkos::finalize();
  MPI_Finalize();
  return fail;
}
