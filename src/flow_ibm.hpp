/// @file
/// @brief flow — host-facing Kokkos IBM Navier-Stokes solver (drop-in flow-style API).
///
/// Assembles the validated cut-cell IBM operators into a runnable solver on a fully-periodic MAC
/// box with immersed SDF solids: per-component backward-Euler implicit diffusion with the
/// Robust-Scaled cut-cell no-slip stencil (buildIbmOverlay + ibmBuildDiffusion + ibmModifyStencil +
/// ibmSolidMask + ibmRbgsSweep), then a rotational incremental-pressure Chorin projection through
/// the open-face-weighted cut-cell pressure Poisson (buildCutcellOp + divergOpen, solved by CG with
/// the constant null space projected out, then projectCorrect; P += (rho/dt)*phi - mu*div(u*)
/// matching CUDA press_update_k). Schemes are a FAITHFUL port of the CUDA flow (point-value
/// cut-cell IBM = ibm_geometry_ext_k<0>; rotational pressure): the velocity field matches CUDA to
/// ~1e-13 (machine precision). Physical units (rho/mu/dt + body force). std::vector setters/getters
/// so a pybind module can drive it. The verify_poiseuille / verify_periodic_spheres mechanism (k
/// matches CUDA to all printed digits), on any backend. NOTE (faithfulness items, see memory): the
/// CG uses a diagonal preconditioner where CUDA uses RB-GS-preconditioned MG-PCG (same converged
/// solution); the pressure operator is stored double where CUDA uses float mreal -- to reconcile in
/// a later port pass.
#ifndef PECLET_FLOW_SDFLOW_IBM_HPP
#define PECLET_FLOW_SDFLOW_IBM_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "peclet/core/geom/device_scene.hpp"

#include "collocated_varrho.hpp"  // rung V8 (WO-T): collocated varRho + face-acceleration kernels
#include "face_props.hpp"
#include "gauge_exact_gradient.hpp"
#include "ghost_projection_debug.hpp"  // opt-in gp row forensics (PECLET_FLOW_GP_DEBUG), no-op off
#include "grid_layout.hpp"
#include "mac_approx_projection.hpp"
#include "mac_cutcell_mg.hpp"
#include "mac_ibm.hpp"
#include "mac_pressure.hpp"
#include "mac_stencils.hpp"
#include "mac_velocity_mg.hpp"
#include "peclet/core/field/field_set.hpp"
#include "property_closures.hpp"
#include "scalar_transport.hpp"
#include "staggered_advection.hpp"
#include "vof/advect_wy.hpp"    // VoF rung V1: the Weymouth-Yue colour advector (its own g=3 block)
#include "vof/colour_field.hpp"  // VoF rung V2a: the G=2 <-> g=3 bridge + the colour ghost policy
#include "vof/colour_bc.hpp"  // VoF rung V-BC: inflow/outflow/backflow colour + the outside mask
#include "vof/curvature_field.hpp"  // VoF rung V3: the HF curvature cascade + PV fallback
#include "vof/momentum_advect.hpp"  // VoF rung V2b: momentum-consistent rho^c u_c transport
#include "vof/surface_tension.hpp"  // VoF rung V4: balanced-force CSF + the capillary dt
#include "vof/phase_change.hpp"    // VoF Part II rungs P0/P1: mass flux, plane-shift regression

namespace peclet::flow {

// Templated on a GridLayout policy (grid_layout.hpp) that supplies the grid-position-dependent
// pieces (currently: the per-component velocity sample offset). IbmSolver == Solver<Staggered> (the
// alias below) is bit-identical to the pre-policy solver; the Colocated policy is added in a later
// phase.
template <class Grid>
class Solver {
 public:
  using FV = Kokkos::View<MReal*, CCMem>;  // velocity operator storage tracks MReal
  static constexpr int G = 2;  // velocity block: Koren advection reach (pressure/MG bridged to g=1)

  Solver(int nx, int ny, int nz) { allocateBlock(nx, ny, nz); }

  // (Re)allocate every per-block buffer for a local inner block of nx*ny*nz. Called by the
  // constructor and by redistribute() after a re-decomposition changes this rank's block size.
  void allocateBlock(int nx, int ny, int nz) {
    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    e_ = C3{nx + 2 * G, ny + 2 * G, nz + 2 * G};
    n_ = (std::size_t)e_.x * e_.y * e_.z;
    e1_ = C3{nx + 2, ny + 2, nz + 2};  // g=1 block for the cut-cell pressure MG
    n1_ = (std::size_t)e1_.x * e1_.y * e1_.z;
    sdf_ = CCField("sdf", n_);
    ox_ = CCField("ox", n_);
    oy_ = CCField("oy", n_);
    oz_ = CCField("oz", n_);
    phi_ = CCField("phi", n_);
    div_ = CCField("div", n_);
    P_ = CCField("P", n_);
    // g=1 scratch for the MG bridge (openness + rhs/phi + PCG vectors)
    ox1_ = CCField("ox1", n1_);
    oy1_ = CCField("oy1", n1_);
    oz1_ = CCField("oz1", n1_);
    rhs1_ = CCField("rhs1", n1_);
    phi1_ = CCField("phi1", n1_);
    r_ = CCField("r", n1_);
    z_ = CCField("z", n1_);
    pp_ = CCField("pp", n1_);
    Ap_ = CCField("Ap", n1_);
    for (int c = 0; c < 3; ++c) {
      C[c].u = CCField("u", n_);
      C[c].b = CCField("b", n_);
      C[c].AC = FV("AC", n_);
      C[c].AW = FV("AW", n_);
      C[c].AE = FV("AE", n_);
      C[c].AS = FV("AS", n_);
      C[c].AN = FV("AN", n_);
      C[c].AB = FV("AB", n_);
      C[c].AT = FV("AT", n_);
      C[c].inhom = CCField("inhom", n_);
      C[c].rscale = CCField("rscale", n_);
      C[c].mask = CCField("mask", n_);
      bcDcorr_[c] = CCField("dcorr", n_);
      bcBrhs_[c] = CCField("brhs", n_);
      const int maxCut = nx * ny * nz;
      using FV32 = Kokkos::View<float*, CCMem>;  // IbmOverlay row data stays float
      C[c].ov = IbmOverlay{Kokkos::View<int*, CCMem>("ci", maxCut),
                           Kokkos::View<int*, CCMem>("nb", maxCut),
                           FV32("dr", maxCut),
                           Kokkos::View<int*, CCMem>("dc", (std::size_t)maxCut * 6),
                           FV32("K", (std::size_t)maxCut * 6),
                           FV32("M", (std::size_t)maxCut * 6),
                           FV32("X", (std::size_t)maxCut * 6),
                           FV32("Nbc", (std::size_t)maxCut * 6),
                           FV32("R", (std::size_t)maxCut * 6)};
      C[c].idMap = Kokkos::View<int*, CCMem>("idMap", n_);
      C[c].counter = Kokkos::View<int, CCMem>("cnt");
      old_[c] = CCField("uOld", n_);    // u^n time base (fixed over the step's Picard sweeps)
      prev_[c] = CCField("uPrev", n_);  // previous Picard iterate (outer-tolerance check)
    }
    if constexpr (Grid::collocated) {  // transient face (MAC) field for the approximate projection
      uf_ = CCField("uf", n_);
      vf_ = CCField("vf", n_);
      wf_ = CCField("wf", n_);
      tgp_ = CCField("tgp", n_);    // scratch: wall-aware transpose gradient (setFaceInterp(2/3))
      wdef_ = CCField("wdef", n_);  // scratch: FV wall viscous-flux defect (setFaceInterp(4))
      fvM_ = CCField("fvM", n_);    // scratch: M·u^k (mode-4 defect matvec)
      fvL_ = CCField("fvL", n_);    // scratch: L_FV(u^k) (mode-4 FV operator apply)
      cs_ = CCField("cs", n_);      // static cell fluid fraction (setFaceInterp(4))
      xcx_ = CCField("xcx", n_);  // static per-face open-centroid wall distance (setFaceInterp(3))
      xcy_ = CCField("xcy", n_);
      xcz_ = CCField("xcz", n_);
    }
    // Register the pre-existing solver fields in the named directory so the multiphysics machinery
    // (scalar transport, property closures) and load-balance redistribution can enumerate the whole
    // set uniformly. adopt() aliases the members (no reallocation, no ownership); all live on the
    // G=2 velocity block and share velHalo_ under MPI.
    fields_.adopt("u", C[0].u, G, peclet::core::Centering::FaceX);
    fields_.adopt("v", C[1].u, G, peclet::core::Centering::FaceY);
    fields_.adopt("w", C[2].u, G, peclet::core::Centering::FaceZ);
    fields_.adopt("p", P_, G, peclet::core::Centering::Cell);
    fields_.adopt("sdf", sdf_, G, peclet::core::Centering::Cell);
  }

  void setRho(double r) { rho_ = r; }
  void setMu(double m) { mu_ = m; }
  void setDt(double d) {
    if (d != dt_) {
      dt_ = d;
      dtDirty_ = true;  // the momentum stencil bakes rho/dt in its diagonal (rebuildStencils);
                        // a mid-run dt change must rebuild it or the operator and RHS disagree
    }
  }
  void setBodyForce(double fx, double fy, double fz) { f_ = {fx, fy, fz}; }
  void setVelocityIterations(int it) { velIters_ = it; }
  // Momentum tolerance stop: end the RB-GS loop once the swept colour's max increment has dropped
  // to rtol of the first sweep's (GS contracts geometrically, so the increment tracks the error).
  // rtol = 0 (default) keeps the legacy fixed-count loop byte-identical. Easy regimes (small
  // nu*dt/dx^2) exit after ~3-5 sweeps; stiff regimes run to the velIters_ cap unchanged. The
  // check is fused into the sweep kernel (no extra memory pass) and the stop decision is
  // rank-uniform (MPI max) so distributed halo exchanges stay in lockstep.
  void setVelocityTolerance(double rtol, int minIters) {
    velTol_ = rtol > 0.0 ? rtol : 0.0;
    velMinIters_ = minIters < 1 ? 1 : minIters;
  }
  long lastMomentumSweeps() const { return lastMomentumSweeps_; }
  // Residual-based momentum stop (opt-in, 0 = off): a component's implicit solve ends once
  // max|b - A u| <= rtol * max|b| over the fluid unknowns (global under MPI). Unlike the update
  // criterion (relative to the FIRST sweep's update, which on a warm-started near-steady step is
  // already at noise level and then costs hundreds of sweeps to shrink by 1e-3) this measures the
  // equation's own convergence, and a converged warm start stops at sweep 1. Available on the
  // stencil paths (IBM / cut-cell domain-BC RB-GS, every velocity-MG mode); the const-coefficient
  // domain-BC smoother keeps the update criterion.
  // rtol > 0: fixed; rtol == 0: the legacy update criterion; rtol < 0 (DEFAULT): FOLLOW THE
  // PRESSURE SOLVER'S TOLERANCE -- the projection is what consumes u*, and it resolves the
  // divergence the momentum residual leaves to its own rtol, so "solve momentum no less
  // accurately than pressure" is the self-consistent choice with no free constant.
  void setVelocityResidualTolerance(double rtol) { velResTol_ = rtol; }
  // Velocity-MG AUTO rule (applies when set_velocity_multigrid was never called): under MPI, once
  // the block is small enough that the momentum RB-GS is halo-latency-bound, take the V-cycle
  // instead (1-2 cycles/component == 2-4 exchanges against 8-9 sweeps x 2). Measured crossover on
  // the FoxBerry bed: RB-GS 2.91 s vs MG 3.32 s/step at 147k cells/rank, MG 0.834 vs 0.844 at
  // 37k; threshold PECLET_FLOW_VMG_AUTO_CELLS (default 65536 cells per rank, 0 = never), and only
  // for global problems of at least PECLET_FLOW_VMG_AUTO_MIN_GLOBAL cells (8M) -- small grids split
  // across ranks keep RB-GS so a distributed run stays exactly the single-rank one.
  void setVelocityMultigridAuto(long cellsPerRank, long minGlobalCells = -1) {
    vmgAutoCells_ = cellsPerRank;
    if (minGlobalCells >= 0)
      vmgAutoMinGlobal_ = minGlobalCells;
  }
  // The tolerance actually in force (resolves the follow-the-pressure default).
  double velocityResidualTolerance() const {
    if (velResTol_ >= 0.0)
      return velResTol_;
    return useChebyshev_ ? chebRtol_ : pcgRtol_;  // FCG shares pcgRtol_; the plain V-cycle driver
                                                  // has no tolerance and takes the PCG default
  }
  // max over components of max|r|/max|b| at exit of the last step's momentum solves (residual
  // mode only; -1 otherwise).
  double lastMomentumResidual() const { return lastMomentumResid_; }
  // Pressure-solve mean-removal scope: "fine" (default — drops the interior-level / post-matvec
  // nullspace projections, ~3x fewer global-reduction latency hits per Krylov iteration; measured
  // winner of the at-scale ablation, iteration counts identical) or "all" (legacy). See CutcellMG.
  void setPressureMeanRemoval(bool all) { mg_.setMeanRemovalScope(all); }
  void setPressureIterations(int it) { presIters_ = it; }
  void setAdvection(bool on) { advect_ = on; }  // explicit high-order advection (default SOU)
  // High-order advection scheme for the (explicit, or deferred-correction) flux: 0 = second-order
  // upwind (SOU, default — 2nd order at smooth extrema too); 1 = Koren TVD (monotone limiter, the
  // legacy CUDA scheme). Only matters when advection is enabled; FOU stays the deferred-correction
  // base.
  void setAdvectionScheme(int s) { advScheme_ = s; }
  // Implicit-FOU deferred-correction advection (CUDA set_implicit_advection): solve the
  // first-order-upwind part of advection implicitly (in the velocity operator) + keep (Koren-FOU)
  // explicit in the RHS -> unconditionally stable for advection (high Re / large dt). Requires the
  // IBM stencil (rebuilt per Picard iteration with the FOU term); the domain-BC path needs
  // velocity-MG (separate milestone).
  void setImplicitAdvection(bool on) { implicitFou_ = on; }
  // Picard outer iterations over the step (CUDA set_outer_iterations): the advecting velocity is
  // lagged at the current iterate u^k while the time base stays u^n. iters>=1; tol>0 stops early on
  // max|du| < tol.
  void setOuterIterations(int iters) { outerIters_ = iters < 1 ? 1 : iters; }
  void setOuterTolerance(double tol) { outerTol_ = tol; }
  long lastOuterIterations() const { return lastOuterIters_; }
  // Velocity (momentum) multigrid for the IBM diffusion solve (CUDA set_velocity_multigrid): the
  // STAIRCASE coarse operator (exact == RB-GS, stiff-stable at large dt). Call before set_solid;
  // built at geometry time.
  void setVelocityMultigrid(bool on, int levels, int vcycles) {
    useVelocityMg_ = on;
    vmgExplicit_ = true;  // an explicit choice disables the AUTO rule
    vmgLevels_ = levels < 1 ? 1 : levels;
    vmgVcycles_ = vcycles < 1 ? 1 : vcycles;
  }
  bool velocityMultigridActive() const { return useVelocityMg_; }
  // Enable the agglomerated GraphAMG bottom solve in the pressure MG: the coarsest level is solved
  // by a mesh-agnostic algebraic multigrid on the operator gathered to rank 0 --
  // decomposition-agnostic, so multilevel convergence works under a WEIGHTED ORB (where the
  // geometric coarse levels can't cleanly coarsen). Applied at the next set_solid / geometry
  // rebuild.
  // Coarse-level (bottom) solve policy: 0 smoothed bottom (default), -1 auto (agglomerate exactly
  // when the geometric hierarchy cannot reach a small enough coarsest grid), 1 always. See CutcellMG.
  void setPressureBottomMode(int mode) {
    pressAgglomMode_ = mode;
    if (cutcellPressure_)
      mg_.setAgglomerationMode(mode);
  }  // Coarse-level telescoping of the pressure multigrid (mac_cutcell_mg.hpp Telescope): when a
  // per-rank block turns odd, merge ORB siblings onto fewer ranks and keep coarsening instead of
  // stopping. Multi-rank only; a no-op single-rank. Takes effect at the next init_mpi/set_solid.
  void setPressureTelescope(bool on) { mg_.setTelescope(on); }
  bool pressureTelescope() const { return mg_.telescope(); }

  void setPressureGraphAmg(bool on) {
    pressGraphAmg_ = on;
    if (cutcellPressure_)
      mg_.setGraphAmgBottom(on);  // propagate live (previously only applied at the next set_solid,
                                  // so toggling after geometry silently had no effect)
  }
  void setPressureLevels(int levels) {
    nLevels_ = levels < 1 ? 1 : levels;
  }  // MG depth (CUDA default 4)
  // Backflow stabilization at outflow faces (Bazilevs 2009 / Esmaily-Moghadam 2011): beta in [0,1]
  // scales the dissipative outflow term that prevents backflow divergence (0 = off). Default 0.2.
  void setBackflowStab(double beta) { backflowBeta_ = beta < 0.0 ? 0.0 : beta; }
  // Deferred-correction advection: on (default) = implicit FOU operator + explicit (HO - FOU)
  // high-order correction (2nd order; HO = SOU by default, or Koren TVD via set_advection_scheme).
  // off = pure implicit FOU (1st order, more dissipative, unconditionally stable) -- useful for
  // very sharp shear layers where the (unlimited SOU) explicit correction overshoots and
  // destabilizes.
  void setDeferredCorrection(bool on) { deferredCorr_ = on; }
  // Chebyshev pressure driver (CUDA set_pressure_chebyshev): communication-light alternative to
  // MG-PCG -- Chebyshev semi-iteration preconditioned by one symmetric V-cycle, no per-iteration
  // global dot-products. Spectral bounds of M^{-1}A are estimated once (lazily) on the first solve
  // and reused every step.
  // Selecting it clears the competing FCG selection (the three Krylov drivers are mutually
  // exclusive, last set wins); `on = false` only deselects Chebyshev, so the solve falls back to
  // whatever else is selected — FCG if set, otherwise MG-PCG.
  void setPressureChebyshev(bool on, int maxit, double rtol) {
    useChebyshev_ = on;
    if (on)
      useFcg_ = false;
    chebMaxit_ = maxit;
    chebRtol_ = rtol;
    chebBoundsSet_ = false;
  }
  // MG-PCG pressure driver (CUDA set_pressure_pcg) + its iteration cap / relative tolerance.
  // `on = true` GENUINELY SELECTS MG-PCG, clearing both competing selections (Chebyshev and FCG),
  // so it works after set_density_mode / set_porous — "last set wins", as CLAUDE.md and the
  // docstring have always claimed. Until 2026-08-30 the flag was silently discarded (WO-H defect 1;
  // the working spelling was set_pressure_chebyshev(False, ...)), which is why every "PCG under
  // varRho/porous" measurement before WO-B actually measured Chebyshev.
  //
  // `on = false` cannot be honoured and therefore THROWS rather than being silently ignored (the
  // failure mode this repair exists to remove): MG-PCG is the terminal fallback of the dispatch in
  // project() — with neither Chebyshev nor FCG selected the solve IS MG-PCG — so "not PCG" is only
  // expressible by selecting another driver. Say which one: set_pressure_chebyshev(True, ...) or
  // set_pressure_fcg(True, ...).
  //
  // Under set_ghost_projection the operator is nonsymmetric and is solved by BiCGStab; this call
  // stays legal there (it is how that path's cap/tolerance is set — pcgMaxit_/pcgRtol_ are shared)
  // and simply does not change which Krylov method the gp branch runs.
  void setPressurePcg(bool on, int maxit, double rtol) {
    if (!on)
      throw std::runtime_error(
          "set_pressure_pcg(False): MG-PCG is the default/terminal pressure driver, so it cannot be "
          "deselected on its own — select the driver you want instead "
          "(set_pressure_chebyshev(True, ...) or set_pressure_fcg(True, ...)).");
    useChebyshev_ = false;  // genuine selection: the three drivers are mutually exclusive
    useFcg_ = false;
    chebBoundsSet_ = false;
    pcgMaxit_ = maxit;
    pcgRtol_ = rtol;
  }
  // FLEXIBLE MG-CG (set_pressure_fcg): the same Krylov driver as MG-PCG with the same V-cycle
  // preconditioner, the same stopping estimate, the same mean removal and the same cap/tolerance
  // (`pcgMaxit_`/`pcgRtol_`, shared deliberately — it is the same solve), differing ONLY in the
  // beta recurrence: Polak-Ribiere `r^T(z_{k+1} - z_k) / r^T z_k` instead of Fletcher-Reeves.
  // Costs one extra level-0 vector and one extra global dot per iteration; buys tolerance of a
  // preconditioner that is not symmetric w.r.t. the fine operator (CutcellMG::solveFCG).
  //
  // All three Krylov drivers are mutually exclusive in both directions and the last set wins:
  // `on` clears `useChebyshev_` here, `setPressureChebyshev(true, ...)` clears `useFcg_`, and
  // `setPressurePcg(true, ...)` clears both. `set_pressure_fcg(false)` returns the solve to MG-PCG.
  // (Before WO-H, setPressurePcg's `on` flag was silently discarded — see its comment.)
  void setPressureFcg(bool on, int maxit, double rtol) {
    if (on && ghostProjection_)
      throw std::runtime_error(
          "set_pressure_fcg: the ghost-projection operator is nonsymmetric and is solved by "
          "BiCGStab; FCG does not apply");
    useFcg_ = on;
    if (on)
      useChebyshev_ = false;  // genuine selection (the pair is exclusive)
    pcgMaxit_ = maxit;
    pcgRtol_ = rtol;
  }
  // EXPERIMENTAL directional ghost-cell projection (second staggered IBM, ghost_projection.hpp):
  // point-based FD divergence with wall-anchored directional closures instead of the
  // openness-weighted cut-cell projection. Call BEFORE set_solid (the overlay is built there).
  // v1: periodic + IBM only, stationary walls (both grids; the collocated variant closes the
  // face-AVERAGED field and adds the gpCenterGrad predictor/correction, face_interp 0 only).
  // Runs multi-rank (initMpi): gp-row ownership is by inner-block cell, the closures read the
  // exchanged g=2 halo, and the fragmentation guard runs on the allgathered GLOBAL sdf (the
  // exact-crossings / openness-override study inputs stay single-rank). The nonsymmetric extended
  // stencil is solved by MG-preconditioned BiCGStab (binary-openness surrogate hierarchy).
  // matrixOrder/rhsOrder select the closure order (1 = linear, 2 = wall-anchored quadratic) for
  // the implicit phi couplings and the divergence RHS/diagnostic respectively:
  //   (2,2) full quadratic (13-point nonsymmetric matrix);
  //   (1,1) linear everywhere (7-point matrix, 1st-order closure);
  //   (1,2) MIXED/deferred: 2nd-order steady constraint with a 7-point near-symmetric matrix —
  //         the operator mismatch converges through the time stepping (measured rate ~0.4).
  void setGhostProjection(bool on, int matrixOrder = 2, int rhsOrder = 2) {
    if constexpr (Grid::collocated) {
      // Collocated ghost mode: the SAME phi matrix/closures on the 1/2-1/2 face-averaged field
      // (the face correction uf -= grad(phi) is the identical substitution), plus the directional
      // gpCenterGrad cell gradient for the predictor -grad(P^n) and the cell correction. Only the
      // plain (mode-0) face map applies — the wall-aware/FV/embed face-interp modes replace the
      // very operators this scheme owns.
      if (on && faceInterp_ != 0)
        faceInterp_ = 0;  // QUARANTINED verification path: it owns the operators mode 9 replaces,
                          // so it selects the plain face map itself rather than throwing on the
                          // (now default) gauge-exact scheme.
    }
    if (on && (porous_ || varRho_ || hasBc_ || useChebyshev_))
      throw std::runtime_error(
          "set_ghost_projection: incompatible with porous/variable-rho/domain-BC/Chebyshev (v1)");
    if (matrixOrder < 1 || matrixOrder > 2 || rhsOrder < 1 || rhsOrder > 2)
      throw std::runtime_error("set_ghost_projection: matrix_order/rhs_order must be 1 or 2");
    if (on && distributed_ && (hasExactCross_ || hasOpenOverride_))
      throw std::runtime_error(
          "set_ghost_projection: exact-crossings/openness-override are single-rank only");
    ghostProjection_ = on;
    colSchemeAuto_ = false;  // explicit selection disables the AUTO default
    gpMatrixOrder_ = matrixOrder;
    gpRhsOrder_ = rhsOrder;
    gpNRows_ = -1;  // takes effect at the next set_solid
  }
  // Analytic-SDF capability: EXACT wall-crossing fractions overriding the linear-interp theta in
  // BOTH the momentum cut-cell overlay and the ghost-projection closures. t is a flat array of
  // size 9*nx*ny*nz, blocks ordered [(c*3 + k)]: for velocity component c, t[(c*3+k)*n + i] is
  // the exact crossing fraction in (0,1) from component c's staggered point at inner cell i
  // toward its +k-axis neighbour point, NaN where the segment has no wall crossing. Computed in
  // Python from the analytic geometry (e.g. line-sphere intersection). Call BEFORE set_solid;
  // pass an empty array to clear. Single-rank only.
  void setExactCrossings(const std::vector<double>& t) {
    const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
    if (t.empty()) {
      hasExactCross_ = false;
      return;
    }
    if (t.size() != 9 * n)
      throw std::runtime_error("set_exact_crossings: expected 9*nx*ny*nz values");
#ifdef PECLET_FLOW_MPI
    if (distributed_)
      throw std::runtime_error("set_exact_crossings: single-rank only");
#endif
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k) {
        tEx_[c][k] = CCField("tEx", n);
        Kokkos::deep_copy(
            tEx_[c][k],
            Kokkos::View<const double*, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                t.data() + ((std::size_t)c * 3 + k) * n, n));
      }
    hasExactCross_ = true;
  }
  // Analytic-SDF capability: EXACT face-openness (aperture) fields overriding the sampled-SDF
  // ccFractionCore openness the cut-cell projection uses. Inner arrays (flat x-fastest,
  // nx*ny*nz); ox[i] = fluid area fraction of the -x face of cell i, etc. Call BEFORE set_solid.
  void setOpennessOverride(const std::vector<double>& ox, const std::vector<double>& oy,
                           const std::vector<double>& oz) {
    const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
    if (ox.empty()) {
      hasOpenOverride_ = false;
      return;
    }
    if (ox.size() != n || oy.size() != n || oz.size() != n)
      throw std::runtime_error("set_openness_override: expected nx*ny*nz values per field");
#ifdef PECLET_FLOW_MPI
    if (distributed_)
      throw std::runtime_error("set_openness_override: single-rank only");
#endif
    oxOverride_ = ox;
    oyOverride_ = oy;
    ozOverride_ = oz;
    hasOpenOverride_ = true;
  }
  // Incremental-rotational pressure (CUDA set_incremental_pressure, default ON): the predictor
  // carries -grad(P^n) and the physical pressure is accumulated rotationally P += (rho/dt)*phi -
  // mu*div(u*). OFF => classical non-incremental Chorin (no -grad(P^n) predictor; P derived on
  // demand as (rho/dt)*phi).
  void setIncrementalPressure(bool on) { incremental_ = on; }
  // Pressure warm-start (CUDA set_pressure_warmstart, default OFF): seed each cut-cell pressure
  // solve from the previous step's projection potential (consecutive phi's are similar along a
  // steady march -> a more converged phi per fixed solver budget) instead of zeroing the initial
  // guess.
  void setPressureWarmstart(bool on) { pwarm_ = on; }
  // Collocated cut-cell treatment of the approximate projection (no effect on the staggered path):
  //   0 = plain ½/½ cell->face averaging + central-difference -grad(P) (default; a consistent
  //       adjoint pair of the WRONG geometry — wall at the solid neighbour's center — first-order
  //       drag at curved walls);
  //   1 = wall-aware cell->face map only (ablation: breaks the adjoint pairing — WORSE, don't use);
  //   2 = wall-aware map + its TRANSPOSE as the predictor -grad(P) and the cell correction
  //       (consistent pair, but face-CENTER point values under-count the open-area flux —
  //       ablation);
  //   3 = mode 2 evaluated at the OPEN-FACE-CENTROID wall distance (static geometry from
  //       buildFaceCentroidDist) — the flux-consistent constraint quadrature (stable, but the
  //       momentum row is still the O(h) axis-by-axis IBM: FV constraint vs FD momentum are
  //       inconsistent);
  //   4 = FULLY-FV: mode-3 projection PLUS the second-order wall viscous-flux deferred correction
  //   on
  //       the momentum (fvViscousApply: μ Σ_a W_a·centroid wall drag via defect correction, W_a
  //       from the divergence-theorem fragment normal o_{a−}−o_{a+}, centroid gradient at the SDF
  //       foot point). Momentum and constraint now share the same finite-volume cut-cell geometry →
  //       targets O(h²).
  //   5 = EMBED (Basilisk embed.h): like mode 4 but the momentum wall drag is the TRUE-NORMAL
  //       image-point gradient embedDirichletGradient (μ·area·d(U)/dn along n̂, O(h²) a-priori)
  //       rather than the axis-by-axis W_a g_a — the reconstruction the mode-4 arc found the O(h)
  //       ceiling in. Keeps the mode-3 (wall-aware, o-adjoint) projection.
  //   6 = EMBED momentum + PLAIN (mode-0) projection: the Basilisk pairing — embed viscous no-slip
  //       with the ½/½ face average, fs-weighted cut-cell Poisson, and central-difference
  //       correction (the mode-1/2/3 wall-aware projection was measured WORSE than plain; embed
  //       drives momentum).
  //   9 = CUTCELL-GHOST HYBRID (the recommended collocated mode for tight-throat porous media):
  //       mode-0's aperture projection unchanged (plain ½/½ map, real openness divergence —
  //       throttles sub-cell throats, symmetric MG-PCG, no fragmentation concern) but the
  //       predictor -grad(P) and the cell correction use the directional gpCenterGrad gradient
  //       (2nd-order one-sided at cut cells, never reads a solid-centered cell's P — the measured
  //       O(1/h) mode-0 defect). Measured: Z&H drag in a −0.04..−0.10% band N=32..128 (NOT clean
  //       2nd order — the pinned-face aperture-constraint truncation floors it — but 7–20× below
  //       mode 0); RCP permeability monotone toward the staggered-cutcell reference
  //       (−13.0/−8.6/−6.2% at Ng=32/44/56) where mode 0 is erratic (−20%..+14%, pathologically
  //       slow settling) and the ghost projection needs its fragmentation guard. See
  //       doc/collocated_second_order_open_problem.md §9.
  //  10 = mode 9 with the OPEN-CENTROID wall-aware constraint quadrature (the mode-3
  //       centerToFaceWallAware map). DEAD ABLATION — kept for the record: O(h) with a worse
  //       constant than mode 9 on Z&H, and DIVERGES on RCP slivers (the mode-3a non-telescoping
  //       row-sum mechanism; the telescoping gpCenterGrad force does not cure the
  //       constraint-side injection). Do not use.
  // Collocated cut-cell projection treatment. THE SUPPORTED VALUES ARE 0 AND 9 — prefer the
  // string API setCollocatedScheme(). 9 ("gauge-exact") is the DEFAULT since 2026-08-18: the
  // aperture constraint (unchanged, throat-safe, symmetric MG-PCG) with the directional
  // gpCenterGrad replacing the two operators measured to be O(1) at cut cells — the -grad(P)
  // predictor and the projection's cell correction. Measured on two periodic sphere beds
  // (peclet-examples benchmarks/porous-scaling, colcmp*/colcmp060*): SECOND ORDER on both
  // (2.36-2.89 over R=5..8, landing at +0.08 % of k_inf at R=16) where mode 0 is first order
  // (0.94-1.20) and additionally fails to reach steady state within 800 steps on 3 of 5 rungs of
  // the phi=0.60 bed; and cheapest of every variant tried, 4.6x faster than the STAGGERED
  // cut-cell reference and 5-6x faster than the directional ghost projection.
  //
  // RETIRED 2026-08-18 (rejected here; the kernels remain but are unreachable, deletion is a
  // follow-up): 1, 2 were pure ablations, and 10 was a documented dead ablation (O(h)
  // with a worse constant on Z&H, divergent on RCP slivers — doc/
  // collocated_second_order_open_problem.md §9.1). 3 and 4 (the FV-constraint variants, 4 with
  // set_fv_relax) survive as ablations reachable only through this integer entry point.
  //
  // 5/6/7 were RE-INSTATED 2026-08-19: they are not ablations, they are the Basilisk embed.h port
  // (true-normal dirichlet_gradient wall drag, openness-weighted cell correction, solid-cut-cell
  // sliver mask — commits db5b4aa/f5fde8c/6d412ec/03a71c6, doc/collocated_embed_port_plan.md).
  // That line is the live candidate for removing the collocated accuracy ceiling, so it must stay
  // reachable. Retiring them was my error.
  void setFaceInterp(int mode) {
    static constexpr int kRetired[] = {1, 2, 10};
    for (int r : kRetired)
      if (mode == r)
        throw std::runtime_error(
            "set_face_interp(" + std::to_string(mode) +
            "): retired 2026-08-18 (ablation / measured divergent). Use "
            "set_collocated_scheme(\"gauge-exact\") — the default — or \"plain\" for the "
            "legacy first-order aperture projection.");
    if (mode != 0 && (mode < 3 || mode > 7) && mode != 9 && (mode < 11 || mode > 13))
      throw std::runtime_error("set_face_interp: unknown mode " + std::to_string(mode));
    if (ghostProjection_ && mode != 0)
      throw std::runtime_error(
          "set_face_interp: incompatible with the ghost projection (set_ghost_projection(False) "
          "first, or use set_collocated_scheme which handles the transition)");
    faceInterp_ = mode;
    colSchemeAuto_ = false;  // explicit selection disables the AUTO default
  }
  // Preferred API for the collocated projection scheme.
  //   "gauge-exact" (default) aperture constraint + directional (gauge-exact) pressure gradient
  //   "plain"                 the legacy plain-average / central-difference path (first order)
  void setCollocatedScheme(const std::string& name) {
    if (name != "ghost" && ghostProjection_) {
      ghostProjection_ = false;  // scheme transition: drop the ghost before selecting a face mode
      gpNRows_ = -1;
    }
    if (name == "gauge-exact") {
      setFaceInterp(9);
      gauge2a_ = false;
    } else if (name == "gauge-2a") {  // EXPERIMENTAL: gauge-exact with the "gradient 2a"
      setFaceInterp(9);               // one-sided branch (see gauge_exact_gradient.hpp)
      gauge2a_ = true;
    } else if (name == "plain") {
      setFaceInterp(0);
      gauge2a_ = false;
    } else if (name == "ghost") {
      // The fluid-only constraint scheme (route 2b, 2026-08): binary-openness divergence +
      // directional closures + gauge-exact gradient. Clean-protocol record: family-free
      // (m1 -> 1e-5 monotone, |P| frozen), NO Layer-1 instability, C2 across dt = 60..1e20 with
      // no stabilizer, both-bed ladders -1.4% -> +0.22% (R=8..24; its own small plateau), Z&H
      // anchor -0.018% at N=128. Costs: BiCGStab (~2.3-2.7x pressure stage; star preconditioner
      // planned), ~1.6 KB/cell overlay (caps single-GPU size), fragmentation guard. The (1,2)
      // mixed mode stays quarantined (march-unstable on >2000-sphere beds).
      setGhostProjection(true, 2, 2);
      gauge2a_ = false;
    } else
      throw std::runtime_error(
          "set_collocated_scheme: expected \"gauge-exact\", \"gauge-2a\", \"plain\" or "
          "\"ghost\", got \"" +
          name + "\"");
  }
  // PM I ablation (Guy-Fogelson): keep the incremental predictor -grad(P^n) but accumulate
  // P += (rho/dt)*phi WITHOUT the rotational -mu*div(u*) term (constant-mu path only; the
  // variable-mu branches keep their own treatment). Default true = shipped behaviour.
  void setRotationalPressure(bool on) { rotationalP_ = on; }
  // Rotational under-relaxation: P += ct*phi - w*mu*div(u*). w = 1 is the shipped Timmermans
  // update; w = 0 is PM I. Shrinking w shrinks the O(1) velocity->pressure off-diagonal that
  // makes the cell-centered approximate projection marginally unstable (Guy-Fogelson eq. 92-94:
  // the destabilizing-perturbation threshold scales ~1/w), at the cost of ~1/w slower pressure
  // relaxation of the smooth modes at large dt. phi = 0 stays the unique fixed point for ANY
  // w > 0, at every dt including dt -> infinity.
  void setRotationalWeight(double w) { rotWeight_ = w; }
  // Wall-banded rotational blend (Frank, 2026-08-20): see the press_wallblend kernel. w0 = 0
  // (default) disables; typical w0 ~ 0.3-0.5. Composes with setRotationalWeight (uniform factor).
  void setRotationalWallWeight(double w0) { rotWallW_ = w0; }
  // Fluid-only pressure constraint (route 2b). Call BEFORE set_solid. Collocated experiment;
  // defaults byte-identical when 0. mode 1 = Design A (close every openness face with a
  // solid-centered side, everywhere); mode 2 = Design B (Kron star elimination: filtered
  // openness feeds the MG hierarchy only, the SPD star overlay restores the throat coupling in
  // the PCG matvec, the divergence keeps the original apertures on fluid rows, and fluid|solid
  // faces are corrected with phibar_s -- see star_elimination.hpp).
  // Aperture estimation order (2026-08-26): 1 = the shipped one-sample linear model (default,
  // byte-identical), 2 = marching-squares (5 trilinear samples/face, triangle-fan; O(h^2),
  // removes the convexity bias measured at +0.59%/+0.27% bed-k at R=8/12 -- tracker row 51).
  // In-solver ceiling is the trilinear field; for ANALYTIC geometry use exact/Saye apertures
  // via set_openness_override (scripts/exact_apertures_spheres.py). Call before set_solid.
  void setApertureOrder(int order) {
    if (order < 1 || order > 2)
      throw std::runtime_error("set_aperture_order: order must be 1 or 2");
    apertureOrder_ = order;
  }
  int apertureOrder() const { return apertureOrder_; }
  void setFluidOnlyConstraint(int mode) {
    if (mode < 0 || mode > 2)
      throw std::runtime_error("set_fluid_only_constraint: mode must be 0, 1 or 2");
    fluidOnlyMode_ = mode;
    if (mode != 0)
      colSchemeAuto_ = false;  // mechanism instruments run on the aperture rails, not AUTO-ghost
  }
  // Filtered rotational update (experimental): P += ct*phi - mu*S(div u*), S = one mask-aware
  // axis-wise (1,2,1)/4 smoothing pass per axis (one-sided 1/2(d_i+d_nbr) toward the open side at
  // a solid-centered neighbour, identity when sandwiched). S annihilates the axis checkerboard
  // including AT wall-adjacent cells; for smooth fields S = I + O(h^2). Steady state unchanged.
  void setRotationalFilter(bool on, double eps = 0.05) {
    rotFilter_ = on;
    rotFilterEps_ = eps;
  }
  // Under-relaxation of the mode-4 FV wall-flux defect correction (1 = full; <1 damps the stiff
  // explicit-lagged wall term). The steady state is independent of this value.
  void setFvRelax(double w) { fvRelax_ = w; }
  // CUDA-only 3-stream concurrent velocity solve (set_velocity_streams): no Kokkos analogue in this
  // port (the default-execution-space kernels are already stream-ordered). Accepted as a no-op for
  // API parity.
  void setVelocityStreams(bool /*on*/) {}
  // Seed/restore the velocity state (CUDA set_state / upload_velocity): u/v/w are inner-cell fields
  // (flat x-fastest, size nx*ny*nz); written into the velocity block + ghosts refreshed (periodic
  // wrap).
  void uploadVelocity(const std::vector<double>& uu, const std::vector<double>& vv,
                      const std::vector<double>& ww) {
    const std::vector<double>* src[3] = {&uu, &vv, &ww};
    CCExec space;
    const int ex = e_.x, ey = e_.y, nx = nx_, ny = ny_, nz = nz_, g = G;
    for (int c = 0; c < 3; ++c) {
      // Upload the inner field once, write it into the inner cells on device, then refresh the
      // periodic ghosts (G4) — the old path mirrored the field down, looped on host, and copied
      // back up.
      CCField din("peclet::flow::vel_in_d", static_cast<std::size_t>(nx_) * ny_ * nz_);
      Kokkos::deep_copy(
          din,
          Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
              src[c]->data(), src[c]->size()));
      CCField u = C[c].u;
      Kokkos::parallel_for(
          "peclet::flow::upload_velocity",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            u((long)(x + g) + (long)(y + g) * ex + (long)(z + g) * (long)ex * ey) =
                din((std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny);
          });
      fillGhosts(C[c].u);
    }
  }
#ifdef PECLET_FLOW_MPI
  // Multi-rank: this rank's IbmSolver is constructed with its LOCAL block dims (= the
  // BlockDecomposer of the GLOBAL grid for this rank); initMpi wires the g=2 velocity-block halo +
  // the global-origin red-black parity, and switches fillGhosts/maxOpenDivergence + the pressure MG
  // (CutcellMG::initMpi) onto their distributed paths. The caller decomposes first (deterministic
  // ORB) to size the constructor; initMpi re-derives it.
  void initMpi(int gnx, int gny, int gnz, MPI_Comm comm) {
    int size = 1;
    MPI_Comm_size(comm, &size);
    // Build the shared decomposition so the pressure MG can derive nested coarse levels
    // (CutcellMG::coarsened) — aligned ORB by default, or coarse-first when
    // set_decomposition_levels/PECLET_FLOW_DECOMP_LEVELS asks for it. Depends only on the global
    // grid and that setting, so it matches mpi_block()'s sizing exactly.
    initMpi(CutcellMG::decomposition(static_cast<std::size_t>(size), gnx, gny, gnz), comm);
  }
  // Shared-decomposition overload: wire the g=2 velocity-block halo from an EXTERNALLY-built ORB
  // (so flow and dem share one BlockDecomposer for coupled runs, and redistribute() can re-init
  // onto a re-decomposed partition). The local block size must already match dec.block(rank).size
  // (set via the constructor / allocateBlock).
  void initMpi(const peclet::core::decomp::BlockDecomposer<3>& dec, MPI_Comm comm) {
    distributed_ = true;
    comm_ = comm;
    const auto& gs = dec.globalSize();
    gnx_ = (int)gs[0];
    gny_ = (int)gs[1];
    gnz_ = (int)gs[2];
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::array<bool, 3> per{true, true, true};
    velHalo_ = std::make_shared<GridHaloTopology<3>>();
    velHalo_->buildTopology(dec, rank, G, per, comm);
    velDev_ = std::make_shared<GridHalo<double>>();
    velDev_->setLabel("velocity g2");
    velDev_->init(*velHalo_);
    // Communication-avoiding momentum sweeps (see smoothComp): the velocity block is g=2 already,
    // so the CA pair needs no new topology — only this float exchange for the stencil ring
    // (operator coefficients are float). Eligible when every rank's block is >= 4 on every axis
    // (rank-uniform: the decomposition is replicated), gated by PECLET_FLOW_CA.
    velDevF_ = std::make_shared<GridHalo<MReal>>();
    velDevF_->init(*velHalo_);
    long minExt = std::numeric_limits<long>::max();
    for (const auto& s : dec.sizes())
      for (int k = 0; k < 3; ++k)
        minExt = std::min(minExt, (long)s[k]);
    caMomentum_ = (caSmoothingMode() & kCaMomentum) && minExt >= 4;
    for (bool& d : momStencilDirty_)
      d = true;
    dec_ =
        std::make_shared<peclet::core::decomp::BlockDecomposer<3>>(dec);  // remember the partition
    const auto oig = velHalo_->indexer().originInclGhost();
    og_ = {(int)oig[0] + G, (int)oig[1] + G,
           (int)oig[2] + G};  // block inner origin -> global parity
    // The colour field's OWN g=3 topology follows the same partition (VOF_PLAN §3 rule 1). Rebuilt
    // here rather than only in enableVof() so the order enable_vof/init_mpi does not matter — and
    // because a redistribute() re-inits through this path. The advector's block carries no state
    // between steps (C's canonical storage is the G=2 registry field "C"), so reallocating it is
    // free of consequence.
    if (vofEnabled_)
      buildVofBlock();
  }
  // Redistribute the solver's state onto a NEW decomposition (dynamic load balancing). Enumerates
  // the registered fields, moves them from the current block layout to the new one (bit-exact via
  // redistributeGridFields), reallocates every buffer to the new block, re-inits the halo +
  // pressure MG on the new partition, and rebuilds all geometry-derived state (openness / IBM
  // overlay / stencils) from the migrated SDF. Velocity + pressure + SDF (+ any registered
  // scalar/property fields) survive; per-step scratch is rebuilt.
  void redistribute(const peclet::core::decomp::BlockDecomposer<3>& newDec) {
    if (!distributed_ || !dec_)
      return;
    int rank = 0;
    MPI_Comm_rank(comm_, &rank);
    const auto ob = dec_->block(rank), nb = newDec.block(rank);
    const int oex = (int)ob.size[0] + 2 * G, oey = (int)ob.size[1] + 2 * G,
              oez = (int)ob.size[2] + 2 * G;
    const int nex = (int)nb.size[0] + 2 * G, ney = (int)nb.size[1] + 2 * G,
              nez = (int)nb.size[2] + 2 * G;

    // 1. gather the surviving registered fields to host padded buffers on the OLD block.
    const auto names = fields_.names();
    std::vector<std::vector<double>> oldHost(names.size()), newHost(names.size());
    for (std::size_t k = 0; k < names.size(); ++k) {
      CCField f = fields_.at(names[k]).data;
      auto h = Kokkos::create_mirror_view(f);
      Kokkos::deep_copy(h, f);
      oldHost[k].assign(h.data(), h.data() + (std::size_t)oex * oey * oez);
      newHost[k].assign((std::size_t)nex * ney * nez, 0.0);
    }
    // 2. redistribute each field OLD -> NEW (host, bit-exact pure data movement).
    std::vector<const double*> op(names.size());
    std::vector<double*> np(names.size());
    for (std::size_t k = 0; k < names.size(); ++k) {
      op[k] = oldHost[k].data();
      np[k] = newHost[k].data();
    }
    peclet::core::decomp::redistributeGridFields<double>(*dec_, newDec, rank, G, op, np, comm_);

    // 3. reallocate every buffer to the new block; re-init the halo + MG on the new partition.
    allocateBlock((int)nb.size[0], (int)nb.size[1], (int)nb.size[2]);
    initMpi(newDec, comm_);
    // scatter a padded host buffer into a registered field's device buffer.
    auto scatterPadded = [&](const std::string& name, const std::vector<double>& src) {
      CCField f = fields_.at(name).data;
      auto h = Kokkos::create_mirror_view(f);
      std::memcpy(h.data(), src.data(), sizeof(double) * (std::size_t)nex * ney * nez);
      Kokkos::deep_copy(f, h);
    };
    // 4. scatter all migrated fields into the fresh (new-block) buffers.
    for (std::size_t k = 0; k < names.size(); ++k)
      scatterPadded(names[k], newHost[k]);
    // 5. rebuild geometry-derived state (openness/IBM/stencils/MG) from the migrated SDF. setSolid
    //    zeroes the velocity + pressure (it is the initial-geometry setup), so re-instate every
    //    non-SDF field afterward from the migrated data.
    setSolid(gatherInner(sdf_), cutcellPressure_);
    for (std::size_t k = 0; k < names.size(); ++k)
      if (names[k] != "sdf")
        scatterPadded(names[k], newHost[k]);
  }
  // Redistribute onto the weighted ORB of per-cell weights `w` (global x-fastest, gnx*gny*gnz). The
  // ergonomic Python entry point for load balancing: the caller passes a weight field (e.g. fluid
  // work + gamma*particle_count) and both flow and dem rebuild the SAME deterministic partition
  // from it. No BlockDecomposer object crosses the language boundary.
  void rebalanceByWeights(const std::vector<peclet::core::Real>& w) {
    if (!distributed_)
      return;
    int size = 1;
    MPI_Comm_size(comm_, &size);
    peclet::core::decomp::BlockDecomposer<3> newDec((std::size_t)size,
                                                    peclet::core::IVec<3>{gnx_, gny_, gnz_}, w);
    redistribute(newDec);
  }
#endif
  // per-face domain BC {face 0..5 = -x,+x,-y,+y,-z,+z}: type 0=periodic,1=no-slip
  // wall,2=Dirichlet/inflow,3=outflow.
  void setDomainBc(int face, int type, double vx, double vy, double vz) {
    bc_[face] = type;
    bcVel_[face][0] = vx;
    bcVel_[face][1] = vy;
    bcVel_[face][2] = vz;
    hasBc_ = false;
    hasOutflow_ = false;
    for (int i = 0; i < 6; ++i) {
      if (bc_[i])
        hasBc_ = true;
      if (bc_[i] == 3)
        hasOutflow_ = true;
    }
  }
  // per-position inlet velocity profile on `face` (CUDA set_domain_bc_profile): prof is (nb,nc,3)
  // on the inner grid of the face's two perpendicular axes; sets the face to inflow (type 2).
  // Resampled (clamp) to the ghost-inclusive face grid so the BC kernel indexes it directly by face
  // position.
  void setDomainBcProfile(int face, const std::vector<double>& prof, int nb, int nc) {
    const int a = face / 2;
    const int dims[3] = {e_.x, e_.y, e_.z};
    const int bax = (a + 1) % 3, cax = (a + 2) % 3;
    const int Lb = dims[bax], Lc = dims[cax];
    CCField pf("bcprof", (std::size_t)Lb * Lc * 3);
    auto h = Kokkos::create_mirror_view(pf);
    auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
    for (int p0 = 0; p0 < Lb; ++p0)
      for (int p1 = 0; p1 < Lc; ++p1) {
        const int ib = cl(p0 - G, nb), ic = cl(p1 - G, nc);
        for (int k = 0; k < 3; ++k)
          h(((long)p0 * Lc + p1) * 3 + k) = prof[((std::size_t)ib * nc + ic) * 3 + k];
      }
    Kokkos::deep_copy(pf, h);
    bcProf_[face] = pf;
    bcProfNc_[face] = Lc;
    bc_[face] = 2;
    hasBc_ = true;  // a profiled face is an inflow
  }
  // all-fluid + domain-BC pressure (CUDA set_pressure_geometry): same path as set_solid with an
  // open SDF.
  void setPressureGeometry(const std::vector<double>& sdfInner) { setSolid(sdfInner, true); }

  // SDF on the inner cells (flat x-fastest, size nx*ny*nz; <0 solid). cutcellPressure enables the
  // open-face-weighted cut-cell projection (off => velocity-only, e.g. unidirectional body-force
  // flow).
  // ---------------------------------------------------------------------------------------
  // ANALYTIC SCENE (Layer 2 of suite/docs/ANALYTIC_SDF_GEOMETRY.md)
  //
  // Geometry has only ever reached flow as an already-sampled field, with "analytic accuracy"
  // supplied as override arrays computed in Python (set_exact_crossings / set_openness_override) --
  // spheres only, and SINGLE-RANK only. A scene set here is device-resident and REPLICATED on every
  // rank, so a rank derives its own block's geometry from it with no communication at all; that is
  // what lifts the single-rank restriction rather than any new exchange.
  //
  // Geometry is expressed in CELL UNITS on the GLOBAL inner grid (cell centre (i,j,k) sits at
  // (i,j,k)), matching scripts/exact_apertures_spheres.py's centers_cells / radii_cells.
  //
  // PERIODICITY is the caller's: the scene is evaluated at global cell coordinates and has no
  // wrap of its own, so a periodic packing must instantiate its images (or use a node whose eval
  // is periodic). Nothing here min-images for you.
  // ---------------------------------------------------------------------------------------

  /// Install an analytic scene from core's flat node/instance encoding
  /// (peclet/core/geom/scene_builder.hpp: 3 ints + 16 reals per node, 2 ints + 17 reals per
  /// instance), held as a core SceneQueryDevice: mode selection (sphere-union fast path vs
  /// general tree walk), candidate-grid acceleration and min-image periodicity all live in CORE
  /// now — flow briefly hand-rolled the decode+upload+eval, which was the wrong layer for it.
  /// Call before set_solid_from_scene().
  ///
  /// `periodic = true` treats the scene as min-image periodic over the GLOBAL inner grid, so a
  /// periodic packing needs ONE instance per body — no 27-image instantiation. `periodic = false`
  /// keeps the open-scene semantics (images are the caller's).
  void setScene(const std::vector<int>& nodeInts, const std::vector<double>& nodeReals,
                const std::vector<int>& instInts, const std::vector<double>& instReals,
                bool periodic = false) {
    namespace g = peclet::core::geom;
    g::SceneBuilder<double> b = g::SceneBuilder<double>::decode(nodeInts, nodeReals, instInts,
                                                               instReals, /*grids=*/{},
                                                               /*pool=*/{});
    if (b.instances().empty())
      throw std::runtime_error("set_scene: at least one instance required");
    for (const auto& nd : b.nodes())
      if (nd.kind == g::kGrid)
        throw std::runtime_error("set_scene: grid leaves are not supported (analytic scenes only)");
    // Global inner-grid extents: the distributed build carries them (gnx_); single-rank (or a
    // non-MPI build) the local block IS the global grid.
    double GX = nx_, GY = ny_, GZ = nz_;
#ifdef PECLET_FLOW_MPI
    if (gnx_ > 0) {
      GX = gnx_;
      GY = gny_;
      GZ = gnz_;
    }
#endif
    g::PeriodicBox<double> box{GX, GY, GZ, periodic};
    sceneB_ = std::make_shared<g::SceneBuilder<double>>(std::move(b));
    sceneOrigin_ = peclet::core::Vec3<double>{0, 0, 0};
    sceneExtent_ = peclet::core::Vec3<double>{GX, GY, GZ};
    scenePeriodic_ = periodic;
    buildSceneQuery();
    hasScene_ = true;
    // Moving-geometry state travels WITH the scene: core's Instance already carries linVel/angVel/
    // center, so a caller can encode motion directly and it arrives here. CENTRE OF ROTATION: the
    // encoded `center` when it is nonzero, otherwise the instance's own translation -- which is
    // what a caller who only placed the body means by "spin it". Set `center` explicitly (or pass
    // it to set_instance_motion) to spin about some other point.
    nInst_ = static_cast<int>(sceneB_->instances().size());
    instCen_.assign((std::size_t)nInst_ * 3, 0.0);
    instLin_.assign((std::size_t)nInst_ * 3, 0.0);
    instAng_.assign((std::size_t)nInst_ * 3, 0.0);
    for (int i = 0; i < nInst_; ++i) {
      const auto& in = sceneB_->instances()[(std::size_t)i];
      const bool haveCen = in.center.x != 0.0 || in.center.y != 0.0 || in.center.z != 0.0;
      const auto c = haveCen ? in.center : in.transform.translation;
      instCen_[3 * (std::size_t)i + 0] = c.x;
      instCen_[3 * (std::size_t)i + 1] = c.y;
      instCen_[3 * (std::size_t)i + 2] = c.z;
      instLin_[3 * (std::size_t)i + 0] = in.linVel.x;
      instLin_[3 * (std::size_t)i + 1] = in.linVel.y;
      instLin_[3 * (std::size_t)i + 2] = in.linVel.z;
      instAng_[3 * (std::size_t)i + 0] = in.angVel.x;
      instAng_[3 * (std::size_t)i + 1] = in.angVel.y;
      instAng_[3 * (std::size_t)i + 2] = in.angVel.z;
    }
    refreshMotionFlag();
    uploadMotion();
  }

  /// Rigid-body motion of one scene instance (Layer 3 rung 2). `lin` is the body's linear
  /// velocity, `ang` its angular velocity about its own centre -- both in CELL UNITS PER TIME, the
  /// same units the velocity field carries, since the scene lives on the global inner grid.
  ///
  /// Setting any nonzero component switches the solver onto the moving-geometry path: the
  /// momentum operator's no-slip datum becomes the local wall velocity instead of zero (rung 2)
  /// and the cut-cell projection gains the wall's own volume flux (rung 3). With every component
  /// zero the solver stays on the static path, bit for bit.
  void setInstanceMotion(int i, const std::array<double, 3>& lin,
                         const std::array<double, 3>& ang, const double* center = nullptr) {
    if (!hasScene_)
      throw std::runtime_error("set_instance_motion: call set_scene first");
    if (i < 0 || i >= nInst_)
      throw std::runtime_error("set_instance_motion: instance index out of range");
    auto& in = sceneB_->instanceRef(i);
    for (int k = 0; k < 3; ++k) {
      instLin_[3 * (std::size_t)i + k] = lin[k];
      instAng_[3 * (std::size_t)i + k] = ang[k];
      if (center)
        instCen_[3 * (std::size_t)i + k] = center[k];
    }
    in.linVel = peclet::core::Vec3<double>{lin[0], lin[1], lin[2]};
    in.angVel = peclet::core::Vec3<double>{ang[0], ang[1], ang[2]};
    in.center = peclet::core::Vec3<double>{instCen_[3 * (std::size_t)i + 0],
                                           instCen_[3 * (std::size_t)i + 1],
                                           instCen_[3 * (std::size_t)i + 2]};
    refreshMotionFlag();
    uploadMotion();
  }

  /// Move one instance (Layer 3 rung 4). Takes effect at the next rebuild_geometry() -- the SDF
  /// field, the cut-cell overlay, the apertures and the pressure operator are ALL derived from the
  /// instance transforms, so a transform change without a rebuild would leave the solver running
  /// on the old geometry with a new wall velocity, which is worse than either.
  ///
  /// The centre of rotation FOLLOWS the body: it is re-anchored to the new translation unless the
  /// caller pinned one explicitly through set_instance_motion.
  void setInstanceTransform(int i, const std::array<double, 3>& translation,
                            const std::array<double, 4>& quat) {
    if (!hasScene_)
      throw std::runtime_error("set_instance_transform: call set_scene first");
    if (i < 0 || i >= nInst_)
      throw std::runtime_error("set_instance_transform: instance index out of range");
    auto& in = sceneB_->instanceRef(i);
    const bool centreTracked =
        instCen_[3 * (std::size_t)i + 0] == in.transform.translation.x &&
        instCen_[3 * (std::size_t)i + 1] == in.transform.translation.y &&
        instCen_[3 * (std::size_t)i + 2] == in.transform.translation.z;
    in.transform.translation =
        peclet::core::Vec3<double>{translation[0], translation[1], translation[2]};
    in.transform.rotation = peclet::core::Quat<double>{quat[0], quat[1], quat[2], quat[3]};
    if (centreTracked) {
      for (int k = 0; k < 3; ++k)
        instCen_[3 * (std::size_t)i + k] = translation[k];
      in.center = in.transform.translation;
    }
    sceneDirty_ = true;
  }

  /// Re-derive ALL geometry from the current instance transforms (Layer 3 rung 4): rebuild the
  /// accelerated scene query, re-sample the SDF, rebuild the cut-cell overlay / apertures /
  /// pressure operator, and re-derive the exact crossings if they were in use.
  ///
  /// The velocity and pressure fields are PRESERVED across the rebuild. set_solid zeroes u, phi
  /// and P by design (it is a setup entry point), which would reset the flow on every step of a
  /// moving-geometry march -- so they are saved and restored around it. That is a full rebuild by
  /// design: the measured 128^3 cost is ~65% momentum/IBM stencils and ~35% pressure/MG, with
  /// scene sampling in the noise, so an incremental path must attack BOTH sides and is deferred.
  ///
  /// FRESH CELLS: a cell uncovered by the body's motion this step inherits the zero the solid held
  /// there, not an extrapolated fluid value. That is the conservative v1 choice (bounded, and the
  /// momentum solve relaxes it within a step at small per-step motion); extrapolation is an open
  /// question recorded in the design note.
  void rebuildGeometry() {
    if (!hasScene_)
      throw std::runtime_error("rebuild_geometry: call set_scene first");
    CCField uSave[3], mSave[3];
    for (int c = 0; c < 3; ++c) {
      uSave[c] = CCField("uSave", n_);
      Kokkos::deep_copy(uSave[c], C[c].u);
      if (freshSeed_) {  // remember which points were SOLID, to find the ones the body uncovers
        mSave[c] = CCField("mSave", n_);
        Kokkos::deep_copy(mSave[c], C[c].mask);
      }
    }
    CCField pSave("pSave", n_);
    Kokkos::deep_copy(pSave, P_);
    if (sceneDirty_) {
      buildSceneQuery();
      sceneDirty_ = false;
    }
    // Crossings BEFORE the solid: the overlay build consumes tEx_, so deriving them first means
    // ONE geometry rebuild per step rather than two.
    if (sceneCrossings_)
      setExactCrossingsFromScene();
    setSolidFromScene(cutcellPressure_);
    for (int c = 0; c < 3; ++c)
      Kokkos::deep_copy(C[c].u, uSave[c]);
    Kokkos::deep_copy(P_, pSave);
    if (freshSeed_)
      seedFreshCells(mSave);
  }

  /// FRESH CELLS: the points a moving body has just uncovered.
  ///
  /// Restoring u across the rebuild hands such a point whatever the SOLID held there -- zero, or a
  /// stale masked value -- rather than a fluid state. The momentum solve relaxes it within a step,
  /// but until it does, a point that should be moving with the wall reads as stopped, and the
  /// discrete reaction charges the body for the difference. That is one of the two textbook
  /// mechanisms behind spurious force oscillations in a moving-boundary IBM (the other being the
  /// abrupt stencil change as the interface crosses a face), and it is why a body translating
  /// through a fixed grid produces a force spike every time it uncovers a row of cells.
  ///
  /// Seeding with the LOCAL WALL VELOCITY is the cheapest defensible choice: uBc_ already holds
  /// the rigid-body velocity of the owning instance evaluated at the wall point nearest each
  /// staggered point, so a just-uncovered point starts moving with the surface that released it
  /// rather than at rest. It is bounded (no extrapolation), needs no new field, and reduces to the
  /// old behaviour exactly when the wall is not moving.
  void seedFreshCells(CCField mOld[3]) {
    if (!hasMotion_)
      return;
    CCExec space;
    const C3 e = e_;
    for (int c = 0; c < 3; ++c) {
      if (mOld[c].extent(0) != n_ || uBc_[c].extent(0) != n_)
        continue;
      CCField u = C[c].u;
      CCConst mo = CCConst(mOld[c]), mn = CCConst(C[c].mask), w = CCConst(uBc_[c]);
      Kokkos::parallel_for(
          "peclet::flow::seed_fresh", Kokkos::RangePolicy<CCExec>(space, 0, (long)n_),
          KOKKOS_LAMBDA(long i) {
            if (mo(i) > 0.5 && mn(i) <= 0.5)
              u(i) = w(i);
          });
    }
    space.fence();
  }

  /// Fresh-cell policy for moving geometry. true (DEFAULT) = seed with the local wall velocity;
  /// false = inherit whatever the solid held, which is what shipped before 2026-08-30. Inert when
  /// nothing moves, so a static run is bit-identical either way. See seedFreshCells.
  void setFreshCellSeed(bool on) { freshSeed_ = on; }
  bool freshCellSeed() const { return freshSeed_; }

  /// Re-derive ONLY the wall-velocity fields and the momentum operator that folds them in.
  ///
  /// The linearised moving-boundary problems -- an oscillating body at vanishing amplitude, a
  /// shear cell driven by counter-moving plates -- change the wall VELOCITY every step while the
  /// geometry never moves. `set_instance_motion` alone does not reach them: `uBc_` (the momentum
  /// operator's no-slip datum) and `uwCell_` (the cut-cell projection's wall flux) are built in
  /// `set_solid_from_scene`, so before this existed such a driver had to call `rebuild_geometry()`
  /// every step and pay a full geometry re-derivation to update a boundary condition.
  ///
  /// SCOPE: the instance TRANSFORMS must be unchanged. Nothing here re-samples the SDF, the
  /// apertures, the ownership field or the pressure operator, so if a body has actually moved this
  /// is silently wrong -- call `rebuild_geometry()` instead. Velocity and pressure are untouched.
  void refreshWallVelocity() {
    if (!hasScene_)
      throw std::runtime_error("refresh_wall_velocity: call set_scene first");
    if (sceneDirty_)
      throw std::runtime_error(
          "refresh_wall_velocity: an instance TRANSFORM changed since the last geometry build -- "
          "this call only refreshes the wall VELOCITY, so the run would continue on stale "
          "geometry. Call rebuild_geometry() instead.");
    buildWallVelocity();
    rebuildStencils();
  }

  /// True when at least one instance carries a nonzero velocity -- i.e. the moving-geometry paths
  /// are live. Everything downstream keys off this, so a driver can assert it.
  bool hasMovingInstance() const { return hasMotion_; }
  int sceneInstanceCount() const { return nInst_; }

  /// Rung 3 on/off. ON (the default) is the correct physics: a rigid body sweeping through a cut
  /// cell injects a wall flux the projection must balance. The switch exists so the Galilean gate
  /// can EXHIBIT the failure the term fixes rather than assert it -- turning it off leaves rung 2's
  /// wall velocity in the momentum operator and a projection that wrongly forces div_open(u) = 0.
  void setWallFluxDivergence(bool on) { wallFluxDiv_ = on; }
  bool wallFluxDivergence() const { return wallFluxDiv_; }

  bool hasScene() const { return hasScene_; }

  /// Per-inner-cell owning instance (Layer 3 rung 1), x-fastest, -1 where no scene has been
  /// sampled yet. Host copy; the device field is what the solver kernels read.
  std::vector<int> getCutOwner() const {
    const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
    std::vector<int> out(n, -1);
    if (cutOwner_.extent(0) != n)
      return out;
    using HostV = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(HostV(out.data(), n), cutOwner_);
    return out;
  }

  /// Sample the scene onto this rank's inner grid and install it as the solid, entirely on device
  /// -- no nx*ny*nz float64 host round trip, and correct on every rank.
  void setSolidFromScene(bool cutcellPressure) {
    if (!hasScene_)
      throw std::runtime_error("set_solid_from_scene: call set_scene first");
    const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
    CCField din("peclet::flow::sceneSdf", n);
    if (cutOwner_.extent(0) != n)
      cutOwner_ = Kokkos::View<int*, CCMem>("peclet::flow::cutOwner", n);
    auto own = cutOwner_;
    const auto q = sceneQ_->view();
    const int nx = nx_, ny = ny_, nz = nz_;
    const C3 og = og_;
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::scene_sample",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const peclet::core::Vec3<double> p{(double)(x + og.x), (double)(y + og.y),
                                            (double)(z + og.z)};
          // evalOwner is ONE traversal returning bitwise eval's value plus the argmin instance, so
          // carrying the ownership field costs nothing over the sample it rides on.
          int oi = -1;
          const std::size_t idx =
              (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny;
          din(idx) = q.evalOwner(p, oi);
          own(idx) = oi;
        });
    space.fence();
    // PERIODIC IMAGES ARE A UNION. The query takes the minimum over an instance's 27 neighbour
    // images -- right for a body straddling a periodic face, and a silent TRAP for a leaf WIDER
    // than the box: its images overlap, and a cavity carved from it (a container wall built as
    // slab-minus-cavity) is refilled wherever a neighbouring image's slab covers it. Measured: a
    // 0.7 L slab minus a 53-cell cavity gave a 38-cell duct; the ten Cate tank ran 30 % narrow
    // through two campaigns and read as "creeping-valued confinement". Detect it EXACTLY: when
    // some instance's bounding sphere spans more than the box on a periodic axis, sample the
    // primary image alone and count the cells whose solid/fluid sign the images changed.
    imageOverlapCells_ = 0;
    if (scenePeriodic_) {
      namespace g = peclet::core::geom;
      const auto hv = sceneB_->view();
      const double lmin = std::fmin(sceneExtent_.x, std::fmin(sceneExtent_.y, sceneExtent_.z));
      bool wide = false;
      for (int i = 0; i < hv.instanceCount; ++i)
        if (2.0 * g::instanceBound(hv, i).r > lmin)
          wide = true;
      if (wide) {
        g::PeriodicBox<double> nobox{sceneExtent_.x, sceneExtent_.y, sceneExtent_.z, false};
        auto qnp = g::SceneQueryDevice<double, CCMem>::build(*sceneB_, sceneOrigin_, sceneExtent_,
                                                            nobox, /*accelerate=*/false);
        const auto qv = qnp.view();
        long cnt = 0;
        Kokkos::parallel_reduce(
            "peclet::flow::scene_image_overlap",
            Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
            KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
              const peclet::core::Vec3<double> p{(double)(x + og.x), (double)(y + og.y),
                                                (double)(z + og.z)};
              int oi = -1;
              const std::size_t idx =
                  (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny;
              const double dp = qv.evalOwner(p, oi);
              if ((dp < 0.0) != (din(idx) < 0.0))
                ++acc;
            },
            cnt);
        space.fence();
        imageOverlapCells_ = cnt;
        if (cnt > 0)
          std::fprintf(stderr,
                       "peclet.flow set_solid_from_scene WARNING: an instance is wider than the "
                       "periodic box, and the UNION of its periodic images changes the solid at "
                       "%ld cells on this rank. If it is a container wall (slab minus cavity), "
                       "keep the slab's half-extent at half the box plus the wall thickness -- "
                       "not more -- or the images refill the cavity. "
                       "periodic_image_overlap_cells() returns this count.\n",
                       cnt);
      }
    }
    setSolidDevice(din, cutcellPressure);
  }

  /// Cells on this rank whose solid/fluid sign was set by a periodic IMAGE of an instance wider
  /// than the box (see setSolidFromScene); 0 when no instance is that wide or the images agree.
  long periodicImageOverlapCells() const { return imageOverlapCells_; }

  /// EXACT wall crossings straight from the scene, on device, on every rank -- the in-solver
  /// replacement for set_exact_crossings + scripts/exact_apertures_spheres.py.
  ///
  /// t[c][a](i) = the fraction in (0,1) along the unit segment from component c's staggered point
  /// at inner cell i toward i + e_a at which the scene's SDF changes sign; NaN where the segment
  /// does not cross (the consumer falls back to the linear-interpolated theta). Bisection, NOT
  /// Newton: contract 2 of the design note only guarantees SIGN correctness for the bound-only
  /// leaves (ellipsoid, superquadric, CSG seams), and a Newton step on a non-distance field can
  /// leave the bracket entirely.
  void setExactCrossingsFromScene() {
    if (!hasScene_)
      throw std::runtime_error("set_exact_crossings_from_scene: call set_scene first");
    const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
    const auto q = sceneQ_->view();
    const int nx = nx_, ny = ny_, nz = nz_;
    const C3 og = og_;
    CCExec space;
    for (int c = 0; c < 3; ++c) {
      // Component c's sample placement comes from the GRID POLICY: staggered puts it on the low
      // face along axis c (offset -1/2 there), collocated at the cell center (offset 0) -- the
      // collocated ghost projection consumes tEx_[c][c] at CENTERS, so hardcoding the staggered
      // offsets here would silently compute crossings from the wrong points on that path.
      const auto po = Grid::offset(c);
      const double offc[3] = {(double)po.x, (double)po.y, (double)po.z};
      for (int a = 0; a < 3; ++a) {
        tEx_[c][a] = CCField("tEx", n);
        CCField t = tEx_[c][a];
        const double ox = offc[0], oy = offc[1], oz = offc[2];
        const int aa = a;
        Kokkos::parallel_for(
            "peclet::flow::scene_crossings",
            Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
            KOKKOS_LAMBDA(int x, int y, int z) {
              const double px = (double)(x + og.x) + ox, py = (double)(y + og.y) + oy,
                           pz = (double)(z + og.z) + oz;
              const double dx = aa == 0 ? 1.0 : 0.0, dy = aa == 1 ? 1.0 : 0.0,
                           dz = aa == 2 ? 1.0 : 0.0;
              const auto f = [&](double s) {
                return q.eval(peclet::core::Vec3<double>{px + s * dx, py + s * dy, pz + s * dz});
              };
              const double f0 = f(0.0), f1 = f(1.0);
              const std::size_t idx =
                  (std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny;
              if ((f0 < 0.0) == (f1 < 0.0)) {  // no sign change -> no crossing on this segment
                t(idx) = Kokkos::Experimental::quiet_NaN_v<double>;
                return;
              }
              double lo = 0.0, hi = 1.0, flo = f0;
              for (int it = 0; it < 52; ++it) {  // bisection to ~1 ulp of the unit interval
                const double mid = 0.5 * (lo + hi);
                const double fm = f(mid);
                if ((fm < 0.0) == (flo < 0.0)) {
                  lo = mid;
                  flo = fm;
                } else {
                  hi = mid;
                }
              }
              t(idx) = 0.5 * (lo + hi);
            });
      }
    }
    space.fence();
    hasExactCross_ = true;
    sceneCrossings_ = true;  // scene-derived: valid on every rank, unlike the host override path
  }

  /// Host entry point: upload the inner SDF once and delegate. Kept so every existing caller and
  /// the Python binding are unchanged.
  void setSolid(const std::vector<double>& sdfInner, bool cutcellPressure) {
    const std::size_t n = (std::size_t)nx_ * ny_ * nz_;
    if (sdfInner.size() != n)
      throw std::runtime_error("set_solid: expected nx*ny*nz values");
    CCField din("peclet::flow::sdfInner_d", n);
    using HostConst = Kokkos::View<const double*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(din, HostConst(sdfInner.data(), n));
    setSolidDevice(din, cutcellPressure);
  }

  /// Device entry point (Layer 2): the inner SDF is ALREADY on device, so geometry never
  /// round-trips through the host. This is the body every set_solid path shares.
  void setSolidDevice(CCField din, bool cutcellPressure) {
    cutcellPressure_ = cutcellPressure;
    // A few setup paths below (the ghost-projection pocket decoupling) are host-side
    // connected-component analyses and genuinely need the inner SDF on the host. Materialise it
    // ONCE, lazily, so the common path keeps the device-resident benefit.
    std::vector<double> sdfInnerHost_;
    auto sdfInner_ = [&]() -> const std::vector<double>& {
      if (sdfInnerHost_.empty()) {
        const std::size_t nI = (std::size_t)nx_ * ny_ * nz_;
        sdfInnerHost_.resize(nI);
        Kokkos::deep_copy(
            Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                sdfInnerHost_.data(), nI),
            din);
      }
      return sdfInnerHost_;
    };
    if constexpr (Grid::collocated) {
      // DEFAULT SWITCH (2026-08-25, user decision after the attractor campaign): the collocated
      // scheme default is AUTO = the GHOST (fluid-only) projection — family-free, unconditionally
      // stable, protocol-independent (doc/collocated_invisible_subspace.md; clean ladders both
      // beds) — falling back to gauge-exact with a stderr notice on the configurations the ghost
      // v1 does not support (porous / variable-rho / domain-BC / Chebyshev / analytic overrides).
      // Any explicit scheme selection (set_collocated_scheme / set_face_interp /
      // set_ghost_projection / set_fluid_only_constraint) disables AUTO.
      if (colSchemeAuto_) {
        const bool ok = !(porous_ || varRho_ || hasBc_ || useChebyshev_ || hasExactCross_ ||
                          hasOpenOverride_ || fluidOnlyMode_ != 0);
        if (ok) {
          ghostProjection_ = true;
          gpMatrixOrder_ = 2;
          gpRhsOrder_ = 2;
          faceInterp_ = 0;  // the ghost owns the operators the face-interp modes replace
          gpNRows_ = -1;
        } else {
          if (ghostProjection_)
            gpNRows_ = -1;
          ghostProjection_ = false;
          faceInterp_ = 9;
          fprintf(stderr,
                  "peclet::flow SolverColocated: AUTO scheme fell back to gauge-exact "
                  "(configuration unsupported by the ghost projection v1). Select explicitly "
                  "with set_collocated_scheme to silence this notice.\n");
        }
      }
    }
#ifdef PECLET_FLOW_MPI
    for (bool& d : momStencilDirty_)  // stencil ring re-exchange for the CA momentum sweeps
      d = true;
#endif
    // Does the geometry actually contain solid? (all-fluid set_pressure_geometry passes sd>0
    // everywhere -> stays false, keeping the channel/BFS path.) Device reduction: din lives on
    // device now, and pulling it back just to scan it would defeat the point.
    {
      const std::size_t nInner = (std::size_t)nx_ * ny_ * nz_;
      int anySolid = 0;
      CCConst dinC(din);
      Kokkos::parallel_reduce(
          "peclet::flow::has_solid", Kokkos::RangePolicy<CCExec>(0, nInner),
          KOKKOS_LAMBDA(const std::size_t i, int& acc) { acc = acc || (dinC(i) < 0.0); }, anySolid);
      hasSolid_ = anySolid != 0;
    }
#ifdef PECLET_FLOW_MPI
    if (distributed_) {  // a solid anywhere in the global domain enables the IBM momentum path
      int local = hasSolid_ ? 1 : 0, global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, comm_);
      hasSolid_ = global != 0;
    }
#endif
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      // Multi-rank: din is THIS rank's LOCAL inner block; fill the inner cells ON DEVICE, then
      // halo-exchange the ghosts (cross-rank + periodic) so the overlay/openness read the
      // neighbour's SDF at the block boundary. (Was a host mirror + triple loop + full H2D.)
      CCExec space;
      const int ex = e_.x, ey = e_.y, nx = nx_, ny = ny_, nz = nz_, g = G;
      CCField sdf = sdf_;
      CCConst dinC(din);
      Kokkos::parallel_for(
          "peclet::flow::sdf_fill_inner",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx, ny, nz}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            sdf((long)(x + g) + (long)(y + g) * ex + (long)(z + g) * (long)ex * ey) =
                dinC((std::size_t)x + (std::size_t)y * nx + (std::size_t)z * (std::size_t)nx * ny);
          });
      space.fence();
      velDev_->exchange(sdf_);
    } else
#endif
    {
      // Single-rank: periodic-wrap gather on device (G4) — fills the whole extended block
      // (inner + periodic ghosts) in one kernel. `din` is already device-resident.
      CCExec space;
      const int ex = e_.x, ey = e_.y, ez = e_.z, nx = nx_, ny = ny_, nz = nz_, g = G;
      CCField sdf = sdf_;
      Kokkos::parallel_for(
          "peclet::flow::sdf_periodic_wrap",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {ex, ey, ez}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const int ix = (((x - g) % nx) + nx) % nx, iy = (((y - g) % ny) + ny) % ny,
                      iz = (((z - g) % nz) + nz) % nz;
            sdf((long)x + (long)y * ex + (long)z * (long)ex * ey) = din(
                (std::size_t)ix + (std::size_t)iy * nx + (std::size_t)iz * (std::size_t)nx * ny);
          });
      space.fence();
    }
    const bool useEx = hasExactCross_ && !Grid::collocated;  // exact-theta arrays are for the
                                                             // staggered point placement
    for (int c = 0; c < 3; ++c) {
      const Off3 off =
          Grid::offset(c);  // velocity-unknown placement (staggered: -1/2 face; collocated: 0)
      C[c].nCut = buildIbmOverlay<0>(
          CCConst(sdf_), e_, G, off, /*Dirichlet*/ 0, C[c].ov, C[c].idMap, C[c].counter,
          useEx ? CCConst(tEx_[c][0]) : CCConst(), useEx ? CCConst(tEx_[c][1]) : CCConst(),
          useEx ? CCConst(tEx_[c][2]) : CCConst(),
          C3{nx_, ny_, nz_});  // SCHEME 0 = point-value (matches CUDA ibm_geometry_ext_k<0>)
      ibmSolidMask(C[c].mask, CCConst(sdf_), e_, off);
      Kokkos::deep_copy(C[c].u, 0.0);
    }
    // MOVING GEOMETRY (rung 2): the wall-velocity fields must exist BEFORE the momentum operator
    // is assembled -- rebuildStencils folds them into the inhomogeneous term. sdf_ and its ghosts
    // are final at this point, which is what the central-difference normals read.
    buildWallVelocity();
    rebuildStencils();
    // Staggered domain BCs bake an implicit-diffusion wall fold; the collocated grid instead uses
    // explicit reflection ghosts (refreshed each smoother sweep), so it needs no fold.
    if (hasBc_ && !Grid::collocated)
      setupBcDiffusion();
#ifdef PECLET_FLOW_MPI
    // AUTO: pick the V-cycle when the per-rank block is small (see setVelocityMultigridAuto). The
    // decision uses the GLOBAL cells / ranks, so every rank agrees without communication. Only on
    // the validated operator modes: IBM-periodic, all-fluid domain-BC (explicit advection), mixed.
    if (!vmgExplicit_ && distributed_ && vmgAutoCells_ > 0) {
      int np = 1;
      MPI_Comm_size(comm_, &np);
      const double perRank = (double)gnx_ * gny_ * gnz_ / (double)np;
      const bool eligible = !varProps_ && !varRho_ && !hasDrag_ && !porous_ && !Grid::collocated &&
                            (!hasBc_ || hasSolid_ || !implicitFou_);
      // np > 1: a single rank has no halo latency to hide (RB-GS is the cheaper solver there) and
      // a distributed np=1 run must stay bit-identical to the single-rank path. Global size floor:
      // the rule is about latency-bound LARGE runs; a small global problem split across ranks
      // (every ctest, every quick check) keeps RB-GS so distributed == single-rank stays exact.
      const double global = (double)gnx_ * gny_ * gnz_;
      useVelocityMg_ = eligible && np > 1 && global >= (double)vmgAutoMinGlobal_ &&
                       perRank < (double)vmgAutoCells_;
      if (useVelocityMg_) {
        vmgLevels_ = 3;
        vmgVcycles_ = 40;
      }
    }
#endif
    if (useVelocityMg_) {  // velocity-MG hierarchy: IBM (staircase/upwind), domain-BC
                           // (const-coeff) or mixed (staircase + folds) mode
#ifdef PECLET_FLOW_MPI
      // Distributed: level 0 on the solver's own decomposition (the g=2 velocity block), coarse
      // levels coarsened in place with the even-block gate (no telescoping here yet -- measured
      // first, see docs/SCALING_ISSUES.md issue 5).
      if (distributed_)
        vmg_.initMpi(*dec_, vmgLevels_, comm_);
      else
#endif
        vmg_.init(nx_, ny_, nz_, vmgLevels_);
      if (hasBc_)
        vmg_.setBC(bc_);
      if (!hasBc_ || hasSolid_) {  // the staircase (IBM / mixed) paths classify by volume fraction
        vmgTheta_ = CCField("vmgTheta", n_);
        vmgClean_ = CCField("vmgClean", n_);
      }
    }
    if (cutcellPressure_) {
      buildOpenness(ox_, oy_, oz_, CCConst(sdf_), e_, 1.0, 1.0, 1.0,
                    apertureOrder_);  // on the g=2 velocity block
      if (hasOpenOverride_) {
        // Analytic-SDF exact apertures (setOpennessOverride): overwrite the sampled-SDF openness
        // with the user-provided inner fields + periodic wrap into the ghost ring (single-rank).
        const std::vector<double>* src[3] = {&oxOverride_, &oyOverride_, &ozOverride_};
        CCField dst[3] = {ox_, oy_, oz_};
        for (int f = 0; f < 3; ++f) {
          CCField din("peclet::flow::openOv_d", (std::size_t)nx_ * ny_ * nz_);
          Kokkos::deep_copy(din, Kokkos::View<const double*, Kokkos::HostSpace,
                                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                                     src[f]->data(), src[f]->size()));
          CCExec space;
          const int ex = e_.x, ey = e_.y, ez = e_.z, nx = nx_, ny = ny_, nz = nz_, g = G;
          CCField o = dst[f];
          Kokkos::parallel_for(
              "peclet::flow::open_override_wrap",
              Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {ex, ey, ez}),
              KOKKOS_LAMBDA(int x, int y, int z) {
                const int ix = (((x - g) % nx) + nx) % nx, iy = (((y - g) % ny) + ny) % ny,
                          iz = (((z - g) % nz) + nz) % nz;
                o((long)x + (long)y * ex + (long)z * (long)ex * ey) =
                    din((std::size_t)ix + (std::size_t)iy * nx +
                        (std::size_t)iz * (std::size_t)nx * ny);
              });
          space.fence();
        }
      }
      if constexpr (Grid::collocated) {  // static open-centroid wall distances (setFaceInterp(3))
        buildFaceCentroidDist(xcx_, xcy_, xcz_, CCConst(sdf_), e_);
        buildCellFraction(cs_, CCConst(sdf_), e_, G);  // cell fluid fraction (setFaceInterp(4))
        if (faceInterp_ >= 5 &&
            faceInterp_ <= 7) {  // EMBED: a solid-CENTRED cut cell (cs>0) is partially fluid and
                                 // holds
          // its reconstructed near-wall velocity — masking it to 0 (the sdf<0 IBM mask) drops the
          // near-wall closure and shifts the whole channel. Re-mask from cs: pin ONLY fully-solid
          // cells (cs≈0), keeping every partial-fluid cut cell live in the embed solve +
          // projection.
          CCConst cs = CCConst(cs_);
          const std::size_t nn = n_;
          for (int c = 0; c < 3; ++c) {
            CCField m = C[c].mask;
            Kokkos::parallel_for(
                "peclet::flow::embed_solid_mask", Kokkos::RangePolicy<CCExec>(0, nn),
                KOKKOS_LAMBDA(std::size_t i) { m(i) = cs(i) < 1e-6 ? 1.0 : 0.0; });
          }
        }
      }
      if (fluidOnlyMode_ == 1) {
        // Mode-14a FLUID-ONLY constraint (setFluidOnlyConstraint): close every face with a
        // solid-CENTERED side in the openness the pressure stack consumes. The aperture operator,
        // the divergence, the face correction and the MG rediscretization all read these fields,
        // so one filter makes constraint/operator/correction consistent by construction: pressure
        // DOFs decouple at solid-centered cells (their rows go empty like solid cells), the
        // invisible multiplier subspace of collocated_invisible_subspace.md S4 ceases to exist,
        // and the operator stays SPD + 7-point (CG + CutcellMG untouched). Closure quality is
        // Neumann-zero at the closed faces (the crude end of the fluid-only family -- measured,
        // not assumed); the consistent-closure variants ride on the gp row machinery instead.
        CCConst sd = CCConst(sdf_);
        CCField oa[3] = {ox_, oy_, oz_};
        C3 e = e_;
        for (int a = 0; a < 3; ++a) {
          CCField o = oa[a];
          const long sa = (a == 0) ? 1 : (a == 1) ? (long)e.x : (long)e.x * e.y;
          Kokkos::parallel_for(
              "peclet::flow::fluid_only_openness",
              Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {1, 1, 1},
                                                             {e.x, e.y, e.z}),
              KOKKOS_LAMBDA(int x, int y, int z) {
                const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
                if (sd(i) < 0.0 || sd(i - sa) < 0.0)
                  o(i) = 0.0;
              });
        }
      }
#ifdef PECLET_FLOW_MPI
      // openness ghosts (the operator + divergence read the +neighbour face) -> exchange across
      // ranks
      if (distributed_) {
        velDev_->exchange(ox_);
        velDev_->exchange(oy_);
        velDev_->exchange(oz_);
      }
#endif
      if (hasBc_) {  // FLUX openness (beta): a face is OPEN only where it carries normal flux --
                     // outflow, or
        B3 e2{e_.x, e_.y, e_.z};
        CCField oa[3] = {ox_, oy_, oz_};  // an inflow with nonzero normal velocity. Walls
        for (int a = 0; a < 3; ++a)
          for (int s = 0; s < 2; ++s) {    // and tangential-only Dirichlet faces (e.g. a
            const int t = bc_[2 * a + s];  // lid: type 2 with zero normal vel) are CLOSED.
            const bool open = (t == 3) || (t == 2 && (bcProf_[2 * a + s].extent(0) > 0 ||
                                                      std::fabs(bcVel_[2 * a + s][a]) > 1e-12));
            if (t != 0 && !open && touchesGlobalFace(2 * a + s))
              bcZeroOpenness(oa[a], e2, G, a, s);  // rank-owned global face only
          }
      }  // the MG re-derives the OPERATOR openness alpha (inflow Neumann -> closed) per level via
         // setBC.
      copyInner(ox1_, e1_, 1, CCConst(ox_), e_, G);  // bridge openness g=2 -> g=1 for the MG
      copyInner(oy1_, e1_, 1, CCConst(oy_), e_, G);
      copyInner(oz1_, e1_, 1, CCConst(oz_), e_, G);
      if (fluidOnlyMode_ == 2) {
        // Design B: the MG hierarchy is built from the FILTERED openness (Design A's operator,
        // the symmetric surrogate preconditioner + the 7-point part of the true operator); the
        // geometric ox_/oy_/oz_ stay ORIGINAL for the divergence and the face correction. Filter
        // the g=1 bridge in place, then build the star overlay from the original apertures.
        if (porous_ || varRho_ || hasBc_ || ghostProjection_ || distributed_ || !Grid::collocated)
          throw std::runtime_error(
              "set_fluid_only_constraint(2): v1 is single-rank periodic collocated only");
        CCExec space;
        CCConst sd = CCConst(sdf_);
        CCField oa1[3] = {ox1_, oy1_, oz1_};
        const C3 e1 = e1_, e2 = e_;
        for (int a = 0; a < 3; ++a) {
          CCField o1 = oa1[a];
          const long sa2 = (a == 0) ? 1 : (a == 1) ? (long)e2.x : (long)e2.x * e2.y;
          Kokkos::parallel_for(
              "peclet::flow::star_filter_bridge",
              Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx_, ny_, nz_}),
              KOKKOS_LAMBDA(int x, int y, int z) {
                const long i2 =
                    (long)(x + G) + (long)(y + G) * e2.x + (long)(z + G) * (long)e2.x * e2.y;
                if (sd(i2) < 0.0 || sd(i2 - sa2) < 0.0)
                  o1((long)(x + 1) + (long)(y + 1) * e1.x + (long)(z + 1) * (long)e1.x * e1.y) =
                      0.0;
              });
        }
        space.fence();
        starCounter_ = Kokkos::View<int, CCMem>("star_counter");
        const C3 nn{nx_, ny_, nz_};
        nStar_ = buildStarOverlay(CCConst(sdf_), CCConst(ox_), CCConst(oy_), CCConst(oz_), e_, G,
                                  nn, StarOverlay{}, starCounter_);
        starOv_ = starMakeOverlay(std::max(nStar_, 1));
        buildStarOverlay(CCConst(sdf_), CCConst(ox_), CCConst(oy_), CCConst(oz_), e_, G, nn,
                         starOv_, starCounter_);
      }
      if (ghostProjection_) {
        // Directional ghost-cell projection: build the closure overlay + the binary (COUPLED)
        // openness. The binary field replaces the geometric openness on the MG rails (the MG
        // hierarchy becomes the symmetric surrogate preconditioner; the overlay delta enters only
        // the fine-level BiCGStab matvec). The geometric ox_/oy_/oz_ above stay for diagnostics.
        if (porous_ || varRho_ || hasBc_)
          throw std::runtime_error(
              "ghost projection: incompatible with porous/variable-rho/domain-BC (v1)");
        const std::size_t nInner = (std::size_t)nx_ * ny_ * nz_;
        gpOv_ = gpMakeOverlay((long)nInner);  // worst-case sizing, like the momentum overlay
        gpIdMap_ = Kokkos::View<int*, CCMem>("gp_idmap", nInner);
        gpCounter_ = Kokkos::View<int, CCMem>("gp_counter");
        oxb_ = CCField("oxb", n_);
        oyb_ = CCField("oyb", n_);
        ozb_ = CCField("ozb", n_);
        gpRh_ = CCField("gpRh", n1_);
        gpT_ = CCField("gpT", n1_);
        gpZ2_ = CCField("gpZ2", n1_);
        if (distributed_)
          gpX2_ = CCField("gpXg2", n_);  // g=2 staging block for the distributed BiCGStab matvec
        // Fragmentation guard: the binary COUPLED-face condition is stricter than aperture
        // connectivity, so tight-throat geometries (e.g. a random close packing with touching
        // spheres) fragment the fluid graph into a main component + tiny pockets at the
        // contacts. Each pocket adds its own null vector that the single global mean-removal
        // cannot handle, and BiCGStab breaks down (measured: fields to ~1e152 on the RCP
        // example). Host BFS over the coupled graph of the INNER sdf; fluid cells outside the
        // largest component are treated as SOLID for the PROJECTION ONLY (sdfGp), decoupling
        // their rows; the momentum step keeps the true sdf.
        // Distributed: connectivity is a GLOBAL property (a pocket can span rank boundaries) and
        // every rank must agree on the main component, so allgather the inner sdf, run the
        // deterministic guard on the global grid identically on every rank, and keep this
        // rank's block of the result.
        std::vector<double> sdfGpHost;
        {
          std::vector<double> work;
          int fx = nx_, fy = ny_, fz = nz_;
          bool verbose = true;
#ifdef PECLET_FLOW_MPI
          int myRank = 0;
          if (distributed_) {
            MPI_Comm_rank(comm_, &myRank);
            int nRanks = 1;
            MPI_Comm_size(comm_, &nRanks);
            fx = gnx_;
            fy = gny_;
            fz = gnz_;
            verbose = myRank == 0;
            std::vector<int> cnts(nRanks), disp(nRanks);
            long acc = 0;
            for (int r = 0; r < nRanks; ++r) {
              const auto b = dec_->block(r);
              cnts[r] = (int)(b.size[0] * b.size[1] * b.size[2]);
              disp[r] = (int)acc;
              acc += cnts[r];
            }
            std::vector<double> flat((std::size_t)acc);
            MPI_Allgatherv(sdfInner_().data(), (int)sdfInner_().size(), MPI_DOUBLE, flat.data(),
                           cnts.data(), disp.data(), MPI_DOUBLE, comm_);
            work.assign((std::size_t)fx * fy * fz, 0.0);
            for (int r = 0; r < nRanks; ++r) {
              const auto b = dec_->block(r);
              const double* src = flat.data() + disp[r];
              for (int z = 0; z < (int)b.size[2]; ++z)
                for (int y = 0; y < (int)b.size[1]; ++y)
                  for (int x = 0; x < (int)b.size[0]; ++x)
                    work[(std::size_t)(x + b.origin[0]) + (std::size_t)(y + b.origin[1]) * fx +
                         (std::size_t)(z + b.origin[2]) * (std::size_t)fx * fy] =
                        src[(std::size_t)x + (std::size_t)y * b.size[0] +
                            (std::size_t)z * (std::size_t)b.size[0] * b.size[1]];
            }
          } else
#endif
            work = sdfInner_();
          const std::size_t nTot = work.size();
          const int nx = fx, ny = fy, nz = fz;
          auto id = [&](int x, int y, int z) {
            return (std::size_t)((x + nx) % nx) + (std::size_t)((y + ny) % ny) * nx +
                   (std::size_t)((z + nz) % nz) * (std::size_t)nx * ny;
          };
          std::vector<int> comp(nTot, -1);
          std::vector<std::size_t> stack;
          int ncomp = 0, mainComp = -1;
          std::size_t mainSize = 0, nActive = 0;
          for (std::size_t seed = 0; seed < nTot; ++seed) {
            if (comp[seed] >= 0 || work[seed] < 0.0)
              continue;
            std::size_t size = 0;
            comp[seed] = ncomp;
            stack.assign(1, seed);
            while (!stack.empty()) {
              const std::size_t c = stack.back();
              stack.pop_back();
              ++size;
              const int x = (int)(c % nx), y = (int)((c / nx) % ny),
                        z = (int)(c / ((std::size_t)nx * ny));
              const int nb[6][3] = {{x - 1, y, z}, {x + 1, y, z}, {x, y - 1, z},
                                    {x, y + 1, z}, {x, y, z - 1}, {x, y, z + 1}};
              for (auto& q : nb) {
                const std::size_t j = id(q[0], q[1], q[2]);
                if (comp[j] >= 0 || work[j] < 0.0)
                  continue;
                // COUPLED face: mean-of-centers face sdf fluid AND both centers fluid
                if (0.5 * (work[c] + work[j]) < 0.0)
                  continue;
                comp[j] = ncomp;
                stack.push_back(j);
              }
            }
            if (size > mainSize) {
              mainSize = size;
              mainComp = ncomp;
            }
            nActive += size;
            ++ncomp;
          }
          if (ncomp > 1) {
            std::size_t pockets = 0;
            for (std::size_t i = 0; i < nTot; ++i)
              if (work[i] >= 0.0 && comp[i] != mainComp) {
                work[i] = -(std::abs(work[i]) * 1.001 + 1e-30);
                ++pockets;
              }
            if (verbose)
              printf("peclet::flow ghost projection: %d fluid components; decoupled %zu pocket "
                     "cells outside the main component (%zu of %zu fluid cells)\n",
                     ncomp, pockets, mainSize, nActive);
          }
#ifdef PECLET_FLOW_MPI
          if (distributed_) {
            const auto b = dec_->block(myRank);
            sdfGpHost.resize(sdfInner_().size());
            for (int z = 0; z < nz_; ++z)
              for (int y = 0; y < ny_; ++y)
                for (int x = 0; x < nx_; ++x)
                  sdfGpHost[(std::size_t)x + (std::size_t)y * nx_ +
                            (std::size_t)z * (std::size_t)nx_ * ny_] =
                      work[(std::size_t)(x + b.origin[0]) + (std::size_t)(y + b.origin[1]) * fx +
                           (std::size_t)(z + b.origin[2]) * (std::size_t)fx * fy];
          } else
#endif
            sdfGpHost = std::move(work);
        }
        sdfGp_ = CCField("peclet::flow::sdfGp", n_);
        CCField sdfGp = sdfGp_;  // the projection's sdf view (pockets decoupled); persisted for
                                 // the collocated gpCenterGrad predictor/correction
#ifdef PECLET_FLOW_MPI
        if (distributed_) {
          // local inner block + halo exchange (cross-rank + periodic), same as the sdf_ upload:
          // the overlay build reads sdfGp ghosts up to +/-2 = G across block boundaries.
          auto h = Kokkos::create_mirror_view(sdfGp_);
          Kokkos::deep_copy(h, sdfGp_);
          for (int z = 0; z < nz_; ++z)
            for (int y = 0; y < ny_; ++y)
              for (int x = 0; x < nx_; ++x)
                h((long)(x + G) + (long)(y + G) * e_.x + (long)(z + G) * (long)e_.x * e_.y) =
                    sdfGpHost[(std::size_t)x + (std::size_t)y * nx_ +
                              (std::size_t)z * (std::size_t)nx_ * ny_];
          Kokkos::deep_copy(sdfGp_, h);
          velDev_->exchange(sdfGp_);
        } else
#endif
        {  // upload + periodic wrap (same pattern as the sdf upload above)
          CCField din("peclet::flow::sdfGpInner_d", nInner);
          Kokkos::deep_copy(din, Kokkos::View<const double*, Kokkos::HostSpace,
                                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                                     sdfGpHost.data(), sdfGpHost.size()));
          CCExec space;
          const int ex = e_.x, ey = e_.y, ez = e_.z, nx = nx_, ny = ny_, nz = nz_, g = G;
          Kokkos::parallel_for(
              "peclet::flow::sdfgp_wrap",
              Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {ex, ey, ez}),
              KOKKOS_LAMBDA(int x, int y, int z) {
                const int ix = (((x - g) % nx) + nx) % nx, iy = (((y - g) % ny) + ny) % ny,
                          iz = (((z - g) % nz) + nz) % nz;
                sdfGp((long)x + (long)y * ex + (long)z * (long)ex * ey) =
                    din((std::size_t)ix + (std::size_t)iy * nx +
                        (std::size_t)iz * (std::size_t)nx * ny);
              });
          space.fence();
        }
        gpBinaryOpenness(oxb_, oyb_, ozb_, CCConst(sdfGp), e_);
        gpNRows_ = buildGpOverlay(CCConst(sdfGp), e_, G, C3{nx_, ny_, nz_}, gpOv_, gpIdMap_,
                                  gpCounter_, gpMatrixOrder_, gpRhsOrder_,
                                  hasExactCross_ ? CCConst(tEx_[0][0]) : CCConst(),
                                  hasExactCross_ ? CCConst(tEx_[1][1]) : CCConst(),
                                  hasExactCross_ ? CCConst(tEx_[2][2]) : CCConst(),
                                  /*useGhost=*/distributed_);
        if (gpDebugLevel() > 0) {  // PECLET_FLOW_GP_DEBUG row forensics (analysis only)
          int gpDbgRank = 0;
#ifdef PECLET_FLOW_MPI
          if (distributed_)
            MPI_Comm_rank(comm_, &gpDbgRank);
#endif
          gpDebugReport(gpOv_, gpNRows_, C3{nx_, ny_, nz_}, gpIdMap_, gpDbgRank);
        }
        copyInner(ox1_, e1_, 1, CCConst(oxb_), e_, G);  // MG surrogate = binary openness
        copyInner(oy1_, e1_, 1, CCConst(oyb_), e_, G);
        copyInner(oz1_, e1_, 1, CCConst(ozb_), e_, G);
      }
      mg_.setBoundaryConditions(bc_);  // per-level wall openness + null-space gating (no-op if
                                       // periodic); BEFORE initMpi — the per-level ghost width
                                       // (CA smoothing) is chosen for the periodic operator only
#ifdef PECLET_FLOW_MPI
      if (distributed_)  // share the level-0 decomposition so the MG block matches this rank's
                         // block
        mg_.initMpi(gnx_, gny_, gnz_, nLevels_, comm_, dec_.get());
      else
#endif
        mg_.init(nx_, ny_, nz_,
                 nLevels_);  // geometric multigrid on the cut-cell openness (MG-PCG pressure)
      mg_.setOpenness(CCConst(ox1_), CCConst(oy1_), CCConst(oz1_), 1.0, 1.0, 1.0);
      // Coarse-solve policy: an explicit set_pressure_graph_amg(True) forces agglomeration,
      // otherwise the mode set by set_pressure_bottom (default auto) decides.
      mg_.setAgglomerationMode(pressGraphAmg_ ? 1 : pressAgglomMode_);
      Kokkos::deep_copy(phi_, 0.0);
      Kokkos::deep_copy(P_, 0.0);
    }
    // Rung V5a (WO-Q): the colour block's cut-cell geometry is derived from the openness and the
    // SDF that were just rebuilt, so set_solid AFTER enable_vof must rebuild it — the same reason
    // initMpi rebuilds the block. Inert (and byte-identical) when VoF is off.
    if (vofEnabled_)
      buildVofBlock();
  }

  void step() {
    const double ts0 = phaseTick();
    tPredictor_ = tMomentum_ = tProjection_ = 0.0;
    lastMomentumSweeps_ = 0;
    lastMomentumResid_ = -1.0;
    mg_.resetAllreduceCounters();
    // Momentum-consistent geometric VoF (rung V2b, WO-K): the colour field and rho^c u_c are
    // advanced TOGETHER, by the same fluxes, at the head of the step — the momentum advection has
    // to precede the predictor that consumes it. The advecting field is u^n, the previous step's
    // projected (discretely divergence-free) output. No-op unless enable_vof_momentum ran; when it
    // has NOT, the colour advection keeps WO-J's slot at the bottom of the step and this path is
    // byte-identical to V2a. See enableVofMomentum().
    advectVofMomentum();
    // Phase change (rungs P0/P1, WO-P01): mdot + the PLIC areas/normals from (C^n, T^n), the
    // divergence source deposit read by project(), and the interface regression applied to the
    // SAME C^n the planes came from. No-op (byte-identical) unless enable_phase_change ran.
    phaseChangeStep();
    // Multiphysics: refresh material properties / body forces from the current fields (frozen over
    // the step). No-op (byte-identical) when no closure is registered.
    updateProperties();
    // WO-G: give the per-cell body force the ghost ring its consumer assumes. This is the ONE point
    // in the step that is after BOTH writers (the closures just above; an external CFD-DEM
    // field_view/exchange_field_add deposit, which happens before step()) and before every consumer
    // (buildRhsVar in the Picard loop). Inert when no force field is registered.
    fillCellForceGhosts();
    // WO-I: and the same for the per-cell drag coefficient, for the same reason one phase earlier.
    // Same call site, same justification: after BOTH writers, before the FIRST consumer (the
    // momentum stencil builds just below). See fillDragBetaGhosts().
    fillDragBetaGhosts();
    // Rung V4 (WO-P): the interface curvature of the colour field this step will run with, and the
    // explicit capillary stability check. No-op (byte-identical) unless set_surface_tension ran.
    updateVofCurvature();
    // eps-conservative porous momentum: the volume-averaged time term is (eps_f rho/dt) u, i.e.
    // the variable-density machinery with the effective density rho_eff = eps*rho, refreshed from
    // the just-deposited eps every step (eps ghosts are already filled by the coupling driver, so
    // the whole-block product has valid ghosts). Without this weight the plain-u momentum lets the
    // projection drag gas along with the moving porosity at zero inertia cost — a spurious energy
    // source that pumps the particles through the drag (measured in the HCS benchmark).
    if (porous_ && porousCons_)
      updateEpsRho();
    // Variable properties / implicit drag: rebuild the diffusion stencil from the current mu/rho
    // and drag_beta fields (the implicit-FOU path rebuilds it per Picard in buildAdvStencil*, so
    // only the non-advective path needs this).
    if (((varProps_ || varRho_ || hasDrag_ || (porous_ && porousCons_)) || dtDirty_) &&
        !implicitAdv())
      rebuildStencils();
    dtDirty_ = false;  // implicit-FOU rebuilds per Picard below (reads dt_ live) — clear either way
    // u^n time base, fixed for the whole step (Picard lags the advecting velocity at u^k, not the
    // base).
    for (int c = 0; c < 3; ++c)
      Kokkos::deep_copy(old_[c], C[c].u);
    if (cutcellPressure_ && incremental_) {
      fillGhosts(P_);
      if (hasBc_)
        pressureBcGhost();
    }  // grad(P^n) for the incremental predictor (once)
    lastOuterIters_ = 0;
    for (int outer = 0; outer < outerIters_; ++outer) {
      const double tp0 = phaseTick();
      lastOuterIters_ = outer + 1;
      if (outerTol_ > 0)
        for (int c = 0; c < 3; ++c)
          Kokkos::deep_copy(prev_[c], C[c].u);
      if (advect_ || hasBc_ || (Grid::collocated && faceInterp_ >= 4 && faceInterp_ <= 7))
        for (int c = 0; c < 3; ++c)
          fillVelGhosts(c,
                        0);  // explicit ghosts (periodic + BC) for advect / mode-4 FV defect matvec
      // A0 (advective cut-wall flux): build the advection's wall-aware velocity inputs. AFTER the
      // ghost exchange, over the extended block, so ghost solid rows carry the wall velocity too.
      // No-op -- and no allocation -- unless a scene instance is moving; static scenes keep reading
      // C[*].u byte for byte. See buildAdvInputs().
      buildAdvInputs();
      // Porous advection-form compensation: the Koren/SOU/FOU advection operators are CONSERVATIVE
      // (flux form, ∇·(u u)), which equals the true advective transport u·∇u only for a solenoidal
      // advecting field. Under the volume-averaged continuity div(eps u)=0 the plain divergence
      // div(u) = -(1/eps) u·grad(eps) != 0, and the flux form silently adds the spurious force
      // +u(div u) — largest where grad(eps) is large (clusters), where it pumps particle kinetic
      // energy through the drag with no physical source (measured: HCS variance rising ~x30 past
      // the clustering plateau). Compensate by subtracting u_f·div(u)_f from the advection in the
      // RHS (the exact identity u·∇u = ∇·(uu) − u∇·u; div(u) at the face = mean of the two cell
      // divergences). Gated on porous_ so every other path is byte-identical.
      if (porous_ && advect_)
        computeDivAdv();
      // rung V8 (WO-T): on the collocated grid with variable density and/or surface tension the
      // predictor carries NO force at all — every force is a face acceleration added after
      // centerToFace (collocated_varrho.hpp). `colocatedFaceForce()` is false on the staggered path
      // and on every validated constant-density collocated path, so this dispatch is inert there.
      const bool coloFF = colocatedFaceForce();
      for (int c = 0; c < 3; ++c)  // RHS from u^n base + advection lagged at u^k
        coloFF ? buildRhsColoFF(c)
               : (vofMomEnabled_
                      ? buildRhsVarMom(c)
                      : (effVarRho() ? buildRhsVar(c)
                                     : (hasCellForce_ ? buildRhsForced(c) : buildRhs(c))));
      // Rung V4 (WO-P): the balanced-force CSF, ADDED to whichever RHS the configuration built --
      // at the same place, and with the same face difference, as the incremental -grad(P^n) just
      // above it. Independent of the time term and the advection form, hence additive rather than a
      // fifth RHS kernel. Gated: byte-identical when surface tension is off.
      if (csfActive() && !coloFF)
        for (int c = 0; c < 3; ++c)
          csfMode_ == 0 ? addCsfRhs(c) : addCsfRhsCellInterp(c);
      // Implicit-FOU: rebuild the IBM velocity stencil = backward-Euler diffusion + rho*FOU(u^k),
      // then re-apply the cut-cell bake. Per Picard iteration (advecting velocity changes). Applies
      // to the IBM (periodic/porous) path when the user opts in, AND ALWAYS to the domain-BC
      // stencil path (inflow/outflow) -- implicitAdv() -> fully-implicit upwind advection (stable
      // at large dt). The velocity-MG BC path keeps its own FOU coarse operator.
      if (implicitAdv() && (!hasBc_ || !useVelocityMg_ || mixedVelocityMg()))
        for (int c = 0; c < 3; ++c)
          (varProps_ || effVarRho()) ? buildAdvStencilVar(c) : buildAdvStencil(c);
      // Outflow backflow stabilization: dissipate reverse flow at the outlet in the momentum
      // operator used by the domain-BC stencil smoother (prevents backflow divergence). Inert
      // without reversal.
      if (bcStencilPath() && backflowBeta_ > 0.0 && hasOutflow_)
        for (int c = 0; c < 3; ++c)
          applyBackflowStab(c);
      // upwind-convective velocity-MG: restrict the (frozen u^k) advecting velocity to the coarse
      // levels ONCE, before the per-component solves update it (shared across the 3 momentum
      // components).
      if (useVelocityMg_ && advect_ &&
          ((implicitFou_ && !hasBc_) || (mixedVelocityMg() && implicitAdv())))
        vmg_.restrictAdvVelocities(advVelView(0), advVelView(1),
                                   advVelView(2));  // A0: same inputs as the fine operator
      const double tp1 = phaseTick();
      tPredictor_ += tp1 - tp0;
      for (int c = 0; c < 3; ++c)
        smoothComp(c);  // per-component IBM implicit-diffusion solve
      // Route (b): stash u* for the reaction-force budget. Every Picard iteration overwrites, so
      // what survives is the LAST momentum solve -- the one whose viscous fluxes, together with
      // the last projection, actually produced u^{n+1} (the time base is u^n for every iteration,
      // so earlier iterations leave no trace in the final state except through P_). A pure
      // deep_copy: no numerical effect on any solve.
      if constexpr (!Grid::collocated) {
        if (hasScene_) {
          for (int c = 0; c < 3; ++c) {
            if (uStar_[c].extent(0) != n_)
              uStar_[c] = CCField("uStar", n_);
            Kokkos::deep_copy(uStar_[c], C[c].u);
          }
          haveUStar_ = true;
        }
      }
      const double tp2 = phaseTick();
      tMomentum_ += tp2 - tp1;
      // The porous (volume-averaged) projection lives entirely on the cut-cell operator rails
      // (divergOpenEps + buildPorousCoeff* into CutcellMG). Without set_solid /
      // set_pressure_geometry there is NO projection at all — the gas would never accelerate to
      // the interstitial velocity in a bed and the drag comes out ~5x too weak (a fluidized bed
      // quietly refuses to fluidize). Fail loudly instead of silently dropping the constraint.
      if (porous_ && !cutcellPressure_)
        throw std::runtime_error(
            "set_porous_continuity(True) requires the cut-cell pressure operator: call "
            "set_solid(...) or set_pressure_geometry(all-fluid SDF) before stepping (a "
            "domain-BC-only box otherwise runs with NO continuity constraint at all)");
      if (cutcellPressure_)
        project();  // cut-cell projection -> incompressible
      tProjection_ += phaseTick() - tp2;
      if (hasBc_)
        for (int c = 0; c < 3; ++c)
          applyVelocityBcComp(c, 0, false);  // re-impose domain BCs (keep outflow)
      if (outerTol_ > 0) {  // outer convergence: max velocity change over this Picard iteration
        double corr = 0.0;
        for (int c = 0; c < 3; ++c)
          corr = Kokkos::fmax(corr, maxAbsDiffInner(CCConst(C[c].u), CCConst(prev_[c])));
        lastOuterCorr_ = corr;
        if (corr < outerTol_)
          break;
      }
    }
    // Geometric VoF (rung V2a): advance the colour field with the just-projected, discretely
    // divergence-free face velocities — the SAME slot, and the same justification, as
    // advanceScalars() below (Weymouth-Yue's exact conservation is conditioned on the advecting
    // field's discrete divergence; see advectVof()). No-op (byte-identical) when VoF is off.
    // Under momentum consistency (V2b) the colour was already advanced at the head of the step,
    // together with the momentum it shares its fluxes with — see advectVofMomentum().
    if (!vofMomEnabled_)
      advectVof();
    // WO-P01: refresh the energy scalar's per-cell Dirichlet set from the colour the energy solve
    // is about to see. No-op unless the thermal mass flux is on.
    pcUpdateThermalMask();
    // Segregated multiphysics: advance any transported scalars with the just-projected
    // divergence-free velocity (properties frozen over the step). No-op (byte-identical) when no
    // scalar is registered.
    advanceScalars();
    tStep_ = phaseTick() - ts0;
  }

  // velocity component c (0=u,1=v,2=w) on the inner cells, flat x-fastest [nx*ny*nz].
  std::vector<double> getVelocity(int c) { return gatherInner(C[c].u); }
  /// Write a component's inner velocity from a host vector (x-fastest, inner region) and
  /// re-impose the solid mask. An initial-condition hook (e.g. a uniform stream around a fixed
  /// body — the Galilean twin of a towed one); u^n is taken from the live field at step start.
  void setVelocity(int c, const std::vector<double>& v) {
    scatterInner(C[c].u, v);
    maskVelocity(c);
  }
  // The divergence-free FACE velocity component (collocated: the projected MAC face field
  // uf_/vf_/wf_, exactly div-free; staggered: C[c].u already lives on the faces). For a periodic
  // bed its mean is the momentum-balance superficial velocity, unperturbed by the openness-aware
  // cell gradient correction (projectCorrectCenter) that biases the cell-field mean at cut cells.
  std::vector<double> getFaceVelocity(int c) {
    if constexpr (Grid::collocated) {
      CCField fa[3] = {uf_, vf_, wf_};
      return gatherInner(fa[c]);
    } else {
      return gatherInner(C[c].u);
    }
  }
  // TEMP DIAGNOSTIC: the face openness (fluid area fraction) used by the cut-cell projection.
  // component c: 0 -> ox_ (low -x face of each inner cell), 1 -> oy_, 2 -> oz_. Grid-independent
  // (built once from the SDF). Exposed to compare the open-weighted superficial flux against the
  // raw velocity mean.
  std::vector<double> getOpenness(int c) {
    CCField o[3] = {ox_, oy_, oz_};
    return gatherInner(o[c]);
  }
  // Diagnostic read-out of the ASSEMBLED momentum-operator diagonal of component c — the float
  // stencil `AC` after the diffusion build, the Robust-Scaled cut-cell bake and, under implicit
  // drag, `addDragDiagonal`'s face drag beta_f — as an x-fastest (nx,ny,nz) inner-region host
  // buffer. Read-only; no solver state is touched. Added for WO-I's
  // `tests/kokkos_mpi/test_dragbeta_ghost_mpi.cpp`, which must gate the FACE drag mean where it is
  // formed: on a periodic box the projection homogenizes a single bad plane into a uniform mean
  // shift of the velocity, so a velocity-only gate sees THAT the drag was wrong but not WHERE.
  std::vector<double> getMomentumDiagonal(int c) {
    auto h = Kokkos::create_mirror_view(C[c].AC);
    Kokkos::deep_copy(h, C[c].AC);
    std::vector<double> out((std::size_t)nx_ * ny_ * nz_);
    for (int z = 0; z < nz_; ++z)
      for (int y = 0; y < ny_; ++y)
        for (int x = 0; x < nx_; ++x)
          out[(std::size_t)x + (std::size_t)y * nx_ + (std::size_t)z * (std::size_t)nx_ * ny_] =
              (double)h((long)(x + G) + (long)(y + G) * e_.x + (long)(z + G) * (long)e_.x * e_.y);
    return out;
  }
  // The openness whose face fluxes the PROJECTION conserves: the binary (COUPLED) openness in
  // ghost-projection mode (oxb_ — the geometric ox_ stays a diagnostic there), the geometric
  // cut-cell openness otherwise. This is what flux bookkeeping downstream of the solve must use
  // (e.g. peclet.pnm's extract_network_flow): sum(o_proj*u*A) over a cell's faces IS the
  // discrete divergence the projection drives to zero.
  std::vector<double> getOpennessProj(int c) {
    const bool gp = ghostProjection_ && oxb_.extent(0) > 0;
    CCField o[3] = {gp ? oxb_ : ox_, gp ? oyb_ : oy_, gp ? ozb_ : oz_};
    return gatherInner(o[c]);
  }
  std::vector<double> getPressure() {
    // Incremental scheme: P_ accumulates the physical pressure. Classical Chorin (!incremental_):
    // derive it on demand from the last projection potential, p = (rho/dt)*phi (CUDA
    // press_from_phi_k).
    if (incremental_)
      return gatherInner(P_);
    std::vector<double> out = gatherInner(phi_);
    const double ct = rho_ / dt_;
    for (double& x : out)
      x *= ct;
    return out;
  }
  // WO-R: the divergence of the field the projection ACTUALLY produced, outflow correction
  // included. `maxOpenDivergence()` below re-imposes the zero-gradient outflow face before
  // measuring (its own comment says so) — which both destroys `bcCorrectOutflow`'s correction as a
  // side effect and reports the divergence of a field the solver never used. On an open-boundary
  // two-phase box that artefact is the dominant number: measured 5e-3, flat in the iteration
  // count, flat in the density ratio and bit-identical in a `-DPECLET_FLOW_MREAL_DOUBLE` build —
  // i.e. not a solver residual at all. This sibling fills the ghosts with `doOutflow = false`,
  // exactly as `step()` does after `project()`, and leaves the velocity field alone.
  //
  // Kept as a SIBLING rather than a change of default: every recorded open-boundary number in the
  // repo was taken with the mutating one, and re-baselining them is not this work order's call.
  double maxOpenDivergenceProjected() {
    if (!cutcellPressure_)
      return 0.0;
    if constexpr (Grid::collocated)
      return maxOpenDivergence();  // the collocated branch already measures the face field
    for (int c = 0; c < 3; ++c)
      fillVelGhostsKeepOutflow(c);
    divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
               CCConst(oz_), div_, e_, G);
    addWallFluxDivergence(div_);
    double m = reduceMaxAbsInner(CCConst(div_));
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      double g = 0;
      MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MAX, comm_);
      return g;
    }
#endif
    return m;
  }
  double maxOpenDivergence() {
    if (!cutcellPressure_)
      return 0.0;
    if constexpr (Grid::collocated) {
      // Report the residual of the PROJECTED face field uf_ (made divergence-free by project(),
      // ghosts filled). Re-averaging the central-difference-corrected CELL field would instead show
      // the inherent O(h^2) approximate-projection cell divergence -- a property of the scheme, not
      // the solver residual. At an outflow, re-impose the zero-gradient face (matching the
      // staggered diagnostic, whose fillVelGhosts overwrites the mass-conserving outflow
      // correction): the operator zeroes the alpha-divergence, but the raw beta-divergence at the
      // open-boundary corner is otherwise spurious.
      if (hasOutflow_) {
        B3 e{e_.x, e_.y, e_.z};
        CCField fa[3] = {uf_, vf_, wf_};
        for (int a = 0; a < 3; ++a)
          if (bc_[2 * a + 1] == 3 && touchesGlobalFace(2 * a + 1))
            bcNeumannGhost(fa[a], e, G, a, 1);
      }
      if (ghostProjection_ && gpNRows_ >= 0) {
        // Ghost mode: the closed point divergence of the projected face field (same kernel pair
        // as the RHS) — the mode's true residual.
        divergOpen(CCConst(uf_), CCConst(vf_), CCConst(wf_), CCConst(oxb_), CCConst(oyb_),
                   CCConst(ozb_), div_, e_, G);
        gpDivergDelta(div_, CCConst(uf_), CCConst(vf_), CCConst(wf_), gpOv_, gpNRows_,
                      C3{nx_, ny_, nz_}, e_, G, distributed_);
      } else
        divergOpen(CCConst(uf_), CCConst(vf_), CCConst(wf_), CCConst(ox_), CCConst(oy_),
                   CCConst(oz_), div_, e_, G);
    } else {
      for (int c = 0; c < 3; ++c)
        fillVelGhosts(c, 0);  // ghosts incl. outflow zero-gradient before the divergence
      if (ghostProjection_ && gpNRows_ >= 0) {
        // Ghost mode: the closed point divergence (same kernels as the RHS) IS the true residual
        // of the mode. (EXPLICIT sliver faces read the corrected stored value here vs u* in the
        // RHS — the only, and rare, departure from the exact identity.)
        divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(oxb_), CCConst(oyb_),
                   CCConst(ozb_), div_, e_, G);
        gpDivergDelta(div_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), gpOv_, gpNRows_,
                      C3{nx_, ny_, nz_}, e_, G, distributed_);
      } else
        divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
                   CCConst(oz_), div_, e_, G);
    }
    // The diagnostic must measure the residual of the constraint the projection actually
    // solved, so it carries the same wall-flux source (rung 3). Without this the moving case
    // would report a "divergence error" that is really the wall flux the solve balances.
    addWallFluxDivergence(div_);
    double m = reduceMaxAbsInner(CCConst(div_));
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      double g = 0;
      MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MAX, comm_);
      return g;
    }
#endif
    return m;
  }
  // Residual of the volume-averaged continuity, max|div(open*eps*u) + d(eps)/dt| — the quantity the
  // porous projection actually drives to zero (NOT the velocity divergence, which is -d(eps)/dt !=
  // 0 in a fluidizing bed). Meaningful only with set_porous_continuity(True); returns 0 otherwise.
  double maxPorousResidual() {
    if (!porous_ || !cutcellPressure_)
      return 0.0;
    for (int c = 0; c < 3; ++c)
      fillVelGhosts(c, 0);
    fillPorousEpsGhosts();  // the SAME eps ghost policy the projection used (the coupling deposit
                            // rewrites the ghosts between project() and this diagnostic)
    divergOpenEps(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
                  CCConst(oz_), CCConst(epsField_), div_, e_, G);
    {  // add back the SAME d(eps)/dt source the projection used (depsdt_ from the last project())
      CCExec space;
      C3 e = e_;  // local copy — capturing e_ in the KOKKOS_LAMBDA would read this-> on the device
      CCField d = div_, dd = depsdt_;
      const bool useDt = porousDepsDt_;
      using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
      Kokkos::parallel_for(
          "peclet::flow::porous_resid", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = (long)x + (long)y * e.x + (long)z * e.x * e.y;
            if (useDt)
              d(i) += dd(i);  // residual of the SAME constraint the projection solved
          });
    }
    double m = reduceMaxAbsInner(CCConst(div_));
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      double g = 0;
      MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MAX, comm_);
      return g;
    }
#endif
    return m;
  }
  long lastPressureIterations() const { return lastPressureIters_; }
  // Per-phase wall times of the last step() in seconds, THIS RANK (device-fenced at each phase
  // boundary): predictor = ghost fills + RHS/advection/stencil builds, momentum = the per-component
  // implicit-diffusion solves, projection = the cut-cell pressure projection; step = the whole
  // step() (remainder = BC re-imposition, Picard bookkeeping, scalars). The allreduce pair is the
  // pressure solve's global-reduction tax (time in / count of MPI_Allreduce; 0 single-rank).
  double lastStepSeconds() const { return tStep_; }
  double lastPredictorSeconds() const { return tPredictor_; }
  double lastMomentumSeconds() const { return tMomentum_; }
  double lastProjectionSeconds() const { return tProjection_; }
  double lastPressureAllreduceSeconds() const { return mg_.allreduceSeconds(); }
  long lastPressureAllreduceCount() const { return mg_.allreduceCount(); }
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }

 private:
  struct Comp {
    CCField u, b, inhom, rscale, mask;
    FV AC, AW, AE, AS, AN, AB, AT;
    IbmOverlay ov;
    Kokkos::View<int*, CCMem> idMap;
    Kokkos::View<int, CCMem> counter;
    int nCut = 0;
  };

 public:  // nvcc forbids extended __host__ __device__ lambdas inside private/protected members.
  // Advection treated implicitly (implicit-FOU upwind + deferred correction): the user opt-in
  // (set_implicit_advection) on any path, OR the DEFAULT on the domain-BC path (inflow/outflow),
  // where explicit advection is unstable. The velocity-MG BC path carries its own FOU coarse
  // operator, so the default does not apply there (it still honours the explicit opt-in).
  // Advection is implicit (first-order upwind in the stencil, stable at large dt) when the user
  // asks for it, and ALWAYS on the domain-BC stencil path -- which includes the mixed velocity MG
  // (solid + domain BCs), so switching that solver on does not silently change the momentum
  // discretization. The all-fluid domain-BC velocity MG (folded const-coefficient operator) keeps
  // explicit advection, as before.
  bool implicitAdv() const {
    return advect_ && (implicitFou_ || (hasBc_ && (!useVelocityMg_ || hasSolid_)));
  }
  // Domain-BC momentum solved via the Robust-Scaled cut-cell / FOU stencil smoother
  // (ibmRbgsStencilColor
  // + reflection-ghost BCs), not the all-fluid const-coeff fold. Needed when (a) an immersed solid
  // is present (cut-cell no-slip must be in the operator), or (b) advection is implicit (the FOU
  // upwind lives in the stencil -> stable at large dt, the fully-implicit design), or (c) any
  // per-cell coefficient lives in the stencil: variable properties, or the implicit CFD-DEM drag
  // diagonal (hasDrag_). Without (c) an all-fluid domain-BC problem fell through to the
  // CONST-COEFFICIENT fold smoother (Ac = rho/dt + 6mu computed inline), which never reads the
  // assembled band -- the drag never entered the momentum operator while the porous projection's
  // w_f=idt/(idt+beta_f) assumed it did, an inconsistency with pressure-loop gain beta*dt/rho (a
  // fixed bed diverged whenever beta > rho/dt; measured gain 3.84 vs predicted 3.85 at beta=77,
  // idt=20).
  // Domain BCs solved with the (unfolded) cut-cell / FOU stencil and reflection ghosts, as
  // opposed to the folded const-coefficient smoother. Decides the RHS treatment too (the fold's
  // RHS correction applies only off this path), so it must agree with the solver actually used:
  // the MIXED velocity MG (solid + domain BCs) runs on this stencil and is therefore ON this path,
  // while the all-fluid domain-BC velocity MG is the folded operator and is not.
  bool bcStencilPath() const {
    return hasBc_ && (hasSolid_ || (!useVelocityMg_ &&
                                    (implicitAdv() || varProps_ || varRho_ || hasDrag_)));
  }
  // The mixed velocity MG: solid + domain BCs, diffusion-dominated constant-property momentum
  // (implicit advection / variable properties / drag stay on RB-GS: their fine stencils are not
  // approximated by the staircase Helmholtz).
  bool mixedVelocityMg() const {
    return hasBc_ && useVelocityMg_ && hasSolid_ && !varProps_ && !varRho_ && !hasDrag_;
  }
  // Fill a property field's ghosts for the face means: periodic/halo base, then zero-gradient
  // (copy) on domain-BC (wall/inflow/outflow) faces — a periodic wrap there would bring the wrong
  // layer's value to the wall face (destabilising, especially for the harmonic mean).
  // Distributed: the override is per-face rank-OWNED (`touchesGlobalFace`), exactly as
  // `applyScalarBc` does — the halo fill runs first (and periodic-wraps the global boundary ghost),
  // the BC overwrite wins on the rank that owns the face. The former `if (!distributed_)` guard
  // keyed on the wrong predicate: it dropped the override at EVERY np including 1.
  void fillPropGhosts(CCField f) {
    fillGhosts(f);
    for (int face = 0; face < 6; ++face)
      if (bc_[face] != 0 && touchesGlobalFace(face))
        applyScalarBcFace(f, face / 2, face % 2, 1, 0.0);  // type 1 = Neumann copy
    vofBcPropGhosts(f);  // WO-R item 5; a no-op unless a VoF inflow colour is set
  }
  void fillMuGhosts() { fillPropGhosts(muField_); }
  // Ghost ring of the per-cell body-force fields ("force_x/y/z") — WO-G.
  //
  // Neither writer of these fields fills their ghosts: `applyClosure` writes the INNER cells only
  // ("ghosts untouched — refilled by the field's own exchange", `property_closures.hpp`), and the
  // external CFD-DEM writer (`field_view` + `exchangeFieldAdd`) folds its ghost-band deposit onto
  // the owners but leaves the ghost band holding that deposit residue. Nothing else exchanged them,
  // so `buildRhsVar`'s face interpolation `0.5*(fb(i) + fb(i - s_c))` read the registration zero (or
  // the residue) on the first inner plane of every block — the face body force came out exactly
  // HALVED at every rank boundary, and single-rank at the periodic wrap plane. Net effect on a
  // periodic axis: a body-force deficit of 1/(2*N_axis) on the whole domain, because the projection
  // removes the non-uniform part and what survives is the (deficient) mean.
  //
  // WHY THE PROPERTY POLICY IS THE RIGHT ONE, even though a body force is not a transported
  // property. `buildRhsVar` face-interpolates the force with the SAME arithmetic mean it uses for
  // the momentum time term and the projection coefficient face-interpolate rho, and the physical
  // content of that pair is the acceleration f_f/rho_f (this three-way consistency is what makes
  // the discrete hydrostatic balance exact — `doc/variable_density_projection.md` §1/§3). Whatever
  // ghost policy rho has, the force must have the SAME one or the ratio breaks at a boundary. rho
  // uses `fillPropGhosts` (halo/periodic base, then Neumann copy on a rank-OWNED domain-BC face),
  // so the force does too, per BC type:
  //   * WALL / inflow (Dirichlet). The ghost feeds only the face force of the wall-NORMAL component
  //     ON the boundary plane — the one unknown the Dirichlet BC pins (`bcVelocityComp`, comp == a:
  //     `at(bf) = wall`) and whose flux openness is 0. So the value is unobservable there today,
  //     which is exactly why the hydrostatic acid test passed at 2.75e-17 with the defect present.
  //     Neumann copy is still the right answer: it is the only choice that keeps f_f/rho_f equal to
  //     the intended acceleration if that pin is relaxed (free-slip / stress BC), and "zero" would
  //     assert that the volumetric source stops at the wall. A body force is a SOURCE, not a flux —
  //     there is no reflection or odd-extension principle to invoke, only extrapolation, and the
  //     piecewise-constant (Neumann) extrapolation is O(h), the same order as rho's own ghost.
  //   * OUTFLOW. Zero-gradient is what every other quantity gets there, and a zero ghost would
  //     halve the body force on the outlet face — this same defect, relocated to the outlet.
  // So the policy does NOT differ per BC type; `fillPropGhosts` is used verbatim.
  //
  // Consumer note: only `buildRhsVar` (variable density, or the eps-conservative porous momentum)
  // reads the ghost. `buildRhsForced` reads the CELL value `fb(i)` alone, so on the constant-density
  // forced path (Boussinesq) this fill is numerically INERT — applied unconditionally anyway, so the
  // field's ghost contract does not depend on which RHS kernel happens to consume it.
  void fillCellForceGhosts() {
    if (!hasCellForce_)
      return;
    for (int c = 0; c < 3; ++c)
      fillPropGhosts(cellForce_[c]);
  }
  // Ghost ring of the per-cell drag coefficient "drag_beta" — WO-I.
  //
  // Same defect class as the body force above, one phase earlier in the step. Under `porous_`,
  // `addDragDiagonal` builds the staggered momentum diagonal from the FACE drag
  //
  //     beta_f(i) = 0.5*(beta(i) + beta(i - s_c))
  //
  // and all three of its call sites (`rebuildStencils`, `buildAdvStencil`, `buildAdvStencilVar`)
  // run at/after the TOP of `step()`. But no writer of `drag_beta` fills its ghosts: `setField` and
  // `applyClosure` write the inner cells only, and the external CFD-DEM writer's driver FOLDS its
  // ghost-band deposit onto the owners and then ZEROES that band (single rank) or leaves the
  // reverse-halo residue in it (MPI). The only `fillPropGhosts(dragBeta_)` used to be inside
  // `project()` — i.e. AFTER the momentum build. So on the first inner plane of every block the
  // momentum diagonal was assembled from a stale/zero ghost while the projection's coefficient
  // (`buildPorousCoeffDrag`/`Cons`) and its correction (`projectCorrectPorous*`) used the freshly
  // exchanged value on that SAME face.
  //
  // That mismatch is exactly what `addDragDiagonal`'s own comment warns about: the incremental
  // pressure loop then has gain (idt + beta_f)/(idt + beta_f^momentum) instead of 1, and "the
  // accumulated pressure diverges exponentially" — here localized to block/wrap boundaries rather
  // than to the bed top. With the CFD-DEM writer's zeroed ghost the momentum diagonal on that plane
  // carried beta/2 against the projection's beta, a factor-2 error in the drag, not round-off.
  //
  // WHY THIS CALL SITE. It is the only point that is after BOTH writers — a closure targeting
  // "drag_beta" (applied by `updateProperties()` immediately above) and the external deposit (which
  // happens before `step()` is entered) — and before the FIRST consumer, the momentum stencil build
  // a few lines below. Nothing inside `step()` writes `drag_beta`.
  //
  // WHY `fillPropGhosts` AND NOT SOME OTHER POLICY. There is no freedom here: `project()` already
  // fills this very field with `fillPropGhosts` (halo/periodic base + Neumann copy on a rank-OWNED
  // domain-BC face), and the whole point is that the momentum diagonal and the projection
  // coefficient must agree on beta_f face by face — the three-way consistency
  // `doc/variable_density_projection.md` §1/§3 states for (time term, body force, projection
  // coefficient) and `doc/porous_drag_scheme.md` §2 states for (diagonal, operator, correction).
  // Any policy other than the one `project()` uses would re-create the mismatch it is fixing.
  //
  // `project()`'s fill is therefore REDUNDANT after this one (nothing writes `drag_beta` in
  // between). It is kept deliberately: removing it is a separate change, `project()` must keep its
  // own ghost contract for any future mid-step writer, and it costs one exchange on the porous path
  // only.
  //
  // Gated on `hasDrag_` (the field exists iff `enableDrag()` ran), not on `porous_`: the field's
  // ghost contract should not depend on which consumer happens to read it. On the non-porous drag
  // path `addDragDiagonal` uses the cell value alone, so the fill is numerically inert there.
  void fillDragBetaGhosts() {
    if (!hasDrag_)
      return;
    fillPropGhosts(dragBeta_);
  }
  // Eps ghost policy for the porous (volume-averaged) machinery. Periodic/halo base fill, then at
  // non-periodic domain faces: wall -> zero-gradient; INFLOW/OUTFLOW -> mirror around 1 so the
  // arithmetic face mean is EXACTLY 1 (the boundary is pure gas: below the distributor and in the
  // freeboard eps = 1, so a prescribed inflow velocity is the SUPERFICIAL gas velocity and its face
  // flux is open_f*1*u — the Kuipers/MFIX distributor convention). Every consumer — the projection
  // RHS divergence, the Poisson coefficients, and maxPorousResidual — must use THIS fill: the
  // external deposit writes its own leakage into these ghosts each step, and any two consumers
  // reading different ghost values enforce two different constraints, which leaves an irreducible
  // residual (eps_f_rhs - eps_f_resid)*u_in pinned at the distributor row and feeds gas at
  // eps_f*U instead of U.
  void fillPorousEpsGhosts() {
    fillGhosts(epsField_);
    for (int face = 0; face < 6; ++face) {
      const int t = bc_[face];
      if (t == 0 || !touchesGlobalFace(face))
        continue;  // rank-owned faces only (see fillPropGhosts)
      if (t == 2 || t == 3)
        applyScalarBcFace(epsField_, face / 2, face % 2, 2, 1.0);  // open face: face eps == 1
      else
        applyScalarBcFace(epsField_, face / 2, face % 2, 1, 0.0);  // wall: zero-gradient
    }
  }
  // --- VoF internals (rung V2a, WO-J) ---------------------------------------------------------
  // Allocate the colour field's own g=3 working block and wire its ghost/all-reduce hooks. Called
  // by enableVof() and again by any path that re-sizes the block (redistribute -> initMpi), since
  // the advector's block must track the solver's.
  void buildVofBlock() {
    vofAdv_.init(nx_, ny_, nz_, 1.0, kVofG);  // h = 1: flow works in cell units
    vofAdv_.cflLimit = vofCflLimit_;
    vofAdv_.interfaceLocalCfl = true;  // WO-J item 4 — see maxCourantInterface()
    const I3 e3 = vofAdv_.extent();
    e3_ = C3{e3.x, e3.y, e3.z};
    vofAdv_.exchange = [this](CCField f) { this->vofFillGhosts(f); };
    vofAdv_.globalMax = nullptr;
#ifdef PECLET_FLOW_MPI
    if (distributed_ && dec_) {
      int rank = 0;
      MPI_Comm_rank(comm_, &rank);
      // Periodic on all three axes, exactly like the velocity halo: the halo owns every interior
      // ghost and wraps the global boundary, and vofFillGhosts then overwrites the out-of-domain
      // ghosts of any non-periodic axis with the globally-clamped value.
      std::array<bool, 3> per{true, true, true};
      vofHalo_ = std::make_shared<GridHaloTopology<3>>();
      vofHalo_->buildTopology(*dec_, rank, kVofG, per, comm_);
      vofDev_ = std::make_shared<GridHalo<double>>();
    vofDev_->setLabel("vof g3");
      vofDev_->init(*vofHalo_);
      MPI_Comm cm = comm_;
      vofAdv_.globalMax = [cm](double v) {
        double r = v;
        MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_MAX, cm);
        return r;
      };
    }
#endif
    buildVofGeometry();  // rung V5a (WO-Q): openness + fluid fraction on the colour block
    // WO-R: the out-of-domain mask and the resampled boundary-colour profiles are properties of
    // THIS block, so they are (re)built with it — and the mask is only INSTALLED on the advector
    // when a VoF boundary colour has actually been set, which is what keeps the V1 flux path
    // bit-identical otherwise (gate G5).
    vofRebuildBcBlock();
    // The half-shifted momentum CVs live on this same block, so they are rebuilt with it.
    vofCurv_.init(nx_, ny_, nz_, kVofG);
    if (vofMomEnabled_) {
      vofMom_.init(vofAdv_, vofRhoG_, vofRhoL_);
      for (int c = 0; c < 3; ++c)
        if (uAdv_[c].extent(0) != n_)
          uAdv_[c] = CCField("uAdv", n_);
    }
  }
  // --- rung V5a (WO-Q): the cut-cell geometry of the colour block ------------------------------
  //
  // The advector needs, on ITS g=3 block and in ITS high-face index convention, the face openness
  // `o_d` and the cell fluid fraction `eps`. Both are built here and both are then run through the
  // colour field's OWN ghost policy (`vofFillGhosts`), which is what makes the classification at
  // the outermost ghost layer the owner's classification rather than a locally-guessed one — the
  // solid-band fill of `cutcell.hpp` reads fluid neighbours at ghost depth 3, so a wrong
  // classification there would be a decomposition dependence in the INNER result.
  //
  // `eps` comes from `buildCellFraction` (mac_approx_projection.hpp: 4^3-subsampled trilinear SDF),
  // which the collocated path already uses for `cs_`. `cs_` is allocated only on the collocated
  // grid, so the staggered path gets its own `vofCs_` here.
  //
  // The openness embed is `vof::copyFaceVelocity`, i.e. THE SAME shifted embed the face velocity
  // uses — because it is the same face. `ox_(i)` is the openness of the `-x` face of cell `i` and
  // the advector wants the `+x` face of cell `i`, exactly the low-face -> high-face shift of
  // `colour_field.hpp`. Using a concentric embed here instead is the openness twin of WO-J's 35 %
  // conservation defect and is what gate G3 exists to catch.
  void buildVofGeometry() {
    if (!vofEnabled_)
      return;
    if (!hasSolid_ || !cutcellPressure_) {
      vofAdv_.disableGeometry();  // all-fluid: the V1 kernels run byte-identically
      vofAdv_.disableWetting();
      vofSolidG2_ = CCField();
      return;
    }
    vofAdv_.enableGeometry();
    if (vofCs_.extent(0) != n_)
      vofCs_ = CCField("vofCs", n_);
    buildCellFraction(vofCs_, CCConst(sdf_), e_, G);  // inner region; ghosts come from the exchange
    copyInner(vofAdv_.epsFraction(), e3_, kVofG, CCConst(vofCs_), e_, G);
    vofExchangeRaw(vofAdv_.epsFraction());
    CCField oa[3] = {ox_, oy_, oz_};
    for (int d = 0; d < 3; ++d) {
      vof::copyFaceVelocity(vofAdv_.faceOpenness(d), I3{e3_.x, e3_.y, e3_.z}, kVofG, oa[d],
                            I3{e_.x, e_.y, e_.z}, G, d);
      vofExchangeRaw(vofAdv_.faceOpenness(d));
    }
    vofAdv_.classifyGeometry();
    vofExchangeRaw(vofAdv_.kindDouble());  // the owner's classification into every ghost layer
    vofAdv_.finalizeGeometry();
    // The G=2 mirror of the classification: the canonical "C" field reports EXACTLY 0 in solid
    // cells (gate G2), while the g=3 working block carries the neutral band fill that the MYC and
    // height-function stencils need. The fill is regenerated deterministically by every
    // `vofFillGhosts`, so nothing is lost by not persisting it.
    if (vofSolidG2_.extent(0) != n_)
      vofSolidG2_ = CCField("vofSolidG2", n_);
    copyInner(vofSolidG2_, e_, G, CCConst(vofAdv_.kindDouble()), e3_, kVofG);
    applyContactAngle();  // rung V5b (WO-S): re-wire the theta field / wall SDF onto the new block
    zeroSolidColour();
  }
  // Zero the canonical G=2 colour field inside solid cells (see buildVofGeometry).
  void zeroSolidColour() {
    if (!vofEnabled_ || !vofAdv_.hasGeometry() || !vofSolidG2_.extent(0) || !vofSolidZero_)
      return;
    // `vofSolidG2_` lives on the EXTENDED G=2 block (copyInner wrote it at (x+G, y+G, z+G)), so it
    // is indexed exactly like cField_ — indexing it as an inner-sized array reads the wrong cells
    // and silently zeroes live fluid colour (measured: 0.5 % of the liquid volume lost per step).
    CCField c = cField_;
    CCConst sl = CCConst(vofSolidG2_);
    const int ex = e_.x, ey = e_.y, g = G;
    Kokkos::parallel_for(
        "peclet::flow::vof_zero_solid",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {0, 0, 0}, {nx_, ny_, nz_}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)(x + g) + (long)(y + g) * ex + (long)(z + g) * (long)ex * ey;
          if (sl(i) > 0.5)
            c(i) = 0.0;
        });
    CCExec().fence();
  }
  // Is axis `a` periodic for the colour field? flow's per-face bc_ is 0 (periodic) on BOTH ends of
  // a periodic axis, so an axis is periodic iff neither of its faces carries a domain BC.
  bool vofAxisPeriodic(int a) const { return bc_[2 * a] == 0 && bc_[2 * a + 1] == 0; }
  // The colour field's ghost policy on its own g=3 block: halo/periodic base, then zero-gradient
  // (globally clamped) on every non-periodic axis. Zero-gradient is the same policy the material
  // properties get (`fillPropGhosts`) — a wall neither creates nor destroys colour, and the MYC
  // stencil of an inner boundary cell must see a plausible continuation rather than a wrap from the
  // far side of the domain. Prescribing C at an inflow face is a V5+ concern (it needs a flux BC,
  // not a ghost value) and is not offered here.
  void vofFillGhosts(CCField f) {
    vofExchangeRaw(f);
    // Rung V5a (WO-Q): the neutral (90 deg) solid-band fill, for the COLOUR field only. It is a
    // stencil device, not transported data — the MYC 3^3 stencil and the V3 height-function
    // columns of a near-wall cell reach into the solid, and leaving those cells at 0 makes every
    // wall perfectly non-wetting. Three passes with a shrinking depth budget (`vof/cutcell.hpp`),
    // then a SECOND exchange so the outermost ghost layer holds its owner's filled value (the
    // passes only reach ghost depth 2, and the curvature cascade reads depth 3).
    if (vofAdv_.hasGeometry() && f.data() == vofAdv_.colour().data()) {
      // Rung V5b (WO-S): the theta-consistent pass 1 reads the FLUID-ONLY Youngs normal of the
      // anchor fluid cell, which the pass may reach at ghost depth 3 — one deeper than a 3^3
      // stencil can be evaluated on this block. Build it on the INNER region and run it through
      // the block's own ghost policy, exactly as the geometry classification is: every read the
      // theta pass then makes is the OWNER's value, which is what keeps the inner fill
      // decomposition-independent.
      if (vofAdv_.hasWetting()) {
        vofAdv_.buildWettingNormals();
        for (int d = 0; d < 3; ++d)
          vofExchangeRaw(vofAdv_.wettingNormal(d));
      }
      vofAdv_.solidBandFill();
      vofExchangeRaw(f);
    }
  }
  void vofExchangeRaw(CCField f) {
    const bool px = vofAxisPeriodic(0), py = vofAxisPeriodic(1), pz = vofAxisPeriodic(2);
#ifdef PECLET_FLOW_MPI
    if (distributed_ && vofDev_)
      vofDev_->exchange(f);
    else
#endif
      vof::periodicFill(f, I3{e3_.x, e3_.y, e3_.z}, kVofG, px, py, pz);
    if (px && py && pz)
      return;
    const I3 gs = vofGlobalSize(), org = vofOrigin();
    vof::clampFill(f, I3{e3_.x, e3_.y, e3_.z}, kVofG, org, gs, px, py, pz);
    vofApplyColourBc(f);  // WO-R: inflow / inletOutlet backflow; a no-op unless one is set
  }
  I3 vofGlobalSize() const {
#ifdef PECLET_FLOW_MPI
    if (distributed_)
      return I3{gnx_, gny_, gnz_};
#endif
    return I3{nx_, ny_, nz_};
  }
  // Global index of this block's inner cell (0,0,0). og_ is exactly that (originInclGhost + G).
  I3 vofOrigin() const { return I3{og_.x, og_.y, og_.z}; }
  // Face velocities -> the advector's g=3 block. The advecting field must be the PROJECTED one
  // (see advectVof), and its ghost ring must be valid because the advector reads the `-d` face of
  // the first inner cell, which is a ghost cell's `+d` face. fillVelGhosts is the solver's own
  // halo+domain-BC fill and is exactly what the Picard loop does at the top of every iteration, so
  // calling it here leaves the velocity ghosts in the state the next consumer would have produced.
  void bridgeVelocityToVof() {
    // Rung V8 (WO-T): on the collocated grid the divergence-free field is the PROJECTED MAC face
    // field uf_/vf_/wf_, not the cell field — and it is in flow's own low-face convention
    // (`uf(i) = 1/2(U(i)+U(i-1))` sits at i-1/2, the -x face of cell i), the SAME convention
    // `getFaceVelocity` reports and `copyFaceVelocity` shifts. So the bridge is the identical call
    // on a different source view; the cell field never enters the colour transport.
    if constexpr (Grid::collocated) {
      CCField fa[3] = {uf_, vf_, wf_};
      for (int c = 0; c < 3; ++c) {
        fillGhosts(fa[c]);  // the face field's own ghost policy (project() does exactly this)
        vof::copyFaceVelocity(vofAdv_.faceVel(c), I3{e3_.x, e3_.y, e3_.z}, kVofG, fa[c],
                              I3{e_.x, e_.y, e_.z}, G, c);
      }
      return;
    }
    for (int c = 0; c < 3; ++c) {
      // KEEP the projection's outflow-face correction when there IS one (see
      // fillVelGhostsKeepOutflow): the full fill would overwrite it with the zero-gradient copy
      // and hand the advector a field that is not divergence-free at the outlet. When no
      // projection has run since the last full fill — the KINEMATIC path, where the caller
      // prescribes the velocity on the inner cells and the boundary face has never been set —
      // the zero-gradient fill is exactly what supplies that face, so run the full one.
      // With no outflow face at all the two are identical.
      if (outflowCorrValid_)
        fillVelGhostsKeepOutflow(c);
      else
        fillVelGhosts(c, 0);
      // The uniform face-velocity seam (getFaceVelocity): staggered C[c].u already lives on the
      // faces. copyFaceVelocity carries the low-face -> high-face index shift; see
      // colour_field.hpp.
      vof::copyFaceVelocity(vofAdv_.faceVel(c), I3{e3_.x, e3_.y, e3_.z}, kVofG, C[c].u,
                            I3{e_.x, e_.y, e_.z}, G, c);
    }
  }
  // Colour: G=2 registry mirror -> the g=3 working block, then the colour field's own ghost policy.
  // Inner cells only in the copy — the two blocks have different ghost extents and each fills its
  // own (the one bridge; see the enableVof note).
  void bridgeColourToVof() {
    copyInner(vofAdv_.colour(), e3_, kVofG, CCConst(cField_), e_, G);
    vofFillGhosts(vofAdv_.colour());
  }

  // Staggered face stride of velocity component c (the -c face of cell i pairs cells i and i-s).
  long strideOf(int c) const { return (c == 0) ? 1 : (c == 1) ? e_.x : (long)e_.x * e_.y; }
  // The face-property accessor for the momentum stencil of component c: mu constant-or-field
  // (arithmetic/harmonic mean), rho constant-or-field (arithmetic face mean for the time diagonal —
  // the same face density the variable-density projection uses).
  // Effective variable density: true varRho, or the eps-conservative porous momentum (rho_eff =
  // eps*rho in epsRho_, refreshed per step by updateEpsRho).
  bool effVarRho() const { return varRho_ || (porous_ && porousCons_); }
  CCField effRhoField() { return varRho_ ? rhoField_ : epsRho_; }
  // --- rung V8 (WO-T): the collocated face-acceleration predictor --------------------------------
  //
  // TRUE exactly on the configurations that used to throw outright on this grid — variable density
  // (`set_density_mode`) and surface tension (which needs `enable_vof`) on `SolverColocated` — so
  // every validated collocated path (constant density, no VoF: `benchmarks/staggered-vs-collocated`,
  // the colocated regression baselines) and the whole staggered solver take the same branches they
  // always did, byte for byte.
  //
  // When it is on, the predictor drops the pressure gradient and EVERY body/interfacial force, and
  // they are re-introduced as a face acceleration on `uf_/vf_/wf_` after `centerToFace` — see
  // `collocated_varrho.hpp` for why the cell balance is not an option here.
  bool colocatedFaceForce() const { return Grid::collocated && (varRho_ || csfActive()); }
  // The AUTO collocated scheme (set in setSolid/setPressureGeometry) picks the GHOST projection when
  // the configuration allows it, and the ghost v1 supports neither variable density nor the V8 face
  // force. `set_density_mode` / `enable_vof` can be called AFTER the geometry, so re-run the same
  // fallback here rather than failing later inside project(). An explicit scheme selection has
  // already cleared colSchemeAuto_ and is left alone (it will hit the loud throw instead).
  void collocatedV8AutoFallback(const char* why) {
    if constexpr (Grid::collocated) {
      if (colSchemeAuto_ && ghostProjection_) {
        ghostProjection_ = false;
        gpNRows_ = -1;
        faceInterp_ = 9;
        fprintf(stderr,
                "peclet::flow SolverColocated: AUTO scheme fell back to gauge-exact (%s is rung V8 "
                "and the ghost projection v1 does not support it). Select explicitly with "
                "set_collocated_scheme to silence this notice.\n",
                why);
      }
    }
  }
  void ensureFaceAcc() {
    for (int c = 0; c < 3; ++c)
      if (faceAcc_[c].extent(0) != n_)
        faceAcc_[c] = CCField("faceAcc", n_);
  }
  // Guard rail for rung V8's scope. The collocated variable-density / face-force path is validated
  // ALL-FLUID (`set_pressure_geometry`); an immersed solid on it would need the cut-cell face
  // acceleration AND the one-sided (gauge-exact / ghost) closures to agree with the face averaging
  // operator, which is a separate derivation. Fail loudly instead of half-supporting it.
  void requireCollocatedFaceForceScope(const char* who) {
    if (!colocatedFaceForce())
      return;
    std::string m(who);
    if (hasSolid_)
      throw std::runtime_error(
          m + ": variable density / surface tension on SolverColocated is rung V8 and is ALL-FLUID "
              "only (set_pressure_geometry). An immersed solid needs the cut-cell face acceleration "
              "and the matching one-sided closures — not this rung.");
    if (ghostProjection_)
      throw std::runtime_error(
          m + ": the ghost projection (v1) does not support variable density; select "
              "set_collocated_scheme(\"gauge-exact\") (the AUTO fallback already does).");
    if (rhoFaceHarmonic_)
      throw std::runtime_error(
          m + ": set_rho_face_harmonic is not wired into the collocated face acceleration (the "
              "face force and the face coefficient would use different rho_f). Staggered only.");
  }
  void updateEpsRho() {
    CCExec space;
    CCField er = epsRho_;
    CCConst ep = CCConst(epsField_);
    const double rho = rho_;
    Kokkos::parallel_for(
        "peclet::flow::eps_rho", Kokkos::RangePolicy<CCExec>(space, 0, n_),
        KOKKOS_LAMBDA(std::size_t i) { er(i) = ep(i) * rho; });
  }
  VarFaceProps makeFaceProps(int c) {
    VarFaceProps fp;
    fp.haveMu = varProps_;
    if (varProps_)
      fp.mu = CCConst(muField_);
    else
      fp.muC = mu_;
    fp.harmMu = harmonicMu_;
    fp.haveRho = effVarRho();
    if (effVarRho()) {
      fp.rho = CCConst(effRhoField());
      fp.idt = 1.0 / dt_;
      // Placement of the velocity unknown: the staggered unknown sits on the -c FACE, so its time
      // diagonal is the arithmetic face mean of rho; the COLLOCATED unknown sits at the cell CENTRE
      // (Grid::offset == 0), so it is rho(i) itself — which is what stride 0 gives,
      // 0.5*(rho(i)+rho(i)) == rho(i) exactly in floating point. Inert until rung V8: `haveRho` was
      // unreachable on the collocated grid (set_density_mode and set_porous_continuity both threw).
      fp.sc = Grid::collocated ? 0 : strideOf(c);
    } else
      fp.rhoIdtC = rho_ / dt_;
    return fp;
  }
  // Mirror the host motion arrays onto the device (KBs; rebuilt only when a driver changes a
  // body's velocity, not per step).
  void buildSceneQuery() {
    namespace g = peclet::core::geom;
    g::PeriodicBox<double> box{sceneExtent_.x, sceneExtent_.y, sceneExtent_.z, scenePeriodic_};
    sceneQ_ = std::make_shared<g::SceneQueryDevice<double, CCMem>>(
        g::SceneQueryDevice<double, CCMem>::build(*sceneB_, sceneOrigin_, sceneExtent_, box));
  }

  void refreshMotionFlag() {
    hasMotion_ = false;
    for (std::size_t k = 0; k < instLin_.size(); ++k)
      if (instLin_[k] != 0.0 || instAng_[k] != 0.0)
        hasMotion_ = true;
  }

  void uploadMotion() {
    const std::size_t m = (std::size_t)nInst_ * 3;
    if (instCenD_.extent(0) != m) {
      instCenD_ = Kokkos::View<double*, CCMem>("instCen", m);
      instLinD_ = Kokkos::View<double*, CCMem>("instLin", m);
      instAngD_ = Kokkos::View<double*, CCMem>("instAng", m);
    }
    if (m == 0)
      return;
    using HostConst =
        Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(instCenD_, HostConst(instCen_.data(), m));
    Kokkos::deep_copy(instLinD_, HostConst(instLin_.data(), m));
    Kokkos::deep_copy(instAngD_, HostConst(instAng_.data(), m));
  }

  peclet::core::geom::InstanceMotionView<double> motionView() const {
    peclet::core::geom::InstanceMotionView<double> mv;
    mv.cen = instCenD_.data();
    mv.lin = instLinD_.data();
    mv.ang = instAngD_.data();
    mv.n = nInst_;
    return mv;
  }

  // MOVING GEOMETRY (Layer 3 rungs 2-3): sample the scene's KINEMATIC WALL VELOCITY onto the grid.
  //
  // At each probe p (component c's staggered point for rung 2; the cell centre for rung 3):
  //   n_hat = central difference of the SAMPLED sdf_, normalised   -- O(h), the v1 fidelity
  //   w     = p - sdf(p) * n_hat                                    -- core's geom::wallPoint
  //   u_w   = geom::instanceVelocity(owner(p), w)
  //
  // THE OWNER IS QUERIED AT p, not read from the cell-centred cutOwner_ field. At a contact
  // between two bodies the staggered point and the cell centre can belong to different ones, and
  // a wall velocity taken from the wrong body is precisely the error this rung exists to avoid.
  //
  // Per-direction crossing-point placement (via tEx_) would put the wall point on the exact
  // crossing instead of along the gradient; that is the documented refinement, deliberately not
  // taken in v1 -- see the design note.
  void buildWallVelocity() {
    if (!hasScene_ || !hasMotion_) {
      // Never moved: the fields stay EMPTY and every consumer takes its old, bit-identical path.
      // MOVED AND THEN STOPPED is different, and was wrong: wallVelView() keys off the field's
      // extent, not hasMotion_, so a previously-built uBc_ would keep being folded into the
      // momentum operator's inhomogeneity after the caller set the velocity back to zero. Zero
      // them instead of stranding them. Allocation state is unchanged either way, so a run that
      // never moves is bit-identical to before.
      for (int c = 0; c < 3; ++c) {
        if (uBc_[c].extent(0) == n_)
          Kokkos::deep_copy(uBc_[c], 0.0);
        if (uwCell_[c].extent(0) == n_)
          Kokkos::deep_copy(uwCell_[c], 0.0);
      }
      return;
    }
    for (int c = 0; c < 3; ++c) {
      if (uBc_[c].extent(0) != n_)
        uBc_[c] = CCField("uBc", n_);
      if (uwCell_[c].extent(0) != n_)
        uwCell_[c] = CCField("uwCell", n_);
    }
    const auto q = sceneQ_->view();
    const auto mv = motionView();
    CCConst sd = CCConst(sdf_);
    const C3 e = e_, og = og_;
    CCExec space;
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    // rung 2: component c's own staggered point, storing component c
    for (int c = 0; c < 3; ++c) {
      const auto po = Grid::offset(c);
      const double ox = po.x, oy = po.y, oz = po.z;
      CCField out = uBc_[c];
      const int cc = c;
      Kokkos::parallel_for(
          "peclet::flow::wall_velocity_stag", MD(space, {0, 0, 0}, {e.x, e.y, e.z}),
          KOKKOS_LAMBDA(int lx, int ly, int lz) {
            const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
            const double sx = (double)lx + ox, sy = (double)ly + oy, sz = (double)lz + oz;
            const double s0 = ccSampleExt(sd, e, sx, sy, sz);
            const peclet::core::Vec3<double> grad{
                0.5 * (ccSampleExt(sd, e, sx + 1, sy, sz) - ccSampleExt(sd, e, sx - 1, sy, sz)),
                0.5 * (ccSampleExt(sd, e, sx, sy + 1, sz) - ccSampleExt(sd, e, sx, sy - 1, sz)),
                0.5 * (ccSampleExt(sd, e, sx, sy, sz + 1) - ccSampleExt(sd, e, sx, sy, sz - 1))};
            const peclet::core::Vec3<double> p{sx - G + og.x, sy - G + og.y, sz - G + og.z};
            const peclet::core::Vec3<double> w = peclet::core::geom::wallPoint(p, s0, grad);
            const peclet::core::Vec3<double> v =
                peclet::core::geom::instanceVelocity(mv, q.owner(p), w, q.box);
            out(i) = cc == 0 ? v.x : (cc == 1 ? v.y : v.z);
          });
    }
    // rung 3: the whole wall velocity at cell centres (the wall-flux divergence source)
    {
      CCField ux = uwCell_[0], uy = uwCell_[1], uz = uwCell_[2];
      Kokkos::parallel_for(
          "peclet::flow::wall_velocity_cell", MD(space, {0, 0, 0}, {e.x, e.y, e.z}),
          KOKKOS_LAMBDA(int lx, int ly, int lz) {
            const long i = (long)lx + (long)ly * e.x + (long)lz * (long)e.x * e.y;
            const double sx = lx, sy = ly, sz = lz;
            const double s0 = ccSampleExt(sd, e, sx, sy, sz);
            const peclet::core::Vec3<double> grad{
                0.5 * (ccSampleExt(sd, e, sx + 1, sy, sz) - ccSampleExt(sd, e, sx - 1, sy, sz)),
                0.5 * (ccSampleExt(sd, e, sx, sy + 1, sz) - ccSampleExt(sd, e, sx, sy - 1, sz)),
                0.5 * (ccSampleExt(sd, e, sx, sy, sz + 1) - ccSampleExt(sd, e, sx, sy, sz - 1))};
            const peclet::core::Vec3<double> p{sx - G + og.x, sy - G + og.y, sz - G + og.z};
            const peclet::core::Vec3<double> w = peclet::core::geom::wallPoint(p, s0, grad);
            const peclet::core::Vec3<double> v =
                peclet::core::geom::instanceVelocity(mv, q.owner(p), w, q.box);
            ux(i) = v.x;
            uy(i) = v.y;
            uz(i) = v.z;
          });
    }
    space.fence();
  }

  // MOVING GEOMETRY rung 3: the wall's own volume flux, folded into the cell divergence.
  //
  // A rigid body sweeping through a cut cell injects a net flux through the WALL part of the
  // cell's fluid boundary; it is zero only integrally over a closed body, never cell by cell.
  // The wall area VECTOR is exact from the aperture identity -- apply the divergence theorem to
  // the constant field e_a over the cell's fluid region and the open-face terms telescope:
  //     A_wall = -(oE - oW, oN - oS, oT - oB)      (in h=1 cell units, matching divergOpen)
  // so continuity over the fluid region reads  div_open(u) + u_w . A_wall = 0. divergOpen has
  // already written the first term into `d`; this adds the second, leaving rhs = -(d) untouched
  // in form. Inert unless a moving instance exists.
  void addWallFluxDivergence(CCField d) {
    if (!hasScene_ || !hasMotion_ || !wallFluxDiv_ || uwCell_[0].extent(0) != n_)
      return;
    CCExec space;
    const C3 e = e_;
    CCConst oxv = CCConst(ox_), oyv = CCConst(oy_), ozv = CCConst(oz_);
    CCConst wx = CCConst(uwCell_[0]), wy = CCConst(uwCell_[1]), wz = CCConst(uwCell_[2]);
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "peclet::flow::wall_flux_div", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)x + (long)y * sy + (long)z * sz;
          const double ax = -(oxv(i + sx) - oxv(i));
          const double ay = -(oyv(i + sy) - oyv(i));
          const double az = -(ozv(i + sz) - ozv(i));
          d(i) += wx(i) * ax + wy(i) * ay + wz(i) * az;
        });
    space.fence();
  }

  /// Hydrodynamic force and torque on each scene instance (Layer 4 rung 2) -- the resolved
  /// CFD-DEM feedback. Returns four 3*nInst blocks: force, torque, and the force split into its
  /// PRESSURE and VISCOUS parts (force == pressure + viscous), because the two carry different
  /// discretisation error and a deficit that sits in one of them localises itself.
  ///
  /// THE SURFACE INTEGRAL, cut cell by cut cell. Over the wall patch inside a cell,
  ///     sigma = -p I + mu (grad u + grad u^T),      dF_body = -(sigma . A_wall)
  /// with A_wall = -(oE-oW, oN-oS, oT-oB) the FLUID-outward wall area vector from the aperture
  /// identity (the same one rung 3's wall flux uses). The minus sign converts it to the BODY's
  /// outward normal, which is the one the traction on the body is taken against. Cells with
  /// A_wall = 0 -- fully open or fully solid -- contribute nothing, so no cut-cell list is needed:
  /// the geometry selects the surface.
  ///
  /// Torque is about the owning instance's centre with the lever arm MIN-IMAGED, for the same
  /// reason instanceVelocity min-images it: a body can own wall cells across a periodic seam.
  ///
  /// NOT bit-reproducible. The accumulation is by atomics over an unordered cell traversal, like
  /// the coupling deposits; expect tolerance-level run-to-run variation, not bitwise equality.
  ///
  /// ACCURACY. Cut-cell force integration is O(h)-noisy: the aperture differences are exact but
  /// the traction is evaluated from a cell-centred pressure and a central-differenced velocity
  /// gradient whose stencil reaches into solid cells near the wall. Measure it (the Zick-Homsy
  /// self-consistency gate does) rather than assuming a tolerance.
  std::vector<double> hydroForceTorque() {
    std::vector<double> out((std::size_t)(nInst_ > 0 ? nInst_ : 0) * 12, 0.0);
    if (!hasScene_ || nInst_ <= 0 || cutOwner_.extent(0) != (std::size_t)nx_ * ny_ * nz_)
      return out;
    const std::size_t m = (std::size_t)nInst_ * 3;
    Kokkos::View<double*, CCMem> Fd("hydroF", m), Td("hydroT", m), Pd("hydroFp", m),
        Vd("hydroFv", m);
    Kokkos::deep_copy(Fd, 0.0);
    Kokkos::deep_copy(Td, 0.0);
    Kokkos::deep_copy(Pd, 0.0);
    Kokkos::deep_copy(Vd, 0.0);
    CCExec space;
    const C3 e = e_, og = og_;
    const int nx = nx_, ny = ny_;
    const double mu = mu_;
    CCConst oxv = CCConst(ox_), oyv = CCConst(oy_), ozv = CCConst(oz_);
    CCConst U = CCConst(C[0].u), Vv = CCConst(C[1].u), W = CCConst(C[2].u);
    CCConst Pf = CCConst(P_);
    CCConst sd = CCConst(sdf_);
    const bool haveWallVel = (uwCell_[0].extent(0) == n_);
    CCConst wcx = haveWallVel ? CCConst(uwCell_[0]) : CCConst();
    CCConst wcy = haveWallVel ? CCConst(uwCell_[1]) : CCConst();
    CCConst wcz = haveWallVel ? CCConst(uwCell_[2]) : CCConst();

    auto own = cutOwner_;
    auto cen = instCenD_;
    const auto box = sceneQ_->view().box;
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "peclet::flow::hydro_force", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long st[3] = {1, e.x, (long)e.x * e.y};
          const long i = (long)x + (long)y * st[1] + (long)z * st[2];
          const double A[3] = {-(oxv(i + st[0]) - oxv(i)), -(oyv(i + st[1]) - oyv(i)),
                               -(ozv(i + st[2]) - ozv(i))};
          if (A[0] == 0.0 && A[1] == 0.0 && A[2] == 0.0)
            return;  // no wall passes through this cell
          const int oi = own((std::size_t)(x - G) + (std::size_t)(y - G) * nx +
                             (std::size_t)(z - G) * (std::size_t)nx * ny);
          if (oi < 0)
            return;
          // Cell-centred velocity from the staggered faces.
          //
          // A SOLID CELL STORES A MASKED ZERO, WHICH IS THE MATERIAL VELOCITY ONLY WHEN THE WALL
          // IS AT REST. Reading it as physical is harmless for static geometry and a sign error
          // for moving geometry: the stencil then sees a spurious shear of the wall speed over one
          // cell across the entire surface. Measured on the Galilean pair -- with the velocity
          // field itself frame-invariant to 7e-7, the integrated force came out +7.08e+01 in the
          // lab frame and -1.71e+02 in a frame boosted by 0.7, a ratio of -2.42. A resolved
          // CFD-DEM loop driven by that does not settle, it runs away. Substituting the wall's own
          // velocity restores frame invariance; with static geometry uwCell_ is empty and this is
          // bit-identical to the plain expression.
          auto uc = [&](int a, long c) {
            if (haveWallVel && sd(c) < 0.0)
              return a == 0 ? wcx(c) : (a == 1 ? wcy(c) : wcz(c));
            const CCConst& F = a == 0 ? U : (a == 1 ? Vv : W);
            return 0.5 * (F(c) + F(c + st[a]));
          };
          // PLAIN CENTRAL DIFFERENCE, as the Layer-4 spec prescribes. It spans 2h while the wall
          // sits a fraction of a cell away, so it under-reads the wall shear -- measured as a
          // RESOLUTION-INDEPENDENT ~29% drag deficit that lives almost entirely in the viscous
          // part. The obvious one-sided repair, differencing to the wall over the crossing
          // distance theta, was TRIED AND IS WORSE: cut cells with theta -> 0 make 1/theta
          // unbounded and the drag came out 17x too large. That is precisely why the momentum
          // operator uses a Robust-Scaled reconstruction rather than a raw one-sided difference,
          // and it is why a correct wall-aware traction has to come from that machinery (or from
          // the discrete reaction the operator already applies) rather than from a patch here.
          // See the design note's OPEN FOR REVIEW.
          double gu[3][3];
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
              gu[a][b] = 0.5 * (uc(a, i + st[b]) - uc(a, i - st[b]));
          const double p = Pf(i);
          double dF[3], dFp[3], dFv[3];
          for (int a = 0; a < 3; ++a) {
            // A_wall is fluid-outward; the traction on the BODY takes -A_wall. Pressure and
            // viscous parts are kept apart because they fail differently: the pressure term reads
            // one cell-centred value, while the viscous term differences a velocity whose stencil
            // reaches into solid cells -- so a deficit that lives entirely in one of them says
            // immediately which.
            dFp[a] = p * A[a];
            double t = 0.0;
            for (int b = 0; b < 3; ++b)
              t += mu * (gu[a][b] + gu[b][a]) * A[b];
            dFv[a] = -t;
            dF[a] = dFp[a] + dFv[a];
          }
          const peclet::core::Vec3<double> r = peclet::core::geom::minImage(
              peclet::core::Vec3<double>{(double)(x - G + og.x) - cen[3 * oi + 0],
                                         (double)(y - G + og.y) - cen[3 * oi + 1],
                                         (double)(z - G + og.z) - cen[3 * oi + 2]},
              box);
          for (int a = 0; a < 3; ++a) {
            Kokkos::atomic_add(&Fd(3 * oi + a), dF[a]);
            Kokkos::atomic_add(&Pd(3 * oi + a), dFp[a]);
            Kokkos::atomic_add(&Vd(3 * oi + a), dFv[a]);
          }
          Kokkos::atomic_add(&Td(3 * oi + 0), r.y * dF[2] - r.z * dF[1]);
          Kokkos::atomic_add(&Td(3 * oi + 1), r.z * dF[0] - r.x * dF[2]);
          Kokkos::atomic_add(&Td(3 * oi + 2), r.x * dF[1] - r.y * dF[0]);
        });
    space.fence();
    using HostV = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(HostV(out.data(), m), Fd);
    Kokkos::deep_copy(HostV(out.data() + m, m), Td);
    Kokkos::deep_copy(HostV(out.data() + 2 * m, m), Pd);
    Kokkos::deep_copy(HostV(out.data() + 3 * m, m), Vd);
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      // Instances are REPLICATED, so each rank integrates only the wall cells inside its own
      // block; the body's total is the sum over ranks.
      std::vector<double> g(out.size(), 0.0);
      MPI_Allreduce(out.data(), g.data(), (int)out.size(), MPI_DOUBLE, MPI_SUM, comm_);
      out.swap(g);
    }
#endif
    return out;
  }

  /// Hydrodynamic force and torque per instance from the DISCRETE REACTION (route (b) of the
  /// design note's OPEN FOR REVIEW 1) -- the recommended source of the resolved CFD-DEM feedback.
  /// Returns two 3*nInst blocks: force, torque about the instance centre.
  ///
  /// THE BUDGET. The composed step at any unmasked staggered face i is, exactly,
  ///     rho/dt (u^{n+1}_i - u^n_i) = sum_nb mu (u*_nb - u*_i) + f_c - grad(pi)_i + F_wall_i
  /// with u* the last momentum solve's iterate (the implicit viscous operator acted on u*, which
  /// is why it is stashed) and pi the effective pressure (P^n predictor + (rho/dt) phi). Define
  ///     R_i = rho/dt (u_i - u^n_i) - f_c - sum_{FLUID nbrs} mu (u*_nb - u*_i)
  /// -- deliberately NOT subtracting the pressure. Then R_i = -grad(pi)_i + F_wall_i, and summed
  /// over the owner region of a body the grad(pi) parts TELESCOPE: interior faces cancel
  /// pairwise, leaving exactly the region-boundary pressure flux plus the wall pressure force --
  /// the control-volume budget, with pressure counted once and in the right place without this
  /// function ever reading a pressure field. F_body(k) = -sum_{owner k} R_i.
  ///
  /// WHY THIS IS THE ACCURATE FORCE, not just the conservative one: the modified cut rows do not
  /// derive from symmetric fluxes, so ANY reconstruction of "the traction" is a choice; the
  /// reaction is the momentum the fluid actually lost, and its accuracy is the (independently
  /// validated, 2nd-order) accuracy of the flow solution it sustains. The traction integral
  /// (hydroForceTorque above) under-reads by a resolution-INDEPENDENT ~29% and is kept as a
  /// diagnostic only.
  ///
  /// EXACTNESS: at steady state, sum over bodies = f_c * N_fluid-momentum-cells per component, to
  /// the momentum solver's residual (the only approximation in the budget; everything else is
  /// identity). Per-body attribution is the control-volume one over the owner partition; the
  /// region-boundary fluxes are counted symmetrically, so they cancel exactly in the total.
  ///
  /// ADVECTION (R0). The explicit high-order advection adds one more RHS term to the same
  /// composed step,  +A_i  with  A_i = rho*(FOU_i - HO_i)  exactly as buildRhs assembled it, so
  /// the budget subtracts A_i alongside f_c. It is STASHED rather than recomputed: recomputing
  /// would read the projected u^{n+1} while the RHS used the Picard iterate u^k, and the two
  /// differ by the projection -- a silent O(1) attribution error. The IMPLICIT upwind path
  /// (implicit_fou / the domain-BC stencil path) instead folds advection into the MATRIX, so the
  /// reaction is no longer of this form; it stays refused.
  ///
  /// THE TORQUE (v3, 2026-08-31): the reaction alone is NOT the physical torque, and the gap is
  /// closed here in closed form. The discrete momentum budget measures the LAPLACIAN-form wall
  /// flux (the operator discretises mu*lap(u) = div(mu grad u)); the physical traction adds the
  /// transposed term mu*(grad u)^T . n. In the interior the two agree for constant mu and a
  /// solenoidal field (div(grad u)^T = grad(div u) = 0), but not as a boundary traction -- and
  /// for an incompressible no-slip flow on a rigid wall moving with angular velocity Omega the
  /// missing traction is computable from WALL DATA ALONE:
  ///
  ///     (grad u)^T . n  =  n x Omega        (pointwise on the wall, exactly)
  ///
  /// Derivation: split grad u on the surface into tangential derivatives -- which equal the
  /// rigid-body field's, grad u_w = [Omega x] -- plus the normal derivative; continuity kills the
  /// n(du/dn . n) piece (n . du/dn = -trace of the tangential part = 0 since [Omega x] is
  /// antisymmetric); what survives is the tangential projection of (n x Omega), which is n x
  /// Omega itself. Verified against the analytic rotlet to 3e-11 pointwise. Its FORCE integral
  /// vanishes over any closed surface (oint n dA = 0), which is why the force identity above
  /// never saw it; its TORQUE integral carries exactly ONE THIRD of the Stokes torque on a
  /// rotating sphere (oint r x (n x Omega) dA = -(8pi/3) a^3 Omega), which is exactly the
  /// resolution-independent -31% the rotating-sphere gate measured before this term (predicted
  /// -33.3%; Maitri et al., Comput. Fluids 175 (2018) 111-128, measured the same 33-34% plateau
  /// on an IBM omitting the same term). The correction below integrates mu * r x (n dA x Omega)
  /// over the cut cells with the EXACT aperture wall-area vectors -- no interior reconstruction,
  /// no near-wall gradient, and identically zero when nothing rotates.
  ///
  /// PER-BODY ATTRIBUTION (v4, 2026-08-31): the owner-boundary pressure flux is now REMOVED from
  /// each body's share. The telescoping of grad(pi) over an owner region leaves the wall pressure
  /// force (physical -- kept) PLUS the flux through the region's boundary against OTHER owners'
  /// regions. Those boundary terms cancel pairwise in the total -- which is why the identity gate
  /// never saw them -- but they are NOT zero per body: they transfer force between attributions
  /// across the owner partition's mid-surfaces. A single instance owns all fluid and has no such
  /// boundary (bit-identical, the settling gate's case); a symmetric array cancels them per body
  /// (the 4-sphere gate's case); an ASYMMETRIC pair does neither. Measured on a sphere translating
  /// through a closed analytic tank -- the ten Cate configuration -- the sphere's attributed drag
  /// was HALF the physical value (lambda 0.62 against a physical floor of 1.36; the identical
  /// sphere in a single-instance periodic box reads 1.42), because the part of the pressure force
  /// transmitted beyond the sphere/tank mid-surface was booked to the tank. The correction
  /// subtracts, for every fluid-fluid staggered face whose two momentum points have different
  /// owners, the face's pi-flux from the side that owned it and adds it to the other -- pairwise,
  /// so the total is untouched to round-off, and exactly nothing changes with fewer than two
  /// instances. pi is read from the accumulated P_ (incremental scheme; the rotational
  /// -mu*div(u*) deviation is the projection residual). Faces are visited once via the +s
  /// convention (each inner momentum point checks only its +s neighbour), which also makes the
  /// pass MPI-clean. The torque uses each side's own lever about its own centre.
  ///
  /// v2 SCOPE, refused loudly: staggered only; implicit advection, porous, variable properties,
  /// domain BCs, ghost projection, drag diagonal, fluid-only star modes all put terms in the
  /// update this budget does not carry, and a missing term here is a silently mis-attributed
  /// force.
  std::vector<double> hydroForceTorqueReaction() {
    std::vector<double> out((std::size_t)(nInst_ > 0 ? nInst_ : 0) * 6, 0.0);
    if (!hasScene_ || nInst_ <= 0)
      return out;
    if constexpr (Grid::collocated)
      throw std::runtime_error("hydro_force_torque_reaction: staggered only (v1)");
    if (implicitAdv() || porous_ || varRho_ || varProps_ || hasBc_ || ghostProjection_ ||
        hasDrag_ || fluidOnlyMode_ != 0)
      throw std::runtime_error(
          "hydro_force_torque_reaction: implicit advection / porous / variable-properties / "
          "domain-BC / ghost-projection / drag / star modes put momentum terms in the step that "
          "this budget does not carry (v2) -- a missing term is a silently mis-attributed force");
    if (!haveUStar_)
      throw std::runtime_error("hydro_force_torque_reaction: call step() first (u* is stashed "
                               "during the step)");
    if (advect_ && (!haveAdvRhs_ || advRhs_[0].extent(0) != n_))
      throw std::runtime_error("hydro_force_torque_reaction: the advective RHS term was not "
                               "stashed -- set_advection was enabled after the last step()");
    // u* ghosts: refresh with the standard fill (periodic wrap single-rank, halo exchange under
    // MPI; hasBc_ is refused above so no BC is imposed). The audit's viscous term reads +-1.
    for (int c = 0; c < 3; ++c)
      fillVelGhostsTo(uStar_[c], c, 0);
    const std::size_t m = (std::size_t)nInst_ * 3;
    Kokkos::View<double*, CCMem> Fd("reactF", m), Td("reactT", m);
    Kokkos::deep_copy(Fd, 0.0);
    Kokkos::deep_copy(Td, 0.0);
    CCExec space;
    const C3 e = e_, og = og_;
    const double idt = rho_ / dt_, mu = mu_;
    const auto q = sceneQ_->view();
    auto cen = instCenD_;
    const auto box = q.box;
    const bool hasFb = hasCellForce_;
    for (int c = 0; c < 3; ++c) {
      CCConst un = CCConst(old_[c]), uc = CCConst(C[c].u), us = CCConst(uStar_[c]),
              mk = CCConst(C[c].mask);
      CCConst fb = hasFb ? CCConst(cellForce_[c]) : CCConst();
      CCConst av = advect_ ? CCConst(advRhs_[c]) : CCConst();
      const double fc = f_[c];
      const auto po = Grid::offset(c);
      const double offx = po.x, offy = po.y, offz = po.z;
      const int cc = c;
      Kokkos::parallel_for(
          "peclet::flow::hydro_reaction",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                         {e.x - G, e.y - G, e.z - G}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long st[3] = {1, e.x, (long)e.x * e.y};
            const long i = (long)x + (long)y * st[1] + (long)z * st[2];
            if (mk(i) > 0.5)
              return;  // solid staggered point: no fluid momentum here
            double R = idt * (uc(i) - un(i)) - fc - (fb.data() ? fb(i) : 0.0) -
                       (av.data() ? av(i) : 0.0);
            for (int a = 0; a < 3; ++a) {
              const long jp = i + st[a], jm = i - st[a];
              if (mk(jp) <= 0.5)
                R -= mu * (us(jp) - us(i));
              if (mk(jm) <= 0.5)
                R -= mu * (us(jm) - us(i));
            }
            const peclet::core::Vec3<double> p{(double)(x - G + og.x) + offx,
                                              (double)(y - G + og.y) + offy,
                                              (double)(z - G + og.z) + offz};
            const int oi = q.owner(p);
            if (oi < 0)
              return;
            const double F = -R;  // force ON the body = minus the wall force on the fluid
            Kokkos::atomic_add(&Fd(3 * oi + cc), F);
            const peclet::core::Vec3<double> r = peclet::core::geom::minImage(
                peclet::core::Vec3<double>{p.x - cen(3 * oi + 0), p.y - cen(3 * oi + 1),
                                           p.z - cen(3 * oi + 2)},
                box);
            // torque of the scalar force F e_c at lever r: r x (F e_c)
            if (cc == 0) {
              Kokkos::atomic_add(&Td(3 * oi + 1), r.z * F);
              Kokkos::atomic_add(&Td(3 * oi + 2), -r.y * F);
            } else if (cc == 1) {
              Kokkos::atomic_add(&Td(3 * oi + 0), -r.z * F);
              Kokkos::atomic_add(&Td(3 * oi + 2), r.x * F);
            } else {
              Kokkos::atomic_add(&Td(3 * oi + 0), r.y * F);
              Kokkos::atomic_add(&Td(3 * oi + 1), -r.x * F);
            }
          });
    }
    space.fence();
    // v3: the transposed-stress wall torque, mu * r x (n dA x Omega) per cut cell. n dA is the
    // exact aperture wall-area vector with the BODY-outward orientation, (oE-oW, oN-oS, oT-oB)
    // per the wallAreaProbe convention (sum x*(oE-oW) = +V_solid). Skipped entirely when no
    // instance moves, so a static run stays bit-identical; a purely TRANSLATING instance has
    // Omega = 0 and contributes exact zeros. The FORCE is deliberately left alone: the term's
    // force integral is identically zero over a closed surface, and adding its discrete
    // counterpart would only inject aperture-level rounding into an exactly-gated identity.
    if (hasMotion_ && cutcellPressure_) {
      CCConst oxv = CCConst(ox_), oyv = CCConst(oy_), ozv = CCConst(oz_);
      auto ang = instAngD_;
      Kokkos::parallel_for(
          "peclet::flow::hydro_reaction_torque_transpose",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                         {e.x - G, e.y - G, e.z - G}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
            const long i = (long)x + (long)y * sy + (long)z * sz;
            const double ax = oxv(i + sx) - oxv(i);
            const double ay = oyv(i + sy) - oyv(i);
            const double az = ozv(i + sz) - ozv(i);
            if (ax == 0.0 && ay == 0.0 && az == 0.0)
              return;  // not a cut cell
            const peclet::core::Vec3<double> p{(double)(x - G + og.x), (double)(y - G + og.y),
                                              (double)(z - G + og.z)};
            const int oi = q.owner(p);
            if (oi < 0)
              return;
            const double wx = ang(3 * oi + 0), wy = ang(3 * oi + 1), wz = ang(3 * oi + 2);
            if (wx == 0.0 && wy == 0.0 && wz == 0.0)
              return;
            // v = (n dA) x Omega  -- the missing traction integrated over this cell's wall patch
            const double vx = ay * wz - az * wy;
            const double vy = az * wx - ax * wz;
            const double vz = ax * wy - ay * wx;
            const peclet::core::Vec3<double> r = peclet::core::geom::minImage(
                peclet::core::Vec3<double>{p.x - cen(3 * oi + 0), p.y - cen(3 * oi + 1),
                                           p.z - cen(3 * oi + 2)},
                box);
            Kokkos::atomic_add(&Td(3 * oi + 0), mu * (r.y * vz - r.z * vy));
            Kokkos::atomic_add(&Td(3 * oi + 1), mu * (r.z * vx - r.x * vz));
            Kokkos::atomic_add(&Td(3 * oi + 2), mu * (r.x * vy - r.y * vx));
          });
      space.fence();
    }
    // v4: owner-boundary attribution correction (see the doc block). Only meaningful with at
    // least two instances and the incremental pressure (P_ holds the physical pressure).
    if (nInst_ > 1 && incremental_ && cutcellPressure_) {
      fillGhosts(P_);
      CCConst pf = CCConst(P_);
      for (int c = 0; c < 3; ++c) {
        CCConst mk = CCConst(C[c].mask);
        const long strd = (c == 0) ? 1 : (c == 1) ? e_.x : (long)e_.x * e_.y;
        const auto po = Grid::offset(c);
        const double offx = po.x, offy = po.y, offz = po.z;
        const int cc = c;
        Kokkos::parallel_for(
            "peclet::flow::hydro_reaction_owner_flux",
            Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                           {e.x - G, e.y - G, e.z - G}),
            KOKKOS_LAMBDA(int x, int y, int z) {
              const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
              const long j = i + strd;                       // the +s neighbour: visit once
              if (mk(i) > 0.5 || mk(j) > 0.5)
                return;                                      // wall faces stay in the wall force
              const peclet::core::Vec3<double> pa{(double)(x - G + og.x) + offx,
                                                 (double)(y - G + og.y) + offy,
                                                 (double)(z - G + og.z) + offz};
              peclet::core::Vec3<double> pb = pa;
              (cc == 0 ? pb.x : cc == 1 ? pb.y : pb.z) += 1.0;
              const int oa = q.owner(pa), ob = q.owner(pb);
              if (oa == ob || oa < 0 || ob < 0)
                return;
              // One-sided staggered gradients: point i reads pi(i) - pi(i-s); point j = i+s
              // reads pi(j) - pi(i). The cell shared by this owner-boundary face is cell i,
              // entering a's telescoped sum(grad pi) with +pi(i) (via point i) and b's with
              // -pi(i) (via point j). F_attr = -sum R carries +sum(grad pi), so a's attribution
              // holds +pi(i) and b's -pi(i) from this face: a pure transfer across the owner
              // partition that belongs to NEITHER wall. Remove it from both, symmetrically --
              // the pairwise cancellation is what keeps the total exact.
              const double flux = pf(i);
              Kokkos::atomic_add(&Fd(3 * oa + cc), -flux);
              Kokkos::atomic_add(&Fd(3 * ob + cc), +flux);
              const peclet::core::Vec3<double> ra = peclet::core::geom::minImage(
                  peclet::core::Vec3<double>{pa.x - cen(3 * oa + 0), pa.y - cen(3 * oa + 1),
                                             pa.z - cen(3 * oa + 2)},
                  box);
              const peclet::core::Vec3<double> rb = peclet::core::geom::minImage(
                  peclet::core::Vec3<double>{pa.x - cen(3 * ob + 0), pa.y - cen(3 * ob + 1),
                                             pa.z - cen(3 * ob + 2)},
                  box);
              if (cc == 0) {
                Kokkos::atomic_add(&Td(3 * oa + 1), ra.z * -flux);
                Kokkos::atomic_add(&Td(3 * oa + 2), -ra.y * -flux);
                Kokkos::atomic_add(&Td(3 * ob + 1), rb.z * +flux);
                Kokkos::atomic_add(&Td(3 * ob + 2), -rb.y * +flux);
              } else if (cc == 1) {
                Kokkos::atomic_add(&Td(3 * oa + 0), -ra.z * -flux);
                Kokkos::atomic_add(&Td(3 * oa + 2), ra.x * -flux);
                Kokkos::atomic_add(&Td(3 * ob + 0), -rb.z * +flux);
                Kokkos::atomic_add(&Td(3 * ob + 2), rb.x * +flux);
              } else {
                Kokkos::atomic_add(&Td(3 * oa + 0), ra.y * -flux);
                Kokkos::atomic_add(&Td(3 * oa + 1), -ra.x * -flux);
                Kokkos::atomic_add(&Td(3 * ob + 0), rb.y * +flux);
                Kokkos::atomic_add(&Td(3 * ob + 1), -rb.x * +flux);
              }
            });
      }
      space.fence();
    }
    using HostV = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::deep_copy(HostV(out.data(), m), Fd);
    Kokkos::deep_copy(HostV(out.data() + m, m), Td);
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      std::vector<double> g(out.size(), 0.0);
      MPI_Allreduce(out.data(), g.data(), (int)out.size(), MPI_DOUBLE, MPI_SUM, comm_);
      out.swap(g);
    }
#endif
    return out;
  }

  /// The number of unmasked (fluid) staggered momentum cells per component -- the exact discrete
  /// datum the reaction identity is stated against: at steady state, sum_bodies F_c = f_c * N_c.
  std::array<long, 3> fluidMomentumCells() {
    std::array<long, 3> out{0, 0, 0};
    CCExec space;
    const C3 e = e_;
    for (int c = 0; c < 3; ++c) {
      CCConst mk = CCConst(C[c].mask);
      long n = 0;
      Kokkos::parallel_reduce(
          "peclet::flow::fluid_cells",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                         {e.x - G, e.y - G, e.z - G}),
          KOKKOS_LAMBDA(int x, int y, int z, long& acc) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            if (mk(i) <= 0.5)
              ++acc;
          },
          n);
      out[(std::size_t)c] = n;
    }
    space.fence();
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      std::array<long, 3> g{0, 0, 0};
      MPI_Allreduce(out.data(), g.data(), 3, MPI_LONG, MPI_SUM, comm_);
      out = g;
    }
#endif
    return out;
  }

  /// R0 DECOMPOSITION PROBE. The reaction identity in its full discrete form is
  ///     sum_bodies F_c  =  f_c*N_c + sum_i fb_i + sum_i A_i  -  sum_i (rho/dt)(u_i - u^n_i)
  /// (every RHS term of the composed step, summed over the FLUID momentum cells; the viscous
  /// fluxes and grad(pi) telescope to zero over the whole fluid region). The Stokes gate drops the
  /// last two terms because they vanish at steady state and A is absent; with advection on they do
  /// not, so this returns them and the identity can be checked term by term instead of being
  /// quietly absorbed. Returns 6 doubles: the three unsteady sums, then the three advective sums.
  ///
  /// sum_i A_i is NOT zero in general and that is a property of the ADVECTION OPERATOR, not of the
  /// budget: the flux form telescopes over the interior, leaving the advective momentum flux
  /// through the fluid region's boundary, which at a cut wall is reconstructed from stencils that
  /// read the masked (wall-velocity) value one or two cells inside the solid. It is an O(h) wall
  /// term and converges away under refinement -- measure it, do not assume it.
  std::vector<double> reactionBudgetTerms() {
    std::vector<double> out(6, 0.0);
    CCExec space;
    const C3 e = e_;
    const double idt = rho_ / dt_;
    const bool haveA = advect_ && haveAdvRhs_ && advRhs_[0].extent(0) == n_;
    for (int c = 0; c < 3; ++c) {
      CCConst un = CCConst(old_[c]), uc = CCConst(C[c].u), mk = CCConst(C[c].mask);
      CCConst av = haveA ? CCConst(advRhs_[c]) : CCConst();
      double su = 0.0, sa = 0.0;
      Kokkos::parallel_reduce(
          "peclet::flow::budget_terms",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                         {e.x - G, e.y - G, e.z - G}),
          KOKKOS_LAMBDA(int x, int y, int z, double& au, double& aa) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            if (mk(i) > 0.5)
              return;
            au += idt * (uc(i) - un(i));
            aa += av.data() ? av(i) : 0.0;
          },
          su, sa);
      out[(std::size_t)c] = su;
      out[(std::size_t)c + 3] = sa;
    }
    space.fence();
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      std::vector<double> g(6, 0.0);
      MPI_Allreduce(out.data(), g.data(), 6, MPI_DOUBLE, MPI_SUM, comm_);
      out.swap(g);
    }
#endif
    return out;
  }

  /// A_wall EXACTNESS PROBE (diagnostic for the Layer-4 force integral). For any smooth field q,
  ///     sum_cells q(x_c) * A_wall,cell  ->  integral over the wall of q n_fluid dA
  /// and taking q = x_a turns that, by the divergence theorem applied to the SOLID interior, into
  /// exactly -V_solid along axis a and 0 on the others. So this returns
  ///     [ sum_c x_c*Ax , sum_c y_c*Ay , sum_c z_c*Az ]  (per axis, summed over all instances)
  /// which must equal -V_solid componentwise if the aperture wall-area vectors are right. It
  /// isolates the GEOMETRY from the traction: a force deficit that shows up here is A_wall's, and
  /// one that does not is the pressure / velocity-gradient reconstruction's.
  std::array<double, 3> wallAreaProbe() {
    std::array<double, 3> out{0, 0, 0};
    if (!hasScene_)
      return out;
    CCExec space;
    const C3 e = e_, og = og_;
    CCConst oxv = CCConst(ox_), oyv = CCConst(oy_), ozv = CCConst(oz_);
    double sx = 0, sy = 0, sz = 0;
    Kokkos::parallel_reduce(
        "peclet::flow::wall_area_probe",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& ax, double& ay, double& az) {
          const long st[3] = {1, e.x, (long)e.x * e.y};
          const long i = (long)x + (long)y * st[1] + (long)z * st[2];
          ax += (double)(x - G + og.x) * -(oxv(i + st[0]) - oxv(i));
          ay += (double)(y - G + og.y) * -(oyv(i + st[1]) - oyv(i));
          az += (double)(z - G + og.z) * -(ozv(i + st[2]) - ozv(i));
        },
        sx, sy, sz);
    space.fence();
    out = {sx, sy, sz};
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      std::array<double, 3> g{0, 0, 0};
      MPI_Allreduce(out.data(), g.data(), 3, MPI_DOUBLE, MPI_SUM, comm_);
      out = g;
    }
#endif
    return out;
  }

  /// Net wall flux this rank injects, sum over inner cells of u_w . A_wall -- the compatibility
  /// datum of the singular pressure problem. Exactly zero for a translating body in a periodic
  /// box (the aperture differences telescope); small but nonzero for rotation and for a body
  /// crossing a non-periodic boundary. Reported, not corrected.
  double wallFluxImbalance() {
    if (!hasScene_ || !hasMotion_ || uwCell_[0].extent(0) != n_)
      return 0.0;
    CCExec space;
    const C3 e = e_;
    CCConst oxv = CCConst(ox_), oyv = CCConst(oy_), ozv = CCConst(oz_);
    CCConst wx = CCConst(uwCell_[0]), wy = CCConst(uwCell_[1]), wz = CCConst(uwCell_[2]);
    double sum = 0.0;
    Kokkos::parallel_reduce(
        "peclet::flow::wall_flux_sum",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)x + (long)y * sy + (long)z * sz;
          acc += wx(i) * -(oxv(i + sx) - oxv(i)) + wy(i) * -(oyv(i + sy) - oyv(i)) +
                 wz(i) * -(ozv(i + sz) - ozv(i));
        },
        sum);
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      double g = 0;
      MPI_Allreduce(&sum, &g, 1, MPI_DOUBLE, MPI_SUM, comm_);
      return g;
    }
#endif
    return sum;
  }

  // Empty when the geometry is static -> ibmModifyStencil takes its scalar u_bc path, unchanged.
  CCConst wallVelView(int c) const {
    return uBc_[c].extent(0) == n_ ? CCConst(uBc_[c]) : CCConst();
  }

  void rebuildStencils() {
    const double idiag = rho_ / dt_, beta = mu_;
    if (varProps_)
      fillMuGhosts();  // face means read mu at i +- stride (boundary inner cells -> ghosts)
    if (varRho_)
      fillPropGhosts(rhoField_);
    for (int c = 0; c < 3; ++c) {
      Kokkos::deep_copy(C[c].rscale, 1.0);
      Kokkos::deep_copy(C[c].inhom, 0.0);
      if (varProps_ || effVarRho())
        ibmBuildDiffusionVar(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e_.x,
                             e_.y, e_.z, G, makeFaceProps(c));
      else
        ibmBuildDiffusion(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e_.x, e_.y,
                          e_.z, beta, idiag);
      ibmModifyStencil(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, C[c].inhom,
                       C[c].rscale, C[c].ov, C[c].nCut, 0.0f, wallVelView(c));
      if (hasDrag_)
        addDragDiagonal(c);
    }
  }
  // copy the nx*ny*nz inner cells between two extended blocks of different ghost width (g=2 <-> g=1
  // MG).
  void copyInner(CCField dst, C3 de, int dg, CCConst src, C3 se, int sg) {
    CCExec space;
    const int NX = nx_, NY = ny_;
    Kokkos::parallel_for(
        "peclet::flow::copyInner", Kokkos::RangePolicy<CCExec>(space, 0, (long)nx_ * ny_ * nz_),
        KOKKOS_LAMBDA(long c) {
          const int ix = (int)(c % NX), iy = (int)((c / NX) % NY), iz = (int)(c / ((long)NX * NY));
          const long di =
              (long)(ix + dg) + (long)(iy + dg) * de.x + (long)(iz + dg) * (long)de.x * de.y;
          const long si =
              (long)(ix + sg) + (long)(iy + sg) * se.x + (long)(iz + sg) * (long)se.x * se.y;
          dst(di) = src(si);
        });
  }
  // Copy the ENTIRE destination block (including its ghost ring) from the source block at per-axis
  // cell offset `off`: dst(x,y,z) <- src(x+off, y+off, z+off). Bridges a G=2 field to the g=1 MG
  // block INCLUDING the g=1 ghosts (off = G-1), so face means at the first inner cell read a valid
  // neighbour. Requires the source ghosts filled (fillGhosts/fillPropGhosts) — under MPI those are
  // the cross-rank values, so the bridge is decomposition-correct.
  void copyBlockShifted(CCField dst, C3 de, CCConst src, C3 se, int off) {
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::copyBlockShifted",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {de.x, de.y, de.z}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long di = (long)x + (long)y * de.x + (long)z * (long)de.x * de.y;
          const long si =
              (long)(x + off) + (long)(y + off) * se.x + (long)(z + off) * (long)se.x * se.y;
          dst(di) = src(si);
        });
  }
  // Fill ghost width G periodically on all 3 axes (x then y then z, covering corners). Distributed:
  // the velocity-block halo (cross-rank + periodic, all ghosts incl. corners).
  void fillGhosts(CCField f) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      velDev_->exchange(f);
      return;
    }
#endif
    fillAxis(f, 0);
    fillAxis(f, 1);
    fillAxis(f, 2);
  }
  // Fused periodic FACE-ghost fill in ONE kernel (vs 3 fillAxis): each inner boundary cell scatters
  // its periodic image to the opposite face ghost, all 3 axes at once. Valid only for
  // FACE-neighbour (7-point) stencils -- it does NOT fill the corner/edge ghosts (which fillAxis's
  // sequential x->y->z does). The IBM RB-GS smoother reads only the 7-point stencil, so this is
  // exact there and cuts the velocity solve's dominant kernel-launch cost (~7200 -> ~2400 fill
  // launches/step) at low resolution. NOT for the Koren advection RHS (reads diagonals) -- keep the
  // full fillGhosts there.
  void fillGhostsFaces(CCField f) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      velDev_->exchange(f);
      return;
    }  // halo gives all ghosts; the 7-pt smoother uses the faces
#endif
    CCExec space;
    C3 e = e_;
    const int Nx = nx_, Ny = ny_, Nz = nz_;
    const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
    CCField ff = f;
    Kokkos::parallel_for(
        "peclet::flow::ibm_facefill", Kokkos::RangePolicy<CCExec>(space, 0, (long)nx_ * ny_ * nz_),
        KOKKOS_LAMBDA(long n) {
          const int ix = (int)(n % Nx), iy = (int)((n / Nx) % Ny), iz = (int)(n / ((long)Nx * Ny));
          const long i = (long)(ix + G) * sx + (long)(iy + G) * sy + (long)(iz + G) * sz;
          if (ix < G)
            ff(i + (long)Nx * sx) = ff(i);
          else if (ix >= Nx - G)
            ff(i - (long)Nx * sx) = ff(i);
          if (iy < G)
            ff(i + (long)Ny * sy) = ff(i);
          else if (iy >= Ny - G)
            ff(i - (long)Ny * sy) = ff(i);
          if (iz < G)
            ff(i + (long)Nz * sz) = ff(i);
          else if (iz >= Nz - G)
            ff(i - (long)Nz * sz) = ff(i);
        });
  }
  void fillAxis(CCField f, int axis) {
    CCExec space;
    C3 e = e_;
    int N3[3] = {nx_, ny_, nz_};
    int dims[3] = {e.x, e.y, e.z};
    long st[3] = {1, e.x, (long)e.x * e.y};
    const int a = axis, b = (axis + 1) % 3, c = (axis + 2) % 3;
    const long sa = st[a], sb = st[b], sc = st[c];
    const int N = N3[a];
    CCField ff = f;
    Kokkos::parallel_for(
        "peclet::flow::ibm_pfill",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
        KOKKOS_LAMBDA(int p0, int p1) {
          const long base = (long)p0 * sb + (long)p1 * sc;
          for (int gl = 0; gl < G; ++gl) {
            ff(base + (long)gl * sa) = ff(base + (long)(gl + N) * sa);
            ff(base + (long)(G + N + gl) * sa) = ff(base + (long)(G + gl) * sa);
          }
        });
  }
  // Cell divergence of the current velocity iterate, on the inner cells + one ghost ring (the RHS
  // compensation reads div at i and i-strd, so faces at the low inner boundary need the ghost-cell
  // value; velocity ghosts were just filled). Porous-only scratch (divAdv_).
  void computeDivAdv() {
    CCExec space;
    C3 e = e_;
    CCField dv = divAdv_;
    CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u);
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "peclet::flow::div_adv",
        MD(space, {G - 1, G - 1, G - 1}, {e.x - G + 1, e.y - G + 1, e.z - G + 1}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long sx = 1, sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)x + (long)y * sy + (long)z * sz;
          dv(i) = (U(i + sx) - U(i)) + (V(i + sy) - V(i)) + (W(i + sz) - W(i));
        });
  }

  /// R0 helper: allocate (once) and arm the per-component advective-term stash the reaction-force
  /// budget consumes. Returns whether the RHS kernel should write it. Off (and untouched) unless a
  /// scene is installed on the staggered grid with explicit advection on, so every other path is
  /// byte-identical and pays no memory.
  // --- A0: wall-aware advection inputs (advective cut-wall flux) -------------------------------
  //
  // The momentum advection kernels (`sadv::advect / advect_sou / advect_fou / fou_operator`) are
  // geometry-blind: `adv_vel` averages the solid-masked rows into the advecting face velocity and
  // the SOU/Koren `PHI` stencils read the advected field up to 2 cells INSIDE the solid.
  // `maskVelocity` pins those rows to 0.0, which IS the wall velocity for a STATIC wall -- so a
  // static scene only carries the O(h) aperture defect (D2, docs/ANALYTIC_SDF_GEOMETRY.md §7 item
  // 8, measured -0.4..-1% on the 4-sphere bed). For a body moving at u_wall the advective term
  // near the body is wrong by O(u_wall): an O(1) local error in exactly the term that produces the
  // finite-Re screening of the confined wall correction (D1). This is the same defect family the
  // fresh-cell seed closed for the TIME term (seedFreshCells); the advective term never got it.
  //
  // The fix does NOT touch the global mask convention -- the masked zeros are load-bearing for the
  // viscous/pressure operators and for the reaction budget's telescoping. Instead the advection
  // gets its own input: a copy of u whose masked rows hold `uBc_`, the local rigid-body wall
  // velocity `v_inst + omega_inst x r` that `buildWallVelocity` already evaluates at component c's
  // staggered points over the WHOLE extended block (ghosts included -- the scene is analytic, so
  // ghost solid rows are computable pointwise and no extra exchange is needed).
  //
  // Both the explicit path (buildRhs / buildRhsForced / buildRhsVar) and the implicit-FOU stencil
  // path (buildAdvStencil / buildAdvStencilVar) read these views, or the deferred correction
  // rho*(aF - aK) would be assembled from two different velocity fields. The advRhs_ stash (the
  // reaction budget's R0 term) therefore carries the corrected term automatically.
  //
  // The fluid rows are the current Picard iterate u^k, so the COPY is per Picard iteration; the
  // WALL rows come from uBc_, which depends only on instance motion and is built once per
  // geometry/motion update.
  //
  // ABLATION: `PECLET_FLOW_ADV_WALLVEL=0` restores the pre-A0 behaviour (masked zeros in the
  // advection inputs) without a rebuild -- the instrument that differences "zeros vs wall velocity
  // in the advective term" directly. Everything else on the moving path is untouched by it.
  static bool advWallVelEnabled() {
    static const bool en = [] {
      const char* v = std::getenv("PECLET_FLOW_ADV_WALLVEL");
      return !(v && v[0] == '0');
    }();
    return en;
  }
  bool advWallInputs() const {
    return advWallVelEnabled() && !Grid::collocated && hasScene_ && hasMotion_ && advect_ &&
           uBc_[0].extent(0) == n_ && C[0].mask.extent(0) == n_;
  }
  void buildAdvInputs() {
    if (!advWallInputs())
      return;
    CCExec space;
    for (int c = 0; c < 3; ++c) {
      if (uwAdv_[c].extent(0) != n_)
        uwAdv_[c] = CCField("uwAdv", n_);
      Kokkos::deep_copy(uwAdv_[c], C[c].u);
      CCField a = uwAdv_[c];
      CCConst m = CCConst(C[c].mask), w = CCConst(uBc_[c]);
      Kokkos::parallel_for(
          "peclet::flow::adv_wall_inputs", Kokkos::RangePolicy<CCExec>(space, 0, (long)n_),
          KOKKOS_LAMBDA(long i) {
            if (m(i) > 0.5)
              a(i) = w(i);
          });
    }
    space.fence();
  }
  /// The velocity view the advection operators must read for component c: the wall-corrected
  /// scratch while an instance is moving, the live field (byte-identical) otherwise.
  CCConst advVelView(int c) const {
    return advWallInputs() ? CCConst(uwAdv_[c]) : CCConst(C[c].u);
  }

  bool ensureAdvStash(int c, bool adv) {
    const bool want = !Grid::collocated && hasScene_ && adv;
    if (want && advRhs_[c].extent(0) != n_)
      advRhs_[c] = CCField("advRhs", n_);
    haveAdvRhs_ = want;
    return want;
  }

  void buildRhs(int c) {
    CCExec space;
    const double idiag = rho_ / dt_, fc = f_[c], rho = rho_;
    C3 e = e_;
    CCField bb = C[c].b, rs = C[c].rscale, P = P_, brhs = bcBrhs_[c], inh = C[c].inhom;
    // A0: U/V/W (the advecting velocities) and aP (the advected field) come from the wall-aware
    // advection inputs -- identical to C[*].u unless an instance is moving. `uu` stays the live
    // field: its only other consumer is the porous advection-form compensation.
    CCConst U = advVelView(0), V = advVelView(1), W = advVelView(2), aP = advVelView(c),
            uu = CCConst(C[c].u), un = CCConst(old_[c]);
    const long strd = (c == 0) ? 1 : (c == 1) ? e_.x : (long)e_.x * e_.y;
    // Pure implicit FOU (no deferred correction): 1st-order upwind carried entirely by the
    // operator, no explicit high-order term in the RHS -- maximally dissipative/stable (diffuses
    // sharp shear layers). Only meaningful on an implicit-advection path.
    const bool pureFou = implicitAdv() && !deferredCorr_;
    const bool incr = cutcellPressure_ && incremental_, adv = advect_ && !pureFou,
               bc = hasBc_ && !bcStencilPath();  // fold RHS only on the const-coeff domain-BC path;
    // on the stencil path (solid and/or implicit advection) the walls enter via reflection ghosts
    // (smoothComp) and the RHS carries the IBM inhom (=0 for no-slip) + the deferred correction.
    // incr predictor carries -grad(P^n).
    const bool ifou =
        implicitAdv() &&
        deferredCorr_;           // deferred correction: keep (HO - FOU) explicit in the RHS
                                 // (implicit on the domain-BC path by default, opt-in elsewhere)
    const int sch = advScheme_;  // 0 = SOU (default), 1 = Koren TVD
    // R0: stash the explicit advective term for the reaction-force budget (staggered scenes only).
    // The LAST Picard iteration overwrites, matching uStar_'s convention -- that is the RHS the
    // final momentum solve actually saw.
    const bool sa = ensureAdvStash(c, adv);
    CCField ar = advRhs_[c];
    // Mode-2 wall-aware pressure force (collocated): -grad(P) = the TRANSPOSE of the wall-aware
    // cell->face constraint interpolation, precomputed per component (the plain path's central
    // difference is the transpose of the plain 1/2-1/2 average, so this keeps the momentum/
    // constraint operators an adjoint pair on both paths).
    const bool tg = Grid::collocated && faceInterp_ >= 2 && faceInterp_ <= 5 && incr;
    // modes 6/7: openness-weighted -grad(P^n) predictor, matching the fs-weighted correction
    const bool wg = Grid::collocated && (faceInterp_ == 6 || faceInterp_ == 7) && incr;
    // mode 11: adjoint-aperture -grad(P^n) predictor G = -(D_a Pi)^T (centerGradAperture) --
    // support-consistent AND adjoint; matches the mode-11 correction so momentum and constraint
    // stay one operator family.
    const bool ag = Grid::collocated && (faceInterp_ >= 11 && faceInterp_ <= 13) && incr;
    // ghost mode (and the mode-9/10 cutcell-ghost hybrids): directional gpCenterGrad predictor —
    // the mode-0 central difference reads the decoupled P=0 at solid-centered cells, a
    // gauge-dependent O(1) gradient error at every cut cell (measured O(1/h) in physical units,
    // ghost_collocated_apriori.py [C2]).
    const bool gg =
        Grid::collocated && (ghostProjection_ || faceInterp_ == 9 || faceInterp_ == 10) && incr;
    if constexpr (Grid::collocated) {
      if (gg) {
        gpCenterGrad(tgp_, CCConst(P_), CCConst(ghostProjection_ ? sdfGp_ : sdf_), c, e_, G, gauge2a_);
      } else if (tg) {
        CCField xcs[3] = {xcx_, xcy_, xcz_};
        CCField oax[3] = {ox_, oy_, oz_};
        transposeGradWallAware(tgp_, CCConst(P_), CCConst(sdf_), CCConst(oax[c]), CCConst(xcs[c]),
                               faceInterp_ >= 3, c, e_, G);
      } else if (wg) {
        CCField oax[3] = {ox_, oy_, oz_};
        centerGradOpen(tgp_, CCConst(P_), CCConst(oax[c]), c, e_, G);
      } else if (ag) {
        CCField oax[3] = {ox_, oy_, oz_};
        if (faceInterp_ == 12)
          centerGradApertureScaled(tgp_, CCConst(P_), CCConst(ox_), CCConst(oy_), CCConst(oz_), c,
                                   e_, G);
        else if (faceInterp_ == 13)
          centerGradOpenCapped(tgp_, CCConst(P_), CCConst(oax[c]), c, apertureFloor_, e_, G);
        else
          centerGradAperture(tgp_, CCConst(P_), CCConst(oax[c]), c, e_, G);
      }
    }
    CCConst gpw = CCConst(tgp_);  // empty view on the staggered path (tg/wg/gg/ag false there)
    // Mode-4 fully-FV momentum via DEFECT CORRECTION: solve M·u^{k+1} = M·u^k − rs·L_FV(u^k) +
    // rs·b_FV so the fixed point satisfies the second-order finite-volume balance L_FV·u* = b_FV
    // exactly, with the (stable, small-cell-safe) IBM matrix M only as preconditioner. fvM_ = M·u^k
    // (stencilMatvec), fvL_ = L_FV(u^k) (fvViscousApply: o_f faces + cs time + centroid wall drag).
    // Interior cells: M = L_FV → the defect vanishes → byte-identical to mode 0. Stokes only
    // (advection folds into the IBM matrix, not yet into L_FV).
    // Porous advection-form compensation (+rho*u_f*div(u)_f): see the step() comment. Off (and the
    // view untouched) on every non-porous path.
    const bool pc = porous_ && advect_;
    CCConst dv = CCConst(divAdv_);
    const bool wd = Grid::collocated && faceInterp_ >= 4 && faceInterp_ <= 7;
    if constexpr (Grid::collocated)
      if (wd) {
        stencilMatvec(fvM_, CCConst(C[c].u), FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE),
                      FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT), e_, G);
        // modes 5/6: TRUE-NORMAL embed wall drag (embedDirichletGradient); mode 4: axis-by-axis W_a
        // g_a
        if (faceInterp_ >= 5)
          embedViscousApply(fvL_, CCConst(C[c].u), CCConst(sdf_), CCConst(cs_), CCConst(ox_),
                            CCConst(oy_), CCConst(oz_), mu_, rho_ / dt_, e_, G);
        else
          fvViscousApply(fvL_, CCConst(C[c].u), CCConst(sdf_), CCConst(cs_), CCConst(ox_),
                         CCConst(oy_), CCConst(oz_), mu_, rho_ / dt_, e_, G);
      }
    CCConst fvM = CCConst(fvM_), fvL = CCConst(fvL_), cs = CCConst(cs_);
    const double fvw = fvRelax_;  // local copy — a KOKKOS_LAMBDA must not read a member (device
                                  // deref of the host `this` pointer = illegal memory access)
    // b = descale*(idiag*u^n - rho*Koren(u^k) + rho*FOU(u^k) + f - grad P^n) - inhom  (+ BC fold
    // brhs). The time base is u^n (Picard); the advecting velocity & advected field are the current
    // iterate u^k.
    ccFor3(
        "rhs", C3{G, G, G}, C3{e.x - G, e.y - G, e.z - G},
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double aK = 0.0, aF = 0.0;
          if (adv) {
            sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y}, Fa{aP, e.x, e.y};
            aK = (sch == 0) ? Grid::advect_sou(c, x, y, z, Ua, Va, Wa, Fa)
                            : Grid::advect(c, x, y, z, Ua, Va, Wa, Fa);
            if (ifou)
              aF = Grid::advect_fou(c, x, y, z, Ua, Va, Wa, Fa);
          }
          if (sa)
            ar(i) = rho * (aF - aK);
          // incremental predictor's -grad(P^n): central-difference cell gradient on the collocated
          // grid (or the wall-aware transpose gradient, mode 2), one-sided face gradient (P at the
          // high cell of the staggered face) on the staggered grid.
          const double gp =
              !incr ? 0.0
              : Grid::collocated
                  ? ((tg || wg || gg || ag) ? gpw(i) : 0.5 * (P((long)i + strd) - P((long)i - strd)))
                  : (P(i) - P((long)i - strd));
          if (wd) {  // FV defect-correction RHS  M·u − ω·rs·(L_FV·u − b_FV),  b_FV = idt·cs·u^n +
                     // cs·(f − grad P). ω<1 damps the (stiff, explicit-lagged) wall-flux
                     // correction; the fixed point L_FV·u* = b_FV is independent of ω.
            const double bfv = idiag * cs(i) * un(i) + cs(i) * (fc - gp);
            bb(i) = fvM(i) - fvw * rs(i) * (fvL(i) - bfv);
          } else {
            const double comp = pc ? rho * uu(i) * 0.5 * (dv(i) + dv((long)i - strd)) : 0.0;
            bb(i) = rs(i) * (idiag * un(i) + fc - rho * aK + rho * aF + comp - gp) +
                    (bc ? brhs(i) : -inh(i));
          }
        });  // BC fold (brhs) on the domain-BC path; -inhom on the IBM path (=0 for no-slip)
  }
  // Sibling of buildRhs adding a per-cell body force fb(i) (Boussinesq buoyancy / CFD-DEM
  // feedback): the constant fc becomes fc + fb(i). Kept as a separate kernel so buildRhs stays
  // byte-identical (no codegen drift on the single-phase path). Selected in step() when
  // hasCellForce_.
  void buildRhsForced(int c) {
    CCExec space;
    const double idiag = rho_ / dt_, fc = f_[c], rho = rho_;
    C3 e = e_;
    CCField bb = C[c].b, rs = C[c].rscale, P = P_, brhs = bcBrhs_[c], inh = C[c].inhom;
    CCConst fb = CCConst(cellForce_[c]);
    // A0: U/V/W (the advecting velocities) and aP (the advected field) come from the wall-aware
    // advection inputs -- identical to C[*].u unless an instance is moving. `uu` stays the live
    // field: its only other consumer is the porous advection-form compensation.
    CCConst U = advVelView(0), V = advVelView(1), W = advVelView(2), aP = advVelView(c),
            uu = CCConst(C[c].u), un = CCConst(old_[c]);
    const long strd = (c == 0) ? 1 : (c == 1) ? e_.x : (long)e_.x * e_.y;
    const bool pureFou = implicitAdv() && !deferredCorr_;
    const bool incr = cutcellPressure_ && incremental_, adv = advect_ && !pureFou,
               bc = hasBc_ && !bcStencilPath();
    const bool ifou = implicitAdv() && deferredCorr_;
    const int sch = advScheme_;
    const bool sa = ensureAdvStash(c, adv);  // R0: see buildRhs
    CCField ar = advRhs_[c];
    // Porous advection-form compensation (+rho*u_f*div(u)_f): see the step() comment.
    const bool pc = porous_ && advect_;
    CCConst dv = CCConst(divAdv_);
    const bool tg = Grid::collocated && faceInterp_ >= 2 && faceInterp_ <= 5 &&
                    incr;  // wall-aware -grad(P) (mode 2/3)
    const bool gg =
        Grid::collocated && (ghostProjection_ || faceInterp_ == 9 || faceInterp_ == 10) &&
        incr;  // directional ghost -grad(P)
    const bool ag =
        Grid::collocated && (faceInterp_ >= 11 && faceInterp_ <= 13) && incr;  // adjoint-aperture
    if constexpr (Grid::collocated) {
      if (gg) {
        gpCenterGrad(tgp_, CCConst(P_), CCConst(ghostProjection_ ? sdfGp_ : sdf_), c, e_, G, gauge2a_);
      } else if (tg) {
        CCField xcs[3] = {xcx_, xcy_, xcz_};
        CCField oax[3] = {ox_, oy_, oz_};
        transposeGradWallAware(tgp_, CCConst(P_), CCConst(sdf_), CCConst(oax[c]), CCConst(xcs[c]),
                               faceInterp_ >= 3, c, e_, G);
      } else if (ag) {
        CCField oax[3] = {ox_, oy_, oz_};
        if (faceInterp_ == 12)
          centerGradApertureScaled(tgp_, CCConst(P_), CCConst(ox_), CCConst(oy_), CCConst(oz_), c,
                                   e_, G);
        else if (faceInterp_ == 13)
          centerGradOpenCapped(tgp_, CCConst(P_), CCConst(oax[c]), c, apertureFloor_, e_, G);
        else
          centerGradAperture(tgp_, CCConst(P_), CCConst(oax[c]), c, e_, G);
      }
    }
    CCConst gpw = CCConst(tgp_);
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "rhs_forced", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double aK = 0.0, aF = 0.0;
          if (adv) {
            sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y}, Fa{aP, e.x, e.y};
            aK = (sch == 0) ? Grid::advect_sou(c, x, y, z, Ua, Va, Wa, Fa)
                            : Grid::advect(c, x, y, z, Ua, Va, Wa, Fa);
            if (ifou)
              aF = Grid::advect_fou(c, x, y, z, Ua, Va, Wa, Fa);
          }
          if (sa)
            ar(i) = rho * (aF - aK);
          const double gp = !incr ? 0.0
                            : Grid::collocated
                                ? ((tg || gg || ag) ? gpw(i)
                                                    : 0.5 * (P((long)i + strd) - P((long)i - strd)))
                                : (P(i) - P((long)i - strd));
          const double comp = pc ? rho * uu(i) * 0.5 * (dv(i) + dv((long)i - strd)) : 0.0;
          bb(i) = rs(i) * (idiag * un(i) + fc + fb(i) - rho * aK + rho * aF + comp - gp) +
                  (bc ? brhs(i) : -inh(i));
        });
  }
  // Variable-density RHS (sibling of buildRhsForced): the time term, the advection weight, and the
  // per-cell body force all use the FACE density of component c (arithmetic mean over the staggered
  // face, matching VarFaceProps::idiag and the projection coefficient — this three-way consistency
  // is what makes discrete hydrostatic balance exact). The cell force fb is face-interpolated for
  // the same reason (a rho*g cell field becomes rho_face*g at the velocity location). Requires the
  // rho ghosts filled (rebuildStencils / buildAdvStencilVar did it this step).
  void buildRhsVar(int c) {
    CCExec space;
    const double idt = 1.0 / dt_, fc = f_[c];
    C3 e = e_;
    CCField bb = C[c].b, rs = C[c].rscale, P = P_, brhs = bcBrhs_[c], inh = C[c].inhom;
    CCConst fb = CCConst(cellForce_[c]);
    CCConst rf = CCConst(effRhoField());
    // A0: U/V/W (the advecting velocities) and aP (the advected field) come from the wall-aware
    // advection inputs -- identical to C[*].u unless an instance is moving. `uu` stays the live
    // field: its only other consumer is the porous advection-form compensation.
    CCConst U = advVelView(0), V = advVelView(1), W = advVelView(2), aP = advVelView(c),
            uu = CCConst(C[c].u), un = CCConst(old_[c]);
    const long strd = strideOf(c);
    const bool pureFou = implicitAdv() && !deferredCorr_;
    const bool incr = cutcellPressure_ && incremental_, adv = advect_ && !pureFou,
               bc = hasBc_ && !bcStencilPath();
    const bool ifou = implicitAdv() && deferredCorr_;
    const int sch = advScheme_;
    // Porous advective-form compensation, weighted by the face density (rho_eff = eps*rho): the
    // eps-weighted ADVECTIVE form eps*rho*(du/dt + u.grad u) IS the conservative volume-averaged
    // momentum given the enforced continuity (the u*[d(eps)/dt + div(eps u)] bracket vanishes).
    const bool pc = porous_ && advect_;
    CCConst dv = CCConst(divAdv_);
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "rhs_var", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double rhoF = 0.5 * (rf(i) + rf(i - strd));  // face density of the velocity unknown
          double aK = 0.0, aF = 0.0;
          if (adv) {
            sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y}, Fa{aP, e.x, e.y};
            aK = (sch == 0) ? Grid::advect_sou(c, x, y, z, Ua, Va, Wa, Fa)
                            : Grid::advect(c, x, y, z, Ua, Va, Wa, Fa);
            if (ifou)
              aF = Grid::advect_fou(c, x, y, z, Ua, Va, Wa, Fa);
          }
          const double gp = !incr              ? 0.0
                            : Grid::collocated ? 0.5 * (P((long)i + strd) - P((long)i - strd))
                                               : (P(i) - P((long)i - strd));
          const double fbF = 0.5 * (fb(i) + fb(i - strd));
          const double comp = pc ? rhoF * uu(i) * 0.5 * (dv(i) + dv((long)i - strd)) : 0.0;
          bb(i) = rs(i) * (rhoF * idt * un(i) + fc + fbF - rhoF * aK + rhoF * aF + comp - gp) +
                  (bc ? brhs(i) : -inh(i));
        });
  }
  // Momentum-consistent sibling of buildRhsVar (rung V2b, WO-K). The validated `buildRhsVar` is not
  // touched; this one differs in exactly one term and drops one.
  //
  //   buildRhsVar :  rho_f/dt * u^n   - rho_f * adv(u^k)   (+ implicit-FOU deferred correction)
  //   this        :  rho_f/dt * u^adv
  //
  // `u^adv` is `(rho^c u_c)/rho^c` after the geometric advection that shared its fluxes with the
  // colour field, so it already contains BOTH the time base and the advection — the Koren/SOU term
  // and the deferred correction are not merely unnecessary here, adding them would advect the
  // momentum twice. Everything else (the incremental -grad(P^n), the face body force, the domain-BC
  // inhomogeneity, the cut-cell rescale) is verbatim, and `rho_f` is the SAME arithmetic face mean
  // the projection coefficient and the face body force use — the three-way consistency that makes
  // hydrostatic balance exact is untouched (see the enableVofMomentum note).
  void buildRhsVarMom(int c) {
    CCExec space;
    const double idt = 1.0 / dt_, fc = f_[c];
    C3 e = e_;
    CCField bb = C[c].b, rs = C[c].rscale, P = P_, brhs = bcBrhs_[c], inh = C[c].inhom;
    CCConst fb = CCConst(cellForce_[c]);
    CCConst rf = CCConst(effRhoField());
    CCConst ua = CCConst(uAdv_[c]);
    const long strd = strideOf(c);
    const bool incr = cutcellPressure_ && incremental_;
    const bool bc = hasBc_ && !bcStencilPath();
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "rhs_var_mom", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double rhoF = 0.5 * (rf(i) + rf(i - strd));
          const double gp = !incr              ? 0.0
                            : Grid::collocated ? 0.5 * (P((long)i + strd) - P((long)i - strd))
                                               : (P(i) - P((long)i - strd));
          const double fbF = 0.5 * (fb(i) + fb(i - strd));
          bb(i) = rs(i) * (rhoF * idt * ua(i) + fc + fbF - gp) + (bc ? brhs(i) : -inh(i));
        });
  }

  // --- rung V8 (WO-T): the collocated predictor when the forces live on the faces ----------------
  //
  // SIBLING of buildRhsVar, reached only when `colocatedFaceForce()` — i.e. only on `SolverColocated`
  // with variable density and/or surface tension, both of which used to throw. It differs from
  // buildRhsVar in exactly three ways, and every one of them is the point of the rung:
  //
  //   * the density weight of the time term and of the advection is the CELL density `rho(i)`, not a
  //     face mean: the collocated velocity unknown IS the cell (`Grid::offset(c) == 0`), and this is
  //     the same placement `VarFaceProps::idiag` now uses for the operator diagonal;
  //   * the incremental `-grad(P^n)` is DROPPED — it is re-applied at the faces, where the pressure
  //     difference `P(i) - P(i-s)` is the projection's own operator;
  //   * the constant body force, the per-cell body force and the CSF are DROPPED for the same
  //     reason. The predictor solves `A u* = (rho/dt) u^n - rho*adv(u^k)` and nothing else.
  //
  // What survives verbatim: the cut-cell rescale `rs`, the domain-BC fold / IBM inhomogeneity, the
  // Koren/SOU advection and its implicit-FOU deferred correction.
  void buildRhsColoFF(int c) {
    CCExec space;
    const double idt = 1.0 / dt_, rhoIdtC = rho_ / dt_, rhoK = rho_;
    C3 e = e_;
    CCField bb = C[c].b, rs = C[c].rscale, brhs = bcBrhs_[c], inh = C[c].inhom;
    const bool haveRho = effVarRho();
    // Unread placeholder when the density is constant (a Kokkos View must still be a live handle).
    CCConst rf = CCConst(haveRho ? effRhoField() : C[c].rscale);
    CCConst U = advVelView(0), V = advVelView(1), W = advVelView(2), aP = advVelView(c),
            un = CCConst(old_[c]);
    const bool pureFou = implicitAdv() && !deferredCorr_;
    const bool adv = advect_ && !pureFou, bc = hasBc_ && !bcStencilPath();
    const bool ifou = implicitAdv() && deferredCorr_;
    const int sch = advScheme_;
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "rhs_colo_ff", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double rhoC = haveRho ? rf(i) : rhoK;
          const double dg = haveRho ? rhoC * idt : rhoIdtC;
          double aK = 0.0, aF = 0.0;
          if (adv) {
            sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y}, Fa{aP, e.x, e.y};
            aK = (sch == 0) ? Grid::advect_sou(c, x, y, z, Ua, Va, Wa, Fa)
                            : Grid::advect(c, x, y, z, Ua, Va, Wa, Fa);
            if (ifou)
              aF = Grid::advect_fou(c, x, y, z, Ua, Va, Wa, Fa);
          }
          bb(i) = rs(i) * (dg * un(i) - rhoC * aK + rhoC * aF) + (bc ? brhs(i) : -inh(i));
        });
  }

  // Add the face acceleration a_f = dt*(f_f - grad_f(P^n))/rho_f to the just-averaged face field,
  // and REMEMBER it in faceAcc_ so the cell counterpart can average exactly the same numbers.
  // Called from project() immediately after centerToFace, before the divergence. See
  // collocated_varrho.hpp.
  void applyFaceAcceleration() {
    requireCollocatedFaceForceScope("project");
    ensureFaceAcc();
    const bool haveRho = effVarRho();
    const bool incr = cutcellPressure_ && incremental_;
    CCField fa[3] = {uf_, vf_, wf_};
    CCField oax[3] = {ox_, oy_, oz_};
    CCConst rho = CCConst(haveRho ? effRhoField() : C[0].rscale);
    for (int c = 0; c < 3; ++c) {
      const long sc = strideOf(c);
      buildFaceAccelVar(faceAcc_[c], CCConst(P_), rho,
                        CCConst(hasCellForce_ ? cellForce_[c] : C[c].rscale), hasCellForce_,
                        CCConst(oax[c]), haveRho, rho_, f_[c], incr, dt_, sc, e_, G);
      if (csfActive())
        addFaceAccelCsf(faceAcc_[c], CCConst(cField_), CCConst(kappaField_), CCConst(kappaBranch_),
                        rho, CCConst(oax[c]), haveRho, rho_, sigmaCsf_, vofAdv_.h(), dt_, sc, e_, G);
      addFaceIncrement(fa[c], CCConst(faceAcc_[c]), e_, G);
    }
  }

  // The cell counterpart of the face path: turn faceAcc_ into the TOTAL face velocity increment of
  // this step (force acceleration minus the projection's own face correction) and give each cell the
  // openness-gated average of its two faces. Called from project() in place of the constant-density
  // cell-correction chain.
  // KNOWN GAP (recorded, not guarded): at an OUTFLOW face `bcCorrectOutflow` adjusts `uf_` after
  // `projectCorrectVar`, and that adjustment is NOT mirrored into faceAcc_, so the cell average at
  // the last row before an outflow face would miss it. Every rung-V8 gate is periodic or walled;
  // an open boundary on the collocated variable-density path is untested (WO-R owns the staggered
  // `bcCorrectOutflowVar`).
  void applyCellFaceAverageCorrection() {
    CCField oax[3] = {ox_, oy_, oz_};
    const bool haveRho = effVarRho();
    CCConst rho = CCConst(haveRho ? effRhoField() : C[0].rscale);
    for (int c = 0; c < 3; ++c) {
      const long sc = strideOf(c);
      faceAccelSubGradPhi(faceAcc_[c], CCConst(phi_), rho, CCConst(oax[c]), haveRho, rho_, sc, e_,
                          G);
      applyCellFaceAverage(C[c].u, CCConst(faceAcc_[c]), CCConst(oax[c]), sc, e_, G);
    }
  }

  // --- balanced-force CSF (rung V4, WO-P) ------------------------------------------------------
  //
  // ADDITIVE to whichever RHS builder just ran (`buildRhs` / `buildRhsForced` / `buildRhsVar` /
  // `buildRhsVarMom`), because the force is independent of which time term and which advection form
  // the configuration selected — and because those four are validated kernel bodies (hard rule 1).
  // It is applied at the same point in the RHS as the incremental `-(P(i) - P(i - s_c))` and
  // carries the same `rs(i)` cut-cell rescale every other RHS term carries.
  //
  // WHY NOT THROUGH THE PER-CELL FORCE FIELD. `cellForce_` is the natural conduit for a body force
  // and its ghosts are sound since WO-G, but its face rule is the ARITHMETIC INTERPOLATION
  // `½(f(i) + f(i - s_c))` of a cell-centred force. That rule is exactly right for `ρg` (the pair
  // `f_f/ρ_f` is then the intended acceleration) and exactly wrong for `σκ∇C`: an interpolated
  // cell-centred `σκ∇C` is not in the range of the discrete gradient operator the projection
  // inverts, so the projection cannot annihilate it and the residue is the classical spurious
  // current. The face value here is instead formed BY the projection's own operator — the same
  // difference `C(i) - C(i - s_c)` that `projectCorrectVar` applies to φ and `buildRhsVar` applies
  // to P. See `vof/surface_tension.hpp` for the full argument; the stationary-droplet gate is what
  // measures it, and it fails loudly on any other choice.
  //
  // `sigmaCsf_ == 0` (the default) never reaches here: `csfActive()` gates the call site, so every
  // non-VoF path is byte-identical.
  void addCsfRhs(int c) {
    CCExec space;
    C3 e = e_;
    CCField bb = C[c].b;
    CCConst rs = CCConst(C[c].rscale), cv = CCConst(cField_), kp = CCConst(kappaField_),
            kb = CCConst(kappaBranch_);
    const long strd = strideOf(c);
    const double sig = sigmaCsf_, h = vofAdv_.h();
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "csf_rhs", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double dC = cv(i) - cv(i - strd);
          if (dC == 0.0)
            return;  // no interface across this face -> no force, and no orphan either
          double kf = 0.0;
          vof::csfFaceCurvature(kp(i - strd), kb(i - strd), kp(i), kb(i), kf);
          bb(i) += rs(i) * vof::csfFaceForce(sig, kf, dC, h);
        });
  }
  // ABLATION (`set_csf_mode(1)`): the same physics discretized the OTHER plausible way — a
  // cell-centred force `f(j) = sigma*kappa(j)*(C(j+s) - C(j-s))/2h` interpolated to the face with
  // the arithmetic mean `1/2 (f(i) + f(i-s))`, exactly as the per-cell body-force machinery would
  // carry a `rho*g` field. It is consistent, it converges, and it is WRONG for surface tension: the
  // face value is no longer in the range of the projection's discrete gradient, so the projection
  // cannot annihilate it. This kernel exists so the difference is a measured number in the ctest
  // rather than an argument — the same role the harmonic-rho_f ablation plays for WO-J's
  // hydrostatic gate. NEVER a production path.
  void addCsfRhsCellInterp(int c) {
    CCExec space;
    C3 e = e_;
    CCField bb = C[c].b;
    CCConst rs = CCConst(C[c].rscale), cv = CCConst(cField_), kp = CCConst(kappaField_),
            kb = CCConst(kappaBranch_);
    const long strd = strideOf(c);
    const double sig = sigmaCsf_, h = vofAdv_.h();
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "csf_rhs_cellinterp", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double f[2] = {0.0, 0.0};
          for (int q = 0; q < 2; ++q) {  // q = 0 -> cell i-s_c, q = 1 -> cell i
            const long j = i - (1 - q) * strd;
            if (!vof::csfKappaDefined(kb(j)))
              continue;
            f[q] = sig * kp(j) * 0.5 * (cv(j + strd) - cv(j - strd)) / h;
          }
          bb(i) += rs(i) * 0.5 * (f[0] + f[1]);
        });
  }
  // Census of the CSF face force over this rank's inner region, on the CURRENT colour + curvature
  // fields: the max |F| per component, and the number of ORPHAN faces — faces across which the
  // colour jumps by more than the wisp threshold but neither cell carries a curvature estimate, so
  // the force was silently dropped. An orphan is a defect (Basilisk's "this should not happen"); it
  // is counted rather than hidden.
  //
  // The threshold matters: the FORCE is applied at every face with `dC != 0` exactly (dropping the
  // round-off jumps would itself break the discrete-gradient identity by O(sigma*kappa*eps), which
  // is 1e-9 and would be visible in the machine-zero gate), but a face whose colour jump is 1e-30
  // is not a missing interface and counting it as one would bury the real thing.
  struct CsfDiagnostics {
    double maxForce[3] = {0.0, 0.0, 0.0};
    long orphanFaces[3] = {0, 0, 0};
    long forcedFaces[3] = {0, 0, 0};
  };
  CsfDiagnostics csfDiagnostics() {
    if (!vofEnabled_ || !kappaField_.extent(0))
      throw std::runtime_error(
          "csf_diagnostics: needs VoF + a curvature field (call set_surface_tension and step, or "
          "compute_vof_curvature)");
    CsfDiagnostics d;
    C3 e = e_;
    CCConst cv = CCConst(cField_), kp = CCConst(kappaField_), kb = CCConst(kappaBranch_);
    const double sig = sigmaCsf_, h = vofAdv_.h(), eps = csfInterfaceEps_;
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    for (int c = 0; c < 3; ++c) {
      const long strd = strideOf(c);
      double mx = 0.0;
      long orph = 0, forced = 0;
      Kokkos::parallel_reduce(
          "csf_diag", MD(CCExec(), {G, G, G}, {e.x - G, e.y - G, e.z - G}),
          KOKKOS_LAMBDA(int x, int y, int z, double& m, long& o, long& f) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            const double dC = cv(i) - cv(i - strd);
            if (dC == 0.0)
              return;
            double kf = 0.0;
            const bool ok = vof::csfFaceCurvature(kp(i - strd), kb(i - strd), kp(i), kb(i), kf);
            if (ok)
              ++f;
            else if (Kokkos::fabs(dC) > eps)
              ++o;
            m = Kokkos::fmax(m, Kokkos::fabs(vof::csfFaceForce(sig, kf, dC, h)));
          },
          Kokkos::Max<double>(mx), orph, forced);
      d.maxForce[c] = mx;
      d.orphanFaces[c] = orph;
      d.forcedFaces[c] = forced;
    }
    return d;
  }

  // Implicit-FOU velocity stencil (CUDA build_adv_stencil_k + ibm_modify_stencil): backward-Euler
  // diffusion (idiag+6beta diag, -beta off) + rho*FOU(u^k) upwind operator (diagonally dominant ->
  // stable at high Re), then the Robust-Scaled cut-cell bake. The advecting velocity u^k = the
  // current C[*].u (ghosts filled).
  void buildAdvStencil(int c) {
    const double idiag = rho_ / dt_, beta = mu_, fouw = rho_;
    C3 e = e_;
    ibmBuildDiffusion(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e.x, e.y, e.z,
                      beta, idiag);
    CCExec space;
    FV AC = C[c].AC, AW = C[c].AW, AE = C[c].AE, AS = C[c].AS, AN = C[c].AN, AB = C[c].AB,
       AT = C[c].AT;
    CCConst U = advVelView(0), V = advVelView(1), W = advVelView(2);  // A0: wall-aware inputs
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "advstencil", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double cC = AC(i), cxm = AW(i), cxp = AE(i), cym = AS(i), cyp = AN(i), czm = AB(i),
                 czp = AT(i);
          sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y};
          Grid::fou_operator(c, x, y, z, Ua, Va, Wa, fouw, cC, cxm, cxp, cym, cyp, czm, czp);
          AC(i) = (MReal)cC;
          AW(i) = (MReal)cxm;
          AE(i) = (MReal)cxp;
          AS(i) = (MReal)cym;
          AN(i) = (MReal)cyp;
          AB(i) = (MReal)czm;
          AT(i) = (MReal)czp;
        });

    Kokkos::deep_copy(C[c].rscale, 1.0);
    Kokkos::deep_copy(C[c].inhom, 0.0);
    ibmModifyStencil(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, C[c].inhom,
                     C[c].rscale, C[c].ov, C[c].nCut, 0.0f, wallVelView(c));
    if (hasDrag_)
      addDragDiagonal(c);
  }
  // Variable-property sibling of buildAdvStencil: VarFaceProps diffusion build (per-face mu, face-
  // density time diagonal) + the FOU upwind weighted by the FACE density (constant path:
  // fouw=rho_). Separate kernel so the validated buildAdvStencil stays byte-identical.
  void buildAdvStencilVar(int c) {
    C3 e = e_;
    if (c == 0) {
      if (varProps_)
        fillMuGhosts();
      if (varRho_)
        fillPropGhosts(rhoField_);
      if (!varRho_ && porous_ && porousCons_)
        updateEpsRho();  // eps ghosts are driver-filled; whole-block product has valid ghosts
    }
    ibmBuildDiffusionVar(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, e.x, e.y,
                         e.z, G, makeFaceProps(c));
    CCExec space;
    FV AC = C[c].AC, AW = C[c].AW, AE = C[c].AE, AS = C[c].AS, AN = C[c].AN, AB = C[c].AB,
       AT = C[c].AT;
    CCConst U = advVelView(0), V = advVelView(1), W = advVelView(2);  // A0: wall-aware inputs
    const bool vr = effVarRho();
    const double rhoC = rho_;
    CCConst rf = vr ? CCConst(effRhoField()) : CCConst();
    const long sc = strideOf(c);
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "advstencil_var", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double cC = AC(i), cxm = AW(i), cxp = AE(i), cym = AS(i), cyp = AN(i), czm = AB(i),
                 czp = AT(i);
          sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y};
          const double fouw = vr ? 0.5 * (rf(i) + rf(i - sc)) : rhoC;
          Grid::fou_operator(c, x, y, z, Ua, Va, Wa, fouw, cC, cxm, cxp, cym, cyp, czm, czp);
          AC(i) = (MReal)cC;
          AW(i) = (MReal)cxm;
          AE(i) = (MReal)cxp;
          AS(i) = (MReal)cym;
          AN(i) = (MReal)cyp;
          AB(i) = (MReal)czm;
          AT(i) = (MReal)czp;
        });
    Kokkos::deep_copy(C[c].rscale, 1.0);
    Kokkos::deep_copy(C[c].inhom, 0.0);
    ibmModifyStencil(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, C[c].inhom,
                     C[c].rscale, C[c].ov, C[c].nCut, 0.0f, wallVelView(c));
    if (hasDrag_)
      addDragDiagonal(c);
  }
  // Backflow stabilization (Bazilevs 2009 / Esmaily-Moghadam 2011) for the NORMAL momentum at
  // outflow faces: add the dissipative diagonal term beta*rho*|min(u.n,0)| where the outflow
  // reverses (fluid re-entering, u.n<0). This removes the spurious kinetic-energy influx that the
  // do-nothing/zero- gradient outflow advects in -- the "backflow divergence" that blows up
  // separated flows (e.g. the BFS recirculation reaching the outlet), worse on finer grids. Purely
  // dissipative (u_ext=0), so it is implicit + unconditionally stable, and INERT where the outlet
  // is outgoing (u.n>=0) -> the channel and any non-reversing outflow stay byte-identical. Applied
  // to C[c].AC after buildAdvStencil (per Picard iteration, lagged at u^k); only the component
  // normal to each outflow face.
  void applyBackflowStab(int c) {
    if (backflowBeta_ <= 0.0 || !hasOutflow_)
      return;
    CCExec space;
    const double beta = backflowBeta_, rho = rho_;
    C3 e = e_;
    int dims[3] = {e.x, e.y, e.z};
    long st[3] = {1, e.x, (long)e.x * e.y};
    FV AC = C[c].AC;
    CCConst u = CCConst(C[c].u);
    const int a = c;  // the normal component of a face on axis a is component a
    for (int s = 0; s < 2; ++s) {
      if (bc_[2 * a + s] != 3 || !touchesGlobalFace(2 * a + s))
        continue;  // rank-owned outflow faces only
      const long sa = st[a];
      const int na = dims[a];
      const int bic = (s == 0) ? G : (na - G - 1);  // outflow-adjacent inner normal-velocity cell
      const double sgn = (s == 0) ? 1.0 : -1.0;     // reversal (u.n<0): u>0 at -a, u<0 at +a
      const int b = (a + 1) % 3, cc = (a + 2) % 3;
      const long sb = st[b], sc = st[cc];
      using MD2 = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>;
      Kokkos::parallel_for(
          "peclet::flow::backflow", MD2(space, {G, G}, {dims[b] - G, dims[cc] - G}),
          KOKKOS_LAMBDA(int p0, int p1) {
            const long i = (long)p0 * sb + (long)p1 * sc + (long)bic * sa;
            const double back =
                sgn * u(i);  // > 0 exactly where the outflow reverses (|min(u.n,0)|)
            if (back > 0.0)
              AC(i) += (MReal)(beta * rho * back);  // dissipative diagonal (u_ext = 0)
          });
    }
  }
  // max|a-b| over inner cells (Picard outer-tolerance check).
  double maxAbsDiffInner(CCConst a, CCConst b) {
    CCExec space;
    C3 e = e_;
    double m = 0;
    Kokkos::parallel_reduce(
        "maxdiff",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double d = Kokkos::fabs(a(i) - b(i));
          if (d > acc)
            acc = d;
        },
        Kokkos::Max<double>(m));
    return m;
  }
  // Shared momentum RB-GS loop: fixed velIters_ sweeps, or (velTol_ > 0) the tolerance stop —
  // colour 0 plain, colour 1 via the fused max-increment kernel, stop once the increment has
  // contracted to velTol_ of the first sweep's. The decision is rank-uniform under MPI (all ranks
  // see the same global max), so per-sweep halo exchanges stay in lockstep.
  // Stencil paths supply `resid` (returns max|b - A u| over this rank's inner fluid cells after
  // a fresh ghost fill) and `bnorm` (max|b|), enabling the residual stop when velResTol_ > 0.
  template <class Fill, class Color, class ColorDu>
  void velSweepLoop(Fill&& fill, Color&& sweepColor, ColorDu&& sweepColorDu,
                    std::function<double()> resid = nullptr, double bnorm = 0.0) {
    double du0 = 0.0;
    int used = velIters_;
    const double vtol = velocityResidualTolerance();
    const bool useRes = vtol > 0.0 && resid;
    auto gmax = [&](double v) {
#ifdef PECLET_FLOW_MPI
      if (distributed_) {
        double g = 0.0;
        MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, comm_);
        return g;
      }
#endif
      return v;
    };
    double scale = 0.0, rPrev = -1.0;
    if (useRes) {
      // The convergence scale from the initial residual: forcing may enter through a Dirichlet
      // ghost (inflow), not b, hence max(|b|, |A u|). NO early return on a converged warm start:
      // the projection needs u* to carry the O(rtol) response of the momentum equation (the
      // hydrostatic acid test drifts by 1e-8 in dP/dz if the solve is skipped), so at least one
      // sweep always runs -- a negligible cost against the 8-9 the solve typically takes.
      fill();
      (void)gmax(resid());
      scale = std::max(gmax(bnorm), gmax(lastAxNorm_));
    }
    for (int it = 0; it < velIters_; ++it) {
      fill();
      sweepColor(0);
      fill();
      if (useRes) {
        sweepColor(1);
        // check on the first sweep, then every 4th (a residual costs about half a sweep)
        if (it == 0 || (it + 1) % 4 == 0 || it + 1 == velIters_) {
          fill();
          const double r = gmax(resid());
          const double ratio = scale > 0 ? r / scale : 0.0;
          // Round-off floor / stagnation guard: a residual at ~1e-14 of the scale, or one that no
          // longer decreases between checks, cannot be improved by more sweeps -- stop rather than
          // chase noise to the cap (an exact case such as the hydrostatic column lands here).
          const bool floor = r <= 1e-14 * scale || (rPrev >= 0.0 && r >= rPrev);
          rPrev = r;
          if (r <= vtol * scale || floor || it + 1 == velIters_) {
            lastMomentumResid_ = std::max(lastMomentumResid_, ratio);  // EXIT ratio per component
            used = it + 1;
            break;
          }
        }
      } else if (velTol_ > 0.0) {
        double du = sweepColorDu(1);
#ifdef PECLET_FLOW_MPI
        if (distributed_) {
          double gd = 0.0;
          MPI_Allreduce(&du, &gd, 1, MPI_DOUBLE, MPI_MAX, comm_);
          du = gd;
        }
#endif
        if (it == 0)
          du0 = du;
        if (it + 1 >= velMinIters_ && du <= velTol_ * du0) {
          used = it + 1;
          break;
        }
      } else {
        sweepColor(1);
      }
    }
    lastMomentumSweeps_ += used;
  }

  VelocityMG::Comm vmgComm() const {
#ifdef PECLET_FLOW_MPI
    return comm_;
#else
    return nullptr;
#endif
  }
  // residual functor + max|b| for the stencil paths of component c (see velSweepLoop)
  // Common tail of a residual evaluation: the held normal-Dirichlet face is imposed, not solved
  // (excluded), remember max|A u| for the convergence scale, return max|r|.
  double finishResidual(int c) {
    if (hasBc_) {
      const int t = bc_[2 * c];
      if ((t == 1 || t == 2) && touchesGlobalFace(2 * c))
        zeroPlane(velRes_, e_, c, G);
    }
    lastAxNorm_ = peclet::flow::maxAbsDiffInner(CCConst(C[c].b), CCConst(velRes_), e_, G);
    return maxAbsInner(CCConst(velRes_), e_, G);
  }
  std::function<double()> stencilResidual(int c, bool exchange = false) {
    if (velocityResidualTolerance() <= 0.0)
      return nullptr;
    if (velRes_.extent(0) != n_)
      velRes_ = CCField("velRes", n_);
    return [this, c, exchange]() {
#ifdef PECLET_FLOW_MPI
      if (exchange && distributed_)
        velDev_->exchange(C[c].u);
#else
      (void)exchange;
#endif
      residualVarPin(velRes_, CCConst(C[c].u), CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                     FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT),
                     CCConst(C[c].mask), e_, G);
      return finishResidual(c);
    };
  }
  // The all-fluid domain-BC smoother's operator (constant coefficients + the boundary fold).
  std::function<double()> constCoeffResidual(int c, double beta, double Ac) {
    if (velocityResidualTolerance() <= 0.0)
      return nullptr;
    if (velRes_.extent(0) != n_)
      velRes_ = CCField("velRes", n_);
    return [this, c, beta, Ac]() {
      const I3 e{e_.x, e_.y, e_.z};
      diffResidual(velRes_, CCConst(C[c].u), CCConst(C[c].b), e, G, beta, Ac,
                   CCConst(bcDcorr_[c]));
      return finishResidual(c);
    };
  }
  double stencilBnorm(int c) {
    return velocityResidualTolerance() > 0.0 ? maxAbsInner(CCConst(C[c].b), e_, G) : 0.0;
  }

  void smoothComp(int c) {
    if constexpr (Grid::collocated) {
      if (hasBc_) {  // collocated domain BC: the (all-fluid) IBM diffusion stencil + cell-centered
                     // wall
        // reflection ghosts refreshed each colour (explicit no-slip; no fold). Converges to the
        // wall value.
        velSweepLoop(
            [&] { fillVelGhostsTo(C[c].u, c, 0); },
            [&](int col) {
              ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                  FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                  FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_, og_, G,
                                  col);
            },
            [&](int col) {
              return ibmRbgsStencilColorDu(C[c].u, CCConst(C[c].b), FPC(C[c].AC),
                                           FPC(C[c].AW), FPC(C[c].AE), FPC(C[c].AS),
                                           FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT),
                                           CCConst(C[c].mask), e_, og_, G, col);
            },
          stencilResidual(c, /*exchange=*/false), stencilBnorm(c));
        return;
      }
    }
    if (mixedVelocityMg()) {
      // MIXED: immersed solid + domain BCs (the packed bed with an inlet/outlet). Fine = the sharp
      // cut-cell stencil, solid pin, clean-fluid exclude + held-face exclude; coarse = staircase
      // Helmholtz + domain-face folds (VelocityMG::setStaircaseBc). The BC hook re-imposes the
      // level-0 velocity BC after every ghost fill, exactly as the RB-GS path's fillVelGhosts(c,1).
      const Off3 off = Grid::offset(c);
      ibmVolfrac(vmgTheta_, CCConst(sdf_), e_, off);
      ibmCleanFluidMask(vmgClean_, CCConst(sdf_), e_, off);
      vmg_.setFineStencil(FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                          FPC(C[c].AB), FPC(C[c].AT));
      // coarse = staircase Helmholtz (+ FOU from the restricted advecting velocity when the fine
      // stencil carries implicit advection) + domain-face folds
      vmg_.setStaircaseBc(c, CCConst(vmgTheta_), CCConst(C[c].mask), CCConst(vmgClean_), mu_,
                          rho_ / dt_, 0.5, /*upwind=*/implicitAdv(), rho_);
      // fold=0: level 0 is the UNFOLDED cut-cell stencil, so its wall ghosts are reflections
      // (exactly the bcStencilPath RB-GS convention); the folded coarse levels hold theirs at 0.
      fillVelGhosts(c, 0);
      vmg_.setBcApplyL0([this, c](CCField x) { applyVelocityBcCompTo(x, c, 0, true); });
      lastMomentumSweeps_ +=
          vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8, velTol_, vmgComm(),
                     velocityResidualTolerance());
      lastMomentumResid_ = std::max(lastMomentumResid_, vmg_.lastResidualRatio());
      maskVelocity(c);
      return;
    }
    if (bcStencilPath()) {
      // Domain BCs solved with the Robust-Scaled cut-cell / FOU stencil (built by setSolid /
      // buildAdvStencil) while refreshing the domain-BC ghosts each colour -- explicit walls/inflow
      // (reflection, fold=0) + outflow zero-gradient. Mirrors the collocated path above. Used for
      // an immersed solid (cut-cell no-slip in the operator) and/or implicit advection (FOU upwind
      // in the stencil -> stable at large dt). The const-coeff fold smoothers below are all-fluid,
      // diffusion-only: they ignore the solid AND run advection explicitly (CFL-limited).
      velSweepLoop(
          [&] { fillVelGhostsTo(C[c].u, c, 0); },
          [&](int col) {
            ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB),
                                FPC(C[c].AT), CCConst(C[c].mask), e_, og_, G, col);
          },
          [&](int col) {
            return ibmRbgsStencilColorDu(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                         FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                         FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_,
                                         og_, G, col);
          },
          stencilResidual(c), stencilBnorm(c));
      return;
    }
    if (hasBc_ &&
        useVelocityMg_) {  // domain-BC velocity multigrid: const-coeff aniso op + no-slip/inflow/
      // outflow boundary fold on every level (CUDA setDiffusionConstAllLevels +
      // setDiffusionBoundaryFold).
      vmg_.setDomainBcOp(c, mu_, rho_ / dt_);  // per component (the fold is component-dependent)
      fillVelGhosts(
          c, 1);  // set the level-0 boundary ghosts (wall fold=0, inflow value, outflow zero-grad)
      // Re-impose the velocity BC on the vel-MG's level-0 iterate each colour/residual (the
      // const-coeff smoother updates the held Dirichlet faces) -> the vel-MG converges to the RB-GS
      // fixed point (not the ~2% drift CUDA's vmg leaves at the boundary corners).
      // (the hook applies the BC only: VelocityMG::fill owns the periodic wrap / halo exchange)
      vmg_.setBcApplyL0([this, c](CCField x) { applyVelocityBcCompTo(x, c, 1, true); });
      lastMomentumSweeps_ +=
          vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8, velTol_, vmgComm(),
                     velocityResidualTolerance());
      lastMomentumResid_ = std::max(lastMomentumResid_, vmg_.lastResidualRatio());
      return;
    }
    if (hasBc_) {  // domain-BC (no immersed solid): CUDA's double const-coeff diff_k + dcorr fold
      // og_ is the GLOBAL block origin (red-black parity); {0,0,0} single-rank, so byte-identical
      // there. It was hard-coded {0,0,0} here, which swaps the colours on any rank whose block
      // origin has odd parity — the only smoother in the file that did not carry og_.
      const I3 e{e_.x, e_.y, e_.z}, og{og_.x, og_.y, og_.z};
      const double beta = mu_, Ac = rho_ / dt_ + 6.0 * mu_;
      velSweepLoop(
          [&] { fillVelGhosts(c, 1); },  // re-impose wall faces (fold) before each color
          [&](int col) {
            diffSmoothColor(C[c].u, CCConst(C[c].b), e, og, G, beta, Ac, col, CCConst(bcDcorr_[c]));
          },
          [&](int col) {
            return diffSmoothColorDu(C[c].u, CCConst(C[c].b), e, og, G, beta, Ac, col,
                                     CCConst(bcDcorr_[c]));
          },
          constCoeffResidual(c, beta, Ac), stencilBnorm(c));
      return;
    }
    if (useVelocityMg_) {  // IBM velocity multigrid: fine = sharp As_[c]; coarse op depends on the
                           // regime.
      vmg_.setFineStencil(FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                          FPC(C[c].AB), FPC(C[c].AT));
      if (implicitFou_ && advect_) {
        // UPWIND-CONVECTIVE coarse op (advection-dominated): aniso const-coeff diffusion + dt*FOU
        // from the restricted advecting velocity (restrictAdvVelocities ran once in step()). No pin
        // / no exclude mask.
        vmg_.buildUpwindCoarse(c, mu_, rho_ / dt_, rho_);
      } else {
        // STAIRCASE coarse op (diffusion-only): theta classification + clean-fluid exclude (exact
        // == RB-GS).
        const Off3 off =
            Grid::offset(c);  // velocity-unknown placement (staggered: -1/2 face; collocated: 0)
        ibmVolfrac(vmgTheta_, CCConst(sdf_), e_, off);
        ibmCleanFluidMask(vmgClean_, CCConst(sdf_), e_, off);
        vmg_.setStaircase(CCConst(vmgTheta_), CCConst(C[c].mask), CCConst(vmgClean_), mu_,
                          rho_ / dt_, 0.5);
      }
      lastMomentumSweeps_ +=
          vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8, velTol_, vmgComm(),
                     velocityResidualTolerance());
      lastMomentumResid_ = std::max(lastMomentumResid_, vmg_.lastResidualRatio());
      maskVelocity(
          c);  // re-impose no-slip at solid (the masked solve leaves them at the pin value)
      return;
    }
    // IBM / periodic: Robust-Scaled cut-cell stencil (float). The 7-point smoother reads faces
    // only -> the fused 1-kernel face fill suffices.
#ifdef PECLET_FLOW_MPI
    if (distributed_ && caMomentum_) {
      // Communication-avoiding pair (the momentum counterpart of CutcellMG::smooth's CA path):
      // ONE 2-deep exchange per red-black pair instead of one per colour — the velocity block is
      // g=2 already. Colour 0 overlaps the exchange with the interior sweep, then sweeps the
      // boundary shell PLUS the 1-deep ghost ring, redundantly recomputing the neighbour's
      // boundary cells from the same operands the neighbour uses (2-deep u ghosts; the ring rows
      // of the stencil/mask/rhs are exchanged below, so they are the owner's bit-exact values).
      // Colour 1 then sweeps with NO exchange: its boundary cells read only colour-0 ring cells,
      // which equal what a fresh exchange would have delivered — bit-identical at half the halo
      // events. The tolerance stop's colour-1 kernel is the ORIGINAL full-inner fused reduction
      // (host pencil form intact), so du matches the blocking path exactly.
      // Stencil + mask ring exchange: once per (re)build. The per-step machinery (implicit-FOU
      // Picard rebuilds, variable properties, implicit drag, eps-conservative porous) rewrites the
      // stencil every solve, so those paths re-exchange every solve — mirrors the step() rebuild
      // gates; a false positive costs 8 extra exchanges, a false negative would break the np>1
      // bit-exactness (the ring rows would read a stale operator).
      const bool perStepStencil =
          implicitAdv() || varProps_ || varRho_ || effVarRho() || hasDrag_;
      if (momStencilDirty_[c] || perStepStencil) {
        for (FV* a : {&C[c].AC, &C[c].AW, &C[c].AE, &C[c].AS, &C[c].AN, &C[c].AB, &C[c].AT})
          velDevF_->exchange(*a);
        velDev_->exchange(C[c].mask);
        momStencilDirty_[c] = false;
      }
      velDev_->exchange(C[c].b);  // rhs ring (owner's inner values); fixed over the sweeps
      const C3 lo{G + 1, G + 1, G + 1}, hi{e_.x - G - 1, e_.y - G - 1, e_.z - G - 1};
      const C3 rlo{G - 1, G - 1, G - 1}, rhi{e_.x - G + 1, e_.y - G + 1, e_.z - G + 1};
      const C3 z0{0, 0, 0};
      velSweepLoop(
          [] {},
          [&](int col) {
            if (col == 0) {
              velDev_->exchangeBegin(C[c].u);
              ibmRbgsStencilColorBox(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                     FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                     FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_, og_,
                                     col, lo, hi, z0, z0);
              velDev_->exchangeEnd(C[c].u);
              ibmRbgsStencilColorBox(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                     FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                     FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_, og_,
                                     col, rlo, rhi, lo, hi);
            } else {
              ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                  FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                  FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_, og_, G,
                                  col);
            }
          },
          [&](int col) {
            return ibmRbgsStencilColorDu(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                         FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                         FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_,
                                         og_, G, col);
          },
          stencilResidual(c, /*exchange=*/true), stencilBnorm(c));
      return;
    }
    if (distributed_) {
      // Overlap the per-colour halo with the interior sweep (the momentum counterpart of the MG
      // smoothers' split, 3ace962): post the exchange, sweep the interior cells — whose 7-point
      // stencil reads no ghost — while the messages fly, complete it, then sweep the boundary
      // shell. A colour's cells never read same-colour cells, so interior-then-shell is
      // bit-identical to the blocking exchange + full sweep; the tolerance stop's max-increment
      // combines the two passes by max (order-independent). Only this periodic/IBM path overlaps:
      // the domain-BC paths re-impose ghost BCs each colour and keep the blocking order (the same
      // decision as VelocityMG's overlap). The exchange is posted INSIDE the colour lambda (fill
      // is a no-op) so the packed send values are exactly the blocking call's.
      const C3 ilo{G, G, G}, ihi{e_.x - G, e_.y - G, e_.z - G};
      const C3 lo{G + 1, G + 1, G + 1}, hi{e_.x - G - 1, e_.y - G - 1, e_.z - G - 1};
      const C3 z0{0, 0, 0};
      velSweepLoop(
          [] {},
          [&](int col) {
            velDev_->exchangeBegin(C[c].u);
            ibmRbgsStencilColorBox(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                   FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                   FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_, og_,
                                   col, lo, hi, z0, z0);
            velDev_->exchangeEnd(C[c].u);
            ibmRbgsStencilColorBox(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                   FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                   FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_, og_,
                                   col, ilo, ihi, lo, hi);
          },
          [&](int col) {
            velDev_->exchangeBegin(C[c].u);
            const double di = ibmRbgsStencilColorDuBox(
                C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE),
                FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT),
                CCConst(C[c].mask), e_, og_, col, lo, hi, z0, z0);
            velDev_->exchangeEnd(C[c].u);
            const double ds = ibmRbgsStencilColorDuBox(
                C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE),
                FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT),
                CCConst(C[c].mask), e_, og_, col, ilo, ihi, lo, hi);
            return di > ds ? di : ds;
          },
          stencilResidual(c, /*exchange=*/true), stencilBnorm(c));
      return;
    }
#endif
    velSweepLoop(
        [&] { fillGhostsFaces(C[c].u); },
        [&](int col) {
          ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                              FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN), FPC(C[c].AB),
                              FPC(C[c].AT), CCConst(C[c].mask), e_, og_, G, col);
        },
        [&](int col) {
          return ibmRbgsStencilColorDu(C[c].u, CCConst(C[c].b), FPC(C[c].AC), FPC(C[c].AW),
                                       FPC(C[c].AE), FPC(C[c].AS), FPC(C[c].AN),
                                       FPC(C[c].AB), FPC(C[c].AT), CCConst(C[c].mask), e_,
                                       og_, G, col);
        },
          stencilResidual(c, /*exchange=*/false), stencilBnorm(c));
  }
  // pressure ghost at domain faces for the incremental predictor's grad(P): zero-gradient (Neumann)
  // at every non-periodic face so grad(P) carries no spurious force there (the periodic fill
  // wrapped the opposite boundary's pressure). Outflow pressure (Dirichlet p=0) is enforced
  // separately in the MG solve.
  void pressureBcGhost() {
    CCExec space;
    C3 e = e_;
    CCField P = P_;
    int dims[3] = {e.x, e.y, e.z};
    long st[3] = {1, e.x, (long)e.x * e.y};
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        if (bc_[2 * a + s] == 0 || !touchesGlobalFace(2 * a + s))
          continue;  // rank-owned global face only (interior ghosts come from the halo)
        const int b = (a + 1) % 3, c = (a + 2) % 3;
        const long sa = st[a], sb = st[b], sc = st[c];
        const int na = dims[a];
        const int bic = (s == 0) ? G : (na - G - 1);
        const int lo = (s == 0) ? 0 : (na - G), hi = (s == 0) ? (G - 1) : (na - 1);
        Kokkos::parallel_for(
            "pbcghost",
            Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {0, 0}, {dims[b], dims[c]}),
            KOKKOS_LAMBDA(int p0, int p1) {
              const long base = (long)p0 * sb + (long)p1 * sc;
              const double pin = P(base + (long)bic * sa);
              for (int ia = lo; ia <= hi; ++ia)
                P(base + (long)ia * sa) = pin;
            });
      }
  }
  // domain-BC velocity ghosts: periodic-fill periodic axes, then apply per-face BCs (fold=0
  // explicit/1 implicit).
  void fillVelGhosts(int comp, int fold) {
    // This is the fill that RE-IMPOSES the zero-gradient outflow face, i.e. the one that erases
    // `bcCorrectOutflow`'s correction (WO-R; see fillVelGhostsKeepOutflow). Record that so the
    // VoF bridge knows whether there is a correction left to preserve.
    outflowCorrValid_ = false;
    fillVelGhostsTo(C[comp].u, comp, fold);
  }
  void applyVelocityBcComp(int comp, int fold, bool doOutflow) {
    applyVelocityBcCompTo(C[comp].u, comp, fold, doOutflow);
  }
  // Field-parameterized variants (so the velocity-MG can re-impose the BC on its own level-0
  // iterate).
  void fillVelGhostsTo(CCField f, int comp, int fold) {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      velDev_->exchange(f);
      applyVelocityBcCompTo(f, comp, fold, true);
      return;
    }
#endif
    for (int a = 0; a < 3; ++a)
      if (bc_[2 * a] == 0 && bc_[2 * a + 1] == 0)
        fillAxis(f, a);
    applyVelocityBcCompTo(f, comp, fold, true);
  }
  // SIBLING of fillVelGhostsTo that does NOT re-impose the zero-gradient OUTFLOW face
  // (`doOutflow = false`, exactly what `step()` already passes after `project()` — "keep
  // outflow"). Used ONLY by `bridgeVelocityToVof`.
  //
  // WHY IT EXISTS (WO-R; a defect found by gate F2). The projection corrects the high-side
  // OUTFLOW normal face separately from every other face (`bcCorrectOutflow`) — that correction
  // IS how mass leaves the domain, and `step()` deliberately re-imposes the domain BCs afterwards
  // with `doOutflow = false` so it survives. `bridgeVelocityToVof` then called the FULL
  // `fillVelGhosts` (`doOutflow = true`), whose `bcOutflowComp` overwrites the boundary face with
  // the zero-gradient copy of the last inner cell — erasing the correction immediately after the
  // projection made it, on every step, whenever VoF is enabled.
  //
  // Two measured consequences, both of which vanish with this sibling
  // (`tests/kokkos/test_vof_bc.cpp` gate F2):
  //   * the field the colour advector is handed is NOT discretely divergence-free at the outflow,
  //     which is precisely the hypothesis Weymouth-Yue's exact conservation rests on;
  //   * `max_open_divergence()`, evaluated after `step()` returns, reports the ERASED field —
  //     measured 4.0 on a stratified outflow box that the projection had actually solved.
  // The outer ghost layers beyond the boundary face are left as the exchange/periodic fill wrote
  // them, and the advector never reads them: its flux sweep along axis d reaches exactly the
  // domain boundary face and no further.
  //
  // Inert for everything that existed: `bridgeVelocityToVof` runs only under `enable_vof`, and no
  // VoF configuration before this rung combined VoF with an outflow face.
  void fillVelGhostsKeepOutflow(int comp) {
    CCField f = C[comp].u;
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      velDev_->exchange(f);
      applyVelocityBcCompTo(f, comp, 0, false);
      return;
    }
#endif
    for (int a = 0; a < 3; ++a)
      if (bc_[2 * a] == 0 && bc_[2 * a + 1] == 0)
        fillAxis(f, a);
    applyVelocityBcCompTo(f, comp, 0, false);
  }
  // Distributed: a rank applies a face's BC iff its block TOUCHES that global face
  // (`touchesGlobalFace`, the same rule the scalar path uses in `applyScalarBc`). Without the test
  // every rank imposed the wall on its OWN block faces, so a partition cutting a walled axis split
  // the domain into independent sub-domains — invisible in the velocity (each sub-domain is
  // separately consistent) and only visible in the pressure. Single-rank the test is always true,
  // so this is byte-identical there.
  void applyVelocityBcCompTo(CCField f, int comp, int fold, bool doOutflow) {
    if (!hasBc_)
      return;
    B3 e{e_.x, e_.y, e_.z};
    if constexpr (Grid::collocated) {
      // Cell-centered velocity: reflect this component about each non-periodic boundary face. Walls
      // (type 1, vel 0) and Dirichlet/lid (type 2, prescribed vel) both use the same reflection;
      // outflow (type 3) and per-position inlet profiles are the inflow/outflow milestone (phase
      // 5b).
      for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s) {
          const int ff = 2 * a + s;
          const int t = bc_[ff];
          if (t == 0 || !touchesGlobalFace(ff))
            continue;  // interior rank boundary: the halo exchange owns those ghosts
          if (t == 3) {
            if (doOutflow)
              bcNeumannGhost(f, e, G, a, s);
            continue;
          }  // outflow: zero-gradient ghost
          if (bcProf_[ff].extent(0) >
              0)  // per-position inlet profile (e.g. the BFS partial parabola)
            bcVelocityColocated(f, e, G, a, s, 0.0, comp, bcProf_[ff], bcProfNc_[ff]);
          else
            bcVelocityColocated(f, e, G, a, s,
                                bcVel_[ff][comp]);  // wall / inflow / lid (Dirichlet)
        }
      return;
    }
    for (int a = 0; a < 3; ++a)
      for (int s = 0; s < 2; ++s) {
        const int ff = 2 * a + s;
        const int t = bc_[ff];
        if (t == 0 || !touchesGlobalFace(ff))
          continue;  // interior rank boundary: the halo exchange owns those ghosts
        if (t == 3) {
          if (doOutflow)
            bcOutflowComp(f, e, G, a, s, comp, fold);
          continue;
        }
        if (bcProf_[ff].extent(0) > 0)
          bcVelocityComp(f, e, G, a, s, comp, 0.0, fold, bcProf_[ff], bcProfNc_[ff]);
        else
          bcVelocityComp(f, e, G, a, s, comp, bcVel_[ff][comp], fold);
      }
  }
  // implicit-diffusion wall fold (CUDA setup_bc_diffusion): dcorr += (wall:+beta tangential /
  // outflow:-beta), brhs += 2*beta*wall (tangential Dirichlet); bake dcorr into the per-component
  // stencil diagonal.
  void setupBcDiffusion() {
    const double beta = mu_;
    B3 e{e_.x, e_.y, e_.z};
    for (int c = 0; c < 3; ++c) {
      Kokkos::deep_copy(bcDcorr_[c], 0.0);
      Kokkos::deep_copy(bcBrhs_[c], 0.0);
      for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s) {
          const int t = bc_[2 * a + s];
          if (!touchesGlobalFace(2 * a + s))
            continue;  // the implicit wall fold belongs to the rank owning that global face
          double dval, bval;
          if (t == 3) {
            dval = -beta;
            bval = 0.0;
          } else if (t != 0 && c != a) {
            dval = beta;
            bval = 2.0 * beta * bcVel_[2 * a + s][c];
          } else
            continue;  // periodic, or the normal component at a wall (held directly)
          bcDiffusionFold(bcDcorr_[c], bcBrhs_[c], e, G, a, s, dval, bval);
        }
      // dcorr is passed to the (double) const-coeff smoother diffSmoothColor each sweep -- matching
      // CUDA diff_k (Ac + dcorr in double), NOT baked into the float stencil.
    }
  }
  // Incremental (rotational) cut-cell projection: solve A phi = -div_open(u*) (RB-GS,
  // mean-removed), u -= grad phi, then accumulate the physical pressure P += (rho/dt)*phi -
  // mu*div(u*) (Timmermans).
  // one mask-aware axis-wise smoothing pass of a cell field (the filtered-rotational S; see
  // setRotationalFilter). Reads the +/-1 axis neighbours' sdf: fluid-fluid -> (1,2,1)/4;
  // one solid side -> 1/2(self + open-side neighbour); both solid -> identity.
  void filterCellField(CCField f, int axis) {
    CCExec space;
    Kokkos::deep_copy(tgp_, f);
    fillGhosts(tgp_);
    CCConst src = CCConst(tgp_);
    CCConst sd = CCConst(sdf_);
    const double eps = rotFilterEps_;
    C3 e = e_;
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "peclet::flow::rot_filter", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long sy = e.x, sz = (long)e.x * e.y;
          const long i = (long)x + (long)y * sy + (long)z * sz;
          const long sa = (axis == 0) ? 1 : (axis == 1) ? sy : sz;
          if (sd(i) < 0.0)
            return;
          const bool am = sd(i - sa) >= 0.0, ap = sd(i + sa) >= 0.0;
          double sm;
          if (am && ap)
            sm = 0.25 * (src(i - sa) + 2.0 * src(i) + src(i + sa));
          else if (ap)
            sm = 0.5 * (src(i) + src(i + sa));
          else if (am)
            sm = 0.5 * (src(i) + src(i - sa));
          else
            sm = src(i);
          // eps-floor blend: S' = eps*I + (1-eps)*S. A pure S has an exact checkerboard null
          // space, which at dt -> infinity (where the (rho/dt)*phi term vanishes) degenerates the
          // fixed point into a frozen-checkerboard family; the floor keeps S' > 0 so
          // (rho/dt + mu S'A) phi = 0 forces phi = 0 at EVERY dt, while still cutting the
          // dangerous mode's feedback gain by ~1/eps.
          f(i) = eps * src(i) + (1.0 - eps) * sm;
        });
  }
  void project() {
    // ghosts incl. domain BCs (outflow zero-gradient) BEFORE the divergence -- matches CUDA
    // apply_velocity_bc before diverg_open, so div(u*) counts the outflow flux (else the rotational
    // pressure pumps the mis-counted outflow divergence and blows up the outflow-wall corner).
    if constexpr (Grid::collocated) {
      // Approximate (MAC) projection: average the cell velocities onto a face field, then project
      // THAT. Use the BC-aware ghost fill (periodic / cross-rank + domain BCs) so the averaged
      // inflow/outflow faces carry the right value -- at open boundaries the flux is counted
      // (closed walls are openness 0).
      for (int c = 0; c < 3; ++c)
        fillVelGhosts(c, 0);
      if ((faceInterp_ >= 1 && faceInterp_ <= 5) || faceInterp_ == 7 ||
          faceInterp_ == 10)  // wall-aware flux map at solid
        centerToFaceWallAware(uf_, vf_, wf_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u),
                              CCConst(sdf_), CCConst(xcx_), CCConst(xcy_), CCConst(xcz_),
                              faceInterp_ >= 3, e_, G);  // faces (modes 1-5,7,10; mode 6/9 = plain)
      else
        centerToFace(uf_, vf_, wf_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), e_, G);
      // rung V8 (WO-T): the body / interfacial forces the predictor deliberately did NOT apply, put
      // on the FACES where the pressure difference lives — the collocated form of the V4 balanced
      // force (Basilisk centered.h). Inert on every constant-density collocated configuration.
      if (colocatedFaceForce())
        applyFaceAcceleration();
      if (ghostProjection_) {
        // Collocated ghost divergence: the SAME binary-openness + closure-delta pair as the
        // staggered path, applied to the 1/2-1/2 face-averaged field (the closures only ever read
        // faces whose two adjacent centers are fluid, so the masked solid-cell zeros never enter
        // except at EXPLICIT slivers — faithfully modelled in the a-priori study).
        if (gpNRows_ < 0)
          throw std::runtime_error("ghost projection: call set_solid after set_ghost_projection");
        if (porous_ || varRho_ || useChebyshev_)
          throw std::runtime_error(
              "ghost projection: porous/variable-rho/Chebyshev unsupported (v1)");
        divergOpen(CCConst(uf_), CCConst(vf_), CCConst(wf_), CCConst(oxb_), CCConst(oyb_),
                   CCConst(ozb_), div_, e_, G);
        gpDivergDelta(div_, CCConst(uf_), CCConst(vf_), CCConst(wf_), gpOv_, gpNRows_,
                      C3{nx_, ny_, nz_}, e_, G, distributed_);
      } else
        divergOpen(CCConst(uf_), CCConst(vf_), CCConst(wf_), CCConst(ox_), CCConst(oy_),
                   CCConst(oz_), div_, e_, G);
    } else {
      for (int c = 0; c < 3; ++c)
        fillVelGhosts(c, 0);
      if (porous_) {            // volume-averaged continuity: div(open*eps*u*), constraint div(eps
                                // u)=-d(eps)/dt
        fillPorousEpsGhosts();  // BEFORE the divergence — one eps ghost policy for RHS,
                                // coefficients and residual (the deposit rewrites these ghosts
                                // every step)
        divergOpenEps(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
                      CCConst(oz_), CCConst(epsField_), div_, e_, G);
      } else if (ghostProjection_) {
        // Directional ghost-cell divergence: binary-openness face differences (COUPLED faces)
        // plus the wall-anchored closures at ghost faces, row-rescaled — the SAME kernel pair
        // serves the RHS here and the diagnostic in maxOpenDivergence (diagnostic == residual).
        if (gpNRows_ < 0)
          throw std::runtime_error("ghost projection: call set_solid after set_ghost_projection");
        if (porous_ || varRho_ || useChebyshev_)
          throw std::runtime_error(
              "ghost projection: porous/variable-rho/Chebyshev unsupported (v1)");
        divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(oxb_), CCConst(oyb_),
                   CCConst(ozb_), div_, e_, G);
        gpDivergDelta(div_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), gpOv_, gpNRows_,
                      C3{nx_, ny_, nz_}, e_, G, distributed_);
      } else
        divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
                   CCConst(oz_), div_, e_, G);
    }
    // MOVING GEOMETRY (rung 3): the wall's own volume flux. Inert without a moving instance.
    addWallFluxDivergence(div_);
    // WO-P01: the phase-change (and any prescribed) divergence source, so the deflated solve
    // delivers div(open u) = S instead of 0. Inert unless enable_phase_change /
    // set_divergence_source ran.
    pcApplyDivergenceSource(div_);
    // Porous continuity source: fold d(eps)/dt into the divergence so the Poisson solves for
    // div(eps u) = -d(eps)/dt (not 0). d(eps)/dt = (eps^{n+1}-eps^n)/dt from the deposited void
    // fraction; stored in depsdt_ (epsPrev_ is overwritten at step end, so the residual reuses
    // this).
    if (porous_) {
      CCExec space;
      C3 e = e_;  // local copy — a KOKKOS_LAMBDA capturing e_ would read this-> on the device
      CCField d = div_, dd = depsdt_, ep = epsField_, epp = epsPrev_;
      const double idt = 1.0 / dt_;
      const bool useDt = porousDepsDt_;
      using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
      Kokkos::parallel_for(
          "peclet::flow::deps_dt", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = (long)x + (long)y * e.x + (long)z * e.x * e.y;
            dd(i) = (ep(i) - epp(i)) * idt;
            if (useDt)
              d(i) += dd(i);  // off -> solve div(eps u)=0 (drop the noisy time-derivative source)
          });
    }
    // bridge -div(u*) (g=2 block) -> the MG rhs (g=1 block); keep div(u*) in div_ for the pressure
    // update
    copyInner(rhs1_, e1_, 1, CCConst(div_), e_, G);
    {
      CCExec space;
      CCField r = rhs1_;
      Kokkos::parallel_for(
          "negdiv", Kokkos::RangePolicy<CCExec>(space, 0, n1_),
          KOKKOS_LAMBDA(std::size_t i) { r(i) = -r(i); });
    }
    if (fluidOnlyMode_ == 2) {
      // Design B: solid rows carry no constraint -- mask their rhs (their operator rows are empty
      // in the filtered 7-point part; the star overlay never adds to them), so phi_s stays 0.
      if (useChebyshev_)
        throw std::runtime_error("set_fluid_only_constraint(2): Chebyshev unsupported (v1)");
      CCExec space;
      CCField r = rhs1_;
      CCConst sd = CCConst(sdf_);
      const C3 e1 = e1_, e2 = e_;
      Kokkos::parallel_for(
          "peclet::flow::star_mask_rhs",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx_, ny_, nz_}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i2 =
                (long)(x + G) + (long)(y + G) * e2.x + (long)(z + G) * (long)e2.x * e2.y;
            if (sd(i2) < 0.0)
              r((long)(x + 1) + (long)(y + 1) * e1.x + (long)(z + 1) * (long)e1.x * e1.y) = 0.0;
          });
    }
    // Variable density: rebuild the Poisson operator with the face coefficients
    // c_f = open_f * rho0/rho_f (rho0 = the scalar rho_, so uniform rho == rho_ reduces exactly to
    // the openness operator). The coefficient fields ride the openness rails: bridge rho to the g=1
    // block INCLUDING its ghost ring, form the coefficients on the inner cells, and hand them to
    // setOpenness, whose per-level ghost fill + boundary re-imposition + coarsening (rediscretized
    // averaging) treat them exactly like openness. Rebuilt every step (rho may be closure/transport
    // driven); Chebyshev bounds are invalidated (stale bounds under changing coefficients diverge
    // silently — PCG is the recommended/default driver here).
    if (varRho_) {
      fillPropGhosts(rhoField_);
      copyBlockShifted(rho1_, e1_, CCConst(rhoField_), e_, G - 1);
      // Face mean of rho: ARITHMETIC by default (the hydrostatic-exactness + series-mobility
      // choice); the harmonic sibling is the opt-in WO-J knob and must be paired with the matching
      // correction in projectVelocities or the projection stops being exact (mac_pressure.hpp).
      if (rhoFaceHarmonic_)
        buildRhoCoeffHarm(cx1_, cy1_, cz1_, CCConst(ox1_), CCConst(oy1_), CCConst(oz1_),
                          CCConst(rho1_), rho_, e1_, 1);
      else
        buildRhoCoeff(cx1_, cy1_, cz1_, CCConst(ox1_), CCConst(oy1_), CCConst(oz1_), CCConst(rho1_),
                      rho_, e1_, 1);
      mg_.setBoundaryConditions(bc_);
      mg_.setOpenness(CCConst(cx1_), CCConst(cy1_), CCConst(cz1_), 1.0, 1.0, 1.0);
      chebBoundsSet_ = false;  // spectrum changed with the coefficients (re-estimated by the solve)
    }
    // Porous continuity: the Poisson operator is eps-weighted (c_f = open_f * eps_f), same rails as
    // the density coefficient above. Rebuilt every step (eps moves with the particles). With
    // implicit CFD-DEM drag the coefficient AND the correction carry the drag-relaxation w_f =
    // idt/(idt+beta_f) (idt = rho/dt) so the pressure correction is consistent with the drag-loaded
    // momentum diagonal A_P = idt+beta (SIMPLE/PISO-with-implicit-drag; stiff drag -> w_f->0 -> the
    // drag holds the velocity, stable). beta==0 reduces exactly to the plain eps-weighted operator.
    if (porous_) {
      // eps ghosts were filled by fillPorousEpsGhosts() before the divergence above — the SAME
      // ghost values must feed the coefficient bridge (face eps == 1 at open domain faces), or the
      // operator and the RHS disagree at the boundary rows.
      copyBlockShifted(eps1_, e1_, CCConst(epsField_), e_, G - 1);
      if (hasDrag_) {
        fillPropGhosts(dragBeta_);
        copyBlockShifted(beta1_, e1_, CCConst(dragBeta_), e_, G - 1);
      } else if (porousCons_) {
        Kokkos::deep_copy(beta1_, 0.0);  // conservative kernels read beta unconditionally when used
      }
      if (porousCons_) {
        // eps-CONSERVATIVE pair: c_f = open * (eps_f rho idt)/(eps_f rho idt + beta_f), matching
        // the eps-weighted momentum diagonal; the eps of the flux cancels the eps of the inertia
        // (see mac_pressure.hpp). Correction: projectCorrectPorousCons below.
        buildPorousCoeffCons(cx1_, cy1_, cz1_, CCConst(ox1_), CCConst(oy1_), CCConst(oz1_),
                             CCConst(eps1_), CCConst(beta1_), hasDrag_, rho_ / dt_, e1_, 1);
      } else if (hasDrag_) {
        buildPorousCoeffDrag(cx1_, cy1_, cz1_, CCConst(ox1_), CCConst(oy1_), CCConst(oz1_),
                             CCConst(eps1_), CCConst(beta1_), rho_ / dt_, e1_, 1);
      } else {
        buildPorousCoeff(cx1_, cy1_, cz1_, CCConst(ox1_), CCConst(oy1_), CCConst(oz1_),
                         CCConst(eps1_), e1_, 1);
      }
      mg_.setBoundaryConditions(bc_);
      mg_.setOpenness(CCConst(cx1_), CCConst(cy1_), CCConst(cz1_), 1.0, 1.0, 1.0);
      chebBoundsSet_ = false;
    }
    // geometric multigrid solve of the cut-cell pressure Poisson A phi = -div(u*) (CUDA
    // mac_multigrid): MG-PCG by default, or the communication-light Chebyshev driver (bounds
    // estimated once, then reused). Warm start (CUDA pwarm_): keep the previous step's phi1_ as the
    // initial guess instead of zeroing.
    if (!pwarm_)
      Kokkos::deep_copy(phi1_, 0.0);
    if (useChebyshev_) {
      if (!chebBoundsSet_) {
        mg_.estimateEigenvalues(CCConst(rhs1_), chebA_, chebB_, 15, 2, 2, 12);
        chebBoundsSet_ = true;
      }
      lastPressureIters_ =
          mg_.solveChebyshev(rhs1_, phi1_, chebMaxit_, chebRtol_, 2, 2, 12, chebA_, chebB_);
    } else if (ghostProjection_) {
      // Nonsymmetric ghost-projection operator (both grids — the phi matrix is identical):
      // BiCGStab, preconditioned by the symmetric binary-openness V-cycle (the hierarchy set up
      // in setSolid); the overlay delta enters the fine-level matvec only. Distributed, the
      // matvec stages the iterate on this solver's g=2 block (gpX2_) whose halo carries the
      // overlay's +/-2 reach.
#ifdef PECLET_FLOW_MPI
      if (distributed_)
        lastPressureIters_ =
            mg_.solveBiCGStab(rhs1_, phi1_, r_, gpRh_, pp_, Ap_, gpT_, z_, gpZ2_, pcgMaxit_,
                              pcgRtol_, 2, 2, 12, gpOv_, gpNRows_, C3{nx_, ny_, nz_}, gpX2_,
                              velDev_.get(), e_);
      else
#endif
        lastPressureIters_ =
            mg_.solveBiCGStab(rhs1_, phi1_, r_, gpRh_, pp_, Ap_, gpT_, z_, gpZ2_, pcgMaxit_,
                              pcgRtol_, 2, 2, 12, gpOv_, gpNRows_, C3{nx_, ny_, nz_});
      // Pin the FREE variables of the binary-openness operator to their design value phi = 0:
      // solid-centered (sdfGp < 0) cells and fully-BC_ONLY overlay rows have a zero row AND zero
      // rhs, so the Krylov iteration leaves an arbitrary (V-cycle-prolongation, iteration-path,
      // and decomposition dependent) value there — invisible to the gp residual (the closures
      // never read those faces) but INJECTED into real near-wall fluid velocities by the plain
      // projectCorrect face gradient. The doc contract of ghost_projection.hpp is "decoupled
      // rows hold phi = 0"; enforce it (measured: without this, np=2 runs differ from the
      // reference by ~1e-2 relative u at sphere-surface faces while agreeing 1e-13 elsewhere).
      {
        CCExec space;
        CCField ph = phi1_;
        CCConst sg = CCConst(sdfGp_);
        auto idMap = gpIdMap_;
        auto ov = gpOv_;
        const C3 e1 = e1_, e2 = e_;
        const int lnx = nx_, lny = ny_;
        Kokkos::parallel_for(
            "peclet::flow::gp_pin_decoupled",
            Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx_, ny_, nz_}),
            KOKKOS_LAMBDA(int x, int y, int z) {
              const long i1 =
                  (long)(x + 1) + (long)(y + 1) * e1.x + (long)(z + 1) * (long)e1.x * e1.y;
              const long i2 =
                  (long)(x + G) + (long)(y + G) * e2.x + (long)(z + G) * (long)e2.x * e2.y;
              bool dec = sg(i2) < 0.0;
              if (!dec) {
                const int s = idMap((long)x + (long)y * lnx + (long)z * (long)lnx * lny);
                if (s >= 0 && ov.coupled(s) == 0)
                  dec = true;
              }
              if (dec)
                ph(i1) = 0.0;
            });
      }
    } else if (useFcg_) {
      // Flexible CG (set_pressure_fcg): identical to the MG-PCG branch below but for the
      // Polak-Ribiere beta, which tolerates a V-cycle preconditioner that is not symmetric w.r.t.
      // the fine operator. Its one extra vector is allocated on first use, so an unselected FCG
      // costs nothing (not even memory).
      if (zp1_.extent(0) != n1_)
        zp1_ = CCField("zp1", n1_);
      lastPressureIters_ =
          mg_.solveFCG(rhs1_, phi1_, r_, pp_, z_, zp1_, Ap_, pcgMaxit_, pcgRtol_, 2, 2, 12,
                       fluidOnlyMode_ == 2 ? &starOv_ : nullptr, nStar_, C3{nx_, ny_, nz_});
    } else {
      lastPressureIters_ =
          mg_.solvePCG(rhs1_, phi1_, r_, pp_, z_, Ap_, pcgMaxit_, pcgRtol_, 2, 2, 12,
                       fluidOnlyMode_ == 2 ? &starOv_ : nullptr, nStar_, C3{nx_, ny_, nz_});
    }
    if (fluidOnlyMode_ == 2) {
      // Pin phi at solid-centered cells to 0 (their rows are unconstrained; the smoother must not
      // leave garbage there -- projectCorrect reads phi_s at fluid|solid faces and the
      // starCorrectFaces fix-up assumes the applied value was exactly 0).
      CCExec space;
      CCField ph = phi1_;
      CCConst sd = CCConst(sdf_);
      const C3 e1 = e1_, e2 = e_;
      Kokkos::parallel_for(
          "peclet::flow::star_pin_solid",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {0, 0, 0}, {nx_, ny_, nz_}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i2 =
                (long)(x + G) + (long)(y + G) * e2.x + (long)(z + G) * (long)e2.x * e2.y;
            if (sd(i2) < 0.0)
              ph((long)(x + 1) + (long)(y + 1) * e1.x + (long)(z + 1) * (long)e1.x * e1.y) = 0.0;
          });
    }
    copyInner(phi_, e_, G, CCConst(phi1_), e1_, 1);  // bridge phi back g=1 -> g=2
    fillGhosts(phi_);
    if (hasOutflow_) {  // hold phi=0 at the outflow ghost so grad(phi) drives the outflow face
                        // (Dirichlet p=0)
      B3 e{e_.x, e_.y, e_.z};
      for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s)
          if (bc_[2 * a + s] == 3 && touchesGlobalFace(2 * a + s))
            bcZeroPressureGhost(phi_, e, G, a, s);
    }
    if constexpr (Grid::collocated) {
      // phi: zero-gradient (Neumann) at non-periodic walls so the cell-centered central-difference
      // correction carries no spurious normal acceleration through the wall (the periodic fill
      // wrapped the opposite boundary's phi). Outflow (Dirichlet p=0) is handled by hasOutflow_
      // above (phase 5b).
      if (hasBc_) {
        B3 e{e_.x, e_.y, e_.z};
        for (int a = 0; a < 3; ++a)
          for (int s = 0; s < 2; ++s) {
            const int t = bc_[2 * a + s];
            if (t != 0 && t != 3 && touchesGlobalFace(2 * a + s))
              bcNeumannGhost(phi_, e, G, a, s);
          }
      }
      // Correct the face field (-> discretely divergence-free; transient this step) and the cell
      // field (central-difference cell gradient).
      // rung V8: per-face 1/rho_f on the gradient, matching the operator coefficient
      // c_f = o_f*rho0/rho_f — the SAME exact-adjoint face correction the staggered path uses.
      if (varRho_)
        projectCorrectVar(uf_, vf_, wf_, CCConst(phi_), CCConst(rhoField_), rho_, e_, G);
      else
        projectCorrect(uf_, vf_, wf_, CCConst(phi_), e_, G);
      if (fluidOnlyMode_ == 2)  // Design B: replace the solid side's phi=0 by phibar_s at
        starCorrectFaces(uf_, vf_, wf_, CCConst(phi_), starOv_, nStar_,  // fluid|solid faces
                         C3{nx_, ny_, nz_}, e_, G, e_, G);
      fillGhosts(uf_);
      fillGhosts(vf_);
      fillGhosts(wf_);    // complete the divergence-free face field (boundary faces)
      if (hasOutflow_) {  // correct the high-side outflow face on the face field so mass leaves
                          // (phi=0 there)
        B3 e{e_.x, e_.y, e_.z};
        CCField fa[3] = {uf_, vf_, wf_};
        for (int a = 0; a < 3; ++a)
          if (bc_[2 * a + 1] == 3 && touchesGlobalFace(2 * a + 1))
            bcCorrectOutflow(fa[a], phi_, e, G, a);
      }
      if (colocatedFaceForce()) {
        // rung V8 (WO-T): the cell sees the AVERAGE of what its two faces saw — the force
        // acceleration minus the projection's own rho-weighted face correction — through the same
        // averaging operator projectCorrectCenter applies to phi differences. A hydrostatic column
        // and a stationary droplet are exactly balanced on the faces, so the cell averages an exact
        // zero. This replaces the whole constant-density cell-correction chain below.
        applyCellFaceAverageCorrection();
      } else if (ghostProjection_ || faceInterp_ == 9 || faceInterp_ == 10) {
        // Ghost cell correction (also the mode-9/10 cutcell-ghost hybrids): the directional
        // gpCenterGrad gradient of phi — 2nd-order one-sided at cut cells, never reads a
        // decoupled (solid/pocket) phi. The same operator supplies the momentum's -grad(P^n)
        // predictor (buildRhs), so the pressure force the momentum feels and the correction stay
        // one operator family.
        for (int cc = 0; cc < 3; ++cc) {
          gpCenterGrad(tgp_, CCConst(phi_), CCConst(ghostProjection_ ? sdfGp_ : sdf_), cc, e_, G, gauge2a_);
          subtractField(C[cc].u, CCConst(tgp_), e_, G);
        }
      } else if (faceInterp_ >= 2 &&
                 faceInterp_ <= 5) {  // modes 2-5: cell correction = the TRANSPOSE of the
        // wall-aware map, keeping (T, Tᵀ) an adjoint pair (transposeGradWallAware)
        CCField xcs[3] = {xcx_, xcy_, xcz_};
        CCField oax[3] = {ox_, oy_, oz_};
        for (int cc = 0; cc < 3; ++cc) {
          transposeGradWallAware(tgp_, CCConst(phi_), CCConst(sdf_), CCConst(oax[cc]),
                                 CCConst(xcs[cc]), faceInterp_ >= 3, cc, e_, G);
          subtractField(C[cc].u, CCConst(tgp_), e_, G);
        }
      } else if (faceInterp_ == 6 ||
                 faceInterp_ == 7) {  // embed: openness-WEIGHTED cell correction
        // (full open-face pressure force at cut cells) — Basilisk centered_grad
        projectCorrectCenterOpen(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(ox_), CCConst(oy_),
                                 CCConst(oz_), e_, G);
      } else if (faceInterp_ >= 11 && faceInterp_ <= 13) {  // adjoint-aperture: cell correction
        // = the TRANSPOSE of the aperture divergence of the 1/2-1/2 average, G = -(D_a Pi)^T
        // (centerGradAperture) -- support-consistent (collapses the invisible subspace) and
        // adjoint (SPSD Uzawa map). Mode 12 = the same times the per-cell openness rescale S(i)
        // (centerGradApertureScaled), the accuracy repair for the 1/2*alpha under-weighting.
        CCField oax[3] = {ox_, oy_, oz_};
        for (int cc = 0; cc < 3; ++cc) {
          if (faceInterp_ == 12)
            centerGradApertureScaled(tgp_, CCConst(phi_), CCConst(ox_), CCConst(oy_), CCConst(oz_),
                                     cc, e_, G);
          else if (faceInterp_ == 13)
            centerGradOpenCapped(tgp_, CCConst(phi_), CCConst(oax[cc]), cc, apertureFloor_, e_, G);
          else
            centerGradAperture(tgp_, CCConst(phi_), CCConst(oax[cc]), cc, e_, G);
          subtractField(C[cc].u, CCConst(tgp_), e_, G);
        }
      } else {
        projectCorrectCenter(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(ox_), CCConst(oy_),
                             CCConst(oz_), e_, G);
      }
    } else {
      if (porous_ && porousCons_)  // eps-conservative gradient rho*idt/(eps_f rho idt + beta_f),
                                   // matching buildPorousCoeffCons (see mac_pressure.hpp)
        projectCorrectPorousCons(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(epsField_),
                                 CCConst(dragBeta_), hasDrag_, rho_ / dt_, e_, G);
      else if (porous_ &&
               hasDrag_)  // drag-relaxed gradient w_f=idt/(idt+beta_f), matching buildPorousCoeffDrag
        projectCorrectPorousDrag(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(dragBeta_),
                                 rho_ / dt_, e_, G);
      else if (varRho_ && rhoFaceHarmonic_)  // the WO-J harmonic knob: coefficient AND correction
        projectCorrectVarHarm(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(rhoField_), rho_, e_,
                              G);
      else if (varRho_)  // per-face 1/rho on the gradient, matching the operator coefficient
        projectCorrectVar(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(rhoField_), rho_, e_, G);
      else
        projectCorrect(C[0].u, C[1].u, C[2].u, CCConst(phi_), e_, G);
      if (hasOutflow_) {  // correct the high-side outflow normal face that projectCorrect misses
                          // (mass leaves)
        B3 e{e_.x, e_.y, e_.z};
        // WO-R item 4: with variable density every OTHER corrected face carries the mobility
        // factor rho0/rho_f (projectCorrectVar) and this one did not — a ratio-sized error on the
        // outflow face, invisible at constant density (rho_f == rho0) which is why the channel/BFS
        // validations never saw it. The porous branches keep the plain correction: their
        // coefficient is the drag/eps relaxation, not 1/rho, and a two-phase porous outlet is not
        // this rung. `!varRho_` is byte-identical.
        // `set_outflow_rho_correction(True)` (or PECLET_FLOW_OUTFLOW_RHO=1) applies the
        // 1/rho_f factor. DEFAULT OFF: measured, it makes the outflow divergence seven orders
        // WORSE, because the operator's outflow-face coefficient is the raw openness — see
        // setOutflowRhoCorrection for the numbers and the mechanism.
        const bool var = varRho_ && !porous_ && outflowRhoCorr_;
        for (int a = 0; a < 3; ++a)
          if (bc_[2 * a + 1] == 3 && touchesGlobalFace(2 * a + 1)) {
            if (var)
              bcCorrectOutflowVar(C[a].u, phi_, rhoField_, rho_, e, G, a, rhoFaceHarmonic_);
            else
              bcCorrectOutflow(C[a].u, phi_, e, G, a);
          }
        outflowCorrValid_ = true;  // the outflow face now carries the mass that leaves
      }
    }
    // the grad(phi) correction also touches solid faces; re-impose no-slip there so the decoupled
    // solid velocity cannot accumulate (matches the CUDA apply_mask/mask_k after correct_k ->
    // stability).
    for (int c = 0; c < 3; ++c)
      maskVelocity(c);
    // Rotational incremental pressure (Timmermans), matching CUDA press_update_k: P += (rho/dt)*phi
    // - mu*div(u*). Classical non-incremental Chorin (!incremental_) skips the accumulation;
    // getPressure() derives p from phi.
    if (incremental_) {
      if (rotFilter_ && rotationalP_)
        for (int a = 0; a < 3; ++a)
          filterCellField(div_, a);  // S(div u*): see setRotationalFilter
      CCExec space;
      CCField P = P_, ph = phi_, d = div_;
      // Pressure under-relaxation (MFIX §10.1): accumulate only omega_p of the increment into the
      // physical pressure P (the velocity correction still uses the full phi to satisfy
      // continuity), so the next step's incremental predictor -grad(P^n) can't overshoot for a
      // stiff drag diagonal. omega_p=1 (default) is the current behaviour; <1 only stabilizes the
      // porous+drag path.
      const double ct = pressUnderRelax_ * rho_ / dt_,
                   mu = rotationalP_ ? rotWeight_ * mu_ : 0.0;
      if (varProps_) {
        // Variable viscosity: the pointwise Timmermans term -mu(i)*div(u*) is inconsistent for
        // heterogeneous mu (see setVariableRotational). Default = constant coefficient chi*mu_min
        // (stable by domination, exact fallback to the uniform-mu scheme); "full" = pointwise
        // (mild contrast only); "off" = plain incremental.
        if (varRotMode_ == 1) {
          CCConst mf = CCConst(muField_);
          const double chi = varRotChi_;
          Kokkos::parallel_for(
              "press_var_full", Kokkos::RangePolicy<CCExec>(space, 0, n_),
              KOKKOS_LAMBDA(std::size_t i) { P(i) += ct * ph(i) - chi * mf(i) * d(i); });
        } else {
          const double muRot = (varRotMode_ == 2) ? 0.0 : varRotChi_ * minMuInner();
          Kokkos::parallel_for(
              "press_var_min", Kokkos::RangePolicy<CCExec>(space, 0, n_),
              KOKKOS_LAMBDA(std::size_t i) { P(i) += ct * ph(i) - muRot * d(i); });
        }
      } else if (rotWallW_ > 0.0 && rotationalP_) {
        // Frank's wall-banded blend (setRotationalWallWeight): at fluid cells with a solid
        // axis-neighbour (the rows whose one-sided gpCenterGrad makes the cell-centered
        // rotational update marginally unstable) use
        //   P += (rho/dt + w*mu/dx^2)*phi - (1-w)*mu*div(u*)
        // (dx = 1 in cell units): the (1-w) shrinks the destabilizing off-diagonal there and the
        // diagonal w*mu gain keeps those rows relaxing at dt -> infinity where rho/dt vanishes.
        // Bulk cells (w = 0) keep the full-speed rotational update. Outer-shell cells keep the
        // bulk formula (their P is ghost/overwritten).
        CCConst sd = CCConst(sdf_);
        const double w0 = rotWallW_, muF = rotWeight_ * mu_;
        C3 e = e_;
        Kokkos::parallel_for(
            "press_wallblend", Kokkos::RangePolicy<CCExec>(space, 0, n_),
            KOKKOS_LAMBDA(std::size_t i) {
              const long sy = e.x, sz = (long)e.x * e.y;
              const int x = (int)(i % e.x), y = (int)((i / e.x) % e.y), z = (int)(i / sz);
              double w = 0.0;
              if (x > 0 && y > 0 && z > 0 && x < e.x - 1 && y < e.y - 1 && z < e.z - 1 &&
                  sd(i) >= 0.0 &&
                  (sd(i - 1) < 0.0 || sd(i + 1) < 0.0 || sd(i - sy) < 0.0 || sd(i + sy) < 0.0 ||
                   sd(i - sz) < 0.0 || sd(i + sz) < 0.0))
                w = w0;
              P(i) += (ct + w * muF) * ph(i) - (1.0 - w) * muF * d(i);
            });
      } else {
        Kokkos::parallel_for(
            "press", Kokkos::RangePolicy<CCExec>(space, 0, n_),
            KOKKOS_LAMBDA(std::size_t i) { P(i) += ct * ph(i) - mu * d(i); });
      }
    }
    // Snapshot eps^{n+1} -> epsPrev_ for the next step's d(eps)/dt (this projection consumed it).
    if (porous_)
      Kokkos::deep_copy(epsPrev_, epsField_);
  }
  void maskVelocity(int c) {
    CCExec space;
    CCField u = C[c].u, m = C[c].mask;
    Kokkos::parallel_for(
        "vmask", Kokkos::RangePolicy<CCExec>(space, 0, n_), KOKKOS_LAMBDA(std::size_t i) {
          if (m(i) > 0.5)
            u(i) = 0.0;
        });
  }
  // Minimum viscosity over the (global, under MPI) inner cells — the provably-stable rotational
  // coefficient for variable viscosity (chi*mu_min <= mu(x) everywhere).
  double minMuInner() {
    CCExec space;
    C3 e = e_;
    CCConst f = CCConst(muField_);
    double m = 1e300;
    Kokkos::parallel_reduce(
        "minmu",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          if (f(i) < acc)
            acc = f(i);
        },
        Kokkos::Min<double>(m));
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      double g = m;
      MPI_Allreduce(&m, &g, 1, MPI_DOUBLE, MPI_MIN, comm_);
      m = g;
    }
#endif
    return m;
  }
  double reduceMaxAbsInner(CCConst f) {
    CCExec space;
    C3 e = e_;
    double m = 0;
    Kokkos::parallel_reduce(
        "maxabs",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double a = Kokkos::fabs(f(i));
          if (a > acc)
            acc = a;
        },
        Kokkos::Max<double>(m));
    return m;
  }
  std::vector<double> gatherInner(CCField fld) {
    auto h = Kokkos::create_mirror_view(fld);
    Kokkos::deep_copy(h, fld);
    std::vector<double> out((std::size_t)nx_ * ny_ * nz_);
    for (int z = 0; z < nz_; ++z)
      for (int y = 0; y < ny_; ++y)
        for (int x = 0; x < nx_; ++x)
          out[(std::size_t)x + (std::size_t)y * nx_ + (std::size_t)z * (std::size_t)nx_ * ny_] =
              h((long)(x + G) + (long)(y + G) * e_.x + (long)(z + G) * (long)e_.x * e_.y);
    return out;
  }
  // Inverse of gatherInner: scatter an x-fastest (nx,ny,nz) inner-region host buffer into the inner
  // cells of a ghosted G=2 field (ghost cells untouched — refill via exchangeField/fillGhosts).
  void scatterInner(CCField fld, const std::vector<double>& in) {
    if (in.size() != (std::size_t)nx_ * ny_ * nz_)
      throw std::runtime_error("flow::setField: array size does not match the inner grid");
    auto h = Kokkos::create_mirror_view(fld);
    Kokkos::deep_copy(h, fld);  // preserve existing ghosts
    for (int z = 0; z < nz_; ++z)
      for (int y = 0; y < ny_; ++y)
        for (int x = 0; x < nx_; ++x)
          h((long)(x + G) + (long)(y + G) * e_.x + (long)(z + G) * (long)e_.x * e_.y) =
              in[(std::size_t)x + (std::size_t)y * nx_ + (std::size_t)z * (std::size_t)nx_ * ny_];
    Kokkos::deep_copy(fld, h);
  }

  // --- Named field registry (multiphysics field container) ------------------------------------
  // Register a new zero-initialised cell-centred field on the G=2 velocity block and return its
  // buffer. Idempotent: re-adding an existing name returns the existing buffer unchanged.
  CCField addField(const std::string& name) {
    if (fields_.has(name))
      return fields_.at(name).data;
    return fields_.add(name, n_, G, peclet::core::Centering::Cell).data;
  }
  bool hasField(const std::string& name) const { return fields_.has(name); }
  CCField fieldView(const std::string& name) { return fields_.at(name).data; }
  std::vector<std::string> fieldNames() const { return fields_.names(); }
  // Ghost-exchange a registered field (cross-rank + periodic under MPI; periodic-only single-rank).
  void exchangeField(const std::string& name) { fillGhosts(fields_.at(name).data); }
  // Add-reduce ("reverse") halo: fold ghost-layer deposits back onto their owner cell (both
  // cross-rank AND periodic self-wrap). This is the coupling primitive for particle->grid
  // deposition (e.g. void fraction / drag reaction) where a particle near a block boundary scatters
  // into ghost cells owned by a neighbour; after this the inner block holds the complete sum.
  // Single-rank non-periodic: a no-op.
  void exchangeFieldAdd(const std::string& name) {
#ifdef PECLET_FLOW_MPI
    if (distributed_ && velHalo_) {
      CCField f = fields_.at(name).data;
      auto h = Kokkos::create_mirror_view(f);
      Kokkos::deep_copy(h, f);
      peclet::core::halo::GridFieldView<double> view{h.data()};
      velHalo_->reverseAdd(view);
      Kokkos::deep_copy(f, h);
    }
#else
    (void)name;
#endif
  }
  // Host round-trip: read a registered field's inner region as an x-fastest (nx,ny,nz) buffer, or
  // write one (ghosts left stale until the next exchangeField).
  std::vector<double> getField(const std::string& name) {
    return gatherInner(fields_.at(name).data);
  }
  void setField(const std::string& name, const std::vector<double>& v) {
    scatterInner(fields_.at(name).data, v);
  }
  // Padded-block extents + ghost width, so a zero-copy field buffer (size ex*ey*ez, x-fastest) can
  // be reshaped in Python.
  std::array<int, 3> blockShape() const { return {e_.x, e_.y, e_.z}; }
  int ghostWidth() const { return G; }
  // Global grid dims (== local dims single-rank). For the CFD-DEM co-decomposition weight field.
  std::array<int, 3> globalResolution() const {
#ifdef PECLET_FLOW_MPI
    if (distributed_)
      return {gnx_, gny_, gnz_};
#endif
    return {nx_, ny_, nz_};
  }
  // This rank's inner-block origin in GLOBAL cells ({0,0,0} single-rank). The deposit-origin shift
  // so particles in global coords land in the local block (gm origin = blockOrigin * h).
  std::array<int, 3> blockOrigin() const { return {og_.x, og_.y, og_.z}; }

  // --- Scalar transport (advection-diffusion) -------------------------------------------------
  // Register a transported scalar `name` with constant diffusivity D (grid units). scheme: 0 FOU,
  // 1 Koren TVD (default), 2 SOU. iters = RB-GS sweeps for the implicit diffusion solve. Its field
  // is registered in the directory (get_field/set_field/field_view). Openness (set_solid /
  // set_pressure_geometry) must be established for transport to occur.
  void addScalar(const std::string& name, double D, int scheme, int iters) {
    ScalarField sc;
    sc.name = name;
    sc.c = addField(name);  // registered, zero-initialised, on the G=2 block
    sc.cOld = CCField(name + "_old", n_);
    sc.b = CCField(name + "_b", n_);
    sc.AC = CCField(name + "_AC", n_);
    sc.AW = CCField(name + "_AW", n_);
    sc.AE = CCField(name + "_AE", n_);
    sc.AS = CCField(name + "_AS", n_);
    sc.AN = CCField(name + "_AN", n_);
    sc.AB = CCField(name + "_AB", n_);
    sc.AT = CCField(name + "_AT", n_);
    sc.D = D;
    sc.scheme = scheme;
    sc.iters = iters < 1 ? 1 : iters;
    scalars_.push_back(sc);
  }
  bool hasScalar(const std::string& name) const {
    for (const auto& sc : scalars_)
      if (sc.name == name)
        return true;
    return false;
  }
  // Per-face scalar BC: face 0..5 = -x,+x,-y,+y,-z,+z; type 0 periodic, 1 Neumann zero-flux
  // (adiabatic), 2 Dirichlet value. Single-rank / non-decomposed domains (distributed BC deferred).
  void setScalarBc(const std::string& name, int face, int type, double value) {
    for (auto& sc : scalars_)
      if (sc.name == name) {
        sc.bc[face] = type;
        sc.bcVal[face] = value;
        return;
      }
    throw std::runtime_error("set_scalar_bc: no scalar named '" + name + "'");
  }
  // Advance all registered scalars one dt with the current divergence-free velocity (also called at
  // the end of step()). Exposed so a test can prescribe a velocity and transport a scalar in
  // isolation.
  void advanceScalars() {
    if (scalars_.empty())
      return;
    const double idt = 1.0 / dt_;
    CCField Uf, Vf, Wf;
    if constexpr (Grid::collocated) {
      Uf = uf_;
      Vf = vf_;
      Wf = wf_;
    } else {
      Uf = C[0].u;
      Vf = C[1].u;
      Wf = C[2].u;
    }
    fillGhosts(Uf);
    fillGhosts(Vf);
    fillGhosts(Wf);  // face velocities need the ±2 advection reach
    for (auto& sc : scalars_) {
      scalarBuildDiffusionOpen(sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, CCConst(ox_),
                               CCConst(oy_), CCConst(oz_), sc.D, idt, e_, G);
      applyScalarBcStencil(sc);  // re-open Dirichlet domain faces (set_domain_bc closes openness)
      // WO-P01: the optional PER-CELL Dirichlet set (interfacial cells at T_sat). Inert — and the
      // operator therefore bit-identical — until a caller allocates the mask.
      const bool hasMask = sc.dmask.extent(0) == n_;
      if (hasMask)
        scalarMaskStencil(sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, CCConst(sc.dmask), e_,
                          G);
      Kokkos::deep_copy(sc.cOld, sc.c);
      scalarFillGhosts(sc);
      scalarBuildRhs(sc.b, CCConst(sc.cOld), CCConst(Uf), CCConst(Vf), CCConst(Wf), CCConst(ox_),
                     CCConst(oy_), CCConst(oz_), idt, sc.scheme, e_, G);
      if (hasMask)
        scalarMaskRhs(sc.b, sc.c, CCConst(sc.dmask), CCConst(sc.dval), e_, G);
      // implicit diffusion: red-black Gauss-Seidel with a ghost fill before each color sweep.
      for (int it = 0; it < sc.iters; ++it) {
        scalarFillGhosts(sc);
        cutcellSmoothColor(sc.c, CCConst(sc.b), sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, e_,
                           og_, G, 0);
        scalarFillGhosts(sc);
        cutcellSmoothColor(sc.c, CCConst(sc.b), sc.AC, sc.AW, sc.AE, sc.AS, sc.AN, sc.AB, sc.AT, e_,
                           og_, G, 1);
      }
      scalarFillGhosts(sc);
    }
  }

  // --- Geometric VoF: the colour field (rung V2a, WO-J) ---------------------------------------
  //
  // WHAT THIS RUNG IS. One phase, transported by geometric (PLIC + Weymouth-Yue) VoF, drives the
  // fluid properties through the ORDINARY property closures, and the existing variable-density
  // projection carries the density jump. No surface tension (V4), and — important — NO
  // MOMENTUM-CONSISTENT TRANSPORT (that is rung V2b / WO-K). Mass and momentum are therefore
  // advected by different fluxes, which multiplies the light phase's acceleration by the heavy
  // phase's density in a mixed cell: a spurious interfacial momentum source of order Δρ. The
  // literature is unambiguous that this breaks down around density ratio 1000 unless the
  // resolution is absurd (Rudman 1998; Arrufat et al., Computers & Fluids 215:104785, 2021 —
  // accurate raindrop at 15 cells/diameter WITH consistency versus ~200 without). SO: **V2a is
  // valid only at modest density ratios.** A ratio-1000 case that is at REST (the hydrostatic acid
  // test) is exact here, because there is no momentum to mis-advect; a ratio-1000 case with motion
  // is not this rung's business.
  //
  // USAGE
  //   s.enable_vof()                              # registers "C" and the g=3 working block
  //   s.set_vof(C0)                               # sharp initial colour, C in [0,1]
  //   s.set_property_model("rho", "linear", "C", [rho_g, rho_l - rho_g])   # rho(C); enables varRho
  //   s.set_property_model("mu",  "linear", "C", [mu_g,  mu_l  - mu_g])    # mu(C) (optional)
  //   s.set_property_model("force_z", "linear", "rho", [0.0, -g])          # gravity
  //
  // STRUCTURE (`suite/docs/VOF_PLAN.md` §3 rule 1: the colour field gets its own g=3 halo and the
  // solver's G = 2 is NOT widened). There are two blocks and the split is deliberate:
  //   * `"C"` is a NORMAL registered G=2 cell field. That is what makes item 3 of the work order
  //     ("ρ(C) and μ(C) through the EXISTING closures, no new closure machinery") possible at all:
  //     `applyClosure` indexes its input and its output with the SAME linear index on the SAME
  //     extent, so a closure input MUST live on the G=2 block. It is also what gives C
  //     get_field/set_field/field_view/exchange_field/redistribute for free.
  //   * the g=3 block is the advector's own working block (`vof::WyAdvector`), with its own
  //     `GridHaloTopology` at width 3 under MPI. MYC needs 3^3 and the donor ring is one cell
  //     outside the inner region (so advection alone needs 2); width 3 is the plan's choice for the
  //     V3 height-function columns.
  // The two blocks exchange INNER REGIONS ONLY (`copyInner`, both ways) and each fills its own
  // ghosts with its own policy — that is the one bridge, and it carries no offset arithmetic
  // beyond `copyInner`'s. The face velocities go the other way, whole-block-embedded
  // (`vof::copyBlockEmbed`), because the advector reads them one cell outside its inner region.
  //
  // STAGGERED ONLY. The collocated path is rung V8 and needs the collocated variable-density
  // projection (which throws today) before it means anything; `enableVof` throws there rather than
  // half-supporting it.
  //
  // NO IMMERSED SOLIDS YET. `VOF_PLAN.md` §3 rule 2 makes C the liquid fraction of the FLUID volume
  // with openness-weighted geometric fluxes (Huang 2025/2026 solid-clipped flux polygons). That is
  // not this work order's scope, and a silently-unweighted flux would leak C into the solid, so
  // `advectVof` throws if an immersed solid is present. An ALL-FLUID `set_pressure_geometry` is
  // fine (and is what the acid test uses) — the check is on `hasSolid_`, i.e. on any inner SDF < 0.
  static constexpr int kVofG = 3;  // the colour field's ghost width (VOF_PLAN §3 rule 1)

  void enableVof() {
    if constexpr (Grid::collocated) {
      // Rung V8 (WO-T): allowed. The colour is advected by the PROJECTED face field uf_/vf_/wf_ —
      // which is what the ABC approximate projection makes exactly divergence-free, i.e. precisely
      // the field Weymouth-Yue's conservation proof needs — and every interfacial force is a face
      // acceleration (collocated_varrho.hpp). ALL-FLUID only at this rung.
      if (hasSolid_)
        throw std::runtime_error(
            "enable_vof: geometric VoF on SolverColocated (rung V8) is ALL-FLUID only — an immersed "
            "solid needs the cut-cell face acceleration and the matching one-sided closures, which "
            "is a later rung. Use the staggered Solver (rung V5a supports cut cells).");
      collocatedV8AutoFallback("geometric VoF on the collocated grid");
    }
    if (vofEnabled_)
      return;
    cField_ = addField("C");  // the G=2 registry mirror (closure input / IO / redistribute)
    vofEnabled_ = true;
    buildVofBlock();
  }
  bool vofEnabled() const { return vofEnabled_; }
  // Initial / prescribed colour field on the inner cells (flat x-fastest, nx*ny*nz), C in [0,1]:
  // the LIQUID fraction of the cell. Enables VoF if it is not on yet. Ghosts are refreshed here so
  // a closure applied before the first step already sees a consistent field.
  void setVof(const std::vector<double>& c) {
    enableVof();
    scatterInner(cField_, c);
    zeroSolidColour();  // rung V5a: solid cells carry no colour (the band fill is regenerated)
    fillPropGhosts(cField_);
  }
  std::vector<double> getVof() { return gatherInner(cField_); }
  // Local (this rank's) colour census: sum / min / max / mixed-cell count / wisp count.
  vof::WyAdvector::Diagnostics vofDiagnostics() {
    if (!vofEnabled_)
      throw std::runtime_error("vof_diagnostics: VoF is not enabled (call enable_vof/set_vof)");
    bridgeColourToVof();
    auto d = vofAdv_.diagnostics();
    // `solidSumC` is the census of the CANONICAL field: the working block's solid cells carry the
    // neutral band fill (reported as `solidFillSum`), while "C" itself is 0 there — that is the
    // quantity gate G2 of WO-Q asks for.
    d.solidSumC = vofSolidColourSum();
    return d;
  }
  // sum of the canonical colour field "C" over SOLID cells of this rank (0 by construction).
  double vofSolidColourSum() {
    if (!vofEnabled_ || !vofAdv_.hasGeometry() || !vofSolidG2_.extent(0))
      return 0.0;
    CCConst c = CCConst(cField_), sl = CCConst(vofSolidG2_);
    const int ex = e_.x, ey = e_.y, g = G;
    double acc = 0.0;
    Kokkos::parallel_reduce(
        "peclet::flow::vof_solid_sum",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {0, 0, 0}, {nx_, ny_, nz_}),
        KOKKOS_LAMBDA(int x, int y, int z, double& a) {
          const long i = (long)(x + g) + (long)(y + g) * ex + (long)(z + g) * (long)ex * ey;
          if (sl(i) > 0.5)
            a += c(i);
        },
        acc);
    Kokkos::fence();
    return acc;
  }
  // Interface-local Courant number max|uf|*dt/h over the faces of mixed cells and their face
  // neighbours, with the CURRENT velocity and dt (an all-reduce max under MPI). This is the number
  // the WY boundedness bound applies to — NOT the global max, which over-throttles badly (V1
  // measured 0.314 in a quiescent Zalesak corner against 0.157 at the interface). Use it to pick
  // dt: `dt_new = dt * cfl_target / vof_max_courant()`.
  double vofMaxCourant() {
    if (!vofEnabled_)
      return 0.0;
    bridgeVelocityToVof();
    bridgeColourToVof();
    const double loc = vofAdv_.maxCourantInterfaceAuto(dt_ / vofAdv_.h());
    return vofAdv_.globalMax ? vofAdv_.globalMax(loc) : loc;
  }
  // The interface-local Courant number of the step just taken (0 before the first step).
  double vofLastCourant() const { return vofEnabled_ ? vofAdv_.lastCfl() : 0.0; }
  // Weymouth-Yue boundedness cap (default 0.25, the PROVEN 3D bound 1/(2(N-1)); 0.5 is the 2D
  // value). `step()` throws when the interface-local Courant number exceeds it.
  void setVofCflLimit(double v) {
    vofCflLimit_ = v;
    if (vofEnabled_)
      vofAdv_.cflLimit = v;
  }
  double vofCflLimit() const { return vofCflLimit_; }

  // --- rung V5a (WO-Q): VoF transport through an immersed solid ---------------------------------
  //
  // The colour advection is openness-weighted (`vof/cutcell.hpp`): the geometric flux of every face
  // is multiplied by the face openness `o_f`, the update is done in FLUID-VOLUME units
  // (`eps_i C_i`), and the dilation term uses the same `o_f a_f` — so the conserved functional
  // `sum_i eps_eff_i C_i` telescopes against the projection's own openness-weighted divergence,
  // exactly as in the uncut case. Solid cells carry no colour (the canonical "C" reads 0 there) and
  // the working block's solid band carries the neutral 90-degree fill for the MYC / height-function
  // stencils.
  //
  // WHAT IT APPROXIMATES: the PLIC polyhedron is reconstructed on the WHOLE unit cell and its slab
  // volume is multiplied by the open area, instead of being clipped against the solid as well
  // (Huang, JCP 2025/2026). Conservative and exact where interface and wall are parallel or the
  // cell is whole; O(1) wrong in the distribution INSIDE a cell whose interface crosses its wall.
  // `vof_diagnostics().clipped_volume` is the tripwire.
  void requireVofGeometry(const char* who) {
    if constexpr (Grid::collocated) {
      // Rung V8 scope, re-checked here because `set_solid` can follow `enable_vof`.
      if (hasSolid_)
        throw std::runtime_error(
            std::string(who) +
            ": geometric VoF on SolverColocated (rung V8) is ALL-FLUID only. The cut-cell colour "
            "transport (rung V5a) is validated on the STAGGERED solver.");
    }
    if (!hasSolid_ || vofAdv_.hasGeometry())
      return;
    std::string m(who);
    m += ": an immersed solid is present but the colour block has no cut-cell geometry. Rung V5a "
         "needs the cut-cell openness, i.e. set_solid(sdf, cutcell_pressure=True) (the staircase "
         "pressure operator has no face openness to weight the geometric fluxes with).";
    throw std::runtime_error(m);
  }
  // Does the colour advection run the cut-cell (openness-weighted) kernels?
  bool vofHasGeometry() const { return vofAdv_.hasGeometry(); }
  // Ablation: drop Weymouth's admissible-interval clamp on the openness-weighted flux
  // (`vof/cutcell.hpp` vofCutFluxClamp). ON by default — the measurement that put it there is in
  // that header. With it off the [0,1] clip becomes the mechanism instead of a tripwire and the
  // conserved functional drifts.
  void setVofCutFluxClamp(bool on) { vofAdv_.cutFluxClamp = on; }
  bool vofCutFluxClamp() const { return vofAdv_.cutFluxClamp; }
  // What the CANONICAL "C" field carries in SOLID cells (the working block always carries the
  // neutral band fill, which is what the MYC / height-function stencils need):
  //   true  (default) 0 — "no colour in the solid", the WO-Q gate. The closures then see gas
  //                   density there and the CSF sees a full colour jump across a wall face.
  //   false           the band fill — a zero-slope continuation of the liquid into the wall.
  // Measured on the G5 cap (D/dx = 24, sigma = 1, mu = 0.05): see the WO-Q findings entry.
  void setVofSolidColourZero(bool on) { vofSolidZero_ = on; }
  bool vofSolidColourZero() const { return vofSolidZero_; }
  // The colour field INCLUDING the neutral solid-band fill, on the inner region — i.e. what the
  // MYC / height-function stencils actually read, as opposed to the canonical "C" (0 in solid).
  // The fill is regenerated here, so this is also the direct gate on its decomposition
  // independence: it must be pointwise BITWISE across np (`tests/kokkos_mpi/test_vof_cutcell_mpi`).
  std::vector<double> getVofFilledColour() {
    if (!vofEnabled_)
      throw std::runtime_error("vof_filled_colour: VoF is not enabled");
    bridgeColourToVof();
    CCField t("vofFilled", n_);
    copyInner(t, e_, G, CCConst(vofAdv_.colour()), e3_, kVofG);
    return gatherInner(t);
  }
  // The cut-cell geometry the colour block runs on, on the inner region: 0 = the cell fluid
  // fraction eps, 1/2/3 = the openness of the +x/+y/+z face of each cell (the ADVECTOR's high-face
  // convention), 4 = the cell classification (1 = solid). All must be bitwise across np.
  std::vector<double> getVofGeometry(int which) {
    if (!vofEnabled_ || !vofAdv_.hasGeometry())
      throw std::runtime_error("vof_geometry: no cut-cell geometry (needs set_solid + enable_vof)");
    CCField t("vofGeom", n_);
    CCConst src;
    if (which == 0)
      src = CCConst(vofAdv_.epsFraction());
    else if (which < 4)
      src = CCConst(vofAdv_.faceOpenness(which - 1));
    else
      src = CCConst(vofAdv_.kindDouble());
    copyInner(t, e_, G, src, e3_, kVofG);
    return gatherInner(t);
  }
  // --- rung V5b (WO-S): static contact angle on SDF solids --------------------------------------
  //
  // The band fill of rung V5a is a stencil device: it decides what the MYC 3^3 stencil and the V3
  // height-function columns of a near-wall cell see INSIDE the solid, and WO-Q's neutral (mean of
  // the fluid face neighbours) rule is the zero-slope continuation, i.e. the 90-degree
  // Afkhami-Bussmann limit. `set_contact_angle` replaces PASS 1 of that fill by the fractions of
  // the plane that continues the fluid-side interface into the solid at the prescribed angle
  // theta, measured THROUGH THE LIQUID (`m . n_w = cos theta`, `vof/wetting.hpp`). Nothing else
  // changes: no force is added at the wall, the V3 cascade and the V4 balanced force are the
  // unmodified ones, and passes 2-3 of the fill are WO-Q's.
  //
  // theta is a per-cell FIELD so the dynamic-angle rung (V6) changes only what fills it. Setting
  // it needs `set_solid(..., cutcell_pressure=True)` + `enable_vof` (there is no wall otherwise);
  // with no call the neutral fill runs and every V5a number is byte-identical.
  void setContactAngle(double thetaDeg) {
    if (!(thetaDeg >= 0.0 && thetaDeg <= 180.0))
      throw std::runtime_error("set_contact_angle: theta must be in [0, 180] degrees");
    contactAngleDeg_ = thetaDeg;
    contactAngleField_.clear();
    contactAngleSet_ = true;
    applyContactAngle();
  }
  // Per-cell contact angle in DEGREES on the inner region (flat x-fastest, nx*ny*nz). Only the
  // value at the SOLID band cell being filled is read, so cells away from a wall are irrelevant.
  void setContactAngleField(const std::vector<double>& thetaDeg) {
    if (thetaDeg.size() != (std::size_t)nx_ * ny_ * nz_)
      throw std::runtime_error("set_contact_angle_field: expected nx*ny*nz values");
    contactAngleField_ = thetaDeg;
    contactAngleSet_ = true;
    applyContactAngle();
  }
  bool contactAngleSet() const { return contactAngleSet_; }
  double contactAngle() const { return contactAngleDeg_; }
  // Which anchor the theta-plane uses (`vof::VofWettingPivot`): 0 volume-consistent (DEFAULT,
  // idempotent), 1 the PLIC centroid p_f (Afkhami-Bussmann), 2 the work order's
  // `c = p_f - sdf(p_f) n_w` (NOT idempotent — measured to be off by 0.26 in cell fraction at
  // theta = 60, gate G0), 3 the contact line on the wall. Ablation only.
  void setContactAnglePivot(int mode) {
    if (mode < 0 || mode > 3)
      throw std::runtime_error("set_contact_angle_pivot: mode must be 0..3");
    contactPivot_ = mode;
    vofAdv_.wettingPivot = mode;
  }
  int contactAnglePivot() const { return contactPivot_; }
  struct ContactAngleDiagnostics {
    long contactCells = 0;    ///< band cells written by the theta plane of their own anchor
    long neighbourCells = 0;  ///< band cells written by the mean of the anchor's MIXED neighbours
    long pureCells = 0;       ///< band cells that took the pure-phase continuation
    long parallelCells = 0;   ///< band cells whose interface was parallel to the wall
    long neutralCells = 0;    ///< band cells that fell back to WO-Q's neutral mean
    long unfilledCells = 0;   ///< SOLID cells pass 1 left untouched (passes 2-3 then fill them)
    double meanApparentAngle = 0.0;  ///< mean measured apparent angle over `contactCells`, degrees
    double setAngle = 0.0;           ///< the prescribed angle, degrees (uniform case)
  };
  // The band census of the CURRENT colour field: how many band cells each branch of the fill wrote
  // and the mean APPARENT angle the fluid-only normal reported at the contact cells (G1's
  // measurement, evaluated on the fill's own data rather than on a post-processed shape).
  ContactAngleDiagnostics contactAngleDiagnostics() {
    if (!vofEnabled_)
      throw std::runtime_error("contact_angle_diagnostics: VoF is not enabled");
    ContactAngleDiagnostics d;
    d.setAngle = contactAngleDeg_;
    if (!vofAdv_.hasWetting())
      return d;
    bridgeColourToVof();  // regenerates the fill, hence the census
    long counts[vof::kVofWetCount];
    long nApp = 0;
    vofAdv_.wettingCensus(counts, d.meanApparentAngle, nApp);
    d.unfilledCells = counts[vof::kVofWetNone];
    d.contactCells = counts[vof::kVofWetTheta];
    d.neighbourCells = counts[vof::kVofWetNeighbour];
    d.pureCells = counts[vof::kVofWetPure];
    d.parallelCells = counts[vof::kVofWetParallel];
    d.neutralCells = counts[vof::kVofWetNeutral];
    return d;
  }
  // Wire the theta field + the wall SDF onto the colour block. Idempotent; called by the setters
  // and again by every geometry rebuild (`buildVofGeometry`), since the block can be re-sized.
  void applyContactAngle() {
    if (!contactAngleSet_ || !vofEnabled_ || !vofAdv_.hasGeometry())
      return;  // remembered; buildVofGeometry calls back once the geometry exists
    vofAdv_.enableWetting();
    vofAdv_.wettingPivot = contactPivot_;
    // (a) the SDF on the colour block: inner region from the solver's own sdf_, then the colour
    //     field's ghost policy, so the central-difference wall normal at ghost depth <= 2 is the
    //     OWNER's (the WO-Q finding-5 argument, applied to the wall normal).
    copyInner(vofAdv_.wallSdf(), e3_, kVofG, CCConst(sdf_), e_, G);
    vofExchangeRaw(vofAdv_.wallSdf());
    // (b) theta, in radians.
    const double toRad = 3.14159265358979323846 / 180.0;
    if (contactAngleField_.empty()) {
      Kokkos::deep_copy(vofAdv_.contactAngle(), contactAngleDeg_ * toRad);
    } else {
      CCField t("thetaG2", n_);
      std::vector<double> rad(contactAngleField_.size());
      for (std::size_t i = 0; i < rad.size(); ++i)
        rad[i] = contactAngleField_[i] * toRad;
      scatterInner(t, rad);
      copyInner(vofAdv_.contactAngle(), e3_, kVofG, CCConst(t), e_, G);
      vofExchangeRaw(vofAdv_.contactAngle());
    }
  }

  // The colour advector itself (its g=3 block, geometry views and planes). For TESTS: gate G3 of
  // `tests/kokkos/test_vof_cutcell.cpp` rebuilds the openness/fraction by an independent route and
  // compares against these.
  const vof::WyAdvector& vofAdvector() const { return vofAdv_; }
  // The sweep permutation index of the NEXT colour advection (`kWySweepPerm[n % 6]`). Exposed so a
  // benchmark can hold the permutation fixed, or resume one, across a restart.
  void setVofStepParity(long n) { vofStep_ = n; }
  long vofStepParity() const { return vofStep_; }

  // KINEMATIC colour advection: advance C ONCE with the solver's CURRENT face velocity and the
  // given dt, with no Navier-Stokes step at all. This is the entry point the advection benchmarks
  // (Zalesak, LeVeque) and the cut-cell conservation gates use — a frozen Stokes field advecting a
  // colour slab is a pure statement about the advection scheme, with the momentum solve and the
  // pressure solve out of the picture.
  //
  // It REFUSES a velocity field that is not discretely divergence-free to 1e-10: Weymouth-Yue's
  // exact conservation is conditional on `sum_f o_f u_f = 0` per cell (the dilation term adds
  // `H(C-1/2)` times that residual to EVERY full cell's budget), so a run on a non-solenoidal field
  // would report a conservation "defect" that is really the caller's velocity. Use the solver's own
  // projected output (run `step()` to a steady state, or call `project()`), never an analytic
  // sample.
  void advectVofKinematic(double dt) {
    if (!vofEnabled_)
      throw std::runtime_error("advect_vof: VoF is not enabled (call enable_vof / set_vof first)");
    requireVofGeometry("advect_vof");
    const double div = maxOpenDivergence();
    if (!(div <= 1e-10)) {
      char msg[320];
      std::snprintf(msg, sizeof(msg),
                    "advect_vof: the current face velocity is not discretely divergence-free "
                    "(max|div(open*u)| = %.6g > 1e-10). Weymouth-Yue conservation is conditional "
                    "on it; project the field first (step() / project()).",
                    div);
      throw std::runtime_error(msg);
    }
    bridgeVelocityToVof();
    bridgeColourToVof();
    vofAdv_.advect(dt, vofStep_++);
    copyInner(cField_, e_, G, CCConst(vofAdv_.colour()), e3_, kVofG);
    zeroSolidColour();
    fillPropGhosts(cField_);
  }
  // Harmonic instead of arithmetic rho_f in the pressure projection (WO-J item 5). DEFAULT OFF and
  // it should stay off — read the long note in mac_pressure.hpp before turning it on: arithmetic
  // rho_f IS the harmonic mean of the mobility 1/rho (the series-correct choice for a normal flux)
  // and is what makes the discrete hydrostatic balance exact, because the momentum time term and
  // the face body force interpolate rho arithmetically and are NOT switched by this flag. Shipped
  // as a measured knob for the coefficient-coarsening question, not as an alternative scheme.
  void setRhoFaceHarmonic(bool on) { rhoFaceHarmonic_ = on; }
  bool rhoFaceHarmonic() const { return rhoFaceHarmonic_; }
  // WO-R item 4 — and the measurement REFUTED the item. `doc/variable_density_projection.md` §4
  // listed the missing `1/rho_f` in `bcCorrectOutflow` as a defect ("fine when the outflow region
  // has rho ~ uniform; revisit with a two-phase outflow case"). This is that two-phase outflow
  // case, and the factor makes things WORSE by seven orders of magnitude:
  //
  //   stratified duct, ratio 10, 5 steps, max|div(open u)| of the PROJECTED field
  //     without the factor (the shipped behaviour)   8.76e-10
  //     with    the factor (`bcCorrectOutflowVar`)   9.24e-03
  //   (tests/kokkos/test_vof_bc.cpp gate F2; at ratio 1 the two are bitwise equal, 1.41e-17.)
  //
  // The mechanism, and why it is not a bug in the sibling kernel: a projection correction removes
  // the discrete divergence only if it uses the SAME face coefficient the operator row used. The
  // interior faces carry `open_f * rho0/rho_f` (`buildRhoCoeff`), but that kernel runs over INNER
  // cells only — the outflow boundary face's coefficient is set by the multigrid's own boundary
  // re-imposition (`bcSetOpenness`, Dirichlet outflow -> open) and is the RAW openness. So the
  // consistent correction at that face is exactly the plain `phi` difference, which is what
  // `bcCorrectOutflow` has always done. The real inconsistency, if one wants it removed, is on
  // the OPERATOR side (give the outflow face the same `rho0/rho_f` the interior faces have, on
  // every level) — a `CutcellMG` change, not a projection-correction one, and not this rung.
  //
  // The sibling therefore ships as a measured ABLATION, DEFAULT OFF. It is bitwise inert at
  // constant density either way (rho_f == rho0 makes the factor exactly 1).
  void setOutflowRhoCorrection(bool on) { outflowRhoCorr_ = on; }
  bool outflowRhoCorrection() const { return outflowRhoCorr_; }

  // --- two-phase open boundaries (rung V-BC, WO-R) ---------------------------------------------
  //
  // Rung V2a gave the colour field one non-periodic ghost rule, `clampFill` (globally-clamped
  // zero-gradient). It is the right rule for a WALL and the wrong one for an INFLOW, where the
  // colour of the incoming fluid is a prescribed datum. Three API calls cover the three domain-BC
  // types; `src/vof/colour_bc.hpp` carries the rules and the reasoning, and this is the plumbing:
  //
  //   set_vof_inflow(face, C)            type-2 face: the colour of the incoming fluid
  //   set_vof_inflow_profile(face, C2d)  the same, per position on the face
  //   set_vof_backflow(face, C)          type-3 face: the colour of fluid that flows back IN
  //   (a type-1 wall keeps `clampFill` = the 90 deg neutral continuation; WO-S replaces it)
  //
  // Setting any of them ARMS the rung: the advector gets the out-of-domain mask (so a boundary
  // donor is fluxed algebraically as `C_donor * a` rather than as a reconstructed PLIC slab — see
  // `wyFaceFluxBc`), the per-face boundary liquid volumes start being accumulated
  // (`vof_diagnostics()['inflow_volume'] / ['outflow_volume']`, and `vof_bc_volumes()` per face),
  // and the property ghosts of an inflow face follow the inflow colour through the closures
  // instead of copying the interior (item 5 below). With none of them set NOTHING changes.
  //
  // The colour of the incoming fluid may be FRACTIONAL and it then means "this fraction of the
  // incoming flux is liquid" — a flux statement, not a sub-cell interface position. That is
  // exactly what the algebraic boundary flux implements.

  /// Colour of the fluid entering through inflow face `f` (0..5 = -x,+x,-y,+y,-z,+z), in [0,1].
  /// The face must already be an inflow (`set_domain_bc(f, 2, ...)`).
  void setVofInflow(int f, double value) {
    checkVofBcFace(f, 2, "set_vof_inflow");
    vofInflowSet_[f] = true;
    vofInflowC_[f] = value;
    vofInflowProfRaw_[f].clear();
    vofBcArm();
  }
  /// Per-position inflow colour on face `f`: `prof` is (nb, nc) on the INNER grid of the face's two
  /// perpendicular axes (the same layout and the same clamp resampling `set_domain_bc_profile`
  /// uses for the velocity).
  void setVofInflowProfile(int f, const std::vector<double>& prof, int nb, int nc) {
    checkVofBcFace(f, 2, "set_vof_inflow_profile");
    if ((int)prof.size() != nb * nc)
      throw std::runtime_error("set_vof_inflow_profile: profile size != nb*nc");
    vofInflowSet_[f] = true;
    vofInflowProfRaw_[f] = prof;
    vofInflowProfNb_[f] = nb;
    vofInflowProfNc_[f] = nc;
    vofBcArm();
  }
  /// `inletOutlet` backflow colour on outflow face `f` (default 0 = gas): where the boundary face
  /// velocity points back INTO the domain, the colour ghost carries this value instead of the
  /// zero-gradient copy (Rusche 2002 thesis section 4; OpenFOAM `inletOutletFvPatchField`). Where
  /// the fluid leaves, zero-gradient is kept and what leaves is what is inside.
  void setVofBackflow(int f, double value) {
    checkVofBcFace(f, 3, "set_vof_backflow");
    vofBackflowSet_[f] = true;
    vofBackflowC_[f] = value;
    vofBcArm();
  }
  bool vofBcActive() const { return vofBcActive_; }
  /// Signed liquid volume that crossed each of the six domain faces during the LAST colour
  /// advection, in cell-volume units, POSITIVE for liquid entering the domain. Local to this rank
  /// (a distributed caller sums them, as it does for every other VoF diagnostic).
  std::vector<double> vofBcVolumes() const {
    return std::vector<double>(vofBcVol_, vofBcVol_ + 6);
  }
  /// The same, accumulated since `enable_vof()` (or the last `resetVofBcVolumes()`). Changing a
  /// boundary colour mid-run deliberately does NOT reset it — a slug injection is exactly the case
  /// where the running total is the quantity of interest.
  std::vector<double> vofBcVolumesTotal() const {
    return std::vector<double>(vofBcVolTotal_, vofBcVolTotal_ + 6);
  }
  void resetVofBcVolumes() {
    for (int f = 0; f < 6; ++f)
      vofBcVol_[f] = vofBcVolTotal_[f] = 0.0;
  }

  // nvcc requires the enclosing member of an extended device lambda to be public; these are
  // implementation detail (see the same note above `patchScalarDirichletFace`).

  /// The colour field's boundary rules, applied at the END of `vofFillGhosts` — i.e. after the
  /// halo/periodic exchange and after `clampFill`, so the BC overwrite wins exactly as the
  /// property/velocity BCs win over the halo fill (WO-F's fill-then-BC order).
  ///
  /// TWO GUARDS, both load-bearing:
  ///  * `vofBcActive_` — with no VoF BC set this returns before touching anything, which is what
  ///    makes gate G5 (every existing VoF ctest bit-identical) hold by construction.
  ///  * the identity test against `vofAdv_.colour()` — `vofFillGhosts` is the advector's generic
  ///    `exchange` hook and rung V2b calls it on the half-shifted colour and on the momentum
  ///    velocity fields too (`momentum_advect.hpp`). A colour BC applied to those would be
  ///    nonsense. (Consequence, recorded rather than hidden: under `enable_vof_momentum` the
  ///    half-shifted colour keeps the zero-gradient band at an inflow face. It matters only when
  ///    the inflow colour differs from the colour of the fluid already at the boundary.)
  void vofApplyColourBc(CCField f) {
    if (!vofBcActive_ || !vofEnabled_)
      return;
    if (f.data() != vofAdv_.colour().data())
      return;
    const I3 e3{e3_.x, e3_.y, e3_.z};
    for (int face = 0; face < 6; ++face) {
      if (!touchesGlobalFace(face))
        continue;  // rank-owned global faces only (the WO-F rule)
      const int a = face / 2, sd = face % 2;
      if (bc_[face] == 2 && vofInflowSet_[face]) {
        if (vofInflowProf3_[face].extent(0))
          vof::bcColourProfile(f, e3, kVofG, a, sd, vofInflowProf3_[face], vofProf3Nc_[face]);
        else
          vof::bcColourConst(f, e3, kVofG, a, sd, vofInflowC_[face]);
      } else if (bc_[face] == 3 && vofBackflowSet_[face]) {
        // reads the face velocity the advector is about to flux with, so it must run after
        // bridgeVelocityToVof() — which it does: advectVof bridges the velocity first.
        vof::bcColourBackflow(f, e3, kVofG, a, sd, vofAdv_.faceVel(a), vofBackflowC_[face]);
      }
    }
  }

  /// WO-R item 5 — the Neumann property policy at an inflow face.
  ///
  /// `fillPropGhosts` copies the inner cell's value into the ghost. At a liquid inlet next to a gas
  /// interior that makes the inlet FACE density (the arithmetic mean of inner and ghost, used by
  /// the momentum time term and by the projection coefficient alike) the interior's density, wrong
  /// by up to the full ratio. rho and mu are closures of C and the colour ghost now carries the
  /// inflow value, so the consistent repair is not a second BC rule but the SAME closure evaluated
  /// on the ghost band: `rho_ghost = rho(C_inflow)` by construction.
  ///
  /// Two cases, both keyed on the field identity:
  ///  * `f` IS the G=2 colour mirror -> put the inflow colour in its ghost band (the Neumann copy
  ///    just overwrote it). This has to happen first, and it does: `advectVof` fills C's ghosts and
  ///    `project()` fills rho's later in the step.
  ///  * `f` is a closure OUTPUT -> re-evaluate that closure on the ghost band.
  /// Anything else (a field with no closure, a hand-set rho) keeps the Neumann copy — there is no
  /// C to derive it from and inventing one would be a silent model.
  void vofBcPropGhosts(CCField f) {
    if (!vofBcActive_ || !vofEnabled_)
      return;
    const bool isColour = cField_.extent(0) && f.data() == cField_.data();
    const I3 e2{e_.x, e_.y, e_.z};
    for (int face = 0; face < 6; ++face) {
      if (bc_[face] != 2 || !vofInflowSet_[face] || !touchesGlobalFace(face))
        continue;
      const int a = face / 2, sd = face % 2;
      if (isColour) {
        if (vofInflowProfG2_[face].extent(0))
          vof::bcColourProfile(f, e2, G, a, sd, vofInflowProfG2_[face], vofProfG2Nc_[face]);
        else
          vof::bcColourConst(f, e2, G, a, sd, vofInflowC_[face]);
      } else {
        for (const auto& cl : closures_)
          if (cl.out.data() == f.data())
            applyClosureFaceGhost(cl, e_, G, a, sd);
      }
    }
  }

  /// (Re)build everything that lives on the g=3 block for this rung: the out-of-domain mask and the
  /// resampled boundary-colour profiles. Called from `buildVofBlock` (so a redistribute/initMpi
  /// re-derives them) and from `vofBcArm` (so a setter takes effect immediately).
  void vofRebuildBcBlock() {
    if (!vofEnabled_)
      return;
    const I3 e3{e3_.x, e3_.y, e3_.z};
    const bool px = vofAxisPeriodic(0), py = vofAxisPeriodic(1), pz = vofAxisPeriodic(2);
    if (vofBcActive_) {
      vofOutside_ = vof::UCField("vof::outside", vofAdv_.size());
      vof::buildOutsideMask(vofOutside_, e3, kVofG, vofOrigin(), vofGlobalSize(), px, py, pz);
      vofAdv_.setOutsideMask(vofOutside_);
    } else {
      vofAdv_.setOutsideMask(vof::UCField());
    }
    // the boundary-flux ledger counts only the GLOBAL domain faces this rank owns
    for (int face = 0; face < 6; ++face)
      vofAdv_.setBcFaceOwned(face, vofBcActive_ && bc_[face] != 0 && touchesGlobalFace(face));
    for (int face = 0; face < 6; ++face) {
      vofInflowProf3_[face] = CCField();
      vofInflowProfG2_[face] = CCField();
      if (vofInflowProfRaw_[face].empty())
        continue;
      vofInflowProf3_[face] =
          resampleFaceScalar(vofInflowProfRaw_[face], vofInflowProfNb_[face],
                             vofInflowProfNc_[face], face, e3_, kVofG, vofProf3Nc_[face]);
      vofInflowProfG2_[face] =
          resampleFaceScalar(vofInflowProfRaw_[face], vofInflowProfNb_[face],
                             vofInflowProfNc_[face], face, e_, G, vofProfG2Nc_[face]);
    }
  }

  /// Clamp-resample a per-position face scalar from the user's (nb, nc) INNER grid onto the
  /// ghost-inclusive (b, c) plane of an extended block, so the fill kernel indexes it directly by
  /// face position. Same rule as `setDomainBcProfile`, one component instead of three.
  CCField resampleFaceScalar(const std::vector<double>& prof, int nb, int nc, int face, C3 ext,
                             int g, int& outNc) {
    const int a = face / 2;
    const int dims[3] = {ext.x, ext.y, ext.z};
    const int bax = (a + 1) % 3, cax = (a + 2) % 3;
    const int Lb = dims[bax], Lc = dims[cax];
    CCField pf("vof::bcprof", (std::size_t)Lb * Lc);
    auto h = Kokkos::create_mirror_view(pf);
    auto cl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
    for (int p0 = 0; p0 < Lb; ++p0)
      for (int p1 = 0; p1 < Lc; ++p1)
        h((long)p0 * Lc + p1) = prof[(std::size_t)cl(p0 - g, nb) * nc + cl(p1 - g, nc)];
    Kokkos::deep_copy(pf, h);
    outNc = Lc;
    return pf;
  }

  /// A VoF boundary colour is only meaningful on a face that already carries the matching domain
  /// BC, and getting that wrong is silent (the ghost band would be written and then never read as
  /// boundary data). Fail loudly instead.
  void checkVofBcFace(int f, int wantType, const char* who) {
    if (f < 0 || f > 5)
      throw std::runtime_error(std::string(who) + ": face must be 0..5 (-x,+x,-y,+y,-z,+z)");
    enableVof();
    if (bc_[f] != wantType)
      throw std::runtime_error(
          std::string(who) + ": face " + std::to_string(f) + " has domain BC type " +
          std::to_string(bc_[f]) + ", not " + std::to_string(wantType) +
          " — call set_domain_bc(face, " + std::to_string(wantType) + ", ...) first (2 = inflow, "
          "3 = outflow).");
  }
  /// Arm the rung: install the mask, rebuild the profiles, and zero the boundary volume ledger.
  void vofBcArm() {
    vofBcActive_ = true;
    vofRebuildBcBlock();
    // Refresh C's G=2 ghost band NOW. The property ghosts derive from it (vofBcPropGhosts) and the
    // first step's `project()` fills rho's ghosts BEFORE the colour advection refills C's, so
    // without this the first step would evaluate rho(C) at the inflow ghost on the pre-BC (Neumann
    // copy) colour — i.e. on the interior's phase.
    if (vofEnabled_ && cField_.extent(0))
      fillPropGhosts(cField_);
  }
  /// Move the advector's per-face boundary volume ledger into the solver's, once per advection.
  void vofHarvestBcVolumes() {
    if (!vofBcActive_)
      return;
    for (int f = 0; f < 6; ++f) {
      vofBcVol_[f] = vofAdv_.bcFaceVolume(f);
      vofBcVolTotal_[f] += vofBcVol_[f];
    }
  }

  // --- interface curvature (rung V3, WO-O) -----------------------------------------------------
  //
  // `compute_vof_curvature()` fills two registered G=2 cell fields from the CURRENT colour field:
  //
  //   "kappa"         kappa = 2H in units of 1/h (cell units). Multiply by 1/h for physical units.
  //                   POSITIVE for a convex blob of liquid: a sphere of liquid of radius R cells
  //                   reads +2/R. WO-P (balanced-force CSF) is the consumer.
  //   "kappa_branch"  which tier of the cascade produced it (`vof::CurvatureBranch`): 0 not
  //                   interfacial, 1 HF, 2 HF in a non-preferred direction, 3 mixed-HF fit (off by
  //                   default), 4/5 the PLIC-volumetric paraboloid fit, 6 NO estimate.
  //
  // Reading "kappa" without reading "kappa_branch" is a mistake: kappa is 0 both where there is no
  // interface (branch 0, correct) and where the cascade could not produce an estimate (branch 6,
  // which must never happen and is loud when it does). The branch field is the difference.
  //
  // The whole cascade is a pure local stencil on the colour field's g = 3 block — no reductions,
  // no new halo — so it is bitwise decomposition-independent by construction. See
  // `vof/curvature.hpp` for the cascade, its literature anchors and its measured branch shares.
  void computeVofCurvature() {
    if (!vofEnabled_)
      throw std::runtime_error(
          "compute_vof_curvature: VoF is not enabled (call enable_vof / set_vof first)");
    if (!kappaField_.extent(0)) {
      kappaField_ = addField("kappa");
      kappaBranch_ = addField("kappa_branch");
    }
    bridgeColourToVof();
    vofCurvStats_ = vofCurv_.compute(vofAdv_.colour());
    copyInner(kappaField_, e_, G, CCConst(vofCurv_.kappa()), e3_, kVofG);
    copyInner(kappaBranch_, e_, G, CCConst(vofCurv_.branch()), e3_, kVofG);
    // kappa is face-interpolated by the V4 surface-tension force exactly as the properties are, so
    // it gets the same rank-aware ghost policy they do (WO-G / WO-I).
    fillPropGhosts(kappaField_);
    fillPropGhosts(kappaBranch_);
  }
  // The branch census of the last `computeVofCurvature()` — LOCAL to this rank (the driver is
  // MPI-free; a distributed caller sums them).
  vof::VofCurvature::Stats vofCurvatureStats() const { return vofCurvStats_; }
  std::vector<double> getVofCurvature() {
    if (!kappaField_.extent(0))
      throw std::runtime_error("vof_curvature: call compute_vof_curvature() first");
    return gatherInner(kappaField_);
  }
  std::vector<double> getVofCurvatureBranch() {
    if (!kappaBranch_.extent(0))
      throw std::runtime_error("vof_curvature_branch: call compute_vof_curvature() first");
    return gatherInner(kappaBranch_);
  }
  // Wendland support width of the PV fallback fit, in cell units (Han et al.: 2.5 with a 5^3
  // stencil; 3.5 recovers first-order spurious-current convergence on a translating droplet and
  // 4.5 over-smooths and destroys it). Exposed for WO-P's sweep.
  void setVofCurvatureWeightWidth(double d) { vofCurv_.weightWidth = d; }
  double vofCurvatureWeightWidth() const { return vofCurv_.weightWidth; }
  // Tier 2b, the mixed height-position fit. OFF by default and it should stay off — see
  // `vof::VofCurvature::useMixedHeightFit` for the measurement that put it there.
  void setVofCurvatureMixedHeightFit(bool on) { vofCurv_.useMixedHeightFit = on; }
  bool vofCurvatureMixedHeightFit() const { return vofCurv_.useMixedHeightFit; }

  // --- balanced-force surface tension (rung V4, WO-P) -------------------------------------------
  //
  // `set_surface_tension(sigma)` turns on the continuum surface force
  //
  //     F_c(i) = sigma * kappa_f(i) * ( C(i) - C(i - s_c) ) / h                                (1)
  //
  // at every staggered velocity unknown, added to the momentum RHS at the same place, in the same
  // units and with the same cut-cell rescale as the incremental scheme's -(P(i) - P(i - s_c)).
  // `kappa_f` is the arithmetic mean of the two cells' curvatures where both carry one, the single
  // available one where only one does (`vof/surface_tension.hpp`).
  //
  // WHY (1) AND NOT AN INTERPOLATED CELL FORCE — the whole content of the rung. `C(i) - C(i - s_c)`
  // is the projection's OWN face difference. With a constant kappa the force is therefore exactly
  // the discrete gradient of `sigma*kappa*C`, i.e. it lies in the range of the operator the
  // projection inverts, so the projection removes it completely and a static drop stays at machine
  // zero. Face-interpolating a cell-centred `sigma*kappa*grad C` — the obvious way to reuse the
  // per-cell body-force machinery — produces a field that is NOT a discrete gradient of anything,
  // the projection cannot annihilate it, and what is left is the classical spurious current
  // (Francois et al. 2006; Popinet 2009). This is the momentum analogue of the three-way rho_f
  // consistency that makes the hydrostatic acid test exact.
  //
  // SIGN. `kappa` is positive for a convex blob of LIQUID (rung V3) and `C` is the liquid fraction,
  // so (1) with a plus sign gives the Young-Laplace overpressure INSIDE the drop: at equilibrium
  // `P = sigma*kappa*C + const`, the discrete solution of the projection, exactly.
  //
  // UNITS. `sigma` is in the solver's own units, in which the cell size is 1 (as are `rho`, `mu`
  // and `set_body_force`). `kappa` is in 1/h and `C(i) - C(i-s)` is `h * dC/dx`, so (1) is a force
  // per unit volume.
  //
  // REQUIREMENTS. VoF must be enabled (staggered only). The curvature cascade runs once per step,
  // at the head, from the SAME colour field the density closure sees.
  void setSurfaceTension(double sigma) {
    if (!(sigma >= 0.0))
      throw std::runtime_error("set_surface_tension: sigma must be >= 0");
    if (sigma > 0.0) {
      enableVof();
      if (!kappaField_.extent(0)) {
        kappaField_ = addField("kappa");
        kappaBranch_ = addField("kappa_branch");
      }
      // Wisp guard on the curvature's interfacial predicate. NOT optional once the curvature feeds
      // a force: Weymouth-Yue leaves round-off colour residue (measured down to -3e-35) in every
      // cell its sweeps touch, those cells satisfy `0 < C < 1`, and the cascade returns |kappa| up
      // to 1e8 for them off a zero-area PLIC polygon. A face between one of them and a real
      // interfacial cell then carries a force eight orders too large. See
      // `vof::VofCurvature::interfaceEps` for the measurement; the V3 default (0) is unchanged for
      // anyone calling `compute_vof_curvature()` without surface tension.
      vofCurv_.interfaceEps = csfInterfaceEps_;
    }
    sigmaCsf_ = sigma;
  }
  double surfaceTension() const { return sigmaCsf_; }
  // The wisp threshold above, exposed so it can be swept/ablated. Default 1e-8; 0 restores the
  // unguarded V3 predicate and, with surface tension on, reproduces the instability it exists for.
  void setVofInterfaceEps(double eps) {
    csfInterfaceEps_ = eps;
    if (sigmaCsf_ > 0.0)
      vofCurv_.interfaceEps = eps;
  }
  double vofInterfaceEps() const { return csfInterfaceEps_; }
  // ABLATION: 0 = the balanced-force face difference (default, the only production mode);
  // 1 = a cell-centred sigma*kappa*grad(C) face-interpolated like an ordinary body force. See
  // `addCsfRhsCellInterp`. Kept so the ctest can measure what the operator pairing is worth.
  void setCsfMode(int m) { csfMode_ = m; }
  int csfMode() const { return csfMode_; }
  bool csfActive() const { return vofEnabled_ && sigmaCsf_ > 0.0 && kappaField_.extent(0) != 0; }

  // INSTRUMENT (not a configuration): stop recomputing the curvature at the head of each step and
  // use whatever is in the "kappa" / "kappa_branch" fields. Together with `set_vof_kappa_constant`
  // this isolates the BALANCED-FORCE identity from the curvature estimator — the exactness gate of
  // this rung, which must hold at machine zero for a curvature that is merely constant, whether or
  // not it is the right one.
  void setVofKappaFrozen(bool on) { kappaFrozen_ = on; }
  bool vofKappaFrozen() const { return kappaFrozen_; }
  // INSTRUMENT: set kappa to a constant over the WHOLE block (inner + ghosts) and mark every cell's
  // branch as a valid estimate, then freeze it. The force (1) is then exactly the discrete gradient
  // of `sigma*kappa*C`, so the projection must annihilate it to round-off from ANY colour field.
  void setVofKappaConstant(double kappa) {
    enableVof();
    if (!kappaField_.extent(0)) {
      kappaField_ = addField("kappa");
      kappaBranch_ = addField("kappa_branch");
    }
    Kokkos::deep_copy(kappaField_, kappa);
    Kokkos::deep_copy(kappaBranch_, (double)vof::kCurvHf);
    kappaFrozen_ = true;
  }

  // The Brackbill (1992) / Denner & van Wachem (2015) capillary time-step limit
  // `sqrt((rho_1 + rho_2) h^3 / (4 pi sigma))`. +inf when surface tension is off.
  //
  // The density SUM is taken from the declared phase pair when momentum consistency is on
  // (`enable_vof_momentum` validated it against the closure), and otherwise from `min(rho) +
  // max(rho)` over the current density field — an MPI_MIN/MPI_MAX pair, so it is exact and
  // decomposition-independent. It is the sum, not a mean: both phases oscillate.
  double capillaryDt() {
    if (!(sigmaCsf_ > 0.0))
      return std::numeric_limits<double>::infinity();
    return vof::capillaryDt(phaseDensitySum(), vofEnabled_ ? vofAdv_.h() : 1.0, sigmaCsf_);
  }
  // Safety factor on the capillary limit: `step()` throws when `dt > factor * capillaryDt()`.
  // Default 1.0 — Denner & van Wachem measured the Brackbill prefactor to BE the stability
  // boundary, so there is no margin built into the formula itself. Set it huge to disable the
  // check, exactly as `set_vof_cfl_limit` is the escape hatch for the Weymouth-Yue cap.
  void setCapillaryCfl(double f) { capillaryCfl_ = f; }
  double capillaryCfl() const { return capillaryCfl_; }

  // Both explicit two-phase step limits at the CURRENT state, and which one binds. This is the
  // number WO-P asks for: at pore-scale capillary numbers the capillary dt, not the Weymouth-Yue
  // CFL, is expected to be the binding constraint, and that decides whether implicit surface
  // tension is ever worth revisiting.
  struct VofStepLimits {
    double courant = 0.0;      ///< the interface-local Courant number at the current dt
    double cflDt = 0.0;        ///< the largest dt the WY boundedness cap admits
    double capillaryDt = 0.0;  ///< the largest dt the Brackbill capillary constraint admits
    double binding = 0.0;      ///< min(cflDt, capillaryCfl * capillaryDt)
    bool capillaryBinds = false;
  };
  VofStepLimits vofStepLimits() {
    VofStepLimits L;
    L.courant = vofMaxCourant();
    L.cflDt = (L.courant > 0.0) ? dt_ * vofCflLimit_ / L.courant
                                : std::numeric_limits<double>::infinity();
    L.capillaryDt = capillaryDt();
    const double cap = capillaryCfl_ * L.capillaryDt;
    L.capillaryBinds = cap < L.cflDt;
    L.binding = L.capillaryBinds ? cap : L.cflDt;
    return L;
  }
  // rho_1 + rho_2 for the capillary limit. Public because nvcc refuses an extended
  // __host__ __device__ lambda inside a private member function (the WO-O build note).
  double phaseDensitySum() {
    if (vofMomEnabled_)
      return vofRhoG_ + vofRhoL_;
    if (!effVarRho())
      return 2.0 * rho_;
    CCExec space;
    C3 e = e_;
    double lo = 1e300, hi = -1e300;
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    // A closure-driven rho field is produced by updateProperties() at the head of the step, so
    // before the FIRST step it is still all zeros and a naive min+max would report 0 — and this is
    // a diagnostic users call while choosing dt, i.e. exactly then. Refresh and retry once in that
    // case; the step's own call site runs after updateProperties() and never takes the branch.
    for (int pass = 0; pass < 2; ++pass) {
      CCConst f = CCConst(effRhoField());
      lo = 1e300;
      hi = -1e300;
      Kokkos::parallel_reduce(
          "rho_minmax", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
          KOKKOS_LAMBDA(int x, int y, int z, double& mn, double& mx) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            mn = Kokkos::fmin(mn, f(i));
            mx = Kokkos::fmax(mx, f(i));
          },
          Kokkos::Min<double>(lo), Kokkos::Max<double>(hi));
      if (hi > 0.0 || pass == 1)
        break;
      updateProperties();
    }
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      double g[2] = {lo, -hi}, r[2];
      MPI_Allreduce(g, r, 2, MPI_DOUBLE, MPI_MIN, comm_);
      lo = r[0];
      hi = -r[1];
    }
#endif
    return lo + hi;
  }
  // Head-of-step curvature refresh + the capillary dt check. No-op unless surface tension is on.
  //
  // The curvature is taken from the colour field the step is ABOUT to run with — the same field
  // `updateProperties()` just turned into rho(C) and mu(C), i.e. C^{n+1} under momentum consistency
  // (the VoF stage ran at the head) and C^n without it (it runs at the tail). Either way kappa,
  // rho and the colour in the force (1) are the same time level, which is what the balance needs.
  void updateVofCurvature() {
    if (!csfActive())
      return;
    if (!kappaFrozen_)
      computeVofCurvature();
    const double cap = capillaryCfl_ * capillaryDt();
    if (!(dt_ <= cap))
      throw std::runtime_error(
          "surface tension: dt = " + std::to_string(dt_) + " exceeds the capillary limit " +
          std::to_string(cap) + " (Brackbill sqrt((rho_1+rho_2) h^3/(4 pi sigma)) = " +
          std::to_string(capillaryDt()) + " x safety factor " + std::to_string(capillaryCfl_) +
          "). Surface tension is EXPLICIT: this is a hard stability boundary (Denner & van Wachem "
          "2015), not a margin. Reduce dt, or raise set_capillary_cfl deliberately.");
  }

  // Advance the colour field one dt with the just-projected face velocities. Called by step()
  // immediately before advanceScalars(); exposed so a test can drive it in isolation.
  //
  // WHY HERE (the ordering, WO-J item 2). Weymouth-Yue conserves volume exactly only against a
  // DISCRETELY divergence-free face field: the dilation term adds H(C-1/2)*div*dt/h to every
  // cell's budget, interior full cells included, so the conservation floor is the advecting field's
  // own discrete divergence residual (WO-E finding 2 — with an analytically-but-not-discretely
  // solenoidal field the floor sits at O(h^2), ten orders above the gate). The only field in the
  // step with that property is the OUTPUT of the projection, which is exactly where
  // `advanceScalars` already sits and for exactly the same reason. u* is not divergence-free; u^n
  // is, but u^n at the top of step n+1 IS u^{n+1} at the bottom of step n, so the two placements
  // are the same point in the timeline and differ only in which side of `updateProperties()` they
  // fall on. Taking the `advanceScalars` slot therefore means the properties of step n are
  // rho(C^n), mu(C^n) — the same time level as the velocity base u^n, and the same segregated
  // contract every other multiphysics field in this solver obeys.
  //
  // HOW WO-K (rung V2b, momentum-consistent transport) DROPS IN. WO-K needs the momentum advection
  // to use the same geometric fluxes, the same sweep order and the SAME frozen dilation flag as
  // this colour advection, on half-shifted control volumes built by clipping the SAME PLIC planes
  // (`plicSlabVolume` is scale/offset invariant, so that is a rescale). Three structural facts make
  // that a local change here:
  //   * everything WO-K must share lives inside `vofAdv_` and survives the call — the frozen flag
  //     `cc_`, the per-sweep planes `mx_/my_/mz_/alpha_`, the face Courant numbers and the
  //     permutation index. WO-K adds a sibling advector for rho^c u_c driven from those SAME
  //     members rather than recomputing anything, so "sharing the fluxes" is a data-flow fact, not
  //     a convention two call sites have to keep.
  //   * the colour advection is ONE call (`advectVof()`), not a set of calls scattered through the
  //     step, so moving it to the head of the predictor (where the momentum advection lives) is a
  //     one-line move, and it moves to a velocity field — u^n — that is the same field it consumes
  //     today, one step earlier in wall-clock and identical in content.
  //   * the half-shifted colour field C^c is a new g=3 field on the SAME advector block, so it
  //     needs no new bridge and no new halo: `vofFillGhosts` already carries any field on that
  //     block, and the velocity is already embedded there (`bridgeVelocityToVof`).
  // What WO-K must NOT do, and what this structure keeps honest: interpolate C to build C^c. The
  // planes are here; clip them.
  void advectVof() {
    if (!vofEnabled_)
      return;
    requireVofGeometry("enable_vof");
    bridgeVelocityToVof();
    bridgeColourToVof();
    vofAdv_.resetBcFaceVolume();  // WO-R: the per-step boundary liquid ledger
    vofAdv_.advect(dt_, vofStep_++);
    vofHarvestBcVolumes();
    // Back to the G=2 registry mirror, then ITS ghost policy: the closures write inner cells only,
    // but the property face means (rho_f in the momentum diagonal, the projection coefficient, the
    // face body force) read the ghost ring, so C's ghosts must be filled with the SAME policy rho
    // uses or the derived rho ghost is inconsistent with the interior at a boundary.
    copyInner(cField_, e_, G, CCConst(vofAdv_.colour()), e3_, kVofG);
    zeroSolidColour();  // rung V5a: the canonical field carries 0 in solid cells, not the fill
    fillPropGhosts(cField_);
  }

  // --- momentum-consistent transport (rung V2b, WO-K) ------------------------------------------
  //
  // Turn on the transport of `rho^c u_c` on the half-shifted MAC control volumes by the SAME
  // geometric fluxes, the same sweep order and one frozen dilation flag as the colour advection of
  // the same step (`vof/momentum_advect.hpp` carries the construction and the consistency identity).
  //
  // The two phase densities are required EXPLICITLY rather than read off the closure: the momentum
  // flux is `rho_g (a - F) + rho_l F` with `F` the geometric LIQUID flux, so the scheme needs to
  // know which density each phase carries, and inferring that from a `rho` field would be a silent
  // dependence on the closure kind. `enableVofMomentum` validates the pair against the registered
  // density field on the spot (see `checkVofPhaseDensities`), so a mismatched call fails loudly.
  //
  // WHY THE ADVECTION MOVES TO THE HEAD OF THE STEP. WO-J placed `advectVof()` in the
  // `advanceScalars()` slot because Weymouth-Yue needs a discretely divergence-free advecting field
  // and the projection's output is the only one in the step. With momentum consistency the SAME
  // fluxes must also carry the momentum, and the momentum advection has to happen before the
  // predictor builds its RHS — so the whole VoF stage moves to the top of `step()`, where the
  // advecting field is `u^n`, i.e. the PREVIOUS step's projected output. That is the same field, one
  // step earlier in wall-clock: `u^{n+1}` at the bottom of step n IS `u^n` at the top of step n+1
  // (WO-J's own note). The one real consequence is at step 0, where `u^0` is whatever the user set
  // and is only divergence-free if the user made it so; every gate in this rung starts from rest or
  // from a uniform field, both exactly divergence-free.
  //
  // With this placement `updateProperties()` then sees `C^{n+1}`, so the step runs with
  // `rho(C^{n+1})`, `mu(C^{n+1})` — the density of the time level the momentum is being advanced TO,
  // which is what the conservative form `rho^{n+1} u^{n+1} = rho^n u^n - div(rho u u) dt + ...`
  // wants.
  //
  // WHAT IS *NOT* CHANGED, DELIBERATELY: the momentum time-term face density, the face body force
  // and the projection coefficient all keep the arithmetic face mean `1/2 (rho(i) + rho(i-s_c))`.
  // That three-way agreement is what makes discrete hydrostatic balance exact
  // (`doc/variable_density_projection.md` §1) and it is a validated kernel. Momentum consistency
  // enters as the ADVECTIVE base velocity `u^adv` replacing `u^n - dt*adv(u^n)`: the momentum
  // equation solved is `rho_f (u* - u^adv)/dt = -grad p + visc + f`, i.e. exactly the conservative
  // update divided through by the face density. Swapping the time term to the clipped `rho^c`
  // instead would break the hydrostatic acid test at O(d rho) — measured and recorded in the WO-K
  // findings.
  void enableVofMomentum(double rhoGas, double rhoLiquid) {
    if constexpr (Grid::collocated)
      throw std::runtime_error(
          "enable_vof_momentum: momentum-consistent VoF transport is STAGGERED-ONLY (rung V2b); the "
          "collocated construction is Favre-averaged face states, rung V8.");
    enableVof();
    if (porous_)
      throw std::runtime_error(
          "enable_vof_momentum: the volume-averaged porous momentum and the momentum-consistent VoF "
          "transport both own the momentum time term; they are not composable at this rung.");
    if (!(rhoGas > 0.0) || !(rhoLiquid > 0.0))
      throw std::runtime_error("enable_vof_momentum: both phase densities must be > 0");
    vofRhoG_ = rhoGas;
    vofRhoL_ = rhoLiquid;
    vofMomEnabled_ = true;
    for (int c = 0; c < 3; ++c)
      if (uAdv_[c].extent(0) != n_)
        uAdv_[c] = CCField("uAdv", n_);
    vofMom_.init(vofAdv_, vofRhoG_, vofRhoL_);
  }
  bool vofMomentumEnabled() const { return vofMomEnabled_; }
  // Floor on rho^c in the recovery divide u = (rho^c u)/rho^c, as a FRACTION of min(rho_g, rho_l)
  // (default 1e-6). rho^c leaves [rho_g, rho_l] only through a wisp in the half-shifted colour, and
  // driving it to zero would need C^c ~ -1/(ratio-1); the floor is a guard, not a model, and
  // `vof_momentum_diagnostics()` reports how many control volumes it actually touched.
  void setVofRhoFloorFrac(double f) { vofMom_.rhoFloorFrac = f; }
  double vofRhoFloorFrac() const { return vofMom_.rhoFloorFrac; }
  double vofRhoFloor() const { return vofMom_.lastRhoFloor(); }
  // MinMod-limited donor reconstruction in the momentum flux. OFF by default — on a control volume
  // a sweep empties, the slope's deviation from the volume's own velocity is amplified by
  // drho*F/rho^c, which is unbounded in the density ratio; measured, it grew the uniform-velocity
  // residual to 2.2e-10 at ratio 1e4 over 50 steps while plain donor-cell upwind stayed flat at
  // 6.7e-16. Harmless at ratio 1e3. See vof/momentum_advect.hpp.
  void setVofMomentumMuscl(bool on) { vofMom_.momentumMuscl = on; }
  // Ablation: the literal reading of "the same frozen dilation flag" (the PRESSURE-cell flag on the
  // shifted control volume instead of its structural analogue).
  void setVofMomentumCellFlag(bool on) { vofMom_.useCellDilationFlag = on; }
  // Ablation: drop the Weymouth flux clamp on the shifted control volume. With it off the
  // half-shifted colour leaves [0,1] by O(a^2) and rho^c goes NEGATIVE at high ratio — the
  // measurement that the clamp is a necessity, not a habit. See vof/momentum_advect.hpp point 3.
  void setVofFluxClamp(bool on) { vofMom_.clampFluxes = on; }
  vof::MomentumConsistentAdvector::Diagnostics vofMomentumDiagnostics() {
    if (!vofMomEnabled_)
      throw std::runtime_error("vof_momentum_diagnostics: enable_vof_momentum was never called");
    return vofMom_.diagnostics();
  }
  // The recovered advected velocity of component c on the inner cells (the momentum RHS's time
  // base). Exposed so a test can gate the uniform-velocity identity on the advection ALONE, with
  // the projection and the momentum solve out of the picture.
  std::vector<double> getVofAdvectedVelocity(int c) {
    if (!vofMomEnabled_)
      throw std::runtime_error("vof_advected_velocity: enable_vof_momentum was never called");
    return gatherInner(uAdv_[c]);
  }

  // The coupled colour + momentum advection. Called from the head of step() when momentum
  // consistency is on; exposed so a test can drive it in isolation.
  void advectVofMomentum() {
    if (!vofMomEnabled_)
      return;
    requireVofGeometry("enable_vof_momentum");
    if (implicitAdv())
      throw std::runtime_error(
          "enable_vof_momentum is incompatible with implicit advection (set_implicit_advection / a "
          "domain-BC stencil path): the momentum advection is already done conservatively by the "
          "VoF fluxes, and the implicit-FOU operator would add a second one.");
    if (!effVarRho())
      throw std::runtime_error(
          "enable_vof_momentum requires the variable-density momentum/projection path: register a "
          "density closure on C (set_property_model('rho','linear','C',[rho_g, rho_l-rho_g])) or "
          "call set_density_mode('variable').");
    if (vofMom_.phaseRhoG() != vofRhoG_ || vofMom_.phaseRhoL() != vofRhoL_)
      vofMom_.setPhaseDensities(vofRhoG_, vofRhoL_);
    bridgeVelocityToVof();
    bridgeColourToVof();
    vofAdv_.resetBcFaceVolume();  // WO-R: the per-step boundary liquid ledger
    vofMom_.advect(vofAdv_, dt_, vofStep_++);
    vofHarvestBcVolumes();
    // Colour back to the G=2 registry mirror (same contract as advectVof), and the advected
    // velocity back onto the solver's velocity index convention.
    copyInner(cField_, e_, G, CCConst(vofAdv_.colour()), e3_, kVofG);
    zeroSolidColour();
    fillPropGhosts(cField_);
    // A plain inner-to-inner copy: the momentum control volumes are indexed in the solver's own
    // low-face convention (vof/momentum_advect.hpp "Indexing"), so CV_c(i) IS the solver's u_c(i)
    // and there is no shift here to get wrong.
    for (int c = 0; c < 3; ++c)
      copyInner(uAdv_[c], e_, G, CCConst(vofMom_.advectedVelocity(c)), e3_, kVofG);
  }


  // --- Phase change (Part II, rungs P0/P1 — WO-P01) --------------------------------------------
  //
  // The kernel set of `suite/docs/VOF_PLAN.md` §9 in its planar form, following Boyd & Ling (2023)
  // and Malan et al. (2021): a mass flux `mdot` on interfacial cells (prescribed at P0, from
  // one-sided pure-cell temperature gradients at P1), interface regression by a PLIC PLANE SHIFT
  // with exact clip-and-redistribute, and the volumetric divergence source shifted into the
  // compact pure-gas layer behind the interface so the interfacial cell's own face velocity stays
  // the LIQUID velocity and Weymouth-Yue advects the colour with a field it is entitled to.
  // Container-free geometry/physics lives in `vof/phase_change.hpp`; this is the block walk.
  //
  // ORDER WITHIN step() (and why). `phaseChangeStep()` runs at the HEAD of the step, before
  // `updateProperties()`:
  //   1. `mdot`, the PLIC area `A_G` and the unit normal `n` are built from (C^n, T^n);
  //   2. the divergence source is deposited into pure gas cells (read by `project()`);
  //   3. the regression is applied to the SAME C^n the planes were reconstructed from.
  // Doing the regression here rather than after the advection is what keeps the plane, the area
  // and the colour it is subtracted from at ONE time level; `updateProperties()` then sees the
  // post-regression colour, so rho(C) and the momentum/projection coefficients are consistent with
  // the interface the step actually runs with. The colour advection keeps its WO-J slot at the
  // bottom of the step (it needs the projected, discretely divergence-free face field).
  //
  // SCOPE at this rung, all enforced with a message: staggered grid, no immersed solid (the
  // solid-clipped flux polygons and the cut-cell source deposit are a later rung), and not
  // composable with `enable_vof_momentum` (both own the head of the step and the momentum flux
  // would have to carry the phase-change mass transfer as well).
  struct PhaseChangeDiagnostics {
    double mdotMin = 0.0, mdotMax = 0.0, mdotMean = 0.0;
    long interfaceCells = 0;
    double removedVolume = 0.0;  ///< sum of dV actually subtracted this step (+ = evaporated)
    double redistributed = 0.0;  ///< |clip deficit| pushed into neighbours this step
    long deficitCells = 0;       ///< cells that clipped at C = 0
    long excessCells = 0;        ///< cells that clipped at C = 1
    double sourceSum = 0.0;      ///< sum of the deposited divergence source over inner cells (1/s)
    long sourceCells = 0;        ///< cells that RECEIVED a deposit
    long fallbackCells = 0;      ///< interfacial cells whose +n walk found no pure gas cell
    double unresolved = 0.0;     ///< clip residue no neighbour could absorb (pushed anyway)
    double minC = 0.0, maxC = 0.0;
    double area = 0.0;  ///< sum of the PLIC polygon areas over interfacial cells (h^2)
  };

  /// Turn on phase change. `rhoG`/`rhoL` are the phase densities used by the regression
  /// (`dV = mdot A dt / rho_l`) and by the divergence source (`S = mdot A (1/rho_g - 1/rho_l)`);
  /// they are given EXPLICITLY rather than read off a closure, exactly as `enable_vof_momentum`
  /// does and for the same reason. `hlv` is the latent heat (J/kg) and is only used by the thermal
  /// mass flux. Registers "mdot" (kg m^-2 s^-1, solver units) and "pc_source" (1/s).
  void enablePhaseChange(double rhoG, double rhoL, double hlv) {
    if constexpr (Grid::collocated)
      throw std::runtime_error(
          "enable_phase_change: rungs P0/P1 are STAGGERED-ONLY (the collocated grid carries every "
          "force as a face acceleration and the source deposit has not been composed with it).");
    enableVof();
    if (hasSolid_)
      throw std::runtime_error(
          "enable_phase_change: an immersed solid is out of scope at rungs P0/P1 (the source "
          "deposit and the regression would need the solid-clipped flux polygons of rung V5a's "
          "follow-on). Use an all-fluid set_pressure_geometry.");
    if (vofMomEnabled_)
      throw std::runtime_error(
          "enable_phase_change is not composable with enable_vof_momentum at this rung: both own "
          "the head of the step, and momentum consistency would have to carry the interfacial mass "
          "transfer in its own fluxes.");
    if (!(rhoG > 0.0) || !(rhoL > 0.0))
      throw std::runtime_error("enable_phase_change: both phase densities must be > 0");
    if (!(hlv > 0.0))
      throw std::runtime_error("enable_phase_change: the latent heat h_lv must be > 0");
    pcRhoG_ = rhoG;
    pcRhoL_ = rhoL;
    pcHlv_ = hlv;
    pcMdot_ = addField("mdot");
    pcSrc_ = addField("pc_source");
    if (pcArea_.extent(0) != n_) {
      pcArea_ = CCField("pc_area", n_);
      pcNrm_[0] = CCField("pc_nx", n_);
      pcNrm_[1] = CCField("pc_ny", n_);
      pcNrm_[2] = CCField("pc_nz", n_);
      pcDep_ = CCField("pc_dep", n_);
      pcTgt_ = CCField("pc_tgt", n_);
      pcCnew_ = CCField("pc_cnew", n_);
      pcDefic_ = CCField("pc_defic", n_);
    }
    pcEnabled_ = true;
  }
  bool phaseChangeEnabled() const { return pcEnabled_; }

  /// Prescribe a UNIFORM mass flux (P0). Overwrites "mdot" on the inner cells and its ghosts.
  void setMassFluxUniform(double v) {
    requirePhaseChange("set_mass_flux_uniform");
    Kokkos::deep_copy(pcMdot_, v);
    pcThermal_ = false;
  }
  /// Prescribe a per-cell mass flux (P0), x-fastest over the inner region.
  void setMassFlux(const std::vector<double>& v) {
    requirePhaseChange("set_mass_flux");
    scatterInner(pcMdot_, v);
    fillPropGhosts(pcMdot_);
    pcThermal_ = false;
  }
  /// P1: compute `mdot` each step from the registered scalar `tname` by the one-sided pure-cell
  /// weighted least-squares gradients of `vof/phase_change.hpp`. `Tsat` is the saturation
  /// temperature, `kg`/`kl` the phase conductivities (W/(cell K)) and `Rint` the interfacial
  /// heat-transfer resistance of the Schrage/IHTR Robin condition `T_G = T_sat + mdot R_int`
  /// (Bureš & Sato 2021); `Rint = 0` is the hard Dirichlet and is the default.
  void setPhaseChangeThermal(const std::string& tname, double Tsat, double kg, double kl,
                             double Rint) {
    requirePhaseChange("set_phase_change_thermal");
    if (!hasScalar(tname))
      throw std::runtime_error("set_phase_change_thermal: no scalar named '" + tname +
                               "' (call add_scalar first)");
    pcTName_ = tname;
    pcTsat_ = Tsat;
    pcKg_ = kg;
    pcKl_ = kl;
    pcRint_ = Rint;
    pcThermal_ = true;
    scalarDirichletMask(tname);  // allocate the per-cell Dirichlet mask + value fields
    pcUpdateThermalMask();
  }
  void setPhaseChangeThermalOff() { pcThermal_ = false; }

  /// A PRESCRIBED extra divergence source (1/s), x-fastest over the inner region, added to the
  /// Poisson RHS exactly like the phase-change deposit: the projection then solves for
  /// `div(open u) = S_pc + S_user`. This is how a CLOSED (periodic) box is made compatible with a
  /// net vapour production: put a balancing sink somewhere the exact solution can absorb it. In a
  /// domain with an outflow face the outflow carries the imbalance and this is not needed.
  void setDivergenceSource(const std::vector<double>& v) {
    if (pcUser_.extent(0) != n_)
      pcUser_ = addField("div_source");
    scatterInner(pcUser_, v);
    fillPropGhosts(pcUser_);
    pcHasUser_ = true;
  }
  void clearDivergenceSource() { pcHasUser_ = false; }

  /// Kinematic entry point (the P0a/P1 driver): build `mdot`/`A_G`/`n` from the current colour and
  /// temperature, deposit the divergence source (for the census only — nothing is projected here)
  /// and apply the interface regression. No Navier-Stokes step, no advection.
  void applyPhaseChange(double dt) {
    requirePhaseChange("apply_phase_change");
    pcBuildInterface();
    pcScatterSource();
    pcRegress(dt);
    pcUpdateThermalMask();
  }

  /// The in-step driver: everything `applyPhaseChange` does, at the head of `step()`.
  /// Byte-identical no-op when phase change is off.
  void phaseChangeStep() {
    if (!pcEnabled_)
      return;
    pcBuildInterface();
    pcScatterSource();
    pcRegress(dt_);
  }

  PhaseChangeDiagnostics phaseChangeDiagnostics() {
    requirePhaseChange("phase_change_diagnostics");
    PhaseChangeDiagnostics d = pcDiag_;
    // colour extrema of the CURRENT field (the regression's own boundedness read-out)
    CCConst c = CCConst(cField_);
    const C3 e = e_;
    double mn = 1e300, mx = -1e300;
    Kokkos::parallel_reduce(
        "peclet::flow::pc_extrema",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& lo, double& hi) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          lo = Kokkos::fmin(lo, c(i));
          hi = Kokkos::fmax(hi, c(i));
        },
        Kokkos::Min<double>(mn), Kokkos::Max<double>(mx));
    Kokkos::fence();
    d.minC = mn;
    d.maxC = mx;
    return d;
  }

  // nvcc requires members holding extended device lambdas to be public.

  /// (1) The interface build: for every inner interfacial cell reconstruct the PLIC plane from the
  /// canonical G=2 colour, store its area and unit normal, evaluate `mdot` (thermal or prescribed),
  /// and decide which pure gas cell will receive the divergence source. Then exchange those
  /// per-cell quantities so the regression's depth-1 ring and the source gather's depth-2 ring see
  /// the OWNER's values — which is what makes both decomposition-independent WITHOUT any
  /// reverse/add halo and without an atomic scatter (bitwise MPI, not a reduction floor).
  void pcBuildInterface() {
    const C3 e = e_;
    const long sy = e_.x, sz = (long)e_.x * e_.y;
    CCField mdot = pcMdot_, area = pcArea_, dep = pcDep_, tgt = pcTgt_;
    CCField nx = pcNrm_[0], ny = pcNrm_[1], nz = pcNrm_[2];
    CCConst c = CCConst(cField_);
    const bool thermal = pcThermal_;
    CCConst T = thermal ? CCConst(scalarField(pcTName_).c) : CCConst(cField_);
    const double eps = pcInterfaceEps_, pureEps = pcPureEps_;
    const double Tsat = pcTsat_, kg = pcKg_, kl = pcKl_, hlv = pcHlv_, Rint = pcRint_;
    const double rhoG = pcRhoG_, rhoL = pcRhoL_;
    long nIface = 0, nFallback = 0;
    double sumArea = 0.0, sumMdot = 0.0;
    Kokkos::parallel_reduce(
        "peclet::flow::pc_build",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, long& nif, long& nfb, double& aacc, double& macc) {
          const long i = (long)x + (long)y * e.x + (long)z * sz;
          area(i) = 0.0;
          dep(i) = 0.0;
          tgt(i) = 0.0;
          nx(i) = 0.0;
          ny(i) = 0.0;
          nz(i) = 0.0;
          if (!vof::pcIsInterfacial(c(i), eps)) {
            if (thermal)
              mdot(i) = 0.0;
            return;
          }
          double st[27];
          for (int kk = -1; kk <= 1; ++kk)
            for (int jj = -1; jj <= 1; ++jj)
              for (int ii = -1; ii <= 1; ++ii)
                st[vof::plicSt(ii + 1, jj + 1, kk + 1)] = c(i + ii + jj * sy + kk * sz);
          double m[3];
          vof::mycNormal(st, m);
          const double al = vof::plicAlpha(m[0], m[1], m[2], c(i));
          const double A = vof::plicArea(m[0], m[1], m[2], al);
          double n[3] = {1.0, 0.0, 0.0};
          if (!(vof::pcUnitNormal(m[0], m[1], m[2], n) > 0.0))
            return;
          const double phic = vof::pcCentreDistance(m[0], m[1], m[2], al);
          double md = mdot(i);
          if (thermal) {
            const double Tg = vof::pcInterfaceTemperature(Tsat, md, Rint);
            vof::PcGradFit fg, fl;
            for (int dz = -2; dz <= 2; ++dz)
              for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                  if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                  const double w = vof::pcGradWeight(dx, dy, dz, n);
                  if (!(w > 0.0))
                    continue;
                  const long j = i + dx + dy * sy + dz * sz;
                  const double cj = c(j);
                  const double phi = vof::pcOffsetDistance(phic, n, dx, dy, dz);
                  if (cj <= pureEps && phi > 0.0)
                    vof::pcGradAdd(fg, w, phi, T(j), Tg);
                  else if (cj >= 1.0 - pureEps && phi < 0.0)
                    vof::pcGradAdd(fl, w, phi, T(j), Tg);
                }
            md = vof::pcMassFlux(kg, vof::pcGradSolve(fg), kl, vof::pcGradSolve(fl), hlv);
            mdot(i) = md;
          }
          area(i) = A;
          nx(i) = n[0];
          ny(i) = n[1];
          nz(i) = n[2];
          ++nif;
          aacc += A;
          macc += md;
          // the divergence source and the pure-gas cell that will carry it
          const double S = vof::pcDivSource(md, A, rhoG, rhoL);
          if (S != 0.0) {
            int tx = 0, ty = 0, tz = 0;
            bool found = false;
            for (int k = 1; k <= 2 && !found; ++k) {
              const int ox = (int)Kokkos::round(k * n[0]);
              const int oy = (int)Kokkos::round(k * n[1]);
              const int oz = (int)Kokkos::round(k * n[2]);
              if (ox == 0 && oy == 0 && oz == 0)
                continue;
              if (c(i + ox + oy * sy + oz * sz) <= pureEps) {
                tx = ox;
                ty = oy;
                tz = oz;
                found = true;
              }
            }
            if (!found)
              ++nfb;  // no pure gas cell within two cells: the source stays in this cell
            dep(i) = S;
            tgt(i) = (double)((tx + 2) + 5 * (ty + 2) + 25 * (tz + 2));
          }
        },
        nIface, nFallback, sumArea, sumMdot);
    Kokkos::fence();
    pcDiag_.interfaceCells = nIface;
    pcDiag_.fallbackCells = nFallback;
    pcDiag_.area = sumArea;
    pcDiag_.mdotMean = nIface > 0 ? sumMdot / (double)nIface : 0.0;
    // extrema of mdot over interfacial cells
    double mn = 0.0, mx = 0.0;
    if (nIface > 0) {
      mn = 1e300;
      mx = -1e300;
      CCConst md = CCConst(pcMdot_), ar = CCConst(pcArea_);
      Kokkos::parallel_reduce(
          "peclet::flow::pc_mdot_extrema",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                         {e.x - G, e.y - G, e.z - G}),
          KOKKOS_LAMBDA(int x, int y, int z, double& lo, double& hi) {
            const long i = (long)x + (long)y * e.x + (long)z * sz;
            if (ar(i) > 0.0) {
              lo = Kokkos::fmin(lo, md(i));
              hi = Kokkos::fmax(hi, md(i));
            }
          },
          Kokkos::Min<double>(mn), Kokkos::Max<double>(mx));
      Kokkos::fence();
    }
    pcDiag_.mdotMin = mn;
    pcDiag_.mdotMax = mx;
    // The exchange that makes the two consumers decomposition-independent. `fillGhosts` is the
    // halo/periodic base; on a NON-periodic domain face the periodic wrap would import the far
    // side's interface as a phantom source/deficit donor, so those ghosts are zeroed.
    fillGhosts(pcMdot_);
    fillGhosts(pcArea_);
    fillGhosts(pcDep_);
    fillGhosts(pcTgt_);
    for (int d = 0; d < 3; ++d)
      fillGhosts(pcNrm_[d]);
    pcZeroDomainGhosts(pcMdot_);
    pcZeroDomainGhosts(pcArea_);
    pcZeroDomainGhosts(pcDep_);
    pcZeroDomainGhosts(pcTgt_);
    for (int d = 0; d < 3; ++d)
      pcZeroDomainGhosts(pcNrm_[d]);
  }

  /// (2) Deposit each interfacial cell's source into its chosen pure-gas cell, as a GATHER (each
  /// receiving cell scans the 5^3 box for donors that named it). A gather rather than an atomic
  /// scatter because the sum then has a fixed order and the result is bitwise reproducible across
  /// decompositions; the donors' `dep`/`tgt` are valid two cells deep thanks to `pcBuildInterface`'s
  /// exchange.
  void pcScatterSource() {
    const C3 e = e_;
    const long sy = e_.x, sz = (long)e_.x * e_.y;
    CCField src = pcSrc_;
    CCConst dep = CCConst(pcDep_), tgt = CCConst(pcTgt_);
    double sum = 0.0;
    long ncell = 0;
    Kokkos::parallel_reduce(
        "peclet::flow::pc_source_gather",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc, long& nc) {
          const long i = (long)x + (long)y * e.x + (long)z * sz;
          double s = 0.0;
          for (int dz = -2; dz <= 2; ++dz)
            for (int dy = -2; dy <= 2; ++dy)
              for (int dx = -2; dx <= 2; ++dx) {
                const long j = i + dx + dy * sy + dz * sz;
                const double dj = dep(j);
                if (dj == 0.0)
                  continue;
                const int code = (int)tgt(j);
                const int tx = code % 5 - 2, ty = (code / 5) % 5 - 2, tz = code / 25 - 2;
                if (tx + dx == 0 && ty + dy == 0 && tz + dz == 0)
                  s += dj;
              }
          src(i) = s;
          acc += s;
          if (s != 0.0)
            ++nc;
        },
        sum, ncell);
    Kokkos::fence();
    pcDiag_.sourceSum = sum;
    pcDiag_.sourceCells = ncell;
    fillPropGhosts(pcSrc_);
  }

  /// (3) The regression: two Jacobi passes over the exchanged per-cell data, so the clip deficit is
  /// redistributed with a FIXED summation order (bitwise across decompositions).
  ///   pass 1 (inner region grown by one, reading only exchanged fields): the raw plane shift
  ///          `C - mdot A dt/rho_l`, clipped into [0,1], with the residue stored;
  ///   pass 2 (inner region): add the clipped colour to the shares of the six face neighbours'
  ///          residues, pushed along `-n` (a liquid deficit) or `+n` (a condensation excess) with
  ///          weights `n_d^2`.
  void pcRegress(double dt) {
    const C3 e = e_;
    const long sy = e_.x, sz = (long)e_.x * e_.y;
    CCField Cf = cField_, cnew = pcCnew_, defic = pcDefic_;
    CCConst md = CCConst(pcMdot_), ar = CCConst(pcArea_);
    const double rhoL = pcRhoL_;
    double removed = 0.0;
    long ndef = 0, nexc = 0;
    double redist = 0.0;
    Kokkos::parallel_reduce(
        "peclet::flow::pc_regress_raw",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {0, 0, 0}, {e.x, e.y, e.z}),
        KOKKOS_LAMBDA(int x, int y, int z, double& rem, long& nd, long& ne, double& rd) {
          const long i = (long)x + (long)y * e.x + (long)z * sz;
          const double A = ar(i);
          if (!(A > 0.0)) {
            cnew(i) = Cf(i);
            defic(i) = 0.0;
            return;
          }
          const double dV = vof::pcRegressVolume(md(i), A, dt, rhoL);
          const double raw = Cf(i) - dV;
          const double cl = Kokkos::fmin(Kokkos::fmax(raw, 0.0), 1.0);
          cnew(i) = cl;
          defic(i) = raw - cl;
          const bool inner = (x >= G && x < e.x - G && y >= G && y < e.y - G && z >= G &&
                              z < e.z - G);
          if (inner) {
            rem += dV;
            if (raw < 0.0)
              ++nd;
            if (raw > 1.0)
              ++ne;
            rd += Kokkos::fabs(raw - cl);
          }
        },
        removed, ndef, nexc, redist);
    Kokkos::fence();
    CCConst cn = CCConst(pcCnew_), df = CCConst(pcDefic_);
    CCConst nxv = CCConst(pcNrm_[0]), nyv = CCConst(pcNrm_[1]), nzv = CCConst(pcNrm_[2]);
    Kokkos::parallel_for(
        "peclet::flow::pc_regress_apply",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * sz;
          double v = cn(i);
          const long st[3] = {1, sy, sz};
          for (int d = 0; d < 3; ++d)
            for (int s = -1; s <= 1; s += 2) {
              const long j = i + (long)s * st[d];  // the neighbour that might push into i
              const double dj = df(j);
              if (dj == 0.0)
                continue;
              // Recompute j's WHOLE allocation here (not just i's share): every receiver runs the
              // identical arithmetic on the identical inputs, so the sum each cell forms has a
              // fixed order and is bitwise independent of the decomposition.
              const double sgn = dj < 0.0 ? -1.0 : 1.0;
              double n[3] = {nxv(j), nyv(j), nzv(j)};
              int step[3];
              double w[3];
              bool avail[3];
              for (int q = 0; q < 3; ++q) {
                const double p = sgn * n[q];
                const int sq = (p > 0.0) ? 1 : ((p < 0.0) ? -1 : 0);
                const double ct = sq == 0 ? 0.0 : cn(j + (long)sq * st[q]);
                avail[q] = sq != 0 && (dj < 0.0 ? (ct > 0.0) : (ct < 1.0));
              }
              vof::pcPushWeights(n, sgn, avail, step, w);
              // j pushes into j + step[d]*e_d; that is i iff step[d] == -s
              if (step[d] == -s)
                v += dj * w[d];
            }
          Cf(i) = v;
        });
    Kokkos::fence();
    pcDiag_.removedVolume = removed;
    pcDiag_.deficitCells = ndef;
    pcDiag_.excessCells = nexc;
    pcDiag_.redistributed = redist;
    // How much residue found no neighbour able to absorb it (pushed anyway, on the unrestricted
    // weights, so conservation holds and the colour goes slightly out of [0,1] instead).
    double unres = 0.0;
    Kokkos::parallel_reduce(
        "peclet::flow::pc_unresolved",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z, double& acc) {
          const long i = (long)x + (long)y * e.x + (long)z * sz;
          const double dj = df(i);
          if (dj == 0.0)
            return;
          const double sgn = dj < 0.0 ? -1.0 : 1.0;
          const long st[3] = {1, sy, sz};
          double n[3] = {nxv(i), nyv(i), nzv(i)};
          bool avail[3];
          for (int q = 0; q < 3; ++q) {
            const double p = sgn * n[q];
            const int sq = (p > 0.0) ? 1 : ((p < 0.0) ? -1 : 0);
            const double ct = sq == 0 ? 0.0 : cn(i + (long)sq * st[q]);
            avail[q] = sq != 0 && (dj < 0.0 ? (ct > 0.0) : (ct < 1.0));
          }
          int step[3];
          double w[3];
          if (!vof::pcPushWeights(n, sgn, avail, step, w))
            acc += Kokkos::fabs(dj);
        },
        unres);
    Kokkos::fence();
    pcDiag_.unresolved = unres;
    fillPropGhosts(cField_);
  }

  /// The per-cell Dirichlet mask of the energy scalar: `T = T_sat + mdot R_int` in every
  /// interfacial cell, released everywhere else. Rebuilt from the CURRENT colour, so a call after
  /// the colour advection is what the energy solve at the bottom of the step sees.
  void pcUpdateThermalMask() {
    if (!pcThermal_)
      return;
    ScalarField& sc = scalarField(pcTName_);
    const C3 e = e_;
    CCField mk = sc.dmask, dv = sc.dval;
    CCConst c = CCConst(cField_), md = CCConst(pcMdot_);
    const double eps = pcInterfaceEps_, Tsat = pcTsat_, Rint = pcRint_;
    Kokkos::parallel_for(
        "peclet::flow::pc_thermal_mask",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const bool on = vof::pcIsInterfacial(c(i), eps);
          mk(i) = on ? 1.0 : 0.0;
          dv(i) = vof::pcInterfaceTemperature(Tsat, md(i), Rint);
        });
    Kokkos::fence();
  }

  /// Zero the two ghost layers on every NON-periodic domain face this rank owns. Used for the
  /// per-cell phase-change data, whose consumers treat a nonzero ghost as a real donor.
  void pcZeroDomainGhosts(CCField f) {
    for (int face = 0; face < 6; ++face) {
      if (bc_[face] == 0 || !touchesGlobalFace(face))
        continue;
      const int a = face / 2, side = face % 2;
      const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
      const int nt1 = (t1 == 0) ? nx_ : (t1 == 1) ? ny_ : nz_;
      const int nt2 = (t2 == 0) ? nx_ : (t2 == 1) ? ny_ : nz_;
      const int na = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
      const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
      const long sa = (a == 0) ? sx : (a == 1) ? sy : sz;
      const long st1 = (t1 == 0) ? sx : (t1 == 1) ? sy : sz;
      const long st2 = (t2 == 0) ? sx : (t2 == 1) ? sy : sz;
      const int aInner = (side == 0) ? G : (G + na - 1);
      const int dir = (side == 0) ? -1 : +1;
      Kokkos::parallel_for(
          "peclet::flow::pc_zero_ghosts",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(CCExec(), {G, G}, {G + nt1, G + nt2}),
          KOKKOS_LAMBDA(int j1, int j2) {
            const long base = (long)aInner * sa + (long)j1 * st1 + (long)j2 * st2;
            for (int L = 1; L <= 2; ++L)
              f(base + (long)dir * L * sa) = 0.0;
          });
    }
    Kokkos::fence();
  }

  /// Subtract the phase-change (and any prescribed) divergence source from `div_` so the deflated
  /// pressure solve delivers `div(open u) = S`. One branch in `project()`, inert when off.
  void pcApplyDivergenceSource(CCField div) {
    if (!pcEnabled_ && !pcHasUser_)
      return;
    const C3 e = e_;
    const bool hasPc = pcEnabled_, hasUser = pcHasUser_;
    CCConst sp = hasPc ? CCConst(pcSrc_) : CCConst(div);
    CCConst su = hasUser ? CCConst(pcUser_) : CCConst(div);
    Kokkos::parallel_for(
        "peclet::flow::pc_div_source",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(CCExec(), {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double s = 0.0;
          if (hasPc)
            s += sp(i);
          if (hasUser)
            s += su(i);
          div(i) -= s;
        });
  }

  void requirePhaseChange(const char* who) const {
    if (!pcEnabled_)
      throw std::runtime_error(std::string(who) + ": phase change is not enabled (call "
                                                  "enable_phase_change first)");
  }
  ScalarField& scalarField(const std::string& name) {
    for (auto& sc : scalars_)
      if (sc.name == name)
        return sc;
    throw std::runtime_error("no scalar named '" + name + "'");
  }
  /// Allocate (idempotently) the per-cell Dirichlet mask + value of a registered scalar. Inert
  /// until allocated: `advanceScalars` branches on `dmask.extent(0)`.
  void scalarDirichletMask(const std::string& name) {
    ScalarField& sc = scalarField(name);
    if (sc.dmask.extent(0) != n_) {
      sc.dmask = CCField(name + "_dmask", n_);
      sc.dval = CCField(name + "_dval", n_);
    }
  }

  // --- Property closures + per-cell body force ------------------------------------------------
  // Register a property/force closure. target: a registered field name — a material property
  // ("mu"/"rho"/…) or a body-force component ("force_x"/"force_y"/"force_z"). kind: LinearMix /
  // BoussinesqForce / ArrheniusMu. in0/in1: input field names (in1 "" if unused). params: up to 4
  // doubles (meaning per kind — property_closures.hpp). Applied at the top of step() in
  // registration order. Targeting a force component turns on the per-cell body-force RHS path.
  void setPropertyModel(const std::string& target, ClosureKind kind, const std::string& in0,
                        const std::string& in1, const std::vector<double>& params) {
    Closure cl;
    cl.kind = kind;
    cl.out = ensureTarget(target);
    cl.in0 = CCConst(fields_.at(in0).data);
    if (!in1.empty())
      cl.in1 = CCConst(fields_.at(in1).data);
    for (int k = 0; k < 4 && k < (int)params.size(); ++k)
      cl.p[k] = params[k];
    closures_.push_back(cl);
    if (target == "mu")  // a closure driving mu turns on variable viscosity
      setPropertyMode(true, harmonicMu_);
    if (target == "rho")  // a closure driving rho turns on the variable-density path
      setDensityMode(true);
  }
  // Enable/disable variable density: binds the "rho" field (creating it seeded with the scalar rho_
  // if absent) into the momentum time term, the advection weight, and the pressure projection
  // (face coefficient open/rho_f + 1/rho_f correction). rho_ (set_rho) becomes the REFERENCE
  // density rho0 of the projection scaling — a uniform rho field == rho_ reduces exactly to the
  // constant solver. Escape hatch: set_field("rho", arr) + set_density_mode(True); or a closure
  // targeting "rho" (e.g. rho = LinearMix of a transported phase fraction) enables it
  // automatically. Staggered grid only (v1); the velocity multigrid (scalar-coefficient) is
  // disabled.
  // Rung V8 (WO-T) lifted the collocated throw. On `SolverColocated` the variable-density path is
  // the ABC approximate projection with the face coefficient `c_f = o_f rho0/rho_f`,
  // `projectCorrectVar` on the FACE field, and a cell correction that is the AVERAGE OF THE TWO FACE
  // CORRECTIONS (never a cell-centred grad(phi)/rho_c); every body / interfacial force becomes a
  // face acceleration added after `centerToFace`. Scope: ALL-FLUID
  // (`set_pressure_geometry`) — an immersed solid still throws, at the first `project()`, and so do
  // the ghost projection and `set_rho_face_harmonic` (see requireCollocatedFaceForceScope).
  // Momentum consistency (`enable_vof_momentum`) is NOT in this rung: the collocated construction
  // needs Favre face states, so the collocated two-phase path is rated to density ratio <= ~100 for
  // cases WITH MOTION (a high-ratio case at REST — hydrostatic, stationary droplet — is exact
  // either way, and is measured at ratio 1000).
  void setDensityMode(bool variable) {
    if (variable)
      collocatedV8AutoFallback("variable density on the collocated grid");
    varRho_ = variable;
    if (variable) {
      if (fields_.has("rho"))
        rhoField_ = fields_.at("rho").data;
      else {
        rhoField_ = addField("rho");
        Kokkos::deep_copy(rhoField_, rho_);
      }
      if (rho1_.extent(0) == 0) {  // g=1 MG-block scratch for the projection coefficients
        rho1_ = CCField("rho1", n1_);
        cx1_ = CCField("cx1", n1_);
        cy1_ = CCField("cy1", n1_);
        cz1_ = CCField("cz1", n1_);
      }
      ensureCellForceAll();  // buildRhsVar reads the per-cell force (zero until a closure sets it)
      useVelocityMg_ = false;  // scalar-coefficient velocity MG (variable-coeff deferred)
      // Pressure driver: CHEBYSHEV by default under variable density — but NOT for the reason this
      // comment used to give ("MG-PCG stalls on the rho-scaled coefficient operator"), which WO-B
      // refuted: the stall it described is a DOMAIN-BC defect at constant density, repaired by
      // WO-H (CutcellMG::applyNeumannGhost), and on the periodic pore-scale operator MG-PCG beats
      // Chebyshev ~10x at density ratio 1e4. The reason that survives measurement is narrower and
      // real: at a high density CONTRAST the arithmetic coarsening of the face coefficient makes
      // the V-cycle preconditioner INDEFINITE (measured on a dense sym(M): a negative pivot from
      // ratio ~1e3), and no Krylov CG survives that, while Chebyshev — which needs only real
      // spectrum bounds, re-estimated on every coefficient rebuild — is healthy on every
      // configuration measured. Coefficient-aware coarsening (VOF_PLAN S3) is what would lift it.
      // An explicit set_pressure_pcg/_fcg/_chebyshev AFTER set_density_mode still wins (last set),
      // and since WO-H set_pressure_pcg's `on` flag genuinely honours that promise.
      useChebyshev_ = true;
      chebBoundsSet_ = false;
    }
  }
  // Enable/disable the volume-averaged (porous) continuity for unresolved CFD-DEM: the projection
  // enforces d(eps)/dt + div(eps u) = 0 instead of div(u)=0, so the velocity is NOT solenoidal
  // where the void fraction changes. Binds the "eps" field (void fraction from the particle
  // deposition; created seeded to 1 if absent). Staggered-only. The coupling deposits eps each step
  // BEFORE step().
  // Has the cut-cell pressure operator been built (set_solid / set_pressure_geometry)? The porous
  // projection requires it — project() throws otherwise; the coupling driver queries this to
  // auto-install an all-fluid geometry.
  bool hasCutcellPressure() const { return cutcellPressure_; }
  void setPorousContinuity(bool on) {
    if constexpr (Grid::collocated) {
      if (on)
        throw std::runtime_error("set_porous_continuity: staggered-only (v1)");
    }
    porous_ = on;
    if (on) {
      if (fields_.has("eps"))
        epsField_ = fields_.at("eps").data;
      else {
        epsField_ = addField("eps");
        Kokkos::deep_copy(epsField_, 1.0);  // no particles -> eps=1 -> reduces to div(u)=0
      }
      if (epsPrev_.extent(0) == 0) {
        epsPrev_ = CCField("epsPrev", n_);
        depsdt_ = CCField("depsdt", n_);
      }
      if (divAdv_.extent(0) == 0)
        divAdv_ = CCField("divAdv", n_);  // cell div(u) for the porous advection-form compensation
      if (epsRho_.extent(0) == 0)
        epsRho_ = CCField("epsRho", n_);  // rho_eff = eps*rho (eps-conservative momentum)
      // The eps-conservative momentum path (porousCons_) routes through buildRhsVar, which reads
      // the per-cell force unconditionally — allocate it (zero) like setDensityMode does. Without
      // this a porous run with NO drag/closure (never the coupled case, which enables drag and
      // thereby the force fields) dereferences an empty device View.
      ensureCellForceAll();
      Kokkos::deep_copy(epsPrev_, epsField_);  // d(eps)/dt=0 on the first step
      if (eps1_.extent(0) == 0)
        eps1_ = CCField("eps1", n1_);
      if (beta1_.extent(0) == 0)
        beta1_ = CCField("beta1", n1_);
      if (rho1_.extent(0) == 0) {  // share the g=1 coefficient scratch with the varRho path
        rho1_ = CCField("rho1", n1_);
        cx1_ = CCField("cx1", n1_);
        cy1_ = CCField("cy1", n1_);
        cz1_ = CCField("cz1", n1_);
      }
      // CHEBYSHEV by default (as for variable density). CAVEAT on the original justification
      // ("MG-PCG stalls on the eps-scaled coefficient operator"): it rests on the same kind of
      // observation WO-B refuted for varRho, and it was recorded through the setter whose `on` flag
      // was a no-op until WO-H — so a "PCG" run made that way actually measured Chebyshev. The
      // default is kept because it is safe (Chebyshev needs only real spectrum bounds and is the
      // one driver healthy on every high-contrast coefficient configuration measured), NOT because
      // the eps-scaled PCG stall has been re-measured; that re-measurement is still owed
      // (doc/vof_workorders.md, WO-B escalation #1). Bounds re-estimated on every coefficient
      // rebuild (chebBoundsSet_ invalidation in project()). An explicit driver set afterwards wins.
      useChebyshev_ = true;
      chebBoundsSet_ = false;
      configurePorousDragSolver();  // if drag already on, switch to GraphAMG+PCG (Chebyshev
                                    // diverges)
    }
  }
  // Reseed eps^n = eps^{n+1} so d(eps)/dt = 0 this step. Call after the FIRST void-fraction
  // deposition (the "eps" field starts empty, so without this step 0 sees a spurious d(eps)/dt from
  // 0 -> eps).
  void syncPorousPrev() {
    if (porous_)
      Kokkos::deep_copy(epsPrev_, epsField_);
  }
  // Include (default) or drop the d(eps)/dt source in the porous projection RHS. Dropping it
  // enforces div(eps u)=0 — useful when eps is a bare per-cell particle deposit whose
  // time-derivative is too jagged and drives the eps-weighted pressure solve unstable.
  void setPorousDepsDt(bool on) { porousDepsDt_ = on; }
  void setPorousConservative(bool on) { porousCons_ = on; }
  // Pressure under-relaxation factor omega_p in (0,1] (MFIX-style); 1.0 = off (default).
  void setPressureUnderRelax(double w) { pressUnderRelax_ = w; }
  // Enable/disable variable-coefficient momentum (variable viscosity). variable=true binds the "mu"
  // field (creating it, seeded with the current scalar mu, if absent) and forces the stencil solve
  // path. harmonic selects the harmonic face mean (continuous shear stress across a viscosity jump)
  // vs arithmetic. Escape hatch: set_field("mu", arr) then set_property_mode(True).
  void setPropertyMode(bool variable, bool harmonic) {
    varProps_ = variable;
    harmonicMu_ = harmonic;
    if (variable) {
      if (fields_.has("mu"))
        muField_ = fields_.at("mu").data;
      else {
        muField_ = addField("mu");
        Kokkos::deep_copy(muField_,
                          mu_);  // default to the scalar mu until a closure/set_field sets it
      }
      useVelocityMg_ =
          false;  // the velocity multigrid takes a scalar mu (variable-coeff vmg deferred)
    }
  }
  // Rotational-pressure treatment under variable viscosity. The Timmermans rotational term
  // P += (rho/dt)phi - mu*div(u*) is only valid for HOMOGENEOUS viscosity (Deteix & Yakoubi, Appl.
  // Math. Lett. 2018 / arXiv:1902.05643): with spatially varying mu the pointwise term is no longer
  // the gradient part of the viscous stress, and the accumulated inconsistency destabilises the
  // incremental scheme at strong contrast (observed: 10x jump + harmonic faces -> divergence).
  // Modes (the incremental predictor -grad(P^n) and P accumulation are kept in ALL of them — that
  // is what enables large-dt / steady-Stokes stepping):
  //   0 "min"  (default): rotational coefficient chi*mu_min — a CONSTANT dominated by the true
  //   local
  //            dissipation everywhere (mu_min <= mu(x)), so the constant-viscosity stability theory
  //            carries over; reduces EXACTLY to the validated scheme when mu is uniform.
  //   1 "full": chi*mu(i) pointwise — better pressure consistency at MILD contrast; not stable at
  //            strong contrast (user's responsibility).
  //   2 "off" : plain incremental (no rotational term) — unconditionally stable, keeps the
  //   artificial
  //            pressure Neumann layer of the non-rotational scheme.
  // The fully consistent variable-viscosity correction (shear-rate projection: an extra Poisson
  // solve for psi with rhs div(div(2 nu D(u)))) is deferred.
  void setVariableRotational(int mode, double chi) {
    varRotMode_ = mode < 0 ? 0 : (mode > 2 ? 2 : mode);
    varRotChi_ = chi < 0.0 ? 0.0 : chi;
  }
  // Tabulated property: out = piecewise-linear interp of (xs, ys) at the input field (xs
  // ascending).
  void setPropertyTable(const std::string& target, const std::string& in0,
                        const std::vector<double>& xs, const std::vector<double>& ys) {
    Closure cl;
    cl.kind = ClosureKind::Table1D;
    cl.out = ensureTarget(target);
    cl.in0 = CCConst(fields_.at(in0).data);
    cl.nTab = (int)std::min(xs.size(), ys.size());
    cl.tabX = CCField(target + "_tabx", cl.nTab);
    cl.tabY = CCField(target + "_taby", cl.nTab);
    auto hx = Kokkos::create_mirror_view(cl.tabX);
    auto hy = Kokkos::create_mirror_view(cl.tabY);
    for (int k = 0; k < cl.nTab; ++k) {
      hx(k) = xs[k];
      hy(k) = ys[k];
    }
    Kokkos::deep_copy(cl.tabX, hx);
    Kokkos::deep_copy(cl.tabY, hy);
    closures_.push_back(cl);
  }
  // Apply all closures (also called at the top of step()). Exposed for testing.
  void updateProperties() {
    for (auto& cl : closures_)
      applyClosure(cl, e_, G);
  }
  // Allocate + register the per-cell body-force fields ("force_x/y/z") and route them into the
  // momentum RHS, for an EXTERNAL writer (CFD-DEM feedback) to fill directly via field_view — no
  // closure needed. buildRhsForced then adds them each step (they persist; the writer overwrites).
  void enableCellForce() { ensureCellForceAll(); }
  // Implicit (semi-implicit) linear drag: a per-cell coefficient field "drag_beta" is added to the
  // momentum diagonal each step, so a drag source −β(u − u_p) is treated implicitly (the fluid
  // solve becomes (ρ/dt + β)u = … + β u_p). The drag TARGET β·u_p goes into the force_x/y/z fields
  // (the RHS). Unconditionally stable for any β (unlike an explicit −β u force, which diverges for
  // the stiff β of a dense particle bed). The external writer (CFD-DEM) fills "drag_beta" +
  // "force_*" via field_view; enableDrag() allocates them and turns the diagonal path on.
  void enableDrag() {
    if (!fields_.has("drag_beta"))
      dragBeta_ = addField("drag_beta");
    else
      dragBeta_ = fields_.at("drag_beta").data;
    ensureCellForceAll();  // force_* carries beta*u_p (the implicit-drag RHS target)
    hasDrag_ = true;
    configurePorousDragSolver();
  }
  // Porous + implicit drag: the drag-relaxation w_f=idt/(idt+beta) makes the pressure coefficient
  // high-ratio (~1 in the freeboard, ->0 in the dense bed). Chebyshev diverges on it; the algebraic
  // GraphAMG coarse solve + PCG is robust. Applied whenever BOTH porous_ and hasDrag_ are on
  // (either set second). An explicit set_pressure_* afterwards still wins.
  void configurePorousDragSolver() {
    if (!(porous_ && hasDrag_))
      return;
    pressGraphAmg_ = true;  // GraphAMG bottom (domain-BC operators: buildAmg skips
                            // the wrap across non-periodic faces and pcgAmg keeps the
                            // mean only when the operator is singular)
    if (cutcellPressure_)   // MG already built (set_solid ran) -> apply now
      mg_.setAgglomerationMode(1);
    useChebyshev_ = false;  // PCG, not Chebyshev (diverges on the high w_f ratio)
    chebBoundsSet_ = false;
  }
  // Add the drag coefficient beta(i) to the (float) momentum diagonal of component c. Called after
  // each stencil (re)build when hasDrag_. All-fluid (rscale==1) is exact; the drag×cut-cell-IBM
  // interaction (rscale≠1) is untested (documented).
  void addDragDiagonal(int c) {
    CCExec space;
    C3 e = e_;
    FV AC = C[c].AC;
    CCConst beta = CCConst(dragBeta_);
    const long sc = strideOf(c);
    // Porous continuity: the projection's operator/correction carry the FACE drag relaxation
    // w_f = idt/(idt + beta_f), beta_f = 1/2(beta(i)+beta(i-sc)) (buildPorousCoeffDrag /
    // projectCorrectPorousDrag). The staggered momentum diagonal of u_c(i) — the face between cells
    // i-sc and i — must carry the SAME beta_f: then a pressure perturbation deltaP produces
    // du* = -grad(deltaP)/(idt+beta_f) and the projection returns phi = -deltaP/idt exactly (same
    // operator), so the incremental predictor cancels pressure errors in one step. With the cell
    // value beta(i) the loop has gain (idt+beta_f)/(idt+beta_cell) at a beta jump (bed top: ~3) and
    // the accumulated pressure diverges exponentially. Non-porous (incompressible drag, w==1 path)
    // keeps the validated cell-beta form.
    const bool faceAvg = porous_;
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "peclet::flow::add_drag_diag", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          const double bd =
              faceAvg ? 0.5 * ((double)beta(i) + (double)beta(i - sc)) : (double)beta(i);
          AC(i) = (MReal)((double)AC(i) + bd);
        });
  }

 private:
  // Resolve a closure target to a registered buffer. A force component allocates ALL three
  // cellForce_ slots (buildRhsForced reads every component) and enables the body-force RHS path.
  CCField ensureTarget(const std::string& name) {
    if (name == "force_x" || name == "force_y" || name == "force_z")
      ensureCellForceAll();
    return addField(name);  // idempotent; returns the (now-existing) buffer
  }
  void ensureCellForceAll() {
    static const char* fn[3] = {"force_x", "force_y", "force_z"};
    for (int c = 0; c < 3; ++c)
      cellForce_[c] = addField(fn[c]);  // zero-initialised, registered
    hasCellForce_ = true;
  }
  // Ghost fill for a scalar: periodic (single-rank) / MPI halo base, then override any domain
  // Dirichlet/Neumann faces.
  void scalarFillGhosts(ScalarField& sc) {
    fillGhosts(sc.c);
    applyScalarBc(sc);
  }
  // Overwrite the ghost band on each Dirichlet/Neumann domain face (both layers, for the ±2
  // advection reach). Distributed: a rank applies a face's BC iff its block TOUCHES that global
  // face. The halo fill runs first (and may periodic-wrap those ghosts); the BC overwrite wins,
  // exactly matching the single-rank fill-then-BC order. Cross-rank ghost CORNERS on a BC face
  // keep their exchanged (pre-BC) values, but the scalar stencils only read axis-aligned ghosts
  // (7-point diffusion + straight ±2 advection reach), so those corners are never consumed.
  void applyScalarBc(ScalarField& sc) {
    for (int f = 0; f < 6; ++f)
      if (sc.bc[f] != 0 && touchesGlobalFace(f))
        applyScalarBcFace(sc.c, f / 2, f % 2, sc.bc[f], sc.bcVal[f]);
  }
  // Does this rank's block touch global domain face f (always true single-rank)?
  bool touchesGlobalFace(int f) const {
#ifdef PECLET_FLOW_MPI
    if (distributed_) {
      const int a = f / 2;
      const int o = (a == 0) ? og_.x : (a == 1) ? og_.y : og_.z;
      const int n = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
      const int gn = (a == 0) ? gnx_ : (a == 1) ? gny_ : gnz_;
      return (f % 2 == 0) ? (o == 0) : (o + n == gn);
    }
#endif
    (void)f;
    return true;
  }
  // Re-open the diffusion face at a Dirichlet domain boundary: set_domain_bc closes the boundary
  // openness (ox_=0), which correctly makes Neumann/adiabatic walls zero-flux but would also cut a
  // Dirichlet wall's heat path. For each Dirichlet face, restore the face coefficient (band = -D,
  // A_C += D); the ghost carries 2*value - inner so the row is the standard Dirichlet operator.
  void applyScalarBcStencil(ScalarField& sc) {
    for (int f = 0; f < 6; ++f) {
      if (sc.bc[f] != 2 || !touchesGlobalFace(f))
        continue;  // only Dirichlet reopens; Neumann/periodic leave the (closed/interior) band
      const int a = f / 2, side = f % 2;
      CCField band = (a == 0)   ? (side == 0 ? sc.AW : sc.AE)
                     : (a == 1) ? (side == 0 ? sc.AS : sc.AN)
                                : (side == 0 ? sc.AB : sc.AT);
      patchScalarDirichletFace(sc.AC, band, sc.D, a, side);
    }
  }
  // nvcc requires member functions that contain extended (device) lambdas to be PUBLIC — the
  // OpenMP/host build accepts them private, so the breakage only shows on the CUDA backend.
 public:
  void patchScalarDirichletFace(CCField AC, CCField band, double D, int a, int side) {
    const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
    const int nt1 = (t1 == 0) ? nx_ : (t1 == 1) ? ny_ : nz_;
    const int nt2 = (t2 == 0) ? nx_ : (t2 == 1) ? ny_ : nz_;
    const int na = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
    const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
    const long sa = (a == 0) ? sx : (a == 1) ? sy : sz;
    const long st1 = (t1 == 0) ? sx : (t1 == 1) ? sy : sz;
    const long st2 = (t2 == 0) ? sx : (t2 == 1) ? sy : sz;
    const int aInner = (side == 0) ? G : (G + na - 1);
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::scalar_bc_stencil",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {G, G}, {G + nt1, G + nt2}),
        KOKKOS_LAMBDA(int j1, int j2) {
          const long i = (long)aInner * sa + (long)j1 * st1 + (long)j2 * st2;
          // base build put band(i) = -D*open_face and A_C += D*open_face; force the face fully open
          // (band -> -D, A_C gains D*(1-open)) without double-counting when it was already open.
          AC(i) += D + band(i);
          band(i) = -D;
        });
  }
  void applyScalarBcFace(CCField c, int a, int side, int type, double val) {
    const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
    const int nt1 = (t1 == 0) ? nx_ : (t1 == 1) ? ny_ : nz_;
    const int nt2 = (t2 == 0) ? nx_ : (t2 == 1) ? ny_ : nz_;
    const int na = (a == 0) ? nx_ : (a == 1) ? ny_ : nz_;
    const long sx = 1, sy = e_.x, sz = (long)e_.x * e_.y;
    const long sa = (a == 0) ? sx : (a == 1) ? sy : sz;
    const long st1 = (t1 == 0) ? sx : (t1 == 1) ? sy : sz;
    const long st2 = (t2 == 0) ? sx : (t2 == 1) ? sy : sz;
    const int aInner = (side == 0) ? G : (G + na - 1);  // inner boundary cell a-index
    const int dir = (side == 0) ? -1 : +1;              // toward the ghost
    CCExec space;
    Kokkos::parallel_for(
        "peclet::flow::scalar_bc_face",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<2>>(space, {G, G}, {G + nt1, G + nt2}),
        KOKKOS_LAMBDA(int j1, int j2) {
          const long base = (long)aInner * sa + (long)j1 * st1 + (long)j2 * st2;
          for (int L = 1; L <= 2; ++L) {
            const long gcell = base + (long)dir * L * sa;
            const long icell = base - (long)dir * (L - 1) * sa;
            c(gcell) = (type == 2) ? (2.0 * val - c(icell)) : c(icell);
          }
        });
  }

 private:
  int nx_, ny_, nz_;
  C3 e_, e1_;
  std::size_t n_, n1_;
  double rho_ = 1.0, mu_ = 0.1, dt_ = 50.0;
  std::array<double, 3> f_{{0, 0, 0}};
  int velIters_ = 200, presIters_ = 20;
  double velTol_ = 0.0;         // momentum tolerance stop (0 = legacy fixed-count loop)
  int velMinIters_ = 2;
  long lastMomentumSweeps_ = 0;  // sweeps actually run last step (summed over components/Picard)
  double velResTol_ = [] {  // residual-based momentum stop: < 0 follows the pressure rtol (DEFAULT
    const char* e = std::getenv("PECLET_FLOW_VRES");  // since 2026-09-02), 0 = update criterion,
    return e ? std::atof(e) : -1.0;                   // > 0 fixed; env override for bisection
  }();
  double lastMomentumResid_ = -1.0;  // max_c max|r|/max|b| at exit (residual mode)
  CCField velRes_;                 // scratch for the stencil-path residual
  double lastAxNorm_ = 0.0;        // max|A u| of the last residual evaluation (scale)
  int pcgMaxit_ = 500;
  double pcgRtol_ = 1e-10;  // cut-cell pressure MG-PCG
  bool useChebyshev_ = false,
       chebBoundsSet_ = false;  // Chebyshev pressure driver (set_pressure_chebyshev)
  bool useFcg_ = false;         // flexible-CG pressure driver (set_pressure_fcg); OFF by default,
                                // so the shipped MG-PCG path is untouched (zp1_ is not allocated)
  CCField zp1_;                 // FCG's extra scratch: allocated lazily at the first FCG solve
  int chebMaxit_ = 120;
  double chebRtol_ = 1e-9, chebA_ = 0.0, chebB_ = 0.0;
  int nLevels_ = 4;             // multigrid depth (CUDA default; set_pressure_multigrid)
  bool pressGraphAmg_ = false;
  // Coarse-solve policy: -1 auto (DEFAULT — agglomerate when the coarsest grid exceeds
  // PECLET_FLOW_AGGLOM_EXTENT on any axis; identical to the smoothed bottom otherwise),
  // 0 smoothed, 1 always. Auto became the default 2026-08-13 after the IBM-path anomaly was
  // fixed (per-fluid-component null-space projection; see ../docs/DECOMPOSITION_AND_MULTIGRID.md).
  int pressAgglomMode_ = -1;
  long lastPressureIters_ = 0;
  CutcellMG mg_;
  // --- multi-rank (MPI) state, gated (single-GPU module never links MPI -> byte-identical when
  // off) ---
  bool distributed_ = false;
  C3 og_{0, 0, 0};  // velocity-block inner origin (global red-black parity); {0,0,0} single-rank
#ifdef PECLET_FLOW_MPI
  std::shared_ptr<GridHaloTopology<3>> velHalo_;  // g=2 velocity-block topology
  std::shared_ptr<GridHalo<double>> velDev_;      // g=2 velocity-block ghost exchange
  std::shared_ptr<GridHalo<MReal>> velDevF_;      // float twin (momentum-stencil ring, CA sweeps)
  bool caMomentum_ = false;  // communication-avoiding momentum sweeps (PECLET_FLOW_CA + extent>=4)
  bool momStencilDirty_[3] = {true, true, true};  // per-component: stencil ring needs an exchange
  std::shared_ptr<peclet::core::decomp::BlockDecomposer<3>>
      dec_;  // current partition (redistribute)
  MPI_Comm comm_ = MPI_COMM_NULL;
  int gnx_ = 0, gny_ = 0, gnz_ = 0;  // communicator + GLOBAL dims
#endif
  int bc_[6] = {0, 0, 0, 0, 0, 0};
  double bcVel_[6][3] = {};
  bool hasBc_ = false, hasOutflow_ = false;  // domain BCs
  bool hasSolid_ =
      false;  // an immersed solid is present (any inner SDF < 0) -- with domain BCs, the
              // momentum solve must use the cut-cell IBM stencil, not the all-fluid fold
  double backflowBeta_ =
      0.2;  // outflow backflow-stabilization coefficient (0 = off; inert unless the
            // outflow reverses, so purely-outgoing outlets stay byte-identical)
  CCField bcProf_[6];
  int bcProfNc_[6] = {0, 0, 0, 0, 0, 0};  // per-position inlet profiles (face grid [Lb*Lc*3])
  CCField bcDcorr_[3], bcBrhs_[3];        // implicit-diffusion face fold (per component)
  bool advect_ = false, cutcellPressure_ = false, implicitFou_ = false;
  bool deferredCorr_ = true;  // deferred-correction advection (off = pure implicit FOU, 1st order)
  int advScheme_ = 0;         // high-order advection: 0 = SOU (default), 1 = Koren TVD
  bool incremental_ = true,
       pwarm_ = false;    // incremental-rotational pressure (CUDA default on) + warm-start
  bool dtDirty_ = false;  // set_dt after set_solid: momentum stencil needs a rebuild
    int faceInterp_ = 9;    // collocated scheme: 9 = gauge-exact (DEFAULT), 0 = plain (legacy)
  double apertureFloor_ = [] {  // mode-13 denominator floor (PECLET_FLOW_APERTURE_FLOOR)
    const char* v = std::getenv("PECLET_FLOW_APERTURE_FLOOR");
    return v ? std::atof(v) : 0.25;
  }();
  bool gauge2a_ = false;   // gauge-exact with the Guy-Fogelson "gradient 2a" one-sided branch
                           // (set_collocated_scheme("gauge-2a"); experimental stall fix).
                           // Single-rank exact; at rank seams the +/-3 stencil falls back to the
                           // 2-point form (decomposition-dependent there until the halo is widened).
  bool rotationalP_ = true;  // false = PM I ablation: drop the -mu*div(u*) Timmermans term from
                             // the incremental pressure accumulation (constant-mu path only)
  bool rotFilter_ = false;   // filtered rotational: smooth div(u*) (mask-aware axis-wise 1-2-1,
                             // one-sided toward the fluid at solid neighbours) before accumulating
                             // -mu*div into P. Kills the wall-normal checkerboard feedback the
                             // cell-centered rotational update is unstable through, keeps the O(1)
                             // pressure-relaxation gain and the phi=0 (dt-free) fixed point.
  double rotFilterEps_ = 0.05;  // S' = eps I + (1-eps) S (see setRotationalFilter)
  double rotWeight_ = 1.0;      // rotational under-relaxation w (setRotationalWeight)
  double rotWallW_ = 0.0;       // wall-banded rotational blend w0 (setRotationalWallWeight)
  int apertureOrder_ = [] {  // face-aperture estimator order (setApertureOrder; DEFAULT 2 =
    // marching-squares since 2026-08-26 -- user decision, kills the convexity bias; 1 = the
    // legacy one-sample model). PECLET_FLOW_APERTURE_ORDER overrides the default (diagnostics).
    const char* v = std::getenv("PECLET_FLOW_APERTURE_ORDER");
    return v ? std::atoi(v) : 2;
  }();
  int fluidOnlyMode_ = 0;  // fluid-only constraint (setFluidOnlyConstraint): 1=A filter, 2=B star
  StarOverlay starOv_;     // mode-B Kron star overlay (built in setSolid)
  Kokkos::View<int, CCMem> starCounter_;
  int nStar_ = 0;
  double fvRelax_ = 1.0;  // mode-4 FV defect-correction under-relaxation (setFvRelax)
  bool useVelocityMg_ = false;
  bool vmgExplicit_ = false;  // set_velocity_multigrid was called (AUTO rule off)
  long vmgAutoCells_ = [] {   // AUTO threshold, cells per rank (0 = never)
    const char* e = std::getenv("PECLET_FLOW_VMG_AUTO_CELLS");
    return e ? std::atol(e) : 65536L;
  }();
  long vmgAutoMinGlobal_ = [] {  // AUTO applies only to global problems at least this large
    const char* e = std::getenv("PECLET_FLOW_VMG_AUTO_MIN_GLOBAL");
    return e ? std::atol(e) : (1L << 23);  // 8M cells
  }();
  int vmgLevels_ = 4, vmgVcycles_ = 8;  // IBM velocity multigrid (staircase)
  VelocityMG vmg_;
  CCField vmgTheta_, vmgClean_;
  int outerIters_ = 1;
  double outerTol_ = 0.0;  // Picard outer iteration (CUDA set_outer_iterations)
  long lastOuterIters_ = 0;
  double lastOuterCorr_ = 0.0;
  // per-step phase timers (seconds, this rank; see lastStepSeconds)
  double tStep_ = 0.0, tPredictor_ = 0.0, tMomentum_ = 0.0, tProjection_ = 0.0;
  // fence-then-read wall clock: phase boundaries must not attribute queued device work to the
  // next phase
  static double phaseTick() {
    Kokkos::fence();
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  CCField sdf_, ox_, oy_, oz_, phi_, div_, P_, ox1_, oy1_, oz1_, rhs1_, phi1_, r_, z_, pp_, Ap_;
  bool ghostProjection_ = false;  // directional ghost-cell projection (the collocated AUTO default)
  bool colSchemeAuto_ = Grid::collocated;  // AUTO scheme resolution at setSolid (cleared by any
                                           // explicit scheme selection)
  GpOverlay gpOv_;                // its per-row overlay (built by setSolid)
  Kokkos::View<int*, CCMem> gpIdMap_;
  Kokkos::View<int, CCMem> gpCounter_;
  int gpNRows_ = -1;         // -1 = overlay not built (set_solid must run with the mode on)
  int gpMatrixOrder_ = 2, gpRhsOrder_ = 2;  // closure order: implicit phi couplings / RHS
  CCField tEx_[3][3];             // exact crossings t[c][k] (inner grid; setExactCrossings)
  bool hasExactCross_ = false;
  bool sceneCrossings_ = false;   // crossings came from the analytic scene (per-rank, no override)
  std::shared_ptr<peclet::core::geom::SceneQueryDevice<double, CCMem>> sceneQ_;
  bool hasScene_ = false;
  // CUT OWNERSHIP (Layer 3 rung 1). Which scene instance owns the nearest surface, per INNER cell
  // (x-fastest, nx*ny*nz), filled by set_solid_from_scene in the same traversal that samples the
  // SDF (core's evalOwner answers both at once, so the field is free). Meaningful everywhere; only
  // the cut cells consume it — moving geometry reads a wall velocity off the owner, and resolved
  // CFD-DEM posts the hydrodynamic force back to it. Empty until a scene is sampled.
  Kokkos::View<int*, CCMem> cutOwner_;
  // MOVING GEOMETRY (Layer 3 rungs 2-3). Per-scene-instance rigid-body motion: host copies (KBs,
  // rank-replicated like the scene itself) plus the device mirrors the kernels capture. All-zero
  // motion => hasMotion_ stays false and EVERY path below is skipped, so a static solver is
  // bit-identical to the build that predates this rung -- "moving with zero velocity" is not the
  // same code path and would not be.
  std::vector<double> instCen_, instLin_, instAng_;  // 3*nInst_ each, world coordinates
  Kokkos::View<double*, CCMem> instCenD_, instLinD_, instAngD_;
  int nInst_ = 0;
  bool hasMotion_ = false;
  CCField uBc_[3];     // R2: wall velocity component c AT component c's staggered points
  CCField uwCell_[3];  // R3: the whole wall velocity at CELL CENTRES (the wall-flux divergence)
  // A0 (advective cut-wall flux): the momentum advection's OWN velocity inputs -- a copy of C[c].u
  // whose solid-masked rows carry uBc_ (the local rigid-body wall velocity) instead of the zeros
  // maskVelocity pins there. Allocated and filled only while an instance is moving; every other
  // configuration keeps reading C[c].u, byte for byte. See buildAdvInputs().
  CCField uwAdv_[3];
  // Route (b) instrumentation: u* = the LAST momentum solve's solution, stashed (ghosts included,
  // exactly as the smoother left them) before the projection overwrites it. The reaction force
  // needs the viscous fluxes AT u* -- the implicit solve acted on u*, not on the projected u.
  CCField uStar_[3];
  bool haveUStar_ = false;
  // R0: the explicit advective term EXACTLY as the last Picard RHS used it, per component and per
  // staggered cell, in the equation's own units (rho*(aF - aK), i.e. before the rscale descale).
  // Written by buildRhs / buildRhsForced when a scene is installed and advection is on; consumed
  // by hydroForceTorqueReaction, which must subtract every non-pressure, non-wall RHS term.
  CCField advRhs_[3];
  bool haveAdvRhs_ = false;
  bool wallFluxDiv_ = true;  // rung 3 on (correct physics); off only to exhibit its absence
  // Fresh-cell seeding (see seedFreshCells). ON by default since 2026-08-30: measured on an
  // oscillating sphere that physically translates through the grid, it removes a
  // RESOLUTION-INDEPENDENT +2.6..2.9% drag bias, cuts the spurious force oscillation 20-50x to
  // within 17% of the non-moving floor, and improves the resolved CFD-DEM loop's total-momentum
  // conservation 95x. set_fresh_cell_seed(False) restores the old behaviour.
  bool freshSeed_ = true;
  // Rung 4: the scene is KEPT (not just its device query), so an instance transform can be updated
  // and the whole geometry re-derived without the caller re-encoding anything.
  std::shared_ptr<peclet::core::geom::SceneBuilder<double>> sceneB_;
  peclet::core::Vec3<double> sceneOrigin_{0, 0, 0}, sceneExtent_{0, 0, 0};
  bool scenePeriodic_ = false;
  long imageOverlapCells_ = 0;  // set_solid_from_scene's periodic-image overlap count
  bool sceneDirty_ = false;  // a transform changed; the device query must be rebuilt
  std::vector<double> oxOverride_, oyOverride_, ozOverride_;  // exact apertures (inner)
  bool hasOpenOverride_ = false;
  CCField oxb_, oyb_, ozb_;  // binary (COUPLED) openness on the g=2 block (ghost divergence)
  CCField sdfGp_;  // the projection's sdf (fragmentation pockets decoupled) — gpCenterGrad reads
                   // it so the collocated predictor/correction never touch a decoupled cell
  CCField gpRh_, gpT_, gpZ2_;  // extra BiCGStab scratch (g=1 block)
  CCField gpX2_;  // distributed BiCGStab matvec staging (g=2 solver block; overlay +/-2 halo)
  CCField uf_, vf_, wf_;    // collocated: transient face (MAC) field (approx projection)
  CCField faceAcc_[3];      // rung V8 (WO-T): the collocated face velocity increment of
                            // this step (force acceleration, then minus the projection's
                            // own face correction). Allocated only on that path.
  CCField tgp_;             // collocated: transpose-gradient scratch (setFaceInterp(2/3))
  CCField wdef_;            // collocated: FV wall viscous-flux defect scratch (setFaceInterp(4))
  CCField fvM_, fvL_, cs_;  // collocated: mode-4 defect scratch (M·u, L_FV·u) + cell fluid fraction
  CCField xcx_, xcy_, xcz_;   // collocated: open-centroid wall distance per face (setFaceInterp(3))
  CCField old_[3], prev_[3];  // u^n time base + previous Picard iterate
  Comp C[3];
  peclet::core::FieldSet fields_;     // named directory of all cell fields (velocity/p/sdf + user)
  std::vector<ScalarField> scalars_;  // transported scalars (advection-diffusion)
  // --- phase change (WO-P01) -------------------------------------------------------------------
  bool pcEnabled_ = false, pcThermal_ = false, pcHasUser_ = false;
  double pcRhoG_ = 1.0, pcRhoL_ = 1.0, pcHlv_ = 1.0;
  double pcTsat_ = 0.0, pcKg_ = 0.0, pcKl_ = 0.0, pcRint_ = 0.0;
  double pcInterfaceEps_ = 1e-12, pcPureEps_ = 1e-12;
  std::string pcTName_;
  CCField pcMdot_, pcSrc_, pcUser_, pcArea_, pcNrm_[3], pcDep_, pcTgt_, pcCnew_, pcDefic_;
  PhaseChangeDiagnostics pcDiag_;
  std::vector<Closure> closures_;     // property/body-force closures (applied at top of step())
  CCField cellForce_[3];  // per-cell momentum body force (Boussinesq / CFD-DEM feedback)
  bool hasCellForce_ = false;
  bool varProps_ = false;    // variable-coefficient momentum (variable viscosity)
  bool harmonicMu_ = false;  // harmonic vs arithmetic face-viscosity mean
  CCField muField_;          // per-cell dynamic viscosity (when varProps_)
  int varRotMode_ = 0;       // rotational term under varProps: 0 chi*mu_min, 1 chi*mu(i), 2 off
  double varRotChi_ = 1.0;   // rotational coefficient scale chi
  bool varRho_ = false;      // variable density (momentum + projection); staggered only
  CCField rhoField_;         // per-cell density (when varRho_); rho_ is the reference rho0
  // --- geometric VoF (rung V2a, WO-J) ---
  bool vofEnabled_ = false;
  bool rhoFaceHarmonic_ = false;  // harmonic instead of arithmetic rho_f in the projection (OFF)
  bool outflowCorrValid_ = false;  // the projection's outflow-face correction is still in u
  // WO-R item 4, and the answer is NO — see setOutflowRhoCorrection. Default OFF;
  // PECLET_FLOW_OUTFLOW_RHO=1 turns the ablation on process-wide.
  bool outflowRhoCorr_ = [] {
    const char* e = std::getenv("PECLET_FLOW_OUTFLOW_RHO");
    return e && e[0] != '0';
  }();
  CCField cField_;                 // the G=2 registry mirror of the colour field ("C")
  CCField vofCs_;                  // rung V5a: cell fluid fraction on the G=2 block (staggered)
  CCField vofSolidG2_;             // rung V5a: 1 where the cell is SOLID (G=2 mirror), else 0
  bool vofSolidZero_ = true;       // rung V5a: canonical "C" is 0 in solid cells (see setter)
  // rung V5b (WO-S): the static contact angle. Unset => the neutral (90 deg) fill of WO-Q, and the
  // whole V5a battery is byte-identical.
  bool contactAngleSet_ = false;
  double contactAngleDeg_ = 90.0;
  int contactPivot_ = vof::kVofPivotVolume;
  std::vector<double> contactAngleField_;
  vof::WyAdvector vofAdv_;         // the g=3 working block: PLIC + Weymouth-Yue sweeps
  C3 e3_{0, 0, 0};                 // extended extents of that block (n + 2*kVofG)
  double vofCflLimit_ = 0.25;      // Weymouth's proven 3D boundedness bound 1/(2(N-1))
  double sigmaCsf_ = 0.0;          // V4: surface-tension coefficient (0 = the force is off)
  double capillaryCfl_ = 1.0;      // safety factor on the Brackbill capillary dt
  double csfInterfaceEps_ = 1e-8;  // V4: wisp threshold on the curvature's interfacial predicate
  int csfMode_ = 0;                // 0 = balanced-force (production), 1 = cell-interp ablation
  bool kappaFrozen_ = false;       // V4 instrument: do not recompute kappa at the head of the step
  long vofStep_ = 0;               // sweep-permutation counter (6-cycle)
  // --- two-phase open boundaries (rung V-BC, WO-R) ---
  bool vofBcActive_ = false;             // any inflow colour / backflow colour set: arms the whole
                                         // rung (the mask, the ghost rules, the property ghosts)
  bool vofInflowSet_[6] = {false, false, false, false, false, false};
  bool vofBackflowSet_[6] = {false, false, false, false, false, false};
  double vofInflowC_[6] = {0, 0, 0, 0, 0, 0};    // uniform inflow colour per face
  double vofBackflowC_[6] = {0, 0, 0, 0, 0, 0};  // inletOutlet backflow colour per face
  std::vector<double> vofInflowProfRaw_[6];      // the user's (nb, nc) profile, kept so the device
  int vofInflowProfNb_[6] = {0, 0, 0, 0, 0, 0};  // views can be rebuilt when the block is rebuilt
  int vofInflowProfNc_[6] = {0, 0, 0, 0, 0, 0};
  CCField vofInflowProf3_[6], vofInflowProfG2_[6];  // resampled onto the g=3 / G=2 face planes
  int vofProf3Nc_[6] = {0, 0, 0, 0, 0, 0}, vofProfG2Nc_[6] = {0, 0, 0, 0, 0, 0};
  vof::UCField vofOutside_;              // the out-of-domain mask on the g=3 block
  double vofBcVol_[6] = {0, 0, 0, 0, 0, 0};       // signed liquid volume of the LAST step, + = in
  double vofBcVolTotal_[6] = {0, 0, 0, 0, 0, 0};  // running total since enable_vof
  // --- curvature (rung V3, WO-O) ---
  vof::VofCurvature vofCurv_;         // the cascade, on the SAME g=3 block as the colour field
  CCField kappaField_, kappaBranch_;  // the G=2 registry mirrors ("kappa", "kappa_branch")
  vof::VofCurvature::Stats vofCurvStats_{};
  // --- momentum-consistent transport (rung V2b, WO-K) ---
  bool vofMomEnabled_ = false;             // rho^c u_c advected by the SAME geometric fluxes as C
  vof::MomentumConsistentAdvector vofMom_;  // the half-shifted CVs, on the SAME g=3 block
  CCField uAdv_[3];  // the recovered advected velocity on the solver's G=2 block (inner cells)
  double vofRhoG_ = 1.0, vofRhoL_ = 1.0;  // the two phase densities (C = 0 / C = 1)
#ifdef PECLET_FLOW_MPI
  std::shared_ptr<GridHaloTopology<3>> vofHalo_;  // the colour field's OWN g=3 topology
  std::shared_ptr<GridHalo<double>> vofDev_;
#endif
  CCField rho1_, cx1_, cy1_, cz1_;  // g=1 MG-block density bridge + projection face coefficients
  bool porous_ = false;             // volume-averaged continuity d(eps)/dt+div(eps u)=0 (CFD-DEM)
  double pressUnderRelax_ = 1.0;    // omega_p for the incremental pressure accumulation (1.0 = off)
  bool porousDepsDt_ = true;        // include the d(eps)/dt source in the projection RHS. Off ->
                                    // enforce div(eps u)=0 (drop the term, which is jagged/noisy
                                    // because eps is a bare per-cell particle deposit; the noisy
                                    // source can drive the eps-weighted pressure solve unstable).
  CCField epsField_, epsPrev_, eps1_, depsdt_;  // eps^{n+1}, eps^n, g=1 bridge, stored d(eps)/dt
  CCField divAdv_;  // cell div(u) — porous advection-form compensation (see buildRhs*)
  CCField epsRho_;  // rho_eff = eps*rho — eps-conservative porous momentum (updateEpsRho per step)
  // eps-CONSERVATIVE porous momentum + projection pair (default): time term (eps_f rho/dt) u,
  // eps_f rho-weighted advective form, projection c_f = open*(eps rho idt)/(eps rho idt + beta)
  // with correction rho idt/(eps rho idt + beta) grad(phi). False = the legacy plain-u pair
  // (for A/B only; it kinematically drags gas with the moving porosity — energy injection).
  bool porousCons_ = true;
  CCField beta1_;         // g=1 bridge of the drag coeff (semi-implicit-drag pressure)
  bool hasDrag_ = false;  // implicit linear drag (CFD-DEM): beta on the momentum diagonal
  CCField dragBeta_;      // per-cell drag coefficient (added to AC; target beta*u_p rides
                          // the force_* cellForce fields)
};

// The staggered MAC solver — THE flow solver, bit-identical to the pre-policy class. Bindings + the
// kokkos_mpi tests reference this name unchanged.
using IbmSolver = Solver<Staggered>;

}  // namespace peclet::flow

#endif  // PECLET_FLOW_SDFLOW_IBM_HPP
