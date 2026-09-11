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
#include "vof/block_container.hpp"  // VoF Part III rung W0: the per-bubble block container
#include "vof/block_exchange.hpp"   // VoF Part III rung W0: the block gather / scatter
#include "vof/colour_field.hpp"  // VoF rung V2a: the G=2 <-> g=3 bridge + the colour ghost policy
#include "vof/colour_bc.hpp"  // VoF rung V-BC: inflow/outflow/backflow colour + the outside mask
#include "vof/curvature_field.hpp"  // VoF rung V3: the HF curvature cascade + PV fallback
#include "vof/momentum_advect.hpp"  // VoF rung V2b: momentum-consistent rho^c u_c transport
#include "vof/surface_tension.hpp"  // VoF rung V4: balanced-force CSF + the capillary dt
#include "vof/phase_change.hpp"    // VoF Part II rungs P0/P1: mass flux, plane-shift regression
#include "vof/wetting_dynamic.hpp"  // VoF rung V6: dynamic contact angle + hysteresis
#include "vof/energy_advect.hpp"   // VoF Part II rung P2/P3: consistent rho c_p T transport
#include "vof/interface_area_field.hpp"  // VoF Part II rung P3c: the cascade-consistent A_Gamma
#include "vof/marching_cubes_field.hpp"  // VoF Part II rung P3d: the JOINED (marching-tet) A_Gamma

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

  Solver(int nx, int ny, int nz);



  // ==========================================================================================
  // PHYSICAL UNITS — the solver takes the world in the caller's own units
  // (suite/docs/PHYSICAL_UNITS_PLAN.md; Phase 1 = isotropic cells)
  //
  // DERIVATION.  The discrete algorithms in this file are index-native: red-black sweeps,
  // cut-cell apertures, multigrid transfers, the halo, PLIC.  Rather than divide every finite
  // difference by a cell size, the solver keeps computing on the UNIT LATTICE and folds the
  // metric into constants at the API boundary (plan §3.2, option B).  Write
  //
  //     x_a = origin_a + h_a * xi_a      (physical position of the index coordinate xi)
  //     t   = tRef * t'                  (tRef = the first dt the caller sets)
  //     u_a = (h_a / tRef) * v_a         (v = index velocity: cells per tRef)
  //     rho = rhoRef * rho'
  //
  // and substitute into  rho (du/dt + u.grad u) = -grad p + mu lap u + F.  Every term then
  // carries the same factor rhoRef*hRef/tRef^2 when the cells are isotropic (h_a == hRef,
  // Phase 1), and dividing it out leaves EXACTLY the equation this file already solves,
  //
  //     rho' (dv/dt' + v.grad_xi v) = -grad_xi p' + mu' lap_xi v + F'
  //
  // with the boundary conversions
  //
  //     rho'   = rho / rhoRef                       dt'    = dt / tRef
  //     mu'    = mu * tRef / (rhoRef*hRef^2)        (the cell diffusion number)
  //     p'     = p * tRef^2 / (rhoRef*hRef^2)       F'     = F * tRef^2 / (rhoRef*h_a)
  //     v_a    = u_a * tRef / h_a                   (the Courant number)
  //     sigma' = sigma * tRef^2 / (rhoRef*hRef^3)   so p' = sigma'*kappa' with kappa' = kappa*hRef
  //     kappa' = kappa * hRef                       div'   = div * tRef
  //     d'     = d / hRef                           (any length: SDF value, slip length)
  //     xi_a   = (x_a - origin_a) / h_a             (any position)
  //
  // Advection is untouched in index space, so the Koren/TVD and implicit-FOU paths do not change;
  // cut-cell apertures and wall crossings are ratios along an axis and are unit-free.
  //
  // WHY THE REFERENCE SCALES (plan decision D3).  They exist so the STORED float operator
  // coefficients stay O(1) whatever unit system the caller uses: mu' is a diffusion number and
  // rho'/dt' is 1 at the first dt, where dividing mu by a physical h^2 would put 1e7 next to 1e8
  // in float storage (docs/SCALING_ISSUES.md #1).  hRef is fixed at construction, rhoRef at the
  // first set_rho and tRef at the first set_dt, and never move again — no stored state is ever
  // rescaled mid-run.  Because the order of those calls must not matter, the PHYSICAL inputs are
  // kept verbatim (rhoPhys_ / muPhys_ / ...) and `refreshUnitDerived()` re-derives the internal
  // ones whenever a scale is first pinned.
  //
  // IDENTITY.  With no extent at all (`extent=None`, the historical cell-unit API) the solver is
  // never armed: hRef = rhoRef = tRef = 1 and every factor below evaluates to exactly 1.0,
  // whatever rho and dt are.  Multiplying a double by 1.0 is the identity in IEEE-754, so every
  // pre-existing script, test and gallery page is bit-identical (plan §5 gate 1).  With
  // extent == cells, dt = 1 and rho = 1 the armed path reproduces the same 1.0 factors — that is
  // the `units_identity` gate.
  // ==========================================================================================
  struct UnitScales {
    bool physical = false;             ///< an extent was given (the scales are armed)
    double h[3] = {1.0, 1.0, 1.0};     ///< cell size per axis = extent / cells
    double org[3] = {0.0, 0.0, 0.0};   ///< physical lower corner of the GLOBAL inner grid
    double ext[3] = {0.0, 0.0, 0.0};   ///< physical extent of the GLOBAL inner grid
    long cells[3] = {0, 0, 0};         ///< the GLOBAL cell counts the extent spans
    double hRef = 1.0;                 ///< reference length  = min_a h_a (== h_a, Phase 1)
    double rhoRef = 1.0;               ///< reference density = the first set_rho
    double tRef = 1.0;                 ///< reference time    = the first set_dt
    bool rhoRefSet = false, tRefSet = false;

    // Every factor below is exactly 1.0 while `physical` is false.
    double lenToInt() const { return 1.0 / hRef; }
    double lenToPhys() const { return hRef; }
    double velToInt(int a) const { return tRef / h[a]; }
    double velToPhys(int a) const { return h[a] / tRef; }
    double timeToInt() const { return 1.0 / tRef; }
    double rhoToInt() const { return 1.0 / rhoRef; }
    double muToInt() const { return tRef / (rhoRef * hRef * hRef); }
    double pToInt() const { return tRef * tRef / (rhoRef * hRef * hRef); }
    double pToPhys() const { return rhoRef * hRef * hRef / (tRef * tRef); }
    double forceToInt(int a) const { return tRef * tRef / (rhoRef * h[a]); }
    double sigmaToInt() const { return tRef * tRef / (rhoRef * hRef * hRef * hRef); }
    double curvToPhys() const { return 1.0 / hRef; }
    double divToPhys() const { return 1.0 / tRef; }
    double timeToPhys() const { return tRef; }
    double angVelToInt() const { return tRef; }  // omega' = omega*tRef (v = omega x r on both sides)
    // A TOTAL force F = int f dV: f' = f*tRef^2/(rhoRef*hRef) and dV' = dV/hRef^3, so
    // F' = F*tRef^2/(rhoRef*hRef^4); a torque carries one more length.
    double forceTotalToPhys() const {
      const double h2 = hRef * hRef;
      return rhoRef * h2 * h2 / (tRef * tRef);
    }
    double torqueToPhys() const { return forceTotalToPhys() * hRef; }

    // ---- Phase 2 (anisotropic cells): the per-axis metric of the discrete operators ----------
    // doc/anisotropic_metric.md §1.1.  h_a' = h_a/hRef >= 1 (exactly 1.0 on the finest axis),
    // w_a = 1/h_a'^2 <= 1 the pressure-gradient / Laplacian weight of axis a, V' = h_x'h_y'h_z'
    // the cell volume in hRef^3, and `aniso` true iff some h_a' != 1.  Every operator fold below
    // is "today's expression times one of these", so on the isotropic path — where they are all
    // EXACTLY 1.0 — the arithmetic is bit-identical (multiplying an IEEE-754 double by 1.0 is the
    // identity).  APPEND new members here; never reorder the struct.
    double hp[3] = {1.0, 1.0, 1.0};   ///< h_a' = h_a / hRef  (>= 1, exactly 1 on the finest axis)
    double w[3] = {1.0, 1.0, 1.0};    ///< w_a  = 1 / h_a'^2  (<= 1)
    double vol = 1.0;                 ///< V'   = h_x' h_y' h_z'  (cell volume in hRef^3)
    bool aniso = false;               ///< some h_a' != 1 — kernels dispatch the per-axis body

    // ---- Phase 3 (the VoF half; flow/doc/anisotropic_vof.md §9) appends TWO more, and reuses
    // the four above verbatim — the two sessions derived the same metric and the rebase kept one
    // copy.  `hpMax` is the Wendland support scale of the curvature/area fits; `vofMetric()` is
    // the per-axis cell size the container-free VoF kernels take (`core/vof/plic.hpp`), which is
    // `{1,1,1}` — hence the pre-Phase-3 arithmetic exactly — whenever the cells are cubes.
    double hpMax = 1.0;               ///< max_a h_a'
    vof::VofMetric vofMetric() const { return vof::VofMetric{{hp[0], hp[1], hp[2]}}; }
  };

  /// The map between the solver's INDEX coordinates and the coordinate system the analytic scene
  /// is written in.  A scene is in the caller's PHYSICAL coordinates once a domain is armed, and
  /// in the historical cell coordinates otherwise (cell centre (i,j,k) at exactly (i,j,k)) — hence
  /// the half-cell in `a`.  Every field is exactly the identity in cell units, and the struct is
  /// captured BY VALUE into device kernels (no Solver state on device).
  struct SceneMap {
    double a[3] = {0.0, 0.0, 0.0};        ///< scene_a = a[a] + b[a]*xi_a  (xi = global index)
    double b[3] = {1.0, 1.0, 1.0};
    double dToInt = 1.0;                  ///< scene signed distance -> index distance (1/hRef)
    double velToInt[3] = {1.0, 1.0, 1.0}; ///< scene velocity -> index velocity
    double angToInt = 1.0;                ///< scene angular velocity -> index (omega*tRef)
  };
  SceneMap sceneMap() const;



  /// Physical-domain constructor (plan §3.3).  `nx,ny,nz` is this rank's block exactly as in the
  /// cell-unit constructor; `extent`/`origin` describe the GLOBAL domain and `globalCells` the
  /// global cell counts the extent spans (equal to nx,ny,nz single-rank — pass the global grid
  /// under MPI, the same numbers `init_mpi` gets).
  Solver(int nx, int ny, int nz, const std::array<double, 3>& extent,
         const std::array<double, 3>& origin, const std::array<long, 3>& globalCells);



  /// Arm the physical domain.  Call before any property, geometry or field call — it fixes hRef,
  /// which every conversion below is built on.
  void setPhysicalDomain(const std::array<double, 3>& extent, const std::array<double, 3>& origin,
                         const std::array<long, 3>& globalCells);



  /// Phase 2 §7 — refuse an ANISOTROPIC domain in a consumer this phase does not carry.  `what`
  /// names the entry point AND the phase/commit that lifts the refusal; this appends the three
  /// spacings and the metric they give, so the message is actionable without a debugger.  A no-op
  /// (and never even formats a string) on the isotropic path, which is every cell-unit run.
  ///
  /// ADMITTED on an anisotropic domain after commit C4 (doc/anisotropic_metric.md §7): the
  /// staggered `Solver` AND `SolverColocated` with sampled or scene geometry, constant or variable
  /// properties, variable density, porous continuity with or without implicit drag, every
  /// domain-BC type, every pressure driver and bottom, the velocity multigrid, scalar transport,
  /// the collocated face-interpolation modes, `hydro_force_torque`, `hydro_force_torque_reaction`
  /// (moving geometry included, since C4b resolved E3), and MPI.  Phase 3 then admitted
  /// `enable_vof` too (flow/doc/anisotropic_vof.md), so nothing in the solver refuses an
  /// anisotropic domain today; this helper stays for the next consumer that needs to.
  void requireIsotropic(const char* what) const;



  bool hasPhysicalDomain() const;


  const UnitScales& unitScales() const;


  /// Cell size per axis (equal on every axis unless the Phase 2 anisotropic path is armed --
  /// see `unitScales().aniso`). 1,1,1 without a physical domain.
  std::array<double, 3> spacing() const;


  /// Physical lower corner of the GLOBAL inner grid.
  std::array<double, 3> domainOrigin() const;


  /// Physical extent of the GLOBAL inner grid (cell counts without a physical domain).
  std::array<double, 3> domainExtent() const;


  /// The GLOBAL cell counts (this rank's block single-rank).
  std::array<long, 3> globalCells() const;


  /// Physical cell-centre coordinates of THIS rank's inner block along `axis` (nx/ny/nz values).
  /// The grid `set_solid` expects an SDF sampled on: meshgrid these three and evaluate.
  std::vector<double> cellCentres(int axis) const;



  /// Re-derive every internal (index-unit) quantity from the stored physical inputs.  Idempotent
  /// and order-free: it reads only the phys_ mirrors and writes only the internal members, so the
  /// caller may set properties before or after the domain, and before or after each other.
  void refreshUnitDerived();



  /// Hand the anisotropic cell metric to every VoF driver that exists (Phase 3).
  ///
  /// The drivers are created lazily (`enableVof`, `enableVofBlocks`, `setPhaseChangeArea`, ...),
  /// so this is called BOTH from `refreshUnitDerived()` — whenever a scale moves — and at the end
  /// of each driver's own set-up. Every driver defaults to the unit metric, so a driver that is
  /// never reached behaves exactly as before.
  void pushVofMetric();



  // (Re)allocate every per-block buffer for a local inner block of nx*ny*nz. Called by the
  // constructor and by redistribute() after a re-decomposition changes this rank's block size.
  void allocateBlock(int nx, int ny, int nz);



  /// Fluid density, in the caller's units. The FIRST call pins the reference density rhoRef when
  /// a physical domain is armed (the internal density is then exactly 1).
  void setRho(double r);


  /// Dynamic viscosity, in the caller's units. Internally the cell diffusion number
  /// mu*tRef/(rhoRef*hRef^2).
  void setMu(double m);


  void setDt(double d);


  /// Body force per unit volume, in the caller's units (e.g. a mean pressure gradient, or rho*g).
  void setBodyForce(double fx, double fy, double fz);


  void setVelocityIterations(int it);


  // Momentum tolerance stop: end the RB-GS loop once the swept colour's max increment has dropped
  // to rtol of the first sweep's (GS contracts geometrically, so the increment tracks the error).
  // rtol = 0 (default) keeps the legacy fixed-count loop byte-identical. Easy regimes (small
  // nu*dt/dx^2) exit after ~3-5 sweeps; stiff regimes run to the velIters_ cap unchanged. The
  // check is fused into the sweep kernel (no extra memory pass) and the stop decision is
  // rank-uniform (MPI max) so distributed halo exchanges stay in lockstep.
  void setVelocityTolerance(double rtol, int minIters);


  long lastMomentumSweeps() const;


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
  void setVelocityResidualTolerance(double rtol);


  // Velocity-MG AUTO rule (applies when set_velocity_multigrid was never called): under MPI, once
  // the block is small enough that the momentum RB-GS is halo-latency-bound, take the V-cycle
  // instead (1-2 cycles/component == 2-4 exchanges against 8-9 sweeps x 2). Measured crossover on
  // the FoxBerry bed: RB-GS 2.91 s vs MG 3.32 s/step at 147k cells/rank, MG 0.834 vs 0.844 at
  // 37k; threshold cellsPerRank (default 65536 cells per rank, 0 = never), and only
  // for global problems of at least minGlobalCells (8M) -- small grids split
  // across ranks keep RB-GS so a distributed run stays exactly the single-rank one.
  void setVelocityMultigridAuto(long cellsPerRank, long minGlobalCells = -1);


  // The tolerance actually in force (resolves the follow-the-pressure default).
  double velocityResidualTolerance() const;


  // max over components of max|r|/max|b| at exit of the last step's momentum solves (residual
  // mode only; -1 otherwise).
  double lastMomentumResidual() const;


  // Pressure-solve mean-removal scope: "fine" (default — drops the interior-level / post-matvec
  // nullspace projections, ~3x fewer global-reduction latency hits per Krylov iteration; measured
  // winner of the at-scale ablation, iteration counts identical) or "all" (legacy). See CutcellMG.
  void setPressureMeanRemoval(bool all);


  void setPressureIterations(int it);


  void setAdvection(bool on);

  // explicit high-order advection (default SOU)
  // High-order advection scheme for the (explicit, or deferred-correction) flux: 0 = second-order
  // upwind (SOU, default — 2nd order at smooth extrema too); 1 = Koren TVD (monotone limiter, the
  // legacy CUDA scheme). Only matters when advection is enabled; FOU stays the deferred-correction
  // base.
  void setAdvectionScheme(int s);


  // Implicit-FOU deferred-correction advection (CUDA set_implicit_advection): solve the
  // first-order-upwind part of advection implicitly (in the velocity operator) + keep (Koren-FOU)
  // explicit in the RHS -> unconditionally stable for advection (high Re / large dt). Requires the
  // IBM stencil (rebuilt per Picard iteration with the FOU term); the domain-BC path needs
  // velocity-MG (separate milestone).
  void setImplicitAdvection(bool on);


  // Picard outer iterations over the step (CUDA set_outer_iterations): the advecting velocity is
  // lagged at the current iterate u^k while the time base stays u^n. iters>=1; tol>0 stops early on
  // max|du| < tol.
  void setOuterIterations(int iters);


  void setOuterTolerance(double tol);


  long lastOuterIterations() const;


  // Velocity (momentum) multigrid for the IBM diffusion solve (CUDA set_velocity_multigrid): the
  // STAIRCASE coarse operator (exact == RB-GS, stiff-stable at large dt). Call before set_solid;
  // built at geometry time.
  void setVelocityMultigrid(bool on, int levels, int vcycles);


  bool velocityMultigridActive() const;


  // Enable the agglomerated GraphAMG bottom solve in the pressure MG: the coarsest level is solved
  // by a mesh-agnostic algebraic multigrid on the operator gathered to rank 0 --
  // decomposition-agnostic, so multilevel convergence works under a WEIGHTED ORB (where the
  // geometric coarse levels can't cleanly coarsen). Applied at the next set_solid / geometry
  // rebuild.
  // Coarse-level (bottom) solve policy: 0 smoothed bottom (default), -1 auto (agglomerate exactly
  // when the geometric hierarchy cannot reach a small enough coarsest grid), 1 always. See CutcellMG.
  void setPressureBottomMode(int mode);

  // Coarse-level telescoping of the pressure multigrid (mac_cutcell_mg.hpp Telescope): when a
  // per-rank block turns odd, merge ORB siblings onto fewer ranks and keep coarsening instead of
  // stopping. Multi-rank only; a no-op single-rank. Takes effect at the next init_mpi/set_solid.
  void setPressureTelescope(bool on);


  bool pressureTelescope() const;


  // Force a telescope at that level even where in-place coarsening is legal (tests: compare the
  // two hierarchies on one problem); -1 = the trigger decides. Set before geometry.
  void setPressureTelescopeForceLevel(int level);


  int pressureTelescopeCount() const;



  void setPressureGraphAmg(bool on);


  void setPressureLevels(int levels);

  // MG depth (CUDA default 4)
  // Backflow stabilization at outflow faces (Bazilevs 2009 / Esmaily-Moghadam 2011): beta in [0,1]
  // scales the dissipative outflow term that prevents backflow divergence (0 = off). Default 0.2.
  void setBackflowStab(double beta);


  // Deferred-correction advection: on (default) = implicit FOU operator + explicit (HO - FOU)
  // high-order correction (2nd order; HO = SOU by default, or Koren TVD via set_advection_scheme).
  // off = pure implicit FOU (1st order, more dissipative, unconditionally stable) -- useful for
  // very sharp shear layers where the (unlimited SOU) explicit correction overshoots and
  // destabilizes.
  void setDeferredCorrection(bool on);


  // Chebyshev pressure driver (CUDA set_pressure_chebyshev): communication-light alternative to
  // MG-PCG -- Chebyshev semi-iteration preconditioned by one symmetric V-cycle, no per-iteration
  // global dot-products. Spectral bounds of M^{-1}A are estimated once (lazily) on the first solve
  // and reused every step.
  // Selecting it clears the competing FCG selection (the three Krylov drivers are mutually
  // exclusive, last set wins); `on = false` only deselects Chebyshev, so the solve falls back to
  // whatever else is selected — FCG if set, otherwise MG-PCG.
  void setPressureChebyshev(bool on, int maxit, double rtol);


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
  void setPressurePcg(bool on, int maxit, double rtol);


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
  void setPressureFcg(bool on, int maxit, double rtol);


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
  void setGhostProjection(bool on, int matrixOrder = 2, int rhsOrder = 2);


  // Analytic-SDF capability: EXACT wall-crossing fractions overriding the linear-interp theta in
  // BOTH the momentum cut-cell overlay and the ghost-projection closures. t is a flat array of
  // size 9*nx*ny*nz, blocks ordered [(c*3 + k)]: for velocity component c, t[(c*3+k)*n + i] is
  // the exact crossing fraction in (0,1) from component c's staggered point at inner cell i
  // toward its +k-axis neighbour point, NaN where the segment has no wall crossing. Computed in
  // Python from the analytic geometry (e.g. line-sphere intersection). Call BEFORE set_solid;
  // pass an empty array to clear. Single-rank only.
  void setExactCrossings(const std::vector<double>& t);


  // Analytic-SDF capability: EXACT face-openness (aperture) fields overriding the sampled-SDF
  // ccFractionCore openness the cut-cell projection uses. Inner arrays (flat x-fastest,
  // nx*ny*nz); ox[i] = fluid area fraction of the -x face of cell i, etc. Call BEFORE set_solid.
  void setOpennessOverride(const std::vector<double>& ox, const std::vector<double>& oy,
                           const std::vector<double>& oz);


  // Incremental-rotational pressure (CUDA set_incremental_pressure, default ON): the predictor
  // carries -grad(P^n) and the physical pressure is accumulated rotationally P += (rho/dt)*phi -
  // mu*div(u*). OFF => classical non-incremental Chorin (no -grad(P^n) predictor; P derived on
  // demand as (rho/dt)*phi).
  void setIncrementalPressure(bool on);


  // Pressure warm-start (CUDA set_pressure_warmstart, default OFF): seed each cut-cell pressure
  // solve from the previous step's projection potential (consecutive phi's are similar along a
  // steady march -> a more converged phi per fixed solver budget) instead of zeroing the initial
  // guess.
  void setPressureWarmstart(bool on);


  // Collocated cut-cell treatment of the approximate projection (no effect on the staggered path).
  // The public API is the string form setCollocatedScheme(); the integer mode is the C++ switch
  // behind it and the Python developer tier's `diagnostics.set_face_interp(5|6)`:
  //   0  "plain"        plain ½/½ cell->face averaging + central-difference -grad(P): a consistent
  //                     adjoint pair of the WRONG geometry (wall at the solid neighbour's centre),
  //                     first-order drag at curved walls. Legacy, kept for reproducing results.
  //   9  "gauge-exact"  the aperture constraint (unchanged, throat-safe, symmetric MG-PCG) with the
  //                     directional gpCenterGrad replacing the two operators measured to be O(1)
  //                     at cut cells -- the -grad(P) predictor and the projection's cell
  //                     correction. SECOND ORDER on two periodic sphere beds (2.36-2.89 over
  //                     R=5..8, +0.08 % of k_inf at R=16) where mode 0 is first order; the
  //                     cheapest scheme measured (doc/collocated_paper_plan.md).
  //   5/6/7 "embed"     the Basilisk embed.h port (commits db5b4aa/f5fde8c/6d412ec/03a71c6,
  //                     doc/history/collocated_embed_port_plan.md): the FV momentum operator with
  //                     the true-normal dirichlet_gradient wall drag applied as a defect
  //                     correction on the IBM matrix (5 = that alone, on the wall-aware face map
  //                     with the transpose-gradient pressure force; 6 = plus the openness-weighted
  //                     -grad(P) predictor and cell correction, on the plain face map; 7 = 6 on
  //                     the wall-aware face map and the solid-cut-cell sliver mask -- the
  //                     COMPLETE port, the public "embed"). The live candidate for removing the
  //                     collocated accuracy ceiling; 5 and 6 are its two intermediate rungs.
  // The "ghost" scheme (setGhostProjection) is not a face-interp mode: it owns the operators the
  // modes replace and forces mode 0 underneath itself.
  // Modes 1, 2, 3, 4, 10, 11, 12 and 13 (pure ablations, the FV-constraint variants and the
  // adjoint-aperture family) and the "gauge-2a" one-sided gradient branch were DELETED with their
  // kernels at 1.0.0 (suite/docs/QUALITY_PLAN.md F); the record of what they measured is in
  // doc/history/collocated_*.md.
  void setFaceInterp(int mode);


  int faceInterp() const;


  // Preferred API for the collocated projection scheme (the strings the Python API takes):
  //   "ghost"        the fluid-only constraint scheme (the AUTO default where supported)
  //   "gauge-exact"  aperture constraint + directional (gauge-exact) pressure gradient (mode 9)
  //   "plain"        the legacy plain-average / central-difference path (mode 0, first order)
  //   "embed"        the complete Basilisk embed.h port (mode 7)
  void setCollocatedScheme(const std::string& name);


  // PM I ablation (Guy-Fogelson): keep the incremental predictor -grad(P^n) but accumulate
  // P += (rho/dt)*phi WITHOUT the rotational -mu*div(u*) term (constant-mu path only; the
  // variable-mu branches keep their own treatment). Default true = shipped behaviour.
  void setRotationalPressure(bool on);


  // Rotational under-relaxation: P += ct*phi - w*mu*div(u*). w = 1 is the shipped Timmermans
  // update; w = 0 is PM I. Shrinking w shrinks the O(1) velocity->pressure off-diagonal that
  // makes the cell-centered approximate projection marginally unstable (Guy-Fogelson eq. 92-94:
  // the destabilizing-perturbation threshold scales ~1/w), at the cost of ~1/w slower pressure
  // relaxation of the smooth modes at large dt. phi = 0 stays the unique fixed point for ANY
  // w > 0, at every dt including dt -> infinity.
  void setRotationalWeight(double w);


  // Wall-banded rotational blend (Frank, 2026-08-20): see the press_wallblend kernel. w0 = 0
  // (default) disables; typical w0 ~ 0.3-0.5. Composes with setRotationalWeight (uniform factor).
  void setRotationalWallWeight(double w0);


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
  void setApertureOrder(int order);


  int apertureOrder() const;


  /// A0 — wall velocity (not zeros) in the advection inputs' masked rows. DEFAULT true.
  void setAdvectionWallVelocity(bool on);


  bool advectionWallVelocity() const;


  /// Communication-avoiding red-black smoothing: kCaMomentum | kCaMg. DEFAULT both.
  /// Must be set BEFORE init_mpi (the momentum half is latched with the halo topology).
  void setCommAvoiding(int mask);


  int commAvoiding() const;


  /// The anisotropic-coarsening aspect threshold theta shared by the pressure and velocity
  /// multigrids (doc/anisotropic_metric.md §5.1). DEFAULT 2.0; only read on an anisotropic metric.
  void setMultigridAspectThreshold(double theta);


  double multigridAspectThreshold() const;


  /// Throw instead of reporting when the pressure preconditioner returns a non-finite
  /// correction (ISSUES sweep item 6). DEFAULT false.
  void setPressureStrict(bool on);


  bool pressureStrict() const;


  /// Neumann (zero-gradient) coarse ghost on wall/inflow faces before the pressure MG's
  /// prolongation — the WO-H symmetry repair. DEFAULT true; false is a measurement ablation.
  void setPressureCoarseGhost(bool on);


  bool pressureCoarseGhost() const;


  /// The `set_pressure_bottom("auto")` criterion: agglomerate once the coarsest GLOBAL grid
  /// exceeds this many cells on any axis. DEFAULT 4.
  void setPressureBottomExtent(int cells);


  int pressureBottomExtent() const;


  /// How the shared level-0 MPI decomposition is built. MUST be set before init_mpi, and the
  /// same values must be handed to `flow.mpi_block` — both derive the same partition.
  void setDecomposition(int levels, double maxImbalance = 1.05);


  int decompositionLevels() const;


  double decompositionMaxImbalance() const;


  void setFluidOnlyConstraint(int mode);


  // Filtered rotational update (experimental): P += ct*phi - mu*S(div u*), S = one mask-aware
  // axis-wise (1,2,1)/4 smoothing pass per axis (one-sided 1/2(d_i+d_nbr) toward the open side at
  // a solid-centered neighbour, identity when sandwiched). S annihilates the axis checkerboard
  // including AT wall-adjacent cells; for smooth fields S = I + O(h^2). Steady state unchanged.
  void setRotationalFilter(bool on, double eps = 0.05);


  // Seed/restore the velocity state (CUDA set_state / upload_velocity): u/v/w are inner-cell fields
  // (flat x-fastest, size nx*ny*nz); written into the velocity block + ghosts refreshed (periodic
  // wrap).
  // ISSUES sweep item 5. On the COLLOCATED grid the colour transport rides the MAC face field
  // `uf_/vf_/wf_` (the only discretely divergence-free field on that grid), which only exists
  // after a projection. `set_state` / `set_velocity` write the CELL field and used to leave the
  // face field untouched — all zeros on a fresh solver — and `advect_vof` then ran happily:
  // its guard measures the FACE field's divergence and a zero field is perfectly solenoidal. So a
  // benchmark loop written the staggered way produced a "conservative" run in which nothing ever
  // moved (measured on LeVeque 32^3: max|u| = 63.3, max|uf| = 0.0, max|C - C0| = 0.0 after
  // `advect_vof`, no throw and no warning).
  //
  // Seeding the face field from the cell field with the SAME `centerToFace` map `project()` uses
  // makes the kinematic entry point mean the same thing on both grids — and it makes the existing
  // divergence guard MEASURE something: an analytic cell field that is not discretely solenoidal
  // on the faces is now rejected instead of silently accepted. It does not make the face field
  // divergence-free (only a projection does), and it is overwritten by the next `project()`, so
  // nothing downstream changes. Staggered: a no-op (the cell field IS the face field).
  void seedFaceFieldFromCells();


  /// Upload an initial velocity field in the caller's PHYSICAL units (converted per component to
  /// the index velocity the kernels carry; the conversion is the identity in cell units).
  void uploadVelocity(const std::vector<double>& uu, const std::vector<double>& vv,
                      const std::vector<double>& ww);

#ifdef PECLET_FLOW_MPI

  // Multi-rank: this rank's IbmSolver is constructed with its LOCAL block dims (= the
  // BlockDecomposer of the GLOBAL grid for this rank); initMpi wires the g=2 velocity-block halo +
  // the global-origin red-black parity, and switches fillGhosts/maxOpenDivergence + the pressure MG
  // (CutcellMG::initMpi) onto their distributed paths. The caller decomposes first (deterministic
  // ORB) to size the constructor; initMpi re-derives it.
  void initMpi(int gnx, int gny, int gnz, MPI_Comm comm);


  // Shared-decomposition overload: wire the g=2 velocity-block halo from an EXTERNALLY-built ORB
  // (so flow and dem share one BlockDecomposer for coupled runs, and redistribute() can re-init
  // onto a re-decomposed partition). The local block size must already match dec.block(rank).size
  // (set via the constructor / allocateBlock).
  void initMpi(const peclet::core::decomp::BlockDecomposer<3>& dec, MPI_Comm comm);


  // Redistribute the solver's state onto a NEW decomposition (dynamic load balancing). Enumerates
  // the registered fields, moves them from the current block layout to the new one (bit-exact via
  // redistributeGridFields), reallocates every buffer to the new block, re-inits the halo +
  // pressure MG on the new partition, and rebuilds all geometry-derived state (openness / IBM
  // overlay / stencils) from the migrated SDF. Velocity + pressure + SDF (+ any registered
  // scalar/property fields) survive; per-step scratch is rebuilt.
  void redistribute(const peclet::core::decomp::BlockDecomposer<3>& newDec);


  // Redistribute onto the weighted ORB of per-cell weights `w` (global x-fastest, gnx*gny*gnz). The
  // ergonomic Python entry point for load balancing: the caller passes a weight field (e.g. fluid
  // work + gamma*particle_count) and both flow and dem rebuild the SAME deterministic partition
  // from it. No BlockDecomposer object crosses the language boundary.
  void rebalanceByWeights(const std::vector<peclet::core::Real>& w);

#endif

  // per-face domain BC {face 0..5 = -x,+x,-y,+y,-z,+z}: type 0=periodic,1=no-slip
  // wall,2=Dirichlet/inflow,3=outflow,4=free-slip/symmetry (zero normal velocity, zero normal
  // derivative of the tangential components, pressure Neumann like a wall; vx/vy/vz ignored).
  // Call order (QUALITY_PLAN F milestone iv, refined 2026-09-10): the TYPE decides the operator
  // openness (wall/inflow closed, outflow open, periodic) and the stencil path, so only a TYPE
  // CHANGE has to precede the geometry; a VALUE update on a face whose type is unchanged (ramping
  // an inflow jet, a lid's tangential speed) is allowed at any time and takes effect the next
  // step -- see the refresh below.
  void setDomainBc(int face, int type, double vx, double vy, double vz);


  // per-position inlet velocity profile on `face` (CUDA set_domain_bc_profile): prof is (nb,nc,3)
  // on the inner grid of the face's two perpendicular axes; sets the face to inflow (type 2).
  // Resampled (clamp) to the ghost-inclusive face grid so the BC kernel indexes it directly by face
  // position. Same call-order rule as set_domain_bc: allowed after the geometry only when the
  // face is ALREADY inflow (a profile VALUE update), since that is the one case that does not
  // change the type the operators were built with.
  void setDomainBcProfile(int face, const std::vector<double>& prof, int nb, int nc);


  // Resample the stored raw inlet profile of `face` onto THIS block's ghost-inclusive face grid.
  void resampleBcProfile(int face);


  // all-fluid + domain-BC pressure (CUDA set_pressure_geometry): same path as set_solid with an
  // open SDF.
  void setPressureGeometry(const std::vector<double>& sdfInner);



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
                bool periodic = false);



  /// The resolved centre of rotation of instance i and whether it is pinned (explicit) or
  /// follows the body's translation.
  std::array<double, 3> instanceCenter(int i) const;


  bool instanceCenterPinned(int i) const;



  /// Rigid-body motion of one scene instance (Layer 3 rung 2). `lin` is the body's linear
  /// velocity, `ang` its angular velocity about its own centre -- both in CELL UNITS PER TIME, the
  /// same units the velocity field carries, since the scene lives on the global inner grid.
  ///
  /// Setting any nonzero component switches the solver onto the moving-geometry path: the
  /// momentum operator's no-slip datum becomes the local wall velocity instead of zero (rung 2)
  /// and the cut-cell projection gains the wall's own volume flux (rung 3). With every component
  /// zero the solver stays on the static path, bit for bit.
  void setInstanceMotion(int i, const std::array<double, 3>& lin,
                         const std::array<double, 3>& ang, const double* center = nullptr);



  /// Move one instance (Layer 3 rung 4). Takes effect at the next rebuild_geometry() -- the SDF
  /// field, the cut-cell overlay, the apertures and the pressure operator are ALL derived from the
  /// instance transforms, so a transform change without a rebuild would leave the solver running
  /// on the old geometry with a new wall velocity, which is worse than either.
  ///
  /// The centre of rotation FOLLOWS the body: it is re-anchored to the new translation unless the
  /// caller pinned one explicitly through set_instance_motion.
  void setInstanceTransform(int i, const std::array<double, 3>& translation,
                            const std::array<double, 4>& quat);



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
  void rebuildGeometry();



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
  void seedFreshCells(CCField mOld[3]);



  /// Fresh-cell policy for moving geometry. true (DEFAULT) = seed with the local wall velocity;
  /// false = inherit whatever the solid held, which is what shipped before 2026-08-30. Inert when
  /// nothing moves, so a static run is bit-identical either way. See seedFreshCells.
  void setFreshCellSeed(bool on);


  bool freshCellSeed() const;



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
  void refreshWallVelocity();



  /// True when at least one instance carries a nonzero velocity -- i.e. the moving-geometry paths
  /// are live. Everything downstream keys off this, so a driver can assert it.
  bool hasMovingInstance() const;


  int sceneInstanceCount() const;



  /// Rung 3 on/off. ON (the default) is the correct physics: a rigid body sweeping through a cut
  /// cell injects a wall flux the projection must balance. The switch exists so the Galilean gate
  /// can EXHIBIT the failure the term fixes rather than assert it -- turning it off leaves rung 2's
  /// wall velocity in the momentum operator and a projection that wrongly forces div_open(u) = 0.
  void setWallFluxDivergence(bool on);


  bool wallFluxDivergence() const;



  bool hasScene() const;



  /// Per-inner-cell owning instance (Layer 3 rung 1), x-fastest, -1 where no scene has been
  /// sampled yet. Host copy; the device field is what the solver kernels read.
  std::vector<int> getCutOwner() const;



  /// Sample the scene onto this rank's inner grid and install it as the solid, entirely on device
  /// -- no nx*ny*nz float64 host round trip, and correct on every rank.
  void setSolidFromScene(bool cutcellPressure);



  /// A MOVING instance whose surface produces no fractional face aperture has no path for its
  /// wall velocity into the momentum operator: the no-slip datum enters ONLY through the cut-cell
  /// fold, so a box face sitting exactly on a grid plane (or a body smaller than a cell) behaves
  /// as a STATIONARY wall and `set_instance_motion` is silently inert -- the shear-driving plates
  /// of the Jeffery-orbit page at y = 16.0 produced max|u| = 0 (peclet-examples ISSUES.md). Count,
  /// per moving instance, the inner cells it owns that touch a fractional aperture; warn on zero.
  void checkMovingInstancesAreCut();


  /// Per instance: cut rows of the momentum operator (all three components) at inner points this
  /// rank owns for it, recounted by set_solid_from_scene / rebuild_geometry when any instance
  /// moves (global under MPI; empty when nothing moves). Zero for a moving instance = its wall
  /// velocity is silently inert.
  std::vector<long> movingInstanceCutCells() const;


  /// Per instance: staggered points where the sampled sdf is exactly zero (see the warning).
  std::vector<long> movingInstanceDegeneratePoints() const;



  /// Cells on this rank whose solid/fluid sign was set by a periodic IMAGE of an instance wider
  /// than the box (see setSolidFromScene); 0 when no instance is that wide or the images agree.
  long periodicImageOverlapCells() const;



  /// EXACT wall crossings straight from the scene, on device, on every rank -- the in-solver
  /// replacement for set_exact_crossings + scripts/exact_apertures_spheres.py.
  ///
  /// t[c][a](i) = the fraction in (0,1) along the unit segment from component c's staggered point
  /// at inner cell i toward i + e_a at which the scene's SDF changes sign; NaN where the segment
  /// does not cross (the consumer falls back to the linear-interpolated theta). Bisection, NOT
  /// Newton: contract 2 of the design note only guarantees SIGN correctness for the bound-only
  /// leaves (ellipsoid, superquadric, CSG seams), and a Newton step on a non-distance field can
  /// leave the bracket entirely.
  void setExactCrossingsFromScene();



  /// Host entry point: upload the inner SDF once and delegate. Kept so every existing caller and
  /// the Python binding are unchanged.
  void setSolid(const std::vector<double>& sdfInner, bool cutcellPressure);



  /// Build the three per-component Robust-Scaled cut-cell overlays + solid masks from the CURRENT
  /// `sdf_` (extracted verbatim from setSolidDevice so that a wall-slip change can rebuild the
  /// closure without re-running the whole geometry setup). `resetU` zeroes the velocity, which the
  /// geometry path wants and a pure closure change must NOT do.
  void buildVelocityOverlays(bool resetU);



  /// Mirror a cell-centred geometry field about every rank-owned FREE-SLIP (type 4) domain face
  /// (the symmetric extension the BC asserts). No-op without a type-4 face.
  void mirrorSdfSlipFaces(CCField f);


  /// Device entry point (Layer 2): the inner SDF is ALREADY on device, so geometry never
  /// round-trips through the host. This is the body every set_solid path shares.
  void setSolidDevice(CCField din, bool cutcellPressure);


  // ---- setSolidDevice stages (QUALITY_PLAN G.1): pure cut-and-paste, each a contiguous
  // block of the original function sharing only member fields and a CCExec. ----
  void setSolidSelectScheme();


  void setSolidUploadSdf(CCField din);


  void setSolidBuildOverlaysAndStencils();


  void setSolidVelocityMgAuto();


  void setSolidInitVelocityMg();


  void setSolidBuildOpenness();


  void setSolidStarOverlay();


  void setSolidGhostProjectionOverlay(CCField din);


  void setSolidInitPressureMg();


  bool geometryBuilt() const;


  void requireNoGeometry(const char* who) const;



  void step();



  /// OUTFLOW REVERSAL CENSUS -- the regime in which the zero-gradient (do-nothing) outflow is
  /// only conditionally energy-stable. Over every rank-owned outflow face plane (the boundary
  /// normal-velocity plane; the tangential components are the boundary-adjacent inner cell's):
  ///   maxReverse   = max(0, -u.n)                        the largest reversed normal velocity,
  ///   fraction     = reversed faces / outlet faces,
  ///   energyInflux = sum_reversed rho |u.n| |u|^2 / 2    (per unit area; h = 1).
  /// The last is the kinetic-energy production the do-nothing outlet admits where the flow
  /// re-enters (Esmaily-Moghadam, Bazilevs & Marsden 2011): a reversed face advects the ghost
  /// value -- with the zero-gradient ghost, the boundary cell's own -- back in, so the
  /// outlet-adjacent momentum row has an advective source with no sink. The backflow
  /// stabilization (applyBackflowStab) adds beta rho |u.n| to that row's diagonal, i.e. removes
  /// beta rho |u.n| |u_n|^2 of it; the analysis' unconditional bound is beta >= 1/2. Measured
  /// on the BFS whose bubble reaches the outlet (S = 16, Re_S = 800, x_r = L, 6000 steps): a
  /// transient excursion of max|u| to 1.28x the inlet peak ON the outlet column, bounded and
  /// finite with beta = 0 (1.909) and with the default 0.2 (1.897) alike -- so at that Re the
  /// stabilization does not decide boundedness; the census is what tells you the regime is
  /// active. Distributed: reduced over the communicator (every rank must call it).
  struct OutflowBackflow {
    double maxReverse = 0.0, fraction = 0.0, energyInflux = 0.0;
    long reversed = 0, total = 0;
  };
  OutflowBackflow outflowBackflow();


  // velocity component c (0=u,1=v,2=w) on the inner cells, flat x-fastest [nx*ny*nz].
  std::vector<double> getVelocity(int c);


  /// Write a component's inner velocity from a host vector (x-fastest, inner region) and
  /// re-impose the solid mask. An initial-condition hook (e.g. a uniform stream around a fixed
  /// body — the Galilean twin of a towed one); u^n is taken from the live field at step start.
  void setVelocity(int c, const std::vector<double>& v);


  // The divergence-free FACE velocity component (collocated: the projected MAC face field
  // uf_/vf_/wf_, exactly div-free; staggered: C[c].u already lives on the faces). For a periodic
  // bed its mean is the momentum-balance superficial velocity, unperturbed by the openness-aware
  // cell gradient correction (projectCorrectCenter) that biases the cell-field mean at cut cells.
  std::vector<double> getFaceVelocity(int c);


  // TEMP DIAGNOSTIC: the face openness (fluid area fraction) used by the cut-cell projection.
  // component c: 0 -> ox_ (low -x face of each inner cell), 1 -> oy_, 2 -> oz_. Grid-independent
  // (built once from the SDF). Exposed to compare the open-weighted superficial flux against the
  // raw velocity mean.
  std::vector<double> getOpenness(int c);


  // Diagnostic read-out of the ASSEMBLED momentum-operator diagonal of component c — the float
  // stencil `AC` after the diffusion build, the Robust-Scaled cut-cell bake and, under implicit
  // drag, `addDragDiagonal`'s face drag beta_f — as an x-fastest (nx,ny,nz) inner-region host
  // buffer. Read-only; no solver state is touched. Added for WO-I's
  // `tests/kokkos_mpi/test_dragbeta_ghost_mpi.cpp`, which must gate the FACE drag mean where it is
  // formed: on a periodic box the projection homogenizes a single bad plane into a uniform mean
  // shift of the velocity, so a velocity-only gate sees THAT the drag was wrong but not WHERE.
  std::vector<double> getMomentumDiagonal(int c);


  // The openness whose face fluxes the PROJECTION conserves: the binary (COUPLED) openness in
  // ghost-projection mode (oxb_ — the geometric ox_ stays a diagnostic there), the geometric
  // cut-cell openness otherwise. This is what flux bookkeeping downstream of the solve must use
  // (e.g. peclet.pnm's extract_network_flow): sum(o_proj*u*A) over a cell's faces IS the
  // discrete divergence the projection drives to zero.
  std::vector<double> getOpennessProj(int c);


  std::vector<double> getPressure();


  // WO-R: the divergence of the field the projection ACTUALLY produced, outflow correction
  // included. `maxOpenDivergence()` below re-imposes the zero-gradient outflow face before
  // measuring (its own comment says so) — which both destroys `bcCorrectOutflow`'s correction as a
  // side effect and reports the divergence of a field the solver never used. On an open-boundary
  // two-phase box that artefact is the dominant number: measured 5e-3, flat in the iteration
  // count, flat in the density ratio and bit-identical in a `-DPECLET_FLOW_OPERATOR_DOUBLE` build —
  // i.e. not a solver residual at all. This sibling fills the ghosts with `doOutflow = false`,
  // exactly as `step()` does after `project()`, and leaves the velocity field alone.
  //
  // Kept as a SIBLING rather than a change of default: every recorded open-boundary number in the
  // repo was taken with the mutating one, and re-baselining them is not this work order's call.
  double maxOpenDivergenceProjected();


  /// The same diagnostic in INDEX units (per tRef), which is what the solver's own guards read.
  double maxOpenDivergenceProjectedInternal();


  double maxOpenDivergence();


  /// The same diagnostic in INDEX units (per tRef), which is what the solver's own guards read.
  double maxOpenDivergenceInternal();


  // Residual of the volume-averaged continuity, max|div(open*eps*u) + d(eps)/dt| — the quantity the
  // porous projection actually drives to zero (NOT the velocity divergence, which is -d(eps)/dt !=
  // 0 in a fluidizing bed). Meaningful only with set_porous_continuity(True); returns 0 otherwise.
  double maxPorousResidual();


  long lastPressureIterations() const;


  // The pressure multigrid's per-level coarsening ratio, one {rx, ry, rz} per level
  // (doc/anisotropic_metric.md §5).  On an isotropic domain this is today's table; on a stretched
  // one the aspect rule defers an axis while it is at least the aspect threshold (2) times
  // coarser than the finest coarsenable one.  Empty until the cut-cell operator exists.
  std::vector<std::array<int, 3>> pressureMgLevelRatios() const;


  // ISSUES sweep item 6: did the last pressure solve break down (non-finite
  // preconditioner output / recurrence scalar)? A failing solve also reports the
  // iteration cap through `lastPressureIterations()`.
  bool pressureSolveFailed() const;


  // Per-phase wall times of the last step() in seconds, THIS RANK (device-fenced at each phase
  // boundary): predictor = ghost fills + RHS/advection/stencil builds, momentum = the per-component
  // implicit-diffusion solves, projection = the cut-cell pressure projection; step = the whole
  // step() (remainder = BC re-imposition, Picard bookkeeping, scalars). The allreduce pair is the
  // pressure solve's global-reduction tax (time in / count of MPI_Allreduce; 0 single-rank).
  double lastStepSeconds() const;


  double lastPredictorSeconds() const;


  double lastMomentumSeconds() const;


  double lastProjectionSeconds() const;


  // ---- WO-V9: the VoF pipeline's own per-stage timers ----------------------------------------
  //
  // `set_vof_timing(True)` arms them; they are OFF by default and, when off, cost one predictable
  // branch per stage and no fence. A phase boundary on a device backend has to fence or the
  // queued work of one stage is billed to the next — the same rule `phaseTick()` already applies
  // to the step's three coarse phases — so the boundaries fence WHEN ARMED and only then. A fence
  // reorders nothing and computes nothing: the gate is that the same run with timing on and off
  // produces bit-identical fields (`tests/kokkos/test_vof_timing.cpp`).
  //
  // The stages are cumulative over steps since the last reset; `steps` counts step() calls with
  // the instrument armed, so every number divides down to a per-step cost.
  struct VofTiming {
    double advect = 0.0;      ///< the colour advection stage as a whole (advectVof)
    double bridge = 0.0;      ///< ... of which: the G=2 <-> g=3 bridges + the C ghost policy
    double momAdvect = 0.0;   ///< the momentum-consistent colour+rho^c u stage (advectVofMomentum)
    double momBridge = 0.0;   ///< ... of which: its bridges
    double curvature = 0.0;   ///< the V3 height-function cascade + the capillary dt check
    double csf = 0.0;         ///< the V4 balanced-force CSF added to the three momentum RHSs
    double phaseChange = 0.0; ///< the Part II phase-change stage (mdot, deposit, regression)
    long steps = 0;
  };
  void setVofTiming(bool on);


  bool vofTiming() const;


  void resetVofTiming();


  const VofTiming& vofTimingReport() const;


  const vof::WyAdvector::Timing& vofKernelTiming() const;


  const vof::VofCurvature::Timing& vofCurvatureTiming() const;


  /// `VofCurvature::useWorklist` — run the height-function cascade and the PV fallback over a
  /// COMPACTED list of the interfacial cells instead of over the whole inner region. See the
  /// binding docstring and the WO-V9 findings for what it is worth and why.
  void setVofCurvatureWorklist(bool on);


  bool vofCurvatureWorklist() const;


  double vofTimingStepSeconds() const;


  double vofTimingPredictorSeconds() const;


  double vofTimingMomentumSeconds() const;


  double vofTimingProjectionSeconds() const;


  /// `WyAdvector::useWorklist` — the compaction of the PLIC reconstruction pass onto the mixed
  /// cells. Pure optimization: off must reproduce the same field bit for bit.
  void setVofWorklist(bool on);


  bool vofWorklist() const;


  double lastPressureAllreduceSeconds() const;


  long lastPressureAllreduceCount() const;


  int nx() const;


  int ny() const;


  int nz() const;



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
  bool implicitAdv() const;


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
  bool bcStencilPath() const;


  // The mixed velocity MG: solid + domain BCs, diffusion-dominated constant-property momentum
  // (implicit advection / variable properties / drag stay on RB-GS: their fine stencils are not
  // approximated by the staircase Helmholtz).
  bool mixedVelocityMg() const;


  // Fill a property field's ghosts for the face means: periodic/halo base, then zero-gradient
  // (copy) on domain-BC (wall/inflow/outflow) faces — a periodic wrap there would bring the wrong
  // layer's value to the wall face (destabilising, especially for the harmonic mean).
  // Distributed: the override is per-face rank-OWNED (`touchesGlobalFace`), exactly as
  // `applyScalarBc` does — the halo fill runs first (and periodic-wraps the global boundary ghost),
  // the BC overwrite wins on the rank that owns the face. The former `if (!distributed_)` guard
  // keyed on the wrong predicate: it dropped the override at EVERY np including 1.
  void fillPropGhosts(CCField f);


  void fillMuGhosts();


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
  void fillCellForceGhosts();


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
  void fillDragBetaGhosts();


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
  void fillPorousEpsGhosts();


  // --- VoF internals (rung V2a, WO-J) ---------------------------------------------------------
  // Allocate the colour field's own g=3 working block and wire its ghost/all-reduce hooks. Called
  // by enableVof() and again by any path that re-sizes the block (redistribute -> initMpi), since
  // the advector's block must track the solver's.
  void buildVofBlock();


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
  // ISSUES sweep item 3. Which domain faces are WETTING WALLS for the colour band fill: a
  // type-1 no-slip or type-4 free-slip face, i.e. an impermeable one, and only while a contact
  // angle is actually set. Returned as a per-face bitmask (bit 2a+s), 0 when the feature is unused
  // -- which is what keeps every existing path byte-identical.
  //
  // A domain wall IS a flat SDF wall sitting exactly on the boundary face; the only reason
  // `set_contact_angle` used to be silently ignored there is that the band fill classifies cells
  // from the colour block's cut-cell GEOMETRY, and an all-fluid solver has none. So the repair is
  // not a new fill rule -- it is a SYNTHESISED geometry for those faces (`applyDomainWallGeometry`)
  // plus the matching wall SDF (`applyContactAngle`), after which WO-S's theta pass, WO-Q's
  // passes 2-3, the branch census and the V6 dynamic angle all run unchanged, with `n_w` coming
  // out of the same central difference and equalling the inward face normal by construction.
  int vofWetWallMask() const;


  // Signed distance (cell units, POSITIVE inside the domain) from a cell centre at GLOBAL index
  // (gx,gy,gz) to the nearest wetting domain wall plane. Container-free so a device lambda can
  // call it. `+inf` when no face is a wetting wall.
  KOKKOS_INLINE_FUNCTION static double vofWallPlaneSdf(int gx, int gy, int gz, I3 gs, int mask);


  void buildVofGeometry();


  // ISSUES sweep item 3: close the colour block's out-of-domain band across every wetting domain
  // wall, so `classifyGeometry` calls those ghost cells SOLID and WO-S's theta pass owns them.
  //
  // The rule is the one the SDF path uses, evaluated on an exact plane: a cell whose centre lies
  // outside the wall has fluid fraction 0, and a face is open only if BOTH of its cells are
  // inside. That closes the boundary face itself (which the flux openness already closes, so
  // nothing in the projection moves) AND all six faces of every band cell, which is what
  // `vofIsSolidCell` requires. Composes with an SDF solid: the solid's eps/openness are simply
  // masked to zero outside the wall.
  void applyDomainWallGeometry(int mask);


  // ISSUES sweep item 3: mark the out-of-domain band SOLID after the classification exchange
  // (see the call site for why the exchange undoes it).
  void imposeDomainWallKind(int mask);


  // ISSUES sweep item 3: the outermost (depth-3) band layer of a wetting DOMAIN wall is the one
  // cell no fill pass can write -- the passes stop at ghost depth 3-k and their 6-point stencil
  // would index outside the g=3 block there -- and the zero-gradient clamp that used to supply it
  // is now skipped (it would wipe the theta band). Continue the band outward instead: depth 3
  // takes depth 2's value, the zero-slope continuation of the theta plane rather than of the first
  // INNER cell's colour. For an SDF wall this cell is an ordinary solid cell and nothing here
  // applies.
  void vofExtendWallBand(CCField f, int mask);


  // Zero the canonical G=2 colour field inside solid cells (see buildVofGeometry).
  void zeroSolidColour();


  // Is axis `a` periodic for the colour field? flow's per-face bc_ is 0 (periodic) on BOTH ends of
  // a periodic axis, so an axis is periodic iff neither of its faces carries a domain BC.
  bool vofAxisPeriodic(int a) const;


  // The colour field's ghost policy on its own g=3 block: halo/periodic base, then zero-gradient
  // (globally clamped) on every non-periodic axis. Zero-gradient is the same policy the material
  // properties get (`fillPropGhosts`) — a wall neither creates nor destroys colour, and the MYC
  // stencil of an inner boundary cell must see a plausible continuation rather than a wrap from the
  // far side of the domain. Prescribing C at an inflow face is a V5+ concern (it needs a flux BC,
  // not a ghost value) and is not offered here.
  void vofFillGhosts(CCField f);


  void vofExchangeRaw(CCField f);


  I3 vofGlobalSize() const;


  // Global index of this block's inner cell (0,0,0). og_ is exactly that (originInclGhost + G).
  I3 vofOrigin() const;


  // Face velocities -> the advector's g=3 block. The advecting field must be the PROJECTED one
  // (see advectVof), and its ghost ring must be valid because the advector reads the `-d` face of
  // the first inner cell, which is a ghost cell's `+d` face. fillVelGhosts is the solver's own
  // halo+domain-BC fill and is exactly what the Picard loop does at the top of every iteration, so
  // calling it here leaves the velocity ghosts in the state the next consumer would have produced.
  void bridgeVelocityToVof();


  // --- rung W0: the block container's view of this rank's patch --------------------------------
  // The owned inner box of every rank, in global cells — the table the block gather/scatter pieces
  // are cut against. It is a function of the CURRENT decomposition, so it is built here rather
  // than inlined at `enable_vof_blocks`: a redistribute has to push the new one through
  // `bindVofBlockPatch` or every piece keeps addressing the previous partition.
  std::vector<vof::VofBox> vofBlockRankBoxes(int size) const;


  // The gather reads the face velocity in the ADVECTOR's high-face convention on the g=3 block,
  // i.e. exactly what `bridgeVelocityToVof` writes, and the scatter writes the union into that
  // same block's colour. `buildVofBlock` reallocates those Views, so the binding is refreshed
  // there — a stale View here would silently gather from freed memory.
  void bindVofBlockPatch();


  // The union colour on the g=3 block -> the canonical registered "C" (+ its ghost policy, the
  // same `fillPropGhosts` rho and mu are derived through).
  void harvestVofBlockUnion();



  // Colour: G=2 registry mirror -> the g=3 working block, then the colour field's own ghost policy.
  // Inner cells only in the copy — the two blocks have different ghost extents and each fills its
  // own (the one bridge; see the enableVof note).
  void bridgeColourToVof();



  // Staggered face stride of velocity component c (the -c face of cell i pairs cells i and i-s).
  long strideOf(int c) const;


  // The face-property accessor for the momentum stencil of component c: mu constant-or-field
  // (arithmetic/harmonic mean), rho constant-or-field (arithmetic face mean for the time diagonal —
  // the same face density the variable-density projection uses).
  // Effective variable density: true varRho, or the eps-conservative porous momentum (rho_eff =
  // eps*rho in epsRho_, refreshed per step by updateEpsRho).
  bool effVarRho() const;


  CCField effRhoField();


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
  bool colocatedFaceForce() const;


  // The AUTO collocated scheme (set in setSolid/setPressureGeometry) picks the GHOST projection when
  // the configuration allows it, and the ghost v1 supports neither variable density nor the V8 face
  // force. `set_density_mode` / `enable_vof` can be called AFTER the geometry, so re-run the same
  // fallback here rather than failing later inside project(). An explicit scheme selection has
  // already cleared colSchemeAuto_ and is left alone (it will hit the loud throw instead).
  void collocatedV8AutoFallback(const char* why);


  void ensureFaceAcc();


  // Guard rail for rung V8's scope. The collocated variable-density / face-force path is validated
  // ALL-FLUID (`set_pressure_geometry`); an immersed solid on it would need the cut-cell face
  // acceleration AND the one-sided (gauge-exact / ghost) closures to agree with the face averaging
  // operator, which is a separate derivation. Fail loudly instead of half-supporting it.
  void requireCollocatedFaceForceScope(const char* who);


  void updateEpsRho();


  VarFaceProps makeFaceProps(int c);


  // Mirror the host motion arrays onto the device (KBs; rebuilt only when a driver changes a
  // body's velocity, not per step).
  void buildSceneQuery();



  void refreshMotionFlag();



  void uploadMotion();



  peclet::core::geom::InstanceMotionView<double> motionView() const;



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
  void buildWallVelocity();



  /// Raw ghost fill of an extended-block field: the rank halo exchange under MPI, the periodic
  /// wrap single-rank. No boundary-condition fold is applied (this is for GEOMETRIC data such as
  /// the wall velocity, not a velocity iterate); non-periodic single-rank ghosts are left as the
  /// kernel computed them.
  void exchangeExtRaw(CCField f);



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
  void addWallFluxDivergence(CCField d);



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
  /// Convert an interleaved result of 3*nInst blocks (force, TORQUE, then further force blocks —
  /// the layout both hydro getters use) from the solver's index units to the caller's. Exactly the
  /// identity in cell units.
  void scaleForceTorque(std::vector<double>& out, std::size_t m) const;



  /// PHASE 2 (anisotropic cells), doc/anisotropic_metric.md §4.4 — ADMITTED since commit C4.  With
  /// `A_a = W_a V'/h_a'` the PHYSICAL fragment area vector in hRef^2 (`W_a = o_{a-} - o_{a+}`, which
  /// is what `A[a]` below holds) and `gu[a][b]` the index-velocity central difference:
  ///
  ///     dFp_a = p' W_a V'/h_a'
  ///     dFv_a = - mu' sum_b W_b (V'/h_b') [ (h_a'/h_b') gu[a][b] + (h_b'/h_a') gu[b][a] ]
  ///
  /// -- the `h_a'/h_b'` pair being the physical strain rate du_a/dx_b + du_b/dx_a written in index
  /// velocities.  Both factors multiply the EXISTING expressions from outside, in the same
  /// association order, so at `V' = h_a' = 1` they are exact 1.0 multiplications and the arithmetic
  /// is bit-identical.  The lever arm `r = rp*sm.dToInt` is already a physical displacement in hRef
  /// units on every axis, and `torqueToPhys` is unchanged.
  std::vector<double> hydroForceTorque();



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
  /// PHASE 2 (anisotropic cells), doc/anisotropic_metric.md §4.4 — ADMITTED since commit C4.  The
  /// momentum row of component `a` is a force DENSITY in the component-`a` normalisation
  /// (rhoRef h_a/tRef^2), so the total force on the body in `forceTotalToPhys` units is
  /// `F_a = - sum_owner R_a h_a' V'` (isotropic `h_a' V' = 1`, so the factor below is an exact 1.0
  /// multiplication and the arithmetic is bit-identical).  The lever arm is already physical.
  /// The v3 transposed-stress WALL TORQUE below carries the metric on its AREA VECTOR, not on its
  /// force component (E3 of `doc/units_escalation.md`, RESOLVED): it is a TRACTION,
  /// `F' = mu' (A' x Omega')` with `A'_b = a_b V'/h_b'` the same physical area vector in hRef^2
  /// the traction paragraph of §4.4 uses, so the per-axis factor belongs to the area component the
  /// cross product consumes.  The "component-a normalisation" that puts `h_a' V'` on `F_a` above
  /// is an argument about MOMENTUM ROWS, and this term is not one.
  std::vector<double> hydroForceTorqueReaction();



  /// The number of unmasked (fluid) staggered momentum cells per component -- the exact discrete
  /// datum the reaction identity is stated against: at steady state, sum_bodies F_c = f_c * N_c.
  std::array<long, 3> fluidMomentumCells();



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
  std::vector<double> reactionBudgetTerms();



  /// A_wall EXACTNESS PROBE (diagnostic for the Layer-4 force integral). For any smooth field q,
  ///     sum_cells q(x_c) * A_wall,cell  ->  integral over the wall of q n_fluid dA
  /// and taking q = x_a turns that, by the divergence theorem applied to the SOLID interior, into
  /// exactly -V_solid along axis a and 0 on the others. So this returns
  ///     [ sum_c x_c*Ax , sum_c y_c*Ay , sum_c z_c*Az ]  (per axis, summed over all instances)
  /// which must equal -V_solid componentwise if the aperture wall-area vectors are right. It
  /// isolates the GEOMETRY from the traction: a force deficit that shows up here is A_wall's, and
  /// one that does not is the pressure / velocity-gradient reconstruction's.
  std::array<double, 3> wallAreaProbe();



  /// Net wall flux this rank injects, sum over inner cells of u_w . A_wall -- the compatibility
  /// datum of the singular pressure problem. Exactly zero for a translating body in a periodic
  /// box (the aperture differences telescope); small but nonzero for rotation and for a body
  /// crossing a non-periodic boundary. Reported, not corrected.
  double wallFluxImbalance();



  // Empty when the geometry is static -> ibmModifyStencil takes its scalar u_bc path, unchanged.
  CCConst wallVelView(int c) const;



  void rebuildStencils();


  // copy the nx*ny*nz inner cells between two extended blocks of different ghost width (g=2 <-> g=1
  // MG).
  void copyInner(CCField dst, C3 de, int dg, CCConst src, C3 se, int sg);


  // Copy the ENTIRE destination block (including its ghost ring) from the source block at per-axis
  // cell offset `off`: dst(x,y,z) <- src(x+off, y+off, z+off). Bridges a G=2 field to the g=1 MG
  // block INCLUDING the g=1 ghosts (off = G-1), so face means at the first inner cell read a valid
  // neighbour. Requires the source ghosts filled (fillGhosts/fillPropGhosts) — under MPI those are
  // the cross-rank values, so the bridge is decomposition-correct.
  void copyBlockShifted(CCField dst, C3 de, CCConst src, C3 se, int off);


  // Fill ghost width G periodically on all 3 axes (x then y then z, covering corners). Distributed:
  // the velocity-block halo (cross-rank + periodic, all ghosts incl. corners).
  void fillGhosts(CCField f);


  // Fused periodic FACE-ghost fill in ONE kernel (vs 3 fillAxis): each inner boundary cell scatters
  // its periodic image to the opposite face ghost, all 3 axes at once. Valid only for
  // FACE-neighbour (7-point) stencils -- it does NOT fill the corner/edge ghosts (which fillAxis's
  // sequential x->y->z does). The IBM RB-GS smoother reads only the 7-point stencil, so this is
  // exact there and cuts the velocity solve's dominant kernel-launch cost (~7200 -> ~2400 fill
  // launches/step) at low resolution. NOT for the Koren advection RHS (reads diagonals) -- keep the
  // full fillGhosts there.
  void fillGhostsFaces(CCField f);


  void fillAxis(CCField f, int axis);


  // Cell divergence of the current velocity iterate, on the inner cells + one ghost ring (the RHS
  // compensation reads div at i and i-strd, so faces at the low inner boundary need the ghost-cell
  // value; velocity ghosts were just filled). Porous-only scratch (divAdv_).
  void computeDivAdv();



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
  // ABLATION: `set_advection_wall_velocity(False)` restores the pre-A0 behaviour (masked zeros in
  // the advection inputs) -- the instrument that differences "zeros vs wall velocity in the
  // advective term" directly. Everything else on the moving path is untouched by it.
  bool advWallInputs() const;


  void buildAdvInputs();


  /// The velocity view the advection operators must read for component c: the wall-corrected
  /// scratch while an instance is moving, the live field (byte-identical) otherwise.
  CCConst advVelView(int c) const;



  bool ensureAdvStash(int c, bool adv);



  void buildRhs(int c);


  // Sibling of buildRhs adding a per-cell body force fb(i) (Boussinesq buoyancy / CFD-DEM
  // feedback): the constant fc becomes fc + fb(i). Kept as a separate kernel so buildRhs stays
  // byte-identical (no codegen drift on the single-phase path). Selected in step() when
  // hasCellForce_.
  void buildRhsForced(int c);


  // Variable-density RHS (sibling of buildRhsForced): the time term, the advection weight, and the
  // per-cell body force all use the FACE density of component c (arithmetic mean over the staggered
  // face, matching VarFaceProps::idiag and the projection coefficient — this three-way consistency
  // is what makes discrete hydrostatic balance exact). The cell force fb is face-interpolated for
  // the same reason (a rho*g cell field becomes rho_face*g at the velocity location). Requires the
  // rho ghosts filled (rebuildStencils / buildAdvStencilVar did it this step).
  void buildRhsVar(int c);


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
  void buildRhsVarMom(int c);



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
  void buildRhsColoFF(int c);



  // Add the face acceleration a_f = dt*(f_f - grad_f(P^n))/rho_f to the just-averaged face field,
  // and REMEMBER it in faceAcc_ so the cell counterpart can average exactly the same numbers.
  // Called from project() immediately after centerToFace, before the divergence. See
  // collocated_varrho.hpp.
  void applyFaceAcceleration();



  // The cell counterpart of the face path: turn faceAcc_ into the TOTAL face velocity increment of
  // this step (force acceleration minus the projection's own face correction) and give each cell the
  // openness-gated average of its two faces. Called from project() in place of the constant-density
  // cell-correction chain.
  // KNOWN GAP (recorded, not guarded): at an OUTFLOW face `bcCorrectOutflow` adjusts `uf_` after
  // `projectCorrectVar`, and that adjustment is NOT mirrored into faceAcc_, so the cell average at
  // the last row before an outflow face would miss it. Every rung-V8 gate is periodic or walled;
  // an open boundary on the collocated variable-density path is untested (WO-R owns the staggered
  // `bcCorrectOutflowVar`).
  void applyCellFaceAverageCorrection();



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
  void addCsfRhs(int c);


  // ABLATION (`set_csf_mode(1)`): the same physics discretized the OTHER plausible way — a
  // cell-centred force `f(j) = sigma*kappa(j)*(C(j+s) - C(j-s))/2h` interpolated to the face with
  // the arithmetic mean `1/2 (f(i) + f(i-s))`, exactly as the per-cell body-force machinery would
  // carry a `rho*g` field. It is consistent, it converges, and it is WRONG for surface tension: the
  // face value is no longer in the range of the projection's discrete gradient, so the projection
  // cannot annihilate it. This kernel exists so the difference is a measured number in the ctest
  // rather than an argument — the same role the harmonic-rho_f ablation plays for WO-J's
  // hydrostatic gate. NEVER a production path.
  void addCsfRhsCellInterp(int c);


  // --- rung W2 (WO-W12): the BLOCK CSF, a sibling of `addCsfRhs` ------------------------------
  //
  // Same force, same place in the RHS, same `rs(i)` cut-cell rescale — but the face value was
  // formed ON THE BLOCKS (each marker's own curvature cascade on its own dense box, the same
  // `csfFaceCurvature` + `csfFaceForce` pair this file's `addCsfRhs` applies to the global field)
  // and scattered into `csfBlkF_` with UNPACK_SUM, so two markers whose bands overlap ADD their
  // forces instead of one of them being lost to the union's `max`. This is TBFsolver's
  // `VOF.f90::computeSurfaceTension` structure (block `stx/sty/stz` -> `boxes_2_grid_vf(...,
  // UNPACK_SUM)`), on the suite's own kernels.
  //
  // WHY THE FORCE AND NOT THE CURVATURE IS SCATTERED. kappa is not additive and the union colour
  // is a `max`, so a face between two overlapping markers has no single (kappa, dC) pair to build
  // a force from; the force is the additive quantity, and forming it where the marker's own colour
  // still exists is the only place the balanced-force pairing (the SAME face difference the
  // projection's gradient uses) is available per marker.
  //
  // Gated on `vofBlockCsf()`, which is false whenever the block container is absent.
  void addCsfRhsBlocks(int c);



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
  CsfDiagnostics csfDiagnostics();



  // Implicit-FOU velocity stencil (CUDA build_adv_stencil_k + ibm_modify_stencil): backward-Euler
  // diffusion (idiag+6beta diag, -beta off) + rho*FOU(u^k) upwind operator (diagonally dominant ->
  // stable at high Re), then the Robust-Scaled cut-cell bake. The advecting velocity u^k = the
  // current C[*].u (ghosts filled).
  void buildAdvStencil(int c);


  // Variable-property sibling of buildAdvStencil: VarFaceProps diffusion build (per-face mu, face-
  // density time diagonal) + the FOU upwind weighted by the FACE density (constant path:
  // fouw=rho_). Separate kernel so the validated buildAdvStencil stays byte-identical.
  void buildAdvStencilVar(int c);


  // Backflow stabilization (Bazilevs 2009 / Esmaily-Moghadam 2011) for the NORMAL momentum at
  // outflow faces: add the dissipative diagonal term beta*rho*|min(u.n,0)| where the outflow
  // reverses (fluid re-entering, u.n<0). This removes the spurious kinetic-energy influx that the
  // do-nothing/zero- gradient outflow advects in -- the "backflow divergence" that blows up
  // separated flows (e.g. the BFS recirculation reaching the outlet), worse on finer grids. Purely
  // dissipative (u_ext=0), so it is implicit + unconditionally stable, and INERT where the outlet
  // is outgoing (u.n>=0) -> the channel and any non-reversing outflow stay byte-identical. Applied
  // to C[c].AC after buildAdvStencil (per Picard iteration, lagged at u^k); only the component
  // normal to each outflow face.
  void applyBackflowStab(int c);


  // max|a-b| over inner cells (Picard outer-tolerance check).
  double maxAbsDiffInner(CCConst a, CCConst b);


  // Shared momentum RB-GS loop: fixed velIters_ sweeps, or (velTol_ > 0) the tolerance stop —
  // colour 0 plain, colour 1 via the fused max-increment kernel, stop once the increment has
  // contracted to velTol_ of the first sweep's. The decision is rank-uniform under MPI (all ranks
  // see the same global max), so per-sweep halo exchanges stay in lockstep.
  // Stencil paths supply `resid` (returns max|b - A u| over this rank's inner fluid cells after
  // a fresh ghost fill) and `bnorm` (max|b|), enabling the residual stop when velResTol_ > 0.
  template <class Fill, class Color, class ColorDu>
  void velSweepLoop(Fill&& fill, Color&& sweepColor, ColorDu&& sweepColorDu,
                    std::function<double()> resid = nullptr, double bnorm = 0.0);



  VelocityMG::Comm vmgComm() const;


  // residual functor + max|b| for the stencil paths of component c (see velSweepLoop)
  // Common tail of a residual evaluation: the held normal-Dirichlet face is imposed, not solved
  // (excluded), remember max|A u| for the convergence scale, return max|r|.
  double finishResidual(int c);


  std::function<double()> stencilResidual(int c, bool exchange = false);


  // The all-fluid domain-BC smoother's operator (per-axis constant coefficients + the boundary
  // fold). `aniso` selects the per-axis body; the isotropic path runs the legacy kernel literally
  // (doc/anisotropic_metric.md §2, trap 4).
  std::function<double()> constCoeffResidual(int c, double bx, double by, double bz, double Ac);


  double stencilBnorm(int c);



  void smoothComp(int c);


  // pressure ghost at domain faces for the incremental predictor's grad(P): zero-gradient (Neumann)
  // at every non-periodic face so grad(P) carries no spurious force there (the periodic fill
  // wrapped the opposite boundary's pressure). Outflow pressure (Dirichlet p=0) is enforced
  // separately in the MG solve.
  void pressureBcGhost();


  // domain-BC velocity ghosts: periodic-fill periodic axes, then apply per-face BCs (fold=0
  // explicit/1 implicit).
  void fillVelGhosts(int comp, int fold);


  void applyVelocityBcComp(int comp, int fold, bool doOutflow);


  // Field-parameterized variants (so the velocity-MG can re-impose the BC on its own level-0
  // iterate). `doOutflow = false` is the SIBLING behaviour merged in here (was
  // `fillVelGhostsKeepOutflow`, used ONLY by `bridgeVelocityToVof` and the two call sites below,
  // always with `fold = 0`): it does NOT re-impose the zero-gradient OUTFLOW face — exactly what
  // `step()` already passes after `project()` ("keep outflow").
  //
  // WHY THE `doOutflow = false` PATH EXISTS (WO-R; a defect found by gate F2). The projection
  // corrects the high-side OUTFLOW normal face separately from every other face
  // (`bcCorrectOutflow`) — that correction IS how mass leaves the domain, and `step()` deliberately
  // re-imposes the domain BCs afterwards with `doOutflow = false` so it survives. `bridgeVelocityToVof`
  // used to call the FULL fill (`doOutflow = true`), whose `bcOutflowComp` overwrites the boundary
  // face with the zero-gradient copy of the last inner cell — erasing the correction immediately
  // after the projection made it, on every step, whenever VoF is enabled.
  //
  // Two measured consequences, both of which vanish with `doOutflow = false`
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
  void fillVelGhostsTo(CCField f, int comp, int fold, bool doOutflow = true);


  // Distributed: a rank applies a face's BC iff its block TOUCHES that global face
  // (`touchesGlobalFace`, the same rule the scalar path uses in `applyScalarBc`). Without the test
  // every rank imposed the wall on its OWN block faces, so a partition cutting a walled axis split
  // the domain into independent sub-domains — invisible in the velocity (each sub-domain is
  // separately consistent) and only visible in the pressure. Single-rank the test is always true,
  // so this is byte-identical there.
  void applyVelocityBcCompTo(CCField f, int comp, int fold, bool doOutflow);


  // implicit-diffusion wall fold (CUDA setup_bc_diffusion): dcorr += (wall:+beta tangential /
  // outflow:-beta), brhs += 2*beta*wall (tangential Dirichlet); bake dcorr into the per-component
  // stencil diagonal.
  void setupBcDiffusion();


  // Incremental (rotational) cut-cell projection: solve A phi = -div_open(u*) (RB-GS,
  // mean-removed), u -= grad phi, then accumulate the physical pressure P += (rho/dt)*phi -
  // mu*div(u*) (Timmermans).
  // one mask-aware axis-wise smoothing pass of a cell field (the filtered-rotational S; see
  // setRotationalFilter). Reads the +/-1 axis neighbours' sdf: fluid-fluid -> (1,2,1)/4;
  // one solid side -> 1/2(self + open-side neighbour); both solid -> identity.
  void filterCellField(CCField f, int axis);


  void project();


  // ---- project() stages (QUALITY_PLAN G.1): pure cut-and-paste, each a contiguous
  // block of the original function sharing only member fields and a CCExec. ----
  void projectAssembleDivergence();


  void projectBuildCoefficients();


  void projectSolve();


  void projectCorrectVelocities();


  void projectPressureUpdate();


  void maskVelocity(int c);


  // Minimum viscosity over the (global, under MPI) inner cells — the provably-stable rotational
  // coefficient for variable viscosity (chi*mu_min <= mu(x) everywhere).
  double minMuInner();


  double reduceMaxAbsInner(CCConst f);


  std::vector<double> gatherInner(CCField fld);


  // Inverse of gatherInner: scatter an x-fastest (nx,ny,nz) inner-region host buffer into the inner
  // cells of a ghosted G=2 field (ghost cells untouched — refill via exchangeField/fillGhosts).
  void scatterInner(CCField fld, const std::vector<double>& in);



  // --- Named field registry (multiphysics field container) ------------------------------------
  // Register a new zero-initialised cell-centred field on the G=2 velocity block and return its
  // buffer. Idempotent: re-adding an existing name returns the existing buffer unchanged.
  CCField addField(const std::string& name);


  bool hasField(const std::string& name) const;


  CCField fieldView(const std::string& name);


  std::vector<std::string> fieldNames() const;


  // Ghost-exchange a registered field (cross-rank + periodic under MPI; periodic-only single-rank).
  void exchangeField(const std::string& name);


  // Add-reduce ("reverse") halo: fold ghost-layer deposits back onto their owner cell (both
  // cross-rank AND periodic self-wrap). This is the coupling primitive for particle->grid
  // deposition (e.g. void fraction / drag reaction) where a particle near a block boundary scatters
  // into ghost cells owned by a neighbour; after this the inner block holds the complete sum.
  // Single-rank non-periodic: a no-op.
  void exchangeFieldAdd(const std::string& name);


  // Host round-trip: read a registered field's inner region as an x-fastest (nx,ny,nz) buffer, or
  // write one (ghosts left stale until the next exchangeField).
  std::vector<double> getField(const std::string& name);


  void setField(const std::string& name, const std::vector<double>& v);


  // Padded-block extents + ghost width, so a zero-copy field buffer (size ex*ey*ez, x-fastest) can
  // be reshaped in Python.
  std::array<int, 3> blockShape() const;


  int ghostWidth() const;


  // Global grid dims (== local dims single-rank). For the CFD-DEM co-decomposition weight field.
  std::array<int, 3> globalResolution() const;


  // This rank's inner-block origin in GLOBAL cells ({0,0,0} single-rank). The deposit-origin shift
  // so particles in global coords land in the local block (gm origin = blockOrigin * h).
  std::array<int, 3> blockOrigin() const;



  // --- Scalar transport (advection-diffusion) -------------------------------------------------
  // Register a transported scalar `name` with constant diffusivity D (grid units). scheme: 0 FOU,
  // 1 Koren TVD (default), 2 SOU. iters = RB-GS sweeps for the implicit diffusion solve. Its field
  // is registered in the directory (get_field/set_field/field_view). Openness (set_solid /
  // set_pressure_geometry) must be established for transport to occur.
  void addScalar(const std::string& name, double D, int scheme, int iters);


  bool hasScalar(const std::string& name) const;


  // Per-face scalar BC: face 0..5 = -x,+x,-y,+y,-z,+z; type 0 periodic, 1 Neumann zero-flux
  // (adiabatic), 2 Dirichlet value. Single-rank / non-decomposed domains (distributed BC deferred).
  void setScalarBc(const std::string& name, int face, int type, double value);


  // Advance all registered scalars one dt with the current divergence-free velocity (also called at
  // the end of step()). Exposed so a test can prescribe a velocity and transport a scalar in
  // isolation.
  void advanceScalars();



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

  void enableVof();


  /// WO-R2 item 4 — the wisp threshold on the advector's mixed-cell predicate and on the
  /// interface Courant band. Default 1e-8 once VoF is enabled; 0 restores the V1 predicate bit
  /// for bit. See `vof::wyIsMixed(c, eps)` and `vof::wyColourJump`.
  void setVofWispEps(double eps);


  double vofWispEps() const;


  /// The value `enableVof` starts from. A standalone `WyAdvector` that a test compares the solver
  /// against must be given the SAME value; `set_vof_wisp_eps(0)` is the "V1 verbatim" ablation.
  static constexpr double defaultVofWispEps();


  bool vofEnabled() const;


  // Initial / prescribed colour field on the inner cells (flat x-fastest, nx*ny*nz), C in [0,1]:
  // the LIQUID fraction of the cell. Enables VoF if it is not on yet. Ghosts are refreshed here so
  // a closure applied before the first step already sees a consistent field.
  void setVof(const std::vector<double>& c);


  std::vector<double> getVof();


  // Local (this rank's) colour census: sum / min / max / mixed-cell count / wisp count.
  vof::WyAdvector::Diagnostics vofDiagnostics();


  // sum of the canonical colour field "C" over SOLID cells of this rank (0 by construction).
  double vofSolidColourSum();


  // Interface-local Courant number max|uf|*dt/h over the faces of mixed cells and their face
  // neighbours, with the CURRENT velocity and dt (an all-reduce max under MPI). This is the number
  // the WY boundedness bound applies to — NOT the global max, which over-throttles badly (V1
  // measured 0.314 in a quiescent Zalesak corner against 0.157 at the interface). Use it to pick
  // dt: `dt_new = dt * cfl_target / vof_max_courant()`.
  double vofMaxCourant();


  // The interface-local Courant number of the step just taken (0 before the first step).
  double vofLastCourant() const;


  // Weymouth-Yue boundedness cap (default 0.25, the PROVEN 3D bound 1/(2(N-1)); 0.5 is the 2D
  // value). `step()` throws when the interface-local Courant number exceeds it.
  void setVofCflLimit(double v);


  double vofCflLimit() const;



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
  void requireVofGeometry(const char* who);


  // Does the colour advection run the cut-cell (openness-weighted) kernels?
  bool vofHasGeometry() const;


  // Ablation: drop Weymouth's admissible-interval clamp on the openness-weighted flux
  // (`vof/cutcell.hpp` vofCutFluxClamp). ON by default — the measurement that put it there is in
  // that header. With it off the [0,1] clip becomes the mechanism instead of a tripwire and the
  // conserved functional drifts.
  void setVofCutFluxClamp(bool on);


  bool vofCutFluxClamp() const;


  // What the CANONICAL "C" field carries in SOLID cells (the working block always carries the
  // neutral band fill, which is what the MYC / height-function stencils need):
  //   true  (default) 0 — "no colour in the solid", the WO-Q gate. The closures then see gas
  //                   density there and the CSF sees a full colour jump across a wall face.
  //   false           the band fill — a zero-slope continuation of the liquid into the wall.
  // Measured on the G5 cap (D/dx = 24, sigma = 1, mu = 0.05): see the WO-Q findings entry.
  void setVofSolidColourZero(bool on);


  bool vofSolidColourZero() const;


  // The colour field INCLUDING the neutral solid-band fill, on the inner region — i.e. what the
  // MYC / height-function stencils actually read, as opposed to the canonical "C" (0 in solid).
  // The fill is regenerated here, so this is also the direct gate on its decomposition
  // independence: it must be pointwise BITWISE across np (`tests/kokkos_mpi/test_vof_cutcell_mpi`).
  std::vector<double> getVofFilledColour();


  // The cut-cell geometry the colour block runs on, on the inner region: 0 = the cell fluid
  // fraction eps, 1/2/3 = the openness of the +x/+y/+z face of each cell (the ADVECTOR's high-face
  // convention), 4 = the cell classification (1 = solid). All must be bitwise across np.
  std::vector<double> getVofGeometry(int which);


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
  void setContactAngle(double thetaDeg);


  // ISSUES sweep item 3: `set_contact_angle` used to be a silent no-op whenever there was nothing
  // for it to bind to -- the theta field simply was never consulted, and
  // `contact_angle_diagnostics()['contact_cells']` reading 0 was the only tell. Say so instead.
  void requireWettingWall() const;


  // Per-cell contact angle in DEGREES on the inner region (flat x-fastest, nx*ny*nz). Only the
  // value at the SOLID band cell being filled is read, so cells away from a wall are irrelevant.
  void setContactAngleField(const std::vector<double>& thetaDeg);


  bool contactAngleSet() const;


  double contactAngle() const;


  // Which anchor the theta-plane uses (`vof::VofWettingPivot`): 0 volume-consistent (DEFAULT,
  // idempotent), 1 the PLIC centroid p_f (Afkhami-Bussmann), 2 the work order's
  // `c = p_f - sdf(p_f) n_w` (NOT idempotent — measured to be off by 0.26 in cell fraction at
  // theta = 60, gate G0), 3 the contact line on the wall. Ablation only.
  void setContactAnglePivot(int mode);


  int contactAnglePivot() const;



  // --- rung V6 (WO-V6): the DYNAMIC contact angle and hysteresis --------------------------------
  //
  // Nothing in the V5b fill changes; only the VALUE of theta per contact cell does. See
  // `vof/wetting_dynamic.hpp` for the model (Afkhami, Zaleski & Bussmann, JCP 228:5370 (2009)):
  //
  //    theta_Delta^3 = theta_e^3 + 9 Ca_cl ln(Delta/lambda),   Ca_cl = mu_l U_cl / sigma
  //
  // with `Delta` the CELL SIZE and `lambda` an EXPLICIT slip length in cells. The explicit slip is
  // the whole point: a VoF contact line's numerical slip is proportional to `Delta`, so without it
  // the imposed angle is silently grid-dependent (VOF_PLAN §6). NEVER report a dynamic-wetting
  // result without stating lambda.
  //
  // @param thetaEDeg   the equilibrium (static base) angle, degrees. Also becomes the static angle.
  // @param slipCells   lambda / Delta, in CELLS. Must lie in (0, 1) — lambda >= Delta would make
  //                    ln(Delta/lambda) <= 0 and REVERSE the correction.
  // @param muLiquid    the LIQUID dynamic viscosity entering Ca_cl (solver units).
  // @param sigma       the surface tension entering Ca_cl; <= 0 means "use set_surface_tension".
  void setContactAngleDynamic(double thetaEDeg, double slipCells, double muLiquid,
                              double sigma = 0.0);


  /// WO-V6b -- the VELOCITY half of the dynamic contact line. Replaces the TANGENTIAL no-slip
  /// Dirichlet datum of the Robust-Scaled cut-cell closure by the Navier condition
  /// `u_t(wall) = lambda du_t/dn`; the wall-NORMAL component stays impermeable (the moving-body
  /// datum if any). `lambdaCells` is lambda/Delta; 0 restores the validated no-slip closure
  /// bit-identically. Shares its value with set_contact_angle_dynamic's cut-off.
  void setWallSlipLength(double lambdaCells);


  /// The Navier slip length in the caller's units (0 = no-slip).
  double wallSlipLength() const;


  /// Cut-cell axes at which a one-cell fluid gap kept the no-slip closure (per component).
  std::array<int, 3> wallSlipSandwichCells() const;


  // theta_a / theta_r, degrees. Composes with the dynamic correction when that is also set: the
  // hysteresis selector picks the BASE angle and Cox-Voinov corrects it, except on the PINNED
  // branch (theta_r <= theta_app <= theta_a), where the apparent angle itself is imposed and the
  // idempotence of the V5b fill (WO-S finding 1) is what makes the contact line stand still.
  void setContactAngleHysteresis(double thetaADeg, double thetaRDeg);


  // Back to the static V5b angle, byte-identically (the driver's views are kept but never read).
  void setContactAngleDynamicOff();


  bool contactAngleDynamic() const;


  bool contactAngleHysteresis() const;


  double contactAngleSlip() const;


  // ABLATION: the 3-point in-wall mean of U_cl (default ON). Off = the raw per-cell MAC velocity.
  void setContactAngleSmoothing(bool on);


  // The angle clamp of the Cox-Voinov cube, degrees (default 1 / 179).
  void setContactAngleClamp(double loDeg, double hiDeg);


  // The sigma the dynamic correction uses: the explicit override if one was given, else the CSF's.
  double effectiveContactSigma() const;


  // The per-cell dynamic-wetting state on the inner region: 0 the IMPOSED angle (degrees),
  // 1 the measured APPARENT angle (degrees), 2 the smoothed U_cl, 3 Ca_cl, 4 the
  // `vof::VofDynamicState`. Non-contact cells read 0 in 1..4 and the static base in 0.
  std::vector<double> getVofDynamicField(int which);


  struct ContactAngleDiagnostics {
    long contactCells = 0;    ///< band cells written by the theta plane of their own anchor
    long neighbourCells = 0;  ///< band cells written by the mean of the anchor's MIXED neighbours
    long pureCells = 0;       ///< band cells that took the pure-phase continuation
    long parallelCells = 0;   ///< band cells whose interface was parallel to the wall
    long neutralCells = 0;    ///< band cells that fell back to WO-Q's neutral mean
    long unfilledCells = 0;   ///< SOLID cells pass 1 left untouched (passes 2-3 then fill them)
    double meanApparentAngle = 0.0;  ///< mean measured apparent angle over `contactCells`, degrees
    double setAngle = 0.0;           ///< the prescribed angle, degrees (uniform case)
    // --- rung V6 (WO-V6), all zero unless a dynamic angle / hysteresis is configured -----------
    long dynamicCells = 0;    ///< band cells the V6 pass produced an angle for
    long pinnedCells = 0;     ///< of those, cells whose contact line is PINNED
    long advancingCells = 0;  ///< theta_app > theta_a
    long recedingCells = 0;   ///< theta_app < theta_r
    double meanImposedTheta = 0.0;    ///< mean IMPOSED angle over the V6 contact cells, degrees
    double meanApparentTheta = 0.0;   ///< mean apparent angle over the same set, degrees
    double maxCaCl = 0.0;             ///< max |Ca_cl| = |mu_l U_cl / sigma|
    double maxContactSpeed = 0.0;     ///< max |U_cl| (smoothed), solver velocity units
  };
  // The band census of the CURRENT colour field: how many band cells each branch of the fill wrote
  // and the mean APPARENT angle the fluid-only normal reported at the contact cells (G1's
  // measurement, evaluated on the fill's own data rather than on a post-processed shape).
  ContactAngleDiagnostics contactAngleDiagnostics();


  // Wire the theta field + the wall SDF onto the colour block. Idempotent; called by the setters
  // and again by every geometry rebuild (`buildVofGeometry`), since the block can be re-sized.
  void applyContactAngle();


  // Cell-centre velocity on the colour block, for the V6 contact-line speed. Built from the
  // solver's own staggered faces (`0.5*(u(i) + u(i+s_c))`, both valid after `fillVelGhosts`) on
  // the INNER region and then run through the colour field's ghost policy, exactly as the wall SDF
  // and the fluid-only normals are — that is what keeps the imposed angle decomposition-
  // independent (WO-S finding 9 applied to a third field).
  void buildVofCellVelocity();



  // The colour advector itself (its g=3 block, geometry views and planes). For TESTS: gate G3 of
  // `tests/kokkos/test_vof_cutcell.cpp` rebuilds the openness/fraction by an independent route and
  // compares against these.
  const vof::WyAdvector& vofAdvector() const;


  // The sweep permutation index of the NEXT colour advection (`kWySweepPerm[n % 6]`). Exposed so a
  // benchmark can hold the permutation fixed, or resume one, across a restart.
  // The sweep permutation is `kWySweepPerm[n % 6]`, so this counter is STATE: a run resumed with
  // it reset takes a different sweep order and its colour differs at the splitting error (measured
  // 6.2e-4 after ONE step of `channel_18`, off a bitwise-identical velocity). The BLOCK container
  // keeps its own counter — `VofBlockSet::step_`, which drives `WyAdvector::advect(dt, step_)` for
  // every marker — so a restart has to set both, and this is the one call that does it.
  void setVofStepParity(long n);


  long vofStepParity() const;



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
  /// `dt` is in the caller's time unit.
  void advectVofKinematic(double dtPhysArg);


  // --- Part III rung W0 (WO-W0): the per-bubble VoF BLOCK container -----------------------------
  //
  // A THIRD container over the same L1 kernels (`suite/docs/VOF_PLAN.md` §10, the TBFsolver
  // `vofBlock` pattern): one bubble = one `WyAdvector` on a small moving global index box with a
  // master rank of its own, and the registered `"C"` the closures see is the UNION
  // `C = max_blocks C_block`, never the source. Two bubbles that touch therefore CANNOT coalesce
  // numerically — coalescence becomes an explicit model decision (rung W4) instead of a numerical
  // accident. See `vof/block_container.hpp` for the three index boxes and the ghost policy, and
  // `vof/block_exchange.hpp` for why the gather is plain Isend/Irecv and not an NBX handshake.
  //
  // W0 stops at KINEMATIC transport: `advect_vof_blocks(dt)` is the block twin of `advect_vof(dt)`
  // and carries the same divergence-free precondition. NS coupling (union -> closures -> varRho
  // projection, per-block curvature + CSF scattered UNPACK_SUM) is rung W12.
  //
  // Scope at W0: all-fluid (no immersed solid — the cut-cell block is W12), and the seeds are
  // spheres given in CELL units.
  void enableVofBlocks(const std::vector<std::array<double, 4>>& seeds);


  // Everything `enable_vof_blocks` does EXCEPT the seeding, so the sphere seeds and the
  // general (`enable_vof_blocks_from_field`) seeds share one code path.
  void prepareVofBlocks();


  // The union the closures see, plus the rung-W1 master assignment on the seeded boxes.
  void finishVofBlocks();


  bool vofBlocksEnabled() const;


  void disableVofBlocks();


  // Kinematic block advection with the CURRENT (projected) face velocity — the block twin of
  // `advect_vof`. Same precondition: Weymouth-Yue conservation is conditional on the face field
  // being discretely divergence-free, so a field that is not is refused rather than silently
  // reported as a conservation defect.
  // `requireSolenoidal` is the KINEMATIC precondition: the Python entry point prescribes the
  // velocity itself, so a field that is not discretely divergence-free is a user error and is
  // refused rather than silently reported as a conservation defect. Inside `step()` the advecting
  // field is the projection's own output and its residual divergence IS the conservation floor --
  // exactly as for the structured `advectVof()`, which has never carried a check here -- so the
  // in-step call passes false. (Rung W2: with variable density the projected residual sits at
  // ~1e-7 without the exact level-0 apply, which would refuse every coupled step.)
  /// `dt` is in the caller's time unit.
  void advectVofBlocks(double dtPhysArg, bool requireSolenoidal = true);


  // Per-bubble Lagrangian census. Only this rank's MASTER blocks carry numbers (volume, centroid,
  // centroid velocity, the central second moments); the box and the master are replicated.
  std::vector<vof::VofBlockStats> vofBlockStats() const;


  // max/mean of the per-rank block-cell load under the CURRENT master assignment (round robin at
  // W0; the weighted-ORB assignment is W1 and these are the numbers to beat). 1.0 = perfect.
  double vofBlockImbalance() const;


  // Blocks mastered by each rank, and the inner cells those blocks carry.
  void vofBlockCensus(std::vector<long>& masters, std::vector<long>& cells) const;



  // --- Part III rung W1 (WO-W12) ----------------------------------------------------------------
  //
  // (a) Master assignment. `mode` 0 = round robin (W0), 1 = LPT greedy on the block cell counts,
  //     2 = weighted ORB over a 1-D block space (core's `BlockDecomposer<1>`). `every` re-runs the
  //     assignment every `every` block steps, MIGRATING the colour of any block that changed
  //     master (nothing else in a block is state). Applied immediately.
  void setVofBlockAssign(int mode, long every);


  int vofBlockAssign() const;


  // The imbalance the CURRENT `assignMode` would give without applying it — so a study can put the
  // three modes side by side on one swarm without perturbing the run.
  double vofBlockImbalanceOf(int mode) const;


  // (b)/(c) instrumentation: device-resident packing on/off, and the block-pool hit census.
  void setVofBlockDeviceStaging(bool on);


  void setVofBlockPool(bool on);


  std::array<long, 2> vofBlockPoolStats() const;



  // General seeding (rung W1): one block per given GLOBAL index box, with the colour taken from
  // the field `set_vof` installed. A sphere seed is a convenience over this; a Hysing bubble, a
  // quasi-2-D cylinder or any scanned marker enters here. The boxes are the BUBBLE extents (the
  // container grows them by the 3-cell margin itself) and must not overlap in a way that makes a
  // cell belong to two markers at seed time -- the gather is a copy, not a union, so a cell inside
  // two boxes would be given to both markers.
  void enableVofBlocksFromField(const std::vector<std::array<int, 6>>& boxes);



  // --- rung W3: checkpoint / restart of the block container ------------------------------------
  //
  // The block's own inner colour is a block's ONLY state, so {box, colour} per block is a complete
  // checkpoint -- and unlike `enableVofBlocksFromField` it is exact when two markers touch (the
  // seeding gather out of the UNION would give each a slice of the other; WO-W12 open item 5).
  std::vector<double> vofBlockColour(long id);


  void enableVofBlocksFromColours(const std::vector<std::array<int, 6>>& boxes,
                                  const std::vector<std::vector<double>>& colours);



  // --- rung W2: the block CSF ------------------------------------------------------------------
  //
  // Turn the surface tension of `set_surface_tension` into a PER-BLOCK force: each marker runs its
  // own curvature cascade on its own box and forms the V4 balanced-force face force there, and the
  // three face fields are summed into the local patch (UNPACK_SUM). Requires the block container.
  void enableVofBlockCsf();


  // Per-block curvature + CSF face force, scattered SUM into the registered face-force fields the
  // RHS reads. Called at the head of every step by `updateVofCurvature()`.
  void computeVofBlockCsf();


  // DIAGNOSTIC: the scattered block CSF face force on this rank's inner cells, component c (the
  // low face of each cell, the same convention `addCsfRhs` uses). This is the field the block mode
  // adds to the RHS; comparing it across decompositions is how a scatter defect is localised.
  std::vector<double> getVofBlockForce(int c);


  // The summed branch census of the last block CSF (LOCAL to this rank).
  vof::VofCurvature::Stats vofBlockCurvatureStats() const;



  // Harmonic instead of arithmetic rho_f in the pressure projection (WO-J item 5). DEFAULT OFF and
  // it should stay off — read the long note in mac_pressure.hpp before turning it on: arithmetic
  // rho_f IS the harmonic mean of the mobility 1/rho (the series-correct choice for a normal flux)
  // and is what makes the discrete hydrostatic balance exact, because the momentum time term and
  // the face body force interpolate rho arithmetically and are NOT switched by this flag. Shipped
  // as a measured knob for the coefficient-coarsening question, not as an alternative scheme.
  void setRhoFaceHarmonic(bool on);


  /// WO-R2 item 3 — the exact (matrix-free, double, flux-form) level-0 operator apply in the
  /// residual and the Krylov matvec. Per solver; `enableVof` turns it on, and this is the
  /// ablation switch.
  void setPressureExactResidual(bool on);


  /// WO-R2 item 1 ABLATION: `set_outflow_operator_coefficient(False)` restores the pre-WO-R2
  /// operator, whose Dirichlet domain-face rows carried the literal openness 1.0 instead of the
  /// variable-density coefficient `open_f*rho0/rho_f`. It exists so the before/after of the
  /// Nusselt film and the outflow divergence stays measurable; there is no reason to set it in
  /// production. ON by default.
  void setOutflowOperatorCoefficient(bool on);


  bool outflowOperatorCoefficient() const;


  bool pressureExactResidual() const;


  bool rhoFaceHarmonic() const;


  // WO-R item 4 asked for the `1/rho_f` factor on the high-side outflow correction;
  // `doc/variable_density_projection.md` §4 listed its absence as a defect. WO-R measured the
  // factor making the outflow divergence SEVEN ORDERS WORSE and recorded the item as refuted —
  // correctly, for the operator as it then was. WO-R2 item 1 fixed that operator
  // (`CutcellMG::setOutflowCoefficient`: the Dirichlet domain-face rows no longer overwrite
  // `buildRhoCoeff`'s `open_f*rho0/rho_f` with the literal 1.0) and the verdict INVERTED:
  //
  //   stratified duct, ratio 10, 5 steps, max|div(open u)| of the PROJECTED field
  //                                    operator = raw openness   operator = rho0/rho_f (WO-R2)
  //     without the factor                    8.76e-10                    9.97e-05
  //     with    the factor (this knob)        9.24e-03                    8.31e-10
  //   (tests/kokkos/test_vof_bc.cpp gate F2; at ratio 1 the two are bitwise equal, 1.40e-17.)
  //
  // The mechanism is one rule: a projection correction removes the discrete divergence only if it
  // uses the SAME face coefficient the operator row used. Both table columns obey it; the fixed
  // operator is the one whose coefficient is also the PHYSICALLY right mobility at the outlet
  // (the low-side outlet had no consistent pairing at all before the fix — see the Nusselt film).
  //
  // DEFAULT ON since WO-R2 (`set_outflow_rho_correction(False)` is the ablation). Bitwise inert at
  // constant density either way (rho_f == rho0 makes the factor exactly 1), and gated on varRho.
  void setOutflowRhoCorrection(bool on);


  bool outflowRhoCorrection() const;



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
  void setVofInflow(int f, double value);


  /// Per-position inflow colour on face `f`: `prof` is (nb, nc) on the INNER grid of the face's two
  /// perpendicular axes (the same layout and the same clamp resampling `set_domain_bc_profile`
  /// uses for the velocity).
  void setVofInflowProfile(int f, const std::vector<double>& prof, int nb, int nc);


  /// `inletOutlet` backflow colour on outflow face `f` (default 0 = gas): where the boundary face
  /// velocity points back INTO the domain, the colour ghost carries this value instead of the
  /// zero-gradient copy (Rusche 2002 thesis section 4; OpenFOAM `inletOutletFvPatchField`). Where
  /// the fluid leaves, zero-gradient is kept and what leaves is what is inside.
  void setVofBackflow(int f, double value);


  bool vofBcActive() const;


  /// Signed liquid volume that crossed each of the six domain faces during the LAST colour
  /// advection, in cell-volume units, POSITIVE for liquid entering the domain. Local to this rank
  /// (a distributed caller sums them, as it does for every other VoF diagnostic).
  std::vector<double> vofBcVolumes() const;


  /// The same, accumulated since `enable_vof()` (or the last `resetVofBcVolumes()`). Changing a
  /// boundary colour mid-run deliberately does NOT reset it — a slug injection is exactly the case
  /// where the running total is the quantity of interest.
  std::vector<double> vofBcVolumesTotal() const;


  void resetVofBcVolumes();



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
  void vofApplyColourBc(CCField f);



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
  void vofBcPropGhosts(CCField f);



  /// (Re)build everything that lives on the g=3 block for this rung: the out-of-domain mask and the
  /// resampled boundary-colour profiles. Called from `buildVofBlock` (so a redistribute/initMpi
  /// re-derives them) and from `vofBcArm` (so a setter takes effect immediately).
  void vofRebuildBcBlock();



  /// Clamp-resample a per-position face scalar from the user's (nb, nc) INNER grid onto the
  /// ghost-inclusive (b, c) plane of an extended block, so the fill kernel indexes it directly by
  /// face position. Same rule as `setDomainBcProfile`, one component instead of three.
  CCField resampleFaceScalar(const std::vector<double>& prof, int nb, int nc, int face, C3 ext,
                             int g, int& outNc);



  /// A VoF boundary colour is only meaningful on a face that already carries the matching domain
  /// BC, and getting that wrong is silent (the ghost band would be written and then never read as
  /// boundary data). Fail loudly instead.
  void checkVofBcFace(int f, int wantType, const char* who);


  /// Arm the rung: install the mask, rebuild the profiles, and zero the boundary volume ledger.
  void vofBcArm();


  /// Move the advector's per-face boundary volume ledger into the solver's, once per advection.
  void vofHarvestBcVolumes();



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
  void computeVofCurvature();


  // The branch census of the last `computeVofCurvature()` — LOCAL to this rank (the driver is
  // MPI-free; a distributed caller sums them).
  vof::VofCurvature::Stats vofCurvatureStats() const;


  /// The interface curvature kappa = 2H in the caller's units, i.e. 1/LENGTH (the internal field
  /// is 1/h; kappa_phys = kappa'/hRef).
  std::vector<double> getVofCurvature();


  std::vector<double> getVofCurvatureBranch();


  // Wendland support width of the PV fallback fit, in cell units (Han et al.: 2.5 with a 5^3
  // stencil; 3.5 recovers first-order spurious-current convergence on a translating droplet and
  // 4.5 over-smooths and destroys it). Exposed for WO-P's sweep.
  void setVofCurvatureWeightWidth(double d);


  double vofCurvatureWeightWidth() const;


  // Tier 2b, the mixed height-position fit. OFF by default and it should stay off — see
  // `vof::VofCurvature::useMixedHeightFit` for the measurement that put it there.
  void setVofCurvatureMixedHeightFit(bool on);


  bool vofCurvatureMixedHeightFit() const;



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
  void setSurfaceTension(double sigma);


  /// The surface tension in the caller's units (what set_surface_tension was given).
  double surfaceTension() const;


  // The wisp threshold above, exposed so it can be swept/ablated. Default 1e-8; 0 restores the
  // unguarded V3 predicate and, with surface tension on, reproduces the instability it exists for.
  void setVofInterfaceEps(double eps);


  double vofInterfaceEps() const;


  // ABLATION: 0 = the balanced-force face difference (default, the only production mode);
  // 1 = a cell-centred sigma*kappa*grad(C) face-interpolated like an ordinary body force. See
  // `addCsfRhsCellInterp`. Kept so the ctest can measure what the operator pairing is worth.
  void setCsfMode(int m);


  int csfMode() const;


  // Rung W2: the CSF may be formed on the BLOCKS instead of the global colour+kappa fields, in
  // which case there is no `kappaField_` at all. `vofBlockCsf()` is false whenever the block
  // container is absent (`vofBlocks_` is null unless `enable_vof_blocks` ran), so the expression
  // below reduces to W0's character for character on every non-block path.
  bool vofBlockCsf() const;


  bool csfActive() const;



  // INSTRUMENT (not a configuration): stop recomputing the curvature at the head of each step and
  // use whatever is in the "kappa" / "kappa_branch" fields. Together with `set_vof_kappa_constant`
  // this isolates the BALANCED-FORCE identity from the curvature estimator — the exactness gate of
  // this rung, which must hold at machine zero for a curvature that is merely constant, whether or
  // not it is the right one.
  void setVofKappaFrozen(bool on);


  bool vofKappaFrozen() const;


  // INSTRUMENT: set kappa to a constant over the WHOLE block (inner + ghosts) and mark every cell's
  // branch as a valid estimate, then freeze it. The force (1) is then exactly the discrete gradient
  // of `sigma*kappa*C`, so the projection must annihilate it to round-off from ANY colour field.
  /// `kappa` is a PHYSICAL curvature, 1/length (the internal field is 1/h; see vof_curvature()).
  void setVofKappaConstant(double kappa);



  // The Brackbill (1992) / Denner & van Wachem (2015) capillary time-step limit
  // `sqrt((rho_1 + rho_2) h^3 / (4 pi sigma))`. +inf when surface tension is off.
  //
  // The density SUM is taken from the declared phase pair when momentum consistency is on
  // (`enable_vof_momentum` validated it against the closure), and otherwise from `min(rho) +
  // max(rho)` over the current density field — an MPI_MIN/MPI_MAX pair, so it is exact and
  // decomposition-independent. It is the sum, not a mean: both phases oscillate.
  /// The Brackbill capillary limit in the CALLER's time unit.
  ///
  /// UNITS NOTE for the whole VoF stack. The colour advector runs on the UNIT LATTICE like every
  /// other operator here (`WyAdvector::h()` stays 1 and its Courant number is the index
  /// `v*dt'`, which IS the physical `u*dt/h`), so nothing inside it changes with the domain.
  /// What crosses the boundary changes: sigma on the way in, kappa and every TIME on the way out.
  /// Check: dt' = sqrt((rho1'+rho2')/(4 pi sigma')) with rho' = rho/rhoRef and
  /// sigma' = sigma*tRef^2/(rhoRef*hRef^3) gives exactly sqrt((rho1+rho2)hRef^3/(4 pi sigma))/tRef.
  double capillaryDt();


  /// The same limit in the solver's index time (per tRef) — what the step's own guards compare to.
  double capillaryDtInternal();


  // Safety factor on the capillary limit: `step()` throws when `dt > factor * capillaryDt()`.
  // Default 1.0 — Denner & van Wachem measured the Brackbill prefactor to BE the stability
  // boundary, so there is no margin built into the formula itself. Set it huge to disable the
  // check, exactly as `set_vof_cfl_limit` is the escape hatch for the Weymouth-Yue cap.
  void setCapillaryCfl(double f);


  double capillaryCfl() const;



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
  /// Both limits at the current state, in the CALLER's time unit (`courant` is dimensionless).
  VofStepLimits vofStepLimits();


  // rho_1 + rho_2 for the capillary limit. Public because nvcc refuses an extended
  // __host__ __device__ lambda inside a private member function (the WO-O build note).
  double phaseDensitySum();


  // Head-of-step curvature refresh + the capillary dt check. No-op unless surface tension is on.
  //
  // The curvature is taken from the colour field the step is ABOUT to run with — the same field
  // `updateProperties()` just turned into rho(C) and mu(C), i.e. C^{n+1} under momentum consistency
  // (the VoF stage ran at the head) and C^n without it (it runs at the tail). Either way kappa,
  // rho and the colour in the force (1) are the same time level, which is what the balance needs.
  // ISSUES sweep item 2. One step at a dt re-picked from the CURRENT state, returning the dt used.
  //
  // Why this exists as an entry point rather than as a recipe in every driver: both limits
  // `vof_step_limits()` reports are INSTANTANEOUS. Re-picking every ten steps -- what the free
  // bubble and falling film studies do, and what the gallery's rising-bubble driver copied -- is
  // safe only where the face field changes slowly. In a packing it is not: a throat jet can grow
  // max|uf| by more than 50 % inside ten steps (measured, examples/bubble-through-packing), and
  // the Weymouth-Yue cap is a hard throw. And `capillary_dt` is itself STATE-dependent (it reads
  // the density field, so the first call on a domain whose closures have not run reports the base
  // `set_rho` value -- measured 7.1x too large on a gas-filled pore-scale domain).
  //
  // dt = min(cfl_target * cfl_dt, capillary_cfl * capillary_dt, dt_max), from the state as it
  // stands at the call, then `set_dt` + `step()`. This is EXACTLY what the three gallery drivers
  // (rising-bubble, bubble-through-packing, trickle-flow-packing) do in Python -- and bitwise so,
  // because IEEE multiplication is monotone, hence `f*min(a,b) == min(f*a,f*b)`.
  //
  // `set_dt` / `step()` semantics are untouched: this is a wrapper, not a mode.
  double stepAdaptive(double cflTarget, double capillaryCflTarget, double dtMax);


  // ISSUES sweep item 1: the two explicit two-phase stability caps, evaluated at the head of
  // `step()` on the state the call STARTS from, so a rejected dt costs nothing.
  //
  // What this can and cannot pre-empt, measured rather than assumed:
  //  * the CAPILLARY cap is a function of dt, sigma, h and the density field only, so hoisting it
  //    is exact -- it fires here for every dt that would have made `updateVofCurvature` throw
  //    (the one difference is that rho is the PREVIOUS step's; the authoritative check inside
  //    `updateVofCurvature`, on this step's rho, is deliberately left in place).
  //  * the Weymouth-Yue cap is a function of the ADVECTING face field. Under
  //    `enable_vof_momentum` the colour rides at the head of the step, so the field checked here
  //    IS the field the advection will use and the pre-check is exact. Without it the advection
  //    sits at the TAIL and rides THIS step's projected output, which does not exist yet -- so the
  //    pre-check runs on u^n (the previous step's projected field, i.e. exactly what the previous
  //    step's advection used) and pre-empts every case where the CAP or the DT moved, but not the
  //    case where the projection itself accelerates the field inside the step. That residual is
  //    what `step_adaptive` (item 2) removes, by re-picking dt from the same limits every step.
  void vofStepPrecheck();


  std::string capillaryThrowMessage(double cap) const;


  void updateVofCurvature();



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
  void advectVof();



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
  /// `rhoGas` / `rhoLiquid` are PHYSICAL densities, the same numbers the rho closure is given.
  void enableVofMomentum(double rhoGasPhys, double rhoLiquidPhys);


  bool vofMomentumEnabled() const;


  // Floor on rho^c in the recovery divide u = (rho^c u)/rho^c, as a FRACTION of min(rho_g, rho_l)
  // (default 1e-6). rho^c leaves [rho_g, rho_l] only through a wisp in the half-shifted colour, and
  // driving it to zero would need C^c ~ -1/(ratio-1); the floor is a guard, not a model, and
  // `vof_momentum_diagnostics()` reports how many control volumes it actually touched.
  void setVofRhoFloorFrac(double f);


  double vofRhoFloorFrac() const;


  double vofRhoFloor() const;


  // MinMod-limited donor reconstruction in the momentum flux. OFF by default — on a control volume
  // a sweep empties, the slope's deviation from the volume's own velocity is amplified by
  // drho*F/rho^c, which is unbounded in the density ratio; measured, it grew the uniform-velocity
  // residual to 2.2e-10 at ratio 1e4 over 50 steps while plain donor-cell upwind stayed flat at
  // 6.7e-16. Harmless at ratio 1e3. See vof/momentum_advect.hpp.
  void setVofMomentumMuscl(bool on);


  // Ablation: the literal reading of "the same frozen dilation flag" (the PRESSURE-cell flag on the
  // shifted control volume instead of its structural analogue).
  void setVofMomentumCellFlag(bool on);


  // Ablation: drop the Weymouth flux clamp on the shifted control volume. With it off the
  // half-shifted colour leaves [0,1] by O(a^2) and rho^c goes NEGATIVE at high ratio — the
  // measurement that the clamp is a necessity, not a habit. See vof/momentum_advect.hpp point 3.
  void setVofFluxClamp(bool on);


  vof::MomentumConsistentAdvector::Diagnostics vofMomentumDiagnostics();


  // The recovered advected velocity of component c on the inner cells (the momentum RHS's time
  // base). Exposed so a test can gate the uniform-velocity identity on the advection ALONE, with
  // the projection and the momentum solve out of the picture.
  std::vector<double> getVofAdvectedVelocity(int c);



  // The coupled colour + momentum advection. Called from the head of step() when momentum
  // consistency is on; exposed so a test can drive it in isolation.
  void advectVofMomentum();




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
    double area = 0.0;    ///< sum of the PLIC polygon areas over interfacial cells (h^2)
    double bandDiv = 0.0; ///< max |div(open u)| over interfacial cells (WO-P23; see pcBandDivergence)
    double Tmin = 0.0, Tmax = 0.0;  ///< energy-scalar extrema (consistent transport only)
    long areaHf = 0;   ///< WO-P3c: cells whose area came from a height function (tiers 1/2a)
    long areaPv = 0;   ///< WO-P3c: cells whose area came from the PV paraboloid (tier 3)
    long areaNone = 0; ///< WO-P3c: cells with NO cascade geometry (kept their MYC PLIC area)
    double areaOrphan = 0.0;  ///< WO-P3d: area the JOINED sheet booked to cells that are not
                              ///< interfacial, i.e. the part of the sum the flux integral drops
    double mdotFit = 0.0;     ///< WO-P3g: the AREA-WEIGHTED one-sided-fit mdot, kept as the
                              ///< diagnostic that item 1 replaced (0 unless the operator flux is on)
    double qOperator = 0.0;   ///< WO-P3g: the heat (W) the operator's Dirichlet rows draw across
                              ///< the interface, i.e. `sum mdot A h_lv` by construction
    double qOrphan = 0.0;     ///< ... the part of it on interfacial cells with NO area, which the
                              ///< regression drops: the ONE non-conservation item 1 cannot remove
  };

  /// **WO-P3f — the ENERGY BUDGET of the phase-change energy solve.** An INSTRUMENT: allocated and
  /// evaluated only under `set_phase_change_budget(true)`, and every number below is a measurement
  /// of the shipped scheme, not a change to it.
  ///
  /// The question it answers. The interfacial cells are Dirichlet rows, so they are OUTSIDE the
  /// energy solve: the "fluid" the energy equation conserves enthalpy over is the UNMASKED set,
  /// and that set changes membership every step as the interface sweeps. A liquid cell that becomes
  /// interfacial LEAVES the solve carrying its superheat `rcp (T - T_sat)`, and an interfacial cell
  /// that becomes pure RE-ENTERS it carrying whatever `pcCarriedValue` left there. Neither transfer
  /// appears anywhere in the latent-heat book-keeping, so on a growing bubble they are a one-signed
  /// enthalpy source/sink of size (cells swept per step) x (superheat one cell from the interface).
  ///
  /// The discrete balance the entries close, over the UNMASKED set and over one `advanceScalars`:
  ///
  ///     sum rcp (T^{n+1} - T*) / dt  =  qGfm + (domain boundary flux) + (solve residual)
  ///
  /// with `qGfm` the heat the plane-anchored (GFM) rows actually deliver ACROSS the interface —
  /// negative when the interface draws heat out of the liquid. `qGfm` is the flux the ENERGY
  /// equation loses; `mdot h_lv A_Gamma` (from the diagnostics' `removed_volume`) is the flux the
  /// REGRESSION books as evaporation. **They are computed by two different discretizations and are
  /// not equal**; their ratio is the scheme's own flux consistency and is the first thing to read.
  struct PhaseChangeBudget {
    double hOpen = 0.0;      ///< sum rcp (T - T_sat) over UNMASKED cells, before the solve
    double hOpenNew = 0.0;   ///< the same, after the solve
    double hLiquid = 0.0;    ///< sum rcp (T - T_sat) over PURE LIQUID cells, before the solve
    double hMasked = 0.0;    ///< the same over MASKED (interfacial) cells
    double dEoverwrite = 0.0;     ///< sum rcp (dval - T) over masked cells: what the Dirichlet
                                  ///< rows inject when they overwrite the transported temperature
    double dEoverwriteNew = 0.0;  ///< the part of it on cells that were NOT masked last step
    double eEnter = 0.0;     ///< sum rcp (T - T_sat) of the cells that JOINED the masked set
    double eLeave = 0.0;     ///< ... and of the cells that LEFT it (carrying `pcCarriedValue`)
    double qGfm = 0.0;       ///< heat INTO the unmasked set across the plane-anchored rows (W)
    double qBehind = 0.0;    ///< WO-P3g: the second-order row's one-sided rescaling of the band
                             ///< BEHIND each interfacial face, which is the rest of the heat the
                             ///< operator transfers across the interface (0 at row order 1)
    long nEnterLiquid = 0;   ///< liquid -> interfacial transitions this step
    long nEnterGas = 0;      ///< gas    -> interfacial
    long nLeaveLiquid = 0;   ///< interfacial -> liquid
    long nLeaveGas = 0;      ///< interfacial -> gas   (the bubble swallowing a cell)
    long nMasked = 0;        ///< size of the masked set
    long calls = 0;          ///< how many energy solves have been instrumented
  };

  /// Turn on phase change. `rhoG`/`rhoL` are the phase densities used by the regression
  /// (`dV = mdot A dt / rho_l`) and by the divergence source (`S = mdot A (1/rho_g - 1/rho_l)`);
  /// they are given EXPLICITLY rather than read off a closure, exactly as `enable_vof_momentum`
  /// does and for the same reason. `hlv` is the latent heat (J/kg) and is only used by the thermal
  /// mass flux. Registers "mdot" (kg m^-2 s^-1, solver units) and "pc_source" (1/s).
  void enablePhaseChange(double rhoG, double rhoL, double hlv);


  bool phaseChangeEnabled() const;



  /// Prescribe a UNIFORM mass flux (P0). Overwrites "mdot" on the inner cells and its ghosts.
  void setMassFluxUniform(double v);


  /// Prescribe a per-cell mass flux (P0), x-fastest over the inner region.
  void setMassFlux(const std::vector<double>& v);


  /// P1: compute `mdot` each step from the registered scalar `tname` by the one-sided pure-cell
  /// weighted least-squares gradients of `vof/phase_change.hpp`. `Tsat` is the saturation
  /// temperature, `kg`/`kl` the phase conductivities (W/(cell K)) and `Rint` the interfacial
  /// heat-transfer resistance of the Schrage/IHTR Robin condition `T_G = T_sat + mdot R_int`
  /// (Bureš & Sato 2021); `Rint = 0` is the hard Dirichlet and is the default.
  void setPhaseChangeThermal(const std::string& tname, double Tsat, double kg, double kl,
                             double Rint);


  void setPhaseChangeThermalOff();



  // --- WO-P23 (rungs P2/P3) --------------------------------------------------------------------

  /// The PLANE-ANCHORED (ghost-fluid) Dirichlet condition, ON by default.
  ///
  /// Rungs P0/P1 pinned the whole interfacial CELL at `T_G`, so the numerical thermal boundary sat
  /// at the cell CENTRE while the mass-flux gradient is fitted from the PLIC PLANE — an O(h)
  /// mismatch that changes sign as the interface sweeps through a cell, and the first-order
  /// component of the P1 Stefan error (WO-P01 finding 6). With this on, the interfacial cell is
  /// instead given the value the one-sided linear profile takes at the cell centre,
  /// `T_cell = T_G + (dT/dn) phi_c`, with `phi_c` the signed normal distance from the plane to the
  /// centre and `dT/dn` the SAME weighted least-squares one-sided fit the mass flux uses, refitted
  /// on the CURRENT colour and temperature. On a saturated side the fit returns 0 and the condition
  /// degenerates to the hard Dirichlet exactly, so it is inert wherever the old form was right.
  /// `set_phase_change_plane_dirichlet(False)` restores the P0/P1 behaviour bit-for-bit.
  void setPhaseChangePlaneDirichlet(bool on);


  bool phaseChangePlaneDirichlet() const;



  /// The QUADRATIC one-sided gradient fit (`T - T_G = G phi + Q phi^2`) instead of the linear one.
  /// This is VOF_PLAN §9 item 1's Aslam quadratic extrapolation in least-squares form: the same
  /// samples and the same stencil reach, one more basis function. Once the plane-anchored Dirichlet
  /// has removed the cell-centre mismatch, the linear fit's `O(T'' h)` curvature bias is the
  /// leading error of the rung.
  void setPhaseChangeQuadraticFit(bool on);


  bool phaseChangeQuadraticFit() const;



  /// **WO-P3c — which geometry the interfacial AREA comes from.** `A_Gamma` enters the plane shift
  /// (`dV = mdot A dt / rho_l`) and the divergence source (`S = mdot A (1/rho_g - 1/rho_l)`), so the
  /// bubble grows as `int mdot dA` and a biased area is a biased growth rate.
  ///
  ///  * `0 = vof::kAreaPlic`   — rungs P0/P1: `plicArea` on the MYC normal. The recorded numbers.
  ///  * `1 = vof::kAreaMetric` — the V3 curvature cascade's own geometry: the height function's
  ///    area element `sqrt(1 + h_x^2 + h_y^2)` on tiers 1/2 and the PV paraboloid's gradient on
  ///    tier 3, applied to the PLIC polygon's projected FOOTPRINT (see `vof/interface_area.hpp`).
  ///  * `2 = vof::kAreaNormal` — the same normals, but the plane is rebuilt on them
  ///    (`plicArea(n*, plicAlpha(n*, C))`) instead of keeping the PLIC footprint.
  ///  * `3 = vof::kAreaFootprint` — the height function's OWN footprint times its own metric, the
  ///    only variant whose cell pieces TILE (see `vof/interface_area.hpp`).
  ///  * `4..7` — **WO-P3d, the JOINED surface**: marching tetrahedra on the cell-centre lattice,
  ///    one watertight sheet whose triangles are booked to cells, so the SUM converges where no
  ///    per-cell construction can (`vof/marching_cubes.hpp`). `4 = kAreaMcColour` (the `C = 1/2`
  ///    level set, whole triangles to the cell holding the centroid), `5 = kAreaMcColourSplit`
  ///    (the same sheet, triangles clipped to each cell's cube), `6 = kAreaMcPlic` and
  ///    `7 = kAreaMcPlicSplit` (the same two deposits on the zero of the PLIC-reconstructed signed
  ///    distance, which is exact on a TILTED plane where the raw `C = 1/2` interpolation is not).
  ///
  /// Modes 1 and 2 are EXACT on a plane (the height function's differences are exact there and the
  /// cascade normal is the MYC one), so every planar gate is unmoved. The default is `0`: on a
  /// well-resolved colour field the MYC area is already within 0.5 % of `4 pi R^2` on a sphere and
  /// the cascade buys nothing measurable — see the WO-P3c findings, which also record that
  /// WO-P3b's 5.5-9.3 % deficit was its probe's own 4^3 sub-sampled initialisation.
  void setPhaseChangeArea(int mode);


  int phaseChangeArea() const;



  /// The summed interfacial area over the inner region, in h^2 (globally reduced under MPI) —
  /// the E7 gallery's `vof_interface_area()`. Uses the CURRENT `set_phase_change_area` geometry,
  /// so the number a page quotes and the number the phase change integrates are the same one.
  /// Needs `enable_vof`; does not need phase change.
  double vofInterfaceArea();



  /// Run the cascade area driver on the (already bridged) g = 3 colour block. Returns the LOCAL
  /// census; the area field stays on the driver for `copyInner`.
  vof::VofInterfaceArea::Stats pcAreaCascadeCompute();



  /// The area field the last `pcAreaCascadeCompute` filled (either driver).
  SField pcAreaCascadeField() const;



  /// Turn on the CONSISTENT energy transport (VOF_PLAN §9 item 6) for the scalar
  /// `set_phase_change_thermal` names: `rho c_p T` is advected with the colour advection's OWN
  /// geometric fluxes (`vof/energy_advect.hpp`) instead of the scalar module's Koren TVD flux, and
  /// the implicit solve carries per-cell `k(C)` (the `k_gas`/`k_liquid` of
  /// `set_phase_change_thermal`) and a `rho c_p(C)` time term instead of a constant diffusivity.
  ///
  /// This is what stops artificial heating at a high `rho c_p` ratio: with two different fluxes the
  /// heat content carried into a mixed cell is divided by a heat capacity built from another flux,
  /// an error of order `d(rho c_p)` — 2000x at water/steam. Requires the thermal mass flux.
  void setPhaseChangeEnergy(double rcpGas, double rcpLiquid);


  /// MinMod-limited donor reconstruction of the face temperature in the consistent energy flux
  /// (`vof/energy_advect.hpp`). OFF by default — see the note there for the measurement.
  void setPhaseChangeEnergyMuscl(bool on);


  bool phaseChangeEnergyMuscl() const;


  void setPhaseChangeEnergyOff();


  bool phaseChangeEnergy() const;



  /// WO-P3f: turn the ENERGY BUDGET instrument on. Allocates one extra cell field and runs two
  /// reductions per energy solve; OFF by default and every kernel is skipped when off, so the
  /// solve is bit-identical. Read with `phase_change_budget()`.
  void setPhaseChangeBudget(bool on);


  bool phaseChangeBudget() const;



  /// WO-P3f: make the per-cell Dirichlet overwrite ENTHALPY-CONSERVING. See `pcCarryDeposit` for
  /// the mechanism and the measurement. OFF by default (the shipped scheme is unchanged).
  void setPhaseChangeCarryConserve(bool on);


  bool phaseChangeCarryConserve() const;



  /// WO-P3f INSTRUMENT: prescribe the interface curvature `kappa = div(n)` the one-sided fits use
  /// to correct their sample distances (`vof::pcCurvedDistance`). 0 (the default) is the shipped
  /// tangent-plane distance and is bitwise inert. This is a PRESCRIBED number, not an estimator:
  /// it exists so the O(h/R) curvature bias of the fit can be measured against a known geometry
  /// before anyone builds a curvature estimator for it.
  /// `kappa` is a PHYSICAL curvature, 1/length (stored as the internal 1/h).
  void setPhaseChangeFitCurvature(double kappa);


  double phaseChangeFitCurvature() const;



  /// **WO-P3g — the SECOND-ORDER interfacial energy operator, as one package.**
  ///
  /// `order = 1` is the shipped WO-P23…P3f scheme, bitwise. `order = 2` turns on, together:
  ///
  ///  1. `set_phase_change_mdot_operator(True)` — `mdot` is the energy operator's OWN interfacial
  ///     flux `q/(h_lv A)` rather than a separate least-squares fit, so the heat the energy
  ///     equation loses and the mass the regression produces are the same discrete number;
  ///  2. `set_phase_change_gfm_order(2)` — the Gibou–Fedkiw three-point row
  ///     (`2/((1+theta) theta)`, `2/(1+theta)`) instead of the two-point `(1/theta, 1)`;
  ///  3. `set_phase_change_curvature_distance(True)` — the row's `theta` and the one-sided fits'
  ///     sample distances are measured to the CURVED interface, with `kappa` from the V3 curvature
  ///     cascade (the same field surface tension uses), not to the tangent plane;
  ///  4. `set_phase_change_carry_conserve(True)` — WO-P3f's enthalpy-conserving Dirichlet
  ///     overwrite, which double-counted before item 1 removed the flux mismatch.
  ///
  /// WO-P3f measured why they only work together: the shipped scheme's 1 % Scriven error is the
  /// residue of a CANCELLATION between (F1) the fit's `O(h/R)` curvature bias `+6 %`, (F2) the
  /// two-point row's `O(h/delta_T)` flux deficit `−5 %`, and (F3) the overwrite's `−0.7 … −4.3 %`
  /// enthalpy destruction — so repairing any ONE of them alone makes the gate worse, measured.
  void setPhaseChangeEnergyOrder(int order);


  int phaseChangeEnergyOrder() const;



  /// WO-P3g item 1: take `mdot` from the energy operator's own interfacial flux instead of the
  /// one-sided least-squares fit. The fit stays as `phase_change_diagnostics()['mdot_fit']`.
  void setPhaseChangeMdotOperator(bool on);


  bool phaseChangeMdotOperator() const;



  /// WO-P3g item 2: the order of the one-sided (ghost-fluid) Dirichlet row. 1 = the shipped
  /// two-point form; 2 = Gibou–Fedkiw's three-point form (`vof::pcGfmRow`).
  void setPhaseChangeGfmOrder(int order);


  int phaseChangeGfmOrder() const;



  /// WO-P3g item 3: measure the GFM row's `theta` and the one-sided fits' sample distances to the
  /// CURVED interface, with the mean curvature taken per cell from the V3 cascade. Supersedes
  /// `set_phase_change_fit_curvature`, which prescribes ONE curvature for the whole field; where
  /// both are on the cascade wins.
  void setPhaseChangeCurvatureDistance(bool on);


  bool phaseChangeCurvatureDistance() const;


  /// **WO-P3f open item 6 / WO-P3g** — the divergence source's 5^3 fallback target, as a setter
  /// (it was only reachable through `set_phase_change_deposit_fallback`). An interfacial cell whose two
  /// along-the-normal candidates (`round(k n)`, k = 1, 2) are BOTH still interfacial keeps its
  /// source, and then carries `div(open u) = S` on its OWN faces — i.e. Weymouth-Yue advects the
  /// colour with a field that is not the liquid velocity, which
  /// `phase_change_diagnostics()['band_div']` reads out directly. With this on, those cells fall
  /// back to the best cell of the `+n` half of the 5^3 box (Malan's collinearity weight). The
  /// order matters: making that search the PRIMARY rule DIVERGES the Scriven bubble (WO-P23), so
  /// it only ever fills holes.
  void setPhaseChangeDepositFallback(bool on);


  bool phaseChangeDepositFallback() const;


  double phaseChangeQOperator() const;


  double phaseChangeQOrphan() const;


  double phaseChangeCarryDeposited() const;


  double phaseChangeCarryLost() const;


  PhaseChangeBudget phaseChangeBudgetValues() const;



  /// The BAND-EXTENDED LIQUID VELOCITY of VOF_PLAN §9 item 3, as a MEASUREMENT rather than a
  /// switch. What that item exists to guarantee is that the field Weymouth-Yue advects the colour
  /// with is the LIQUID velocity at every interfacial cell — which the source deposit already
  /// delivers when it lands in the compact pure-gas layer behind the interface (WO-P01's P0b row
  /// measured the interfacial cell's own faces at the liquid velocity to 1e-19). The quantity that
  /// decides whether an extension is needed is therefore the discrete divergence of the ADVECTING
  /// field at the interfacial cells: it is exactly zero iff no deposit sits on a face WY reads.
  /// `phase_change_diagnostics()['band_div']` reports `max |div(open u)|` over interfacial cells;
  /// a nonzero value there, times the frozen dilation flag, is the volume the colour update
  /// creates, so it is the direct read-out and not a proxy.
  double pcBandDivergence();



  /// Refresh `k(C)` and `(rho c_p)(C)` on the G=2 block from the current colour.
  void pcUpdateEnergyProps();



  /// The colour block's ghost policy WITHOUT the colour-specific rules (no solid-band fill, no VoF
  /// boundary colour): the halo/periodic exchange plus the non-periodic zero-gradient clamp. This
  /// is the temperature's policy on the g=3 block.
  void vofExchangeScalar(CCField f);



  /// A PRESCRIBED extra divergence source (1/s), x-fastest over the inner region, added to the
  /// Poisson RHS exactly like the phase-change deposit: the projection then solves for
  /// `div(open u) = S_pc + S_user`. This is how a CLOSED (periodic) box is made compatible with a
  /// net vapour production: put a balancing sink somewhere the exact solution can absorb it. In a
  /// domain with an outflow face the outflow carries the imbalance and this is not needed.
  void setDivergenceSource(const std::vector<double>& v);


  void clearDivergenceSource();



  /// WO-P23: an AUTO-BALANCED sink region for the phase-change divergence source. `w` is a
  /// non-negative weight per inner cell (x-fastest); after every deposit the solver subtracts
  /// `(global sum of the phase-change source) * w(i) / (global sum of w)` from the source field, so
  /// the Poisson RHS is EXACTLY compatible in a closed domain, every step, with no user
  /// bookkeeping. This is the generalisation of WO-P01's hand-set sink plane, and it is what makes
  /// a closed-box phase-change run possible without the variable-density OUTFLOW operator (whose
  /// density-ratio inconsistency is WO-R2's subject): the sink is a region of the LIQUID far from
  /// the interface, where the exact solution simply has the liquid leaving.
  void setDivergenceSink(const std::vector<double>& w);


  void clearDivergenceSink();



  /// Kinematic entry point (the P0a/P1 driver): build `mdot`/`A_G`/`n` from the current colour and
  /// temperature, deposit the divergence source (for the census only — nothing is projected here)
  /// and apply the interface regression. No Navier-Stokes step, no advection.
  /// `dt` is in the caller's time unit.
  void applyPhaseChange(double dtPhysArg);



  /// The in-step driver: everything `applyPhaseChange` does, at the head of `step()`.
  /// Byte-identical no-op when phase change is off.
  void phaseChangeStep();



  PhaseChangeDiagnostics phaseChangeDiagnostics();



  // nvcc requires members holding extended device lambdas to be public.

  /// (1) The interface build: for every inner interfacial cell reconstruct the PLIC plane from the
  /// canonical G=2 colour, store its area and unit normal, evaluate `mdot` (thermal or prescribed),
  /// and decide which pure gas cell will receive the divergence source. Then exchange those
  /// per-cell quantities so the regression's depth-1 ring and the source gather's depth-2 ring see
  /// the OWNER's values — which is what makes both decomposition-independent WITHOUT any
  /// reverse/add halo and without an atomic scatter (bitwise MPI, not a reduction floor).
  void pcBuildInterface();



  /// (2) Deposit each interfacial cell's source into its chosen pure-gas cell, as a GATHER (each
  /// receiving cell scans the 5^3 box for donors that named it). A gather rather than an atomic
  /// scatter because the sum then has a fixed order and the result is bitwise reproducible across
  /// decompositions; the donors' `dep`/`tgt` are valid two cells deep thanks to `pcBuildInterface`'s
  /// exchange.
  void pcScatterSource();



  /// (3) The regression: two Jacobi passes over the exchanged per-cell data, so the clip deficit is
  /// redistributed with a FIXED summation order (bitwise across decompositions).
  ///   pass 1 (inner region grown by one, reading only exchanged fields): the raw plane shift
  ///          `C - mdot A dt/rho_l`, clipped into [0,1], with the residue stored;
  ///   pass 2 (inner region): add the clipped colour to the shares of the six face neighbours'
  ///          residues, pushed along `-n` (a liquid deficit) or `+n` (a condensation excess) with
  ///          weights `n_d^2`.
  void pcRegress(double dt);



  /// The per-cell Dirichlet mask of the energy scalar: `T = T_sat + mdot R_int` in every
  /// interfacial cell, released everywhere else. Rebuilt from the CURRENT colour, so a call after
  /// the colour advection is what the energy solve at the bottom of the step sees.
  /// **WO-P3g item 3** — the V3 curvature cascade's `kappa` on the G = 2 phase-change block, from
  /// the CURRENT colour. Same driver, same g = 3 block and the same sign convention as surface
  /// tension: `kappa = 2H` in 1/h, positive for a convex blob of LIQUID, i.e. `-2/R` for a gas
  /// bubble — which is exactly `div(n)` for the PLIC normal, the convention `pcCurvedDistance` and
  /// `pcGfmThetaK` are derived in. Where the cascade produces no estimate it leaves 0, and both
  /// consumers then fall back to the tangent-plane distance, which is the shipped behaviour.
  void pcUpdateCurvature();



  /// **WO-P3g** — 1 on every cell that CARRIES A ROW in the energy solve, 0 otherwise.
  ///
  /// The operator-flux `mdot` gathers, from the interfacial side, the Dirichlet couplings of its
  /// PURE face neighbours. A neighbour that is a ghost belonging to another RANK does carry a row
  /// (that rank builds it), and dropping it would make `mdot` decomposition-dependent; a ghost
  /// beyond a NON-PERIODIC domain face carries none, and counting it invents heat. `dmask` cannot
  /// tell the two apart — `pcZeroDomainGhosts` makes both look pure — so this field does, by the
  /// same construction: 1 on the inner region, exchanged (so a rank's ghosts inherit the owner's 1
  /// and a periodic wrap keeps it), then zeroed on the non-periodic domain ghosts.
  ///
  /// Measured on the planar carry probe (64^3, an interface that spans the whole y-z
  /// cross-section, so its edge cells sit ON the domain boundary): without it those cells' `mdot`
  /// reads 2.115e-3 against the exact 2.000e-3, +5.8 %, and the operator flux exceeds the energy
  /// solve's own by 5.3e-4 of the total.
  void pcBuildInDomain();



  void pcUpdateThermalMask();



  /// Zero the two ghost layers on every NON-periodic domain face this rank owns. Used for the
  /// per-cell phase-change data, whose consumers treat a nonzero ghost as a real donor.
  void pcZeroDomainGhosts(CCField f);



  /// **WO-P3f — the enthalpy the per-cell Dirichlet overwrite destroys, returned to the phase it
  /// came from.** An OPTION (`set_phase_change_carry_conserve`), OFF by default.
  ///
  /// The leak. An interfacial cell's row is the identity `T = dval` (`scalarMaskRhs`), so whatever
  /// the geometric energy transport left in that cell is DISCARDED every step and replaced by the
  /// carried value `pcCarriedValue`. Over a cell's whole interfacial lifetime that telescopes to
  /// `rho c_p (T_entry - dval_exit)`: a liquid cell that the interface sweeps enters with its
  /// superheat and leaves as vapour at `T_sat`, and the difference goes nowhere. It is one-signed
  /// on a growing bubble and it scales with the cells swept per step, i.e. with `R^2 dR/dt` —
  /// exactly the shape of the P3 flux deficit. **Measured** by `phase_change_budget()`'s
  /// `d_overwrite` on the Scriven scene at 128^3: `-0.7 %` of `mdot h_lv A_Gamma` at Ja = 0.5 and
  /// `-4.3 %` at Ja = 2 (a destruction), against a growth deficit of 1.0 / 1.5 %.
  ///
  /// (What is NOT a leak, and the instrument says so: a cell CHANGING CLASS. `e_enter` is the
  /// enthalpy of the cells that became interfacial this step, but those cells do not move and
  /// their temperature is still in the field — the relabelling transfers nothing. The `e_enter`
  /// column is a flux between two BOOKS, not an energy sink, and the two genuine
  /// non-conservations of the rung are this overwrite and the `q_gfm` / `mdot h_lv A` mismatch.)
  ///
  /// The repair. Before the overwrite, hand `dE_j = rho c_p(C_j) (T_j - dval_j)` to cell `j`'s
  /// face neighbours that are still IN the solve, weighted by `n_d^2` — the same allocation the
  /// clip-and-redistribute uses, and a fixed-order GATHER (each receiving cell recomputes its
  /// donors' decision) so it is decomposition-independent with no reverse-add halo, the WO-P01
  /// pattern. Which SIDE of the interface receives is decided per axis and locally: of the two
  /// pure neighbours along that axis, the one whose deviation from `T_Gamma` has the same sign as
  /// the interfacial cell's own. That is the phase the discarded enthalpy came from — the
  /// superheated LIQUID on an evaporating bubble (P3), the superheated VAPOUR on the Stefan
  /// problem (P1) — with no scene-specific rule anywhere.
  ///
  /// Reads: the donor at depth 1 and the donor's own neighbours at depth 2, all inside the G = 2
  /// halo and all exchanged before this runs. Writes: inner unmasked cells only.
  void pcCarryDeposit(ScalarField& sc);



  /// WO-P3f: consume `pcCarrySrc_` into the energy solve's time base. Called from
  /// `advanceScalars` right after `cOld` is taken, so the deposit enters that step's RHS.
  void pcCarryApply(ScalarField& sc);



  /// WO-P3f, the energy-budget instrument, part 1: the state BEFORE the energy solve, and the
  /// class-change accounting against the previous step. Called from `advanceScalars` after the time
  /// base `cOld` is taken and BEFORE `scalarMaskRhs` overwrites the masked cells with `dval`, which
  /// is the exact moment the transported temperature of a newly interfacial cell is discarded.
  void pcBudgetPre(ScalarField& sc);



  /// WO-P3f, part 2: the state AFTER the energy solve, the heat the plane-anchored rows actually
  /// delivered across the interface, and the class snapshot the NEXT step compares against.
  ///
  /// `qGfm` mirrors `scalarMaskGfm` exactly (same `theta`, same face conductivity choice, same
  /// openness) and accumulates `k open (T_Gamma - T_i)/theta` — the heat flowing INTO the unmasked
  /// set, i.e. NEGATIVE while a bubble grows into superheated liquid. It is the energy equation's
  /// own interfacial flux, and `mdot h_lv A_Gamma` is the regression's; the instrument exists to
  /// compare them.
  void pcBudgetPost(ScalarField& sc);



  /// Subtract the phase-change (and any prescribed) divergence source from `div_` so the deflated
  /// pressure solve delivers `div(open u) = S`. One branch in `project()`, inert when off.
  void pcApplyDivergenceSource(CCField div);



  /// The colour tolerance that decides which cells are INTERFACIAL, and it has to be at least the
  /// colour advector's own wisp tolerance.
  ///
  /// **This is a real interaction bug and the measurement is in the findings.** WO-R2 item 4 made
  /// `enable_vof` set `WyAdvector::wispEps = 1e-8`, so the advector treats a cell with
  /// `C <= 1e-8` (or `>= 1 - 1e-8`) as PURE for reconstruction and flux. Phase change used its own
  /// `1e-12`, so over the band `1e-12 < C < 1e-8` the two disagreed about what an interface IS:
  /// the phase-change driver reconstructed a plane, gave the cell an area, an `mdot`, a Dirichlet
  /// row and a divergence-source deposit, while Weymouth-Yue moved that cell's colour
  /// algebraically as a pure phase. On the P3 Scriven bubble the disagreement DIVERGES the run —
  /// `R(t)` error 48 % and the study's dt-collapse guard trips at step 34, against **2.002 %** and
  /// 80 clean steps with `set_vof_wisp_eps(0.0)`. The planar P0/P1/P2 gates never saw it (their
  /// interface has no wisps), which is exactly why it had to be found on the curved case.
  double pcEffInterfaceEps() const;


  /// …and the mirror statement: a cell the ADVECTOR treats as a pure phase is a pure phase here
  /// too, or the source deposit's "find a pure gas cell" walk rejects exactly the cells the colour
  /// field has already emptied and the source is left in an interfacial cell (measured:
  /// `band_div` 2.2e+02 on the P2 sucking gate with only the interfacial tolerance raised).
  double pcEffPureEps() const;



  void requirePhaseChange(const char* who) const;


  ScalarField& scalarField(const std::string& name);


  /// Allocate (idempotently) the per-cell Dirichlet mask + value of a registered scalar. Inert
  /// until allocated: `advanceScalars` branches on `dmask.extent(0)`.
  void scalarDirichletMask(const std::string& name);



  // --- Property closures + per-cell body force ------------------------------------------------
  // Register a property/force closure. target: a registered field name — a material property
  // ("mu"/"rho"/…) or a body-force component ("force_x"/"force_y"/"force_z"). kind: LinearMix /
  // BoussinesqForce / ArrheniusMu. in0/in1: input field names (in1 "" if unused). params: up to 4
  // doubles (meaning per kind — property_closures.hpp). Applied at the top of step() in
  // registration order. Targeting a force component turns on the per-cell body-force RHS path.
  void setPropertyModel(const std::string& target, ClosureKind kind, const std::string& in0,
                        const std::string& in1, const std::vector<double>& params);


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
  void setDensityMode(bool variable);


  // Enable/disable the volume-averaged (porous) continuity for unresolved CFD-DEM: the projection
  // enforces d(eps)/dt + div(eps u) = 0 instead of div(u)=0, so the velocity is NOT solenoidal
  // where the void fraction changes. Binds the "eps" field (void fraction from the particle
  // deposition; created seeded to 1 if absent). Staggered-only. The coupling deposits eps each step
  // BEFORE step().
  // Has the cut-cell pressure operator been built (set_solid / set_pressure_geometry)? The porous
  // projection requires it — project() throws otherwise; the coupling driver queries this to
  // auto-install an all-fluid geometry.
  bool hasCutcellPressure() const;


  void setPorousContinuity(bool on);


  // Reseed eps^n = eps^{n+1} so d(eps)/dt = 0 this step. Call after the FIRST void-fraction
  // deposition (the "eps" field starts empty, so without this step 0 sees a spurious d(eps)/dt from
  // 0 -> eps).
  void syncPorousPrev();


  // Include (default) or drop the d(eps)/dt source in the porous projection RHS. Dropping it
  // enforces div(eps u)=0 — useful when eps is a bare per-cell particle deposit whose
  // time-derivative is too jagged and drives the eps-weighted pressure solve unstable.
  void setPorousDepsDt(bool on);


  void setPorousConservative(bool on);


  // Pressure under-relaxation factor omega_p in (0,1] (MFIX-style); 1.0 = off (default).
  void setPressureUnderRelax(double w);


  // Enable/disable variable-coefficient momentum (variable viscosity). variable=true binds the "mu"
  // field (creating it, seeded with the current scalar mu, if absent) and forces the stencil solve
  // path. harmonic selects the harmonic face mean (continuous shear stress across a viscosity jump)
  // vs arithmetic. Escape hatch: set_field("mu", arr) then set_property_mode(True).
  void setPropertyMode(bool variable, bool harmonic);


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
  void setVariableRotational(int mode, double chi);


  // Tabulated property: out = piecewise-linear interp of (xs, ys) at the input field (xs
  // ascending).
  void setPropertyTable(const std::string& target, const std::string& in0,
                        const std::vector<double>& xs, const std::vector<double>& ys);


  // Apply all closures (also called at the top of step()). Exposed for testing.
  void updateProperties();


  // Allocate + register the per-cell body-force fields ("force_x/y/z") and route them into the
  // momentum RHS, for an EXTERNAL writer (CFD-DEM feedback) to fill directly via field_view — no
  // closure needed. buildRhsForced then adds them each step (they persist; the writer overwrites).
  void enableCellForce();


  // Implicit (semi-implicit) linear drag: a per-cell coefficient field "drag_beta" is added to the
  // momentum diagonal each step, so a drag source −β(u − u_p) is treated implicitly (the fluid
  // solve becomes (ρ/dt + β)u = … + β u_p). The drag TARGET β·u_p goes into the force_x/y/z fields
  // (the RHS). Unconditionally stable for any β (unlike an explicit −β u force, which diverges for
  // the stiff β of a dense particle bed). The external writer (CFD-DEM) fills "drag_beta" +
  // "force_*" via field_view; enableDrag() allocates them and turns the diagonal path on.
  void enableDrag();


  // Porous + implicit drag: the drag-relaxation w_f=idt/(idt+beta) makes the pressure coefficient
  // high-ratio (~1 in the freeboard, ->0 in the dense bed). Chebyshev diverges on it; the algebraic
  // GraphAMG coarse solve + PCG is robust. Applied whenever BOTH porous_ and hasDrag_ are on
  // (either set second). An explicit set_pressure_* afterwards still wins.
  void configurePorousDragSolver();


  // Add the drag coefficient beta(i) to the (float) momentum diagonal of component c. Called after
  // each stencil (re)build when hasDrag_. All-fluid (rscale==1) is exact; the drag×cut-cell-IBM
  // interaction (rscale≠1) is untested (documented).
  void addDragDiagonal(int c);



 private:
  // === dynamic load balancing: making EVERY per-block allocation follow the new block ===========
  //
  // `allocateBlock` re-creates the buffers it names explicitly and re-`adopt`s the five aliased
  // registry entries ("u"/"v"/"w"/"p"/"sdf"). Everything the solver has grown SINCE construction
  // does not follow it:
  //
  //   1. the FieldSet's OWN storage (`FieldSet::add` — every transported scalar, every property /
  //      body-force closure target, VoF's "C" / "kappa" / "kappa_branch", the phase-change fields);
  //   2. the member handles that ALIAS one of those records (`cField_`, `rhoField_`, `muField_`,
  //      `epsField_`, `cellForce_[]`, `ScalarField::c`, every `Closure`'s in/out views);
  //   3. the lazily-allocated per-block scratch (`vofCs_`, `uAdv_`, the phase-change and porous
  //      work arrays, the g=1 MG-bridge coefficients, …), each allocated at the `n_` / `n1_` of
  //      whichever block was current when its feature was first switched on.
  //
  // `redistribute` then memcpy'd the NEW block's padded extent into (1)'s OLD allocation: measured
  // by AddressSanitizer as a 184320-byte WRITE into a 115328-byte Kokkos HostSpace region at
  // `scatterPadded`, i.e. exactly the `free(): invalid pointer` /
  // `malloc(): unsorted double linked list corrupted` that made `rebalance_by_weights` unusable on
  // any run owning a registry field. (2) and (3) are the same defect one step later — a stale view
  // indexed with the NEW `nx_`/`ny_`/`nz_`.
  //
  // `resizeForBlock()` is the root fix and runs the three passes in this order, immediately after
  // `allocateBlock` and BEFORE `initMpi` (which rebuilds the VoF block and writes `cField_`).
  void resizeForBlock();


  // Pass 1: the FieldSet's own storage. Fresh, zero-initialised, same name/ghost/centering; the
  // migrated data is scattered back into it by `redistribute` step 4. Aliased records
  // (`ownStorage == false`) are left alone — `allocateBlock` has just re-adopted them.
  void reallocOwnedFields();


  // Pass 2: re-resolve every member handle that aliases a registry record. A handle whose record
  // does not exist is left as it is (the feature was never switched on, so the handle is empty).
  void rebindFieldAliases();


  // Pass 3: the lazily-allocated per-block scratch. A view that was never allocated (extent 0)
  // STAYS unallocated — "inert until its feature is enabled" is load-bearing all over this class
  // (every consumer branches on `extent(0)`), so resizing an empty view would switch a feature on.
  // Views that carry state across a step are re-derived below rather than left zeroed.
  static void resizeIfAllocated(CCField& f, const char* label, std::size_t n);


  void resizeBlockScratch();


  // Resolve a closure target to a registered buffer. A force component allocates ALL three
  // cellForce_ slots (buildRhsForced reads every component) and enables the body-force RHS path.
  CCField ensureTarget(const std::string& name);


  void ensureCellForceAll();


  // Ghost fill for a scalar: periodic (single-rank) / MPI halo base, then override any domain
  // Dirichlet/Neumann faces.
  void scalarFillGhosts(ScalarField& sc);


  // Overwrite the ghost band on each Dirichlet/Neumann domain face (both layers, for the ±2
  // advection reach). Distributed: a rank applies a face's BC iff its block TOUCHES that global
  // face. The halo fill runs first (and may periodic-wrap those ghosts); the BC overwrite wins,
  // exactly matching the single-rank fill-then-BC order. Cross-rank ghost CORNERS on a BC face
  // keep their exchanged (pre-BC) values, but the scalar stencils only read axis-aligned ghosts
  // (7-point diffusion + straight ±2 advection reach), so those corners are never consumed.
  void applyScalarBc(ScalarField& sc);


  // Does this rank's block touch global domain face f (always true single-rank)?
  bool touchesGlobalFace(int f) const;


  // Re-open the diffusion face at a Dirichlet domain boundary: set_domain_bc closes the boundary
  // openness (ox_=0), which correctly makes Neumann/adiabatic walls zero-flux but would also cut a
  // Dirichlet wall's heat path. For each Dirichlet face, restore the face coefficient (band = -D,
  // A_C += D); the ghost carries 2*value - inner so the row is the standard Dirichlet operator.
  /// Variable-k sibling: at a Dirichlet domain face the reopened band takes the boundary CELL's
  /// own `k(C)` instead of the constant `D`. Same rule, same rows; only the coefficient differs.
  void applyScalarBcStencilVar(ScalarField& sc);


  void applyScalarBcStencil(ScalarField& sc);


  // nvcc requires member functions that contain extended (device) lambdas to be PUBLIC — the
  // OpenMP/host build accepts them private, so the breakage only shows on the CUDA backend.
 public:
  void patchScalarDirichletFaceVar(CCField AC, CCField band, CCField kc, int a, int side);


  void patchScalarDirichletFace(CCField AC, CCField band, double Din, int a, int side);


  void applyScalarBcFace(CCField c, int a, int side, int type, double val);



 private:
  int nx_, ny_, nz_;
  C3 e_, e1_;
  std::size_t n_, n1_;
  double rho_ = 1.0, mu_ = 0.1, dt_ = 50.0;  // INTERNAL (index-unit) values; see UnitScales
  UnitScales u_;                            // the metric + reference scales (all 1 in cell units)
  // The caller's own values, kept verbatim so refreshUnitDerived() can re-derive the internal
  // ones in any call order. Equal to the internal ones on the cell-unit path.
  double rhoPhys_ = 1.0, muPhys_ = 0.1, dtPhys_ = 50.0;
  std::array<double, 3> fPhys_{{0, 0, 0}};
  double sigmaPhys_ = 0.0;
  double slipPhys_ = 0.0;
  double bcVelPhys_[6][3] = {};
  std::array<double, 3> f_{{0, 0, 0}};
  int velIters_ = 200, presIters_ = 20;
  double velTol_ = 0.0;         // momentum tolerance stop (0 = legacy fixed-count loop)
  int velMinIters_ = 2;
  long lastMomentumSweeps_ = 0;  // sweeps actually run last step (summed over components/Picard)
  // Residual-based momentum stop (setVelocityResidualTolerance): < 0 follows the pressure rtol
  // (DEFAULT since 2026-09-02), 0 = the update criterion, > 0 = a fixed tolerance.
  double velResTol_ = -1.0;
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
  // setPressureBottomExtent cells on any axis; identical to the smoothed bottom otherwise),
  // 0 smoothed, 1 always. Auto became the default 2026-08-13 after the IBM-path anomaly was
  // fixed (per-fluid-component null-space projection; see ../docs/DECOMPOSITION_AND_MULTIGRID.md).
  int pressAgglomMode_ = -1;
  long lastPressureIters_ = 0;
  bool lastPressureFailed_ = false;  // ISSUES sweep item 6
  // ISSUES sweep item 5: has the collocated MAC face field ever been built (by a projection
  // or by `seedFaceFieldFromCells`)? Meaningless on the staggered grid.
  bool faceFieldValid_ = false;
  // ISSUES sweep items 1+2: `step_adaptive` has just evaluated the two limits on this state, so
  // `step()`'s head-of-step pre-check can skip its own (identical) evaluation. One-shot.
  bool vofPrecheckDone_ = false;
  CutcellMG mg_;
  // --- multi-rank (MPI) state, gated (single-GPU module never links MPI -> byte-identical when
  // off) ---
  // rung W0 (WO-W0): the per-bubble block container. Null unless enable_vof_blocks ran, so
  // every existing VoF path is byte-identical.
  std::shared_ptr<vof::VofBlockSet> vofBlocks_;
  std::shared_ptr<vof::VofBlockExchange> vofBlockExch_;
  // rung W2: the block CSF face force -- on the g=3 patch (the scatter's target) and mirrored onto
  // the G=2 registry block the RHS reads. Allocated only by `enable_vof_block_csf`.
  SField vofBlkF_[3];
  CCField csfBlkF_[3];
  bool distributed_ = false;
  C3 og_{0, 0, 0};  // velocity-block inner origin (global red-black parity); {0,0,0} single-rank
#ifdef PECLET_FLOW_MPI
  std::shared_ptr<GridHaloTopology<3>> velHalo_;  // g=2 velocity-block topology
  std::shared_ptr<GridHalo<double>> velDev_;      // g=2 velocity-block ghost exchange
  std::shared_ptr<GridHalo<MReal>> velDevF_;      // float twin (momentum-stencil ring, CA sweeps)
  bool caMomentum_ = false;  // communication-avoiding momentum sweeps (setCommAvoiding, extent>=4)
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
  bool backflowWarned_ = false;  // the reversed-outflow-without-stabilization warning, once
  double backflowBeta_ =
      0.2;  // outflow backflow-stabilization coefficient (0 = off; inert unless the
            // outflow reverses, so purely-outgoing outlets stay byte-identical)
  CCField bcProf_[6];
  int bcProfNc_[6] = {0, 0, 0, 0, 0, 0};  // per-position inlet profiles (face grid [Lb*Lc*3])
  std::vector<double> bcProfRaw_[6];      // the user's (nb, nc, 3) profile, kept so the resampled
  int bcProfNb_[6] = {0, 0, 0, 0, 0, 0};  // buffer above can be rebuilt when the block is resized
  int bcProfRawNc_[6] = {0, 0, 0, 0, 0, 0};
  CCField bcDcorr_[3], bcBrhs_[3];        // implicit-diffusion face fold (per component)
  bool advect_ = false, cutcellPressure_ = false, implicitFou_ = false;
  bool deferredCorr_ = true;  // deferred-correction advection (off = pure implicit FOU, 1st order)
  int advScheme_ = 0;         // high-order advection: 0 = SOU (default), 1 = Koren TVD
  bool incremental_ = true,
       pwarm_ = false;    // incremental-rotational pressure (CUDA default on) + warm-start
  bool dtDirty_ = false;  // set_dt/set_rho/set_mu after set_solid: momentum stencil needs a rebuild
  // Set once setSolidDevice has built the operators. The settings that are FOLDED INTO that build
  // (domain BCs, aperture order, exact crossings / openness overrides, the fluid-only constraint,
  // the ghost overlay) refuse a later call instead of being silently ignored (QUALITY_PLAN F).
  bool geometryBuilt_ = false;
  int faceInterp_ = 9;  // collocated scheme: 9 = gauge-exact (DEFAULT), 0 = plain, 5/6/7 = embed
  // A0: fill the advection inputs' masked (solid) rows with the WALL velocity instead of zeros.
  // ON by default; setAdvectionWallVelocity(false) is the pre-A0 ablation. See advWallInputs.
  bool advWallVel_ = true;
  // WO-R2 item 1 — the variable-density coefficient on the operator's Dirichlet (outflow)
  // domain-face rows. ON by default; see setOutflowOperatorCoefficient.
  bool outflowOpCoeff_ = true;
  // P1 — the exact double flux-form level-0 apply (setPressureExactResidual; enableVof turns it
  // on). Mirrored into mg_, and read directly by the star overlay's additive delta.
  bool exactResidual_ = false;
  // Communication-avoiding smoothing mask (kCaMomentum | kCaMg), see setCommAvoiding. Both on by
  // default; the momentum half is additionally gated on the block extent (caMomentum_).
  int caMode_ = kCaBoth;
  // The shared level-0 MPI decomposition (setDecomposition): 0 = the aligned ORB, >= 2 =
  // coarse-first with that depth, taking the deepest candidate within maxImbalance.
  int decompLevels_ = 0;
  double decompMaxImbalance_ = 1.05;
  double aspectTheta_ = 2.0;  // mirror of the two multigrids' threshold
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
  // Face-aperture estimator order (setApertureOrder; DEFAULT 2 = marching-squares since
  // 2026-08-26 -- user decision, kills the convexity bias; 1 = the legacy one-sample model).
  int apertureOrder_ = 2;
  int fluidOnlyMode_ = 0;  // fluid-only constraint (setFluidOnlyConstraint): 1=A filter, 2=B star
  StarOverlay starOv_;     // mode-B Kron star overlay (built in setSolid)
  Kokkos::View<int, CCMem> starCounter_;
  int nStar_ = 0;
  bool useVelocityMg_ = false;
  bool vmgExplicit_ = false;  // set_velocity_multigrid was called (AUTO rule off)
  long vmgAutoCells_ = 65536L;         // AUTO threshold, cells per rank (0 = never)
  long vmgAutoMinGlobal_ = 1L << 23;   // AUTO applies only to global problems >= 8M cells
  int vmgLevels_ = 4, vmgVcycles_ = 8;  // IBM velocity multigrid (staircase)
  VelocityMG vmg_;
  CCField vmgTheta_, vmgClean_;
  int outerIters_ = 1;
  double outerTol_ = 0.0;  // Picard outer iteration (CUDA set_outer_iterations)
  long lastOuterIters_ = 0;
  double lastOuterCorr_ = 0.0;
  // per-step phase timers (seconds, this rank; see lastStepSeconds)
  double tStep_ = 0.0, tPredictor_ = 0.0, tMomentum_ = 0.0, tProjection_ = 0.0;
  // WO-V9: cumulative VoF-stage timers (armed by set_vof_timing; inert and unfenced when off)
  bool vofTiming_ = false;
  VofTiming vt_;
  double tStepSum_ = 0.0, tPredSum_ = 0.0, tMomSum_ = 0.0, tProjSum_ = 0.0;
  double vofTick() const;


  void vofAdd(double& acc, double t0);


  // fence-then-read wall clock: phase boundaries must not attribute queued device work to the
  // next phase
  static double phaseTick();


  CCField sdf_, ox_, oy_, oz_, phi_, div_, P_, ox1_, oy1_, oz1_, rhs1_, phi1_, r_, z_, pp_, Ap_;
  bool ghostProjection_ = false;  // directional ghost-cell projection (the collocated AUTO default)
  bool colSchemeAuto_ = Grid::collocated;  // AUTO scheme resolution at setSolid (cleared by any
                                           // explicit scheme selection)
  GpOverlayMReal gpOv_;            // its per-row overlay (built by setSolid), follows MReal (G.6)
  Kokkos::View<int*, CCMem> gpIdMap_;
  Kokkos::View<int, CCMem> gpCounter_;
  int gpNRows_ = -1;         // -1 = overlay not built (set_solid must run with the mode on)
  int gpMatrixOrder_ = 2, gpRhsOrder_ = 2;  // closure order: implicit phi couplings / RHS
  CCField tEx_[3][3];             // exact crossings t[c][k] (inner grid; setExactCrossings)
  bool hasExactCross_ = false;
  // WO-V6b: the Navier slip length (in cells) shared with the dynamic-wetting cut-off, and the
  // switch that makes the MOMENTUM wall closure use it.
  double slipLambda_ = 0.0;
  bool wallSlip_ = false;
  Kokkos::View<int, CCMem> slipSkipDev_;
  std::array<int, 3> slipSandwich_{0, 0, 0};
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
  std::vector<long> movingCutCells_;  // per-instance cut-cell counts (moving scenes only)
  std::vector<int> instCenPinned_;   // 1 = centre of rotation pinned (explicit), 0 = follows the body
  std::vector<long> movingDegenerate_;  // per-instance exactly-on-lattice staggered points
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
  CCField tgp_;             // collocated: cell pressure-gradient scratch
  CCField fvM_, fvL_, cs_;  // collocated: embed defect scratch (M·u, L_FV·u) + cell fluid fraction
  CCField xcx_, xcy_, xcz_;  // collocated: open-centroid wall distance per face (wall-aware map)
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
  // --- WO-P23 (rungs P2/P3) --------------------------------------------------------------------
  bool pcPlaneDir_ = true;   // plane-anchored (GFM) Dirichlet rows instead of pinning the cell
  double pcGfmThMin_ = 0.1, pcGfmThMax_ = 1.9;  // the GFM distance clamp (cells)
  bool pcQuadFit_ = true;    // quadratic (Aslam) one-sided gradient fit (WO-P23 default)
  // WO-P3d (2026-09-03, coordinator's decision): the DEFAULT is the joined sheet (marching
  // tetrahedra on the PLIC signed-distance level set, centroid deposit). Gate re-derived: it is at
  // the floor on every a-priori geometry (sphere 1e-4, tilted planes 2e-6, cylinder order 2.04),
  // does not drift with wisps under advection, is byte-identical on every planar phase-change gate,
  // makes the P2 MPI gate bitwise, and removes the pure-gas deposit fallback (band_div 2e-3 ->
  // 6e-12). The P3 (Scriven) 1 % gate is still open at 1.0-1.5 % and is no longer explained by
  // the area (WO-P3e). `set_phase_change_area(0)` reproduces the P0-P3c numbers.
  int pcAreaMode_ = vof::kAreaMcPlic;  // WO-P3c/P3d: which geometry A_Gamma comes from
  vof::VofInterfaceArea pcAreaC_;    // the cascade area driver (g = 3 block; lazily initialised)
  vof::VofMcArea pcAreaMc_;          // WO-P3d: the joined marching-tet area driver
  double pcMcOrphanArea_ = 0.0;      // WO-P3d: area booked to non-interfacial cells
  CCField pcAreaCg2_;                // its inner values on the G = 2 phase-change block
  // WO-P23 ablation (`set_phase_change_deposit_fallback`): give the interfacial
  // cells whose along-the-normal deposit candidates are BOTH still interfacial a target from the
  // 5^3 box instead of leaving the source in place. Default OFF — see the findings.
  bool pcDepositFallback_ = false;
  bool pcEnergy_ = false;    // consistent rho c_p T transport + variable k(C)/rho c_p(C) operator
  double pcRcpG_ = 1.0, pcRcpL_ = 1.0;
  CCField pcKcell_, pcRcp_;  // k(C) and (rho c_p)(C) on the G=2 block, refreshed per step
  vof::VofEnergyAdvector vofEnergy_;
  double pcBandDiv_ = 0.0;   // max |div(open u)| over interfacial cells, last diagnostics call
  CCField pcGn_[3], pcGphi_, pcTgam_;  // plane normal, centre distance, T_Gamma (WO-P23)
  CCField pcSink_;           // auto-balanced sink weights (WO-P23)
  // WO-P3f: the energy-budget instrument. `pcClsPrev_` holds the PREVIOUS step's class of every
  // cell (0 gas, 1 interfacial/masked, 2 liquid) so the class CHANGES can be counted and their
  // enthalpy accounted; both are unallocated and every kernel is skipped unless the flag is on.
  bool pcBudgetOn_ = false;
  PhaseChangeBudget pcBudget_;
  CCField pcClsPrev_;
  // WO-P3f: conserve the enthalpy the per-cell Dirichlet OVERWRITE destroys (see
  // `pcCarryDeposit`). Off by default; `pcCarrySrc_` is unallocated until it is turned on.
  // WO-P3f: the PRESCRIBED interface curvature `div(n)` used to correct the one-sided fit's
  // sample distances (`vof::pcCurvedDistance`). 0 = the shipped tangent-plane distance, bitwise.
  double pcFitKappa_ = 0.0;
  bool pcCarryConserve_ = false;
  CCField pcCarrySrc_;
  double pcCarryDeposited_ = 0.0, pcCarryLost_ = 0.0;
  // --- WO-P3g: the second-order interfacial energy operator -----------------------------------
  // Three independent pieces, each an option so the ablation can be measured, and one package
  // switch (`set_phase_change_energy_order`) that turns all of them plus WO-P3f's `carry_conserve`
  // on together. Every one is bitwise inert at its default.
  bool pcMdotOperator_ = false;  // item 1: mdot from the energy operator's OWN interfacial flux
  int pcGfmOrder_ = 1;           // item 2: 2 = the Gibou-Fedkiw three-point row
  bool pcCurvDist_ = false;      // item 3: curvature-consistent distances, kappa from the cascade
  CCField pcKappa_;              // the V3 cascade's kappa on the G = 2 block (item 3)
  CCField pcMdotFit_;            // the one-sided-fit mdot, kept as a DIAGNOSTIC under item 1
  CCField pcInDomain_;           // 1 where a cell CARRIES A ROW (inside the global domain), else 0
  bool pcMaskFresh_ = false;     // is the Dirichlet/plane geometry current for THIS colour field?
  double pcQOperator_ = 0.0;     // sum of the operator's interfacial heat (W), last build
  double pcQOrphan_ = 0.0;       // ... the part on cells with no interfacial AREA (dropped)
  double pcMdotFitMean_ = 0.0;   // area-weighted mean of the diagnostic fit mdot
  bool pcHasSink_ = false;
  double pcSinkW_ = 0.0;
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
  // WO-R2 item 4 — see setVofWispEps / defaultVofWispEps.
  double vofWispEps_ = defaultVofWispEps();
  bool rhoFaceHarmonic_ = false;  // harmonic instead of arithmetic rho_f in the projection (OFF)
  bool outflowCorrValid_ = false;  // the projection's outflow-face correction is still in u
  // WO-R item 4 measured NO against the DEFECTIVE operator; WO-R2 item 1 fixed the operator and
  // the answer flipped to YES (measured: with the fix, the projected outflow divergence at ratio
  // 10 is 8.31e-10 WITH the factor and 9.97e-05 without). DEFAULT ON;
  // `set_outflow_rho_correction(False)` restores the plain correction as the ablation.
  bool outflowRhoCorr_ = true;
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
  // rung V6 (WO-V6): the dynamic angle / hysteresis producer for the theta field. Inert unless
  // set_contact_angle_dynamic / _hysteresis is called (`vofDyn_.active()`), in which case the
  // whole V5b battery is byte-identical.
  vof::VofDynamicWetting vofDyn_;
  double contactSigmaOverride_ = 0.0;
  bool contactSigmaWarned_ = false;
  CCField vofDynVel_[3];
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

// The twelve domain headers hold the out-of-line member DEFINITIONS, and since G.8 only the two
// instantiation TUs need them: everything else linking `peclet_flow_solver` gets the declarations
// above and calls into the library.  Narrowing the include to those TUs is what makes an edit to one
// domain header rebuild FOUR objects instead of all 49 (measured: 9m10s -> 4m35s of CPU at -j8).
//
// The one thing this forbids: a member TEMPLATE of `Solver` may not be defined in a domain header
// and called from a consumer -- an explicit instantiation of the class does not cover member
// templates, so the consumer would need the definition and would fail to link.  There is exactly one
// member template on the class today (the VoF `template <class Fill, class Color, class ColorDu>`
// pair), it is used only from inside the class, and the whole battery links.  If you add one that a
// test or the bindings must call, declare it in this file and define it HERE, above this point --
// not in a domain header.
#ifdef PECLET_FLOW_INSTANTIATING
#include "flow_ibm_core.hpp"
#include "flow_ibm_project.hpp"
#include "flow_ibm_scene.hpp"
#include "flow_ibm_geometry.hpp"
#include "flow_ibm_hydro.hpp"
#include "flow_ibm_vof.hpp"
#include "flow_ibm_phase_change.hpp"
#include "flow_ibm_closures.hpp"
#include "flow_ibm_scalars.hpp"
#include "flow_ibm_bc.hpp"
#include "flow_ibm_mpi.hpp"
#include "flow_ibm_diagnostics.hpp"
#endif  // PECLET_FLOW_INSTANTIATING


// The ONE compiled instantiation (suite/docs/QUALITY_PLAN.md §3.G.8).  `Solver<Grid>` is a ~12 k-line
// class template whose 517 out-of-line members are defined in the twelve domain headers included just
// above (in the instantiation TUs only); every
// consumer used to instantiate all of it at -O3 for itself (45 test executables plus the bindings,
// for both grids), which is what made a full rebuild ~45-50 CPU-minutes and made an edit to any one
// domain header invalidate all 45.  These declarations suppress that implicit instantiation; the
// definitions are compiled once, in src/flow_solver_staggered.cpp and src/flow_solver_colocated.cpp,
// into the static library every consumer links (cmake/PecletFlowSolver.cmake).
//
// Nothing is hidden by this.  The class definition above is complete, so a consumer can still derive
// from it (the bindings' `BoundSolver final : Solver<Grid>` inherits the constructor) and the
// standard exempts inline functions -- everything defined inside the class body -- from an explicit
// instantiation declaration, so the small accessors are instantiated and inlined in the consumer
// exactly as before.  What moves is the out-of-line bulk, compiled from the same source under the
// same flags (no LTO, no -march: there is no cross-TU optimization to lose).
//
// PECLET_FLOW_MPI must match between the instantiation TUs and their consumers -- it changes the
// class.  The build makes that structural: the macro is a PUBLIC property of the MPI library.
#ifndef PECLET_FLOW_INSTANTIATING
namespace peclet::flow {
extern template class Solver<Staggered>;
extern template class Solver<Colocated>;
}  // namespace peclet::flow
#endif

#endif  // PECLET_FLOW_SDFLOW_IBM_HPP
