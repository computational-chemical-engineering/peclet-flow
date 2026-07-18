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

#include <array>
#include <cmath>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

#include "face_props.hpp"
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

namespace peclet::flow {

// Templated on a GridLayout policy (grid_layout.hpp) that supplies the grid-position-dependent
// pieces (currently: the per-component velocity sample offset). IbmSolver == Solver<Staggered> (the
// alias below) is bit-identical to the pre-policy solver; the Colocated policy is added in a later
// phase.
template <class Grid>
class Solver {
 public:
  using FV = Kokkos::View<float*, CCMem>;
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
      C[c].ov = IbmOverlay{Kokkos::View<int*, CCMem>("ci", maxCut),
                           Kokkos::View<int*, CCMem>("nb", maxCut),
                           FV("dr", maxCut),
                           Kokkos::View<int*, CCMem>("dc", (std::size_t)maxCut * 6),
                           FV("K", (std::size_t)maxCut * 6),
                           FV("M", (std::size_t)maxCut * 6),
                           FV("X", (std::size_t)maxCut * 6),
                           FV("Nbc", (std::size_t)maxCut * 6),
                           FV("R", (std::size_t)maxCut * 6)};
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
  void setDt(double d) { dt_ = d; }
  void setBodyForce(double fx, double fy, double fz) { f_ = {fx, fy, fz}; }
  void setVelocityIterations(int it) { velIters_ = it; }
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
    vmgLevels_ = levels < 1 ? 1 : levels;
    vmgVcycles_ = vcycles < 1 ? 1 : vcycles;
  }
  // Enable the agglomerated GraphAMG bottom solve in the pressure MG: the coarsest level is solved
  // by a mesh-agnostic algebraic multigrid on the operator gathered to rank 0 --
  // decomposition-agnostic, so multilevel convergence works under a WEIGHTED ORB (where the
  // geometric coarse levels can't cleanly coarsen). Applied at the next set_solid / geometry
  // rebuild.
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
  void setPressureChebyshev(bool on, int maxit, double rtol) {
    useChebyshev_ = on;
    chebMaxit_ = maxit;
    chebRtol_ = rtol;
    chebBoundsSet_ = false;
  }
  // MG-PCG pressure tolerance/iteration cap (CUDA set_pressure_pcg). The Kokkos cut-cell pressure
  // solve is MG-PCG by default; this just sets its bounds (the `on` flag is accepted for API
  // parity).
  void setPressurePcg(bool /*on*/, int maxit, double rtol) {
    pcgMaxit_ = maxit;
    pcgRtol_ = rtol;
  }
  // EXPERIMENTAL directional ghost-cell projection (second staggered IBM, ghost_projection.hpp):
  // point-based FD divergence with wall-anchored directional closures instead of the
  // openness-weighted cut-cell projection. Call BEFORE set_solid (the overlay is built there).
  // v1: single-rank, periodic + IBM only, stationary walls (both grids; the collocated variant
  // closes the face-AVERAGED field and adds the gpCenterGrad predictor/correction, face_interp 0
  // only). The nonsymmetric extended
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
        throw std::runtime_error("set_ghost_projection: collocated ghost requires face_interp 0");
    }
    if (on && (porous_ || varRho_ || hasBc_ || useChebyshev_))
      throw std::runtime_error(
          "set_ghost_projection: incompatible with porous/variable-rho/domain-BC/Chebyshev (v1)");
    if (matrixOrder < 1 || matrixOrder > 2 || rhsOrder < 1 || rhsOrder > 2)
      throw std::runtime_error("set_ghost_projection: matrix_order/rhs_order must be 1 or 2");
#ifdef PECLET_FLOW_MPI
    if (on && distributed_)
      throw std::runtime_error("set_ghost_projection: single-rank only (v1)");
#endif
    ghostProjection_ = on;
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
  void setFaceInterp(int mode) {
    if (ghostProjection_ && mode != 0)
      throw std::runtime_error("set_face_interp: incompatible with the ghost projection");
    faceInterp_ = mode;
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
    peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size),
                                                 peclet::core::IVec<3>{gnx, gny, gnz});
    initMpi(dec, comm);
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
    velDev_->init(*velHalo_);
    dec_ =
        std::make_shared<peclet::core::decomp::BlockDecomposer<3>>(dec);  // remember the partition
    const auto oig = velHalo_->indexer().originInclGhost();
    og_ = {(int)oig[0] + G, (int)oig[1] + G,
           (int)oig[2] + G};  // block inner origin -> global parity
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
  void setSolid(const std::vector<double>& sdfInner, bool cutcellPressure) {
    cutcellPressure_ = cutcellPressure;
    hasSolid_ =
        false;  // does the geometry actually contain solid? (all-fluid set_pressure_geometry
    for (double v :
         sdfInner)  // passes sd>0 everywhere -> stays false, keeping the channel/BFS path)
      if (v < 0.0) {
        hasSolid_ = true;
        break;
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
      // Multi-rank: sdfInner is THIS rank's LOCAL inner block; fill the inner cells, then
      // halo-exchange the ghosts (cross-rank + periodic) so the overlay/openness read the
      // neighbour's SDF at the block boundary.
      auto h = Kokkos::create_mirror_view(sdf_);
      Kokkos::deep_copy(h, sdf_);
      for (int z = 0; z < nz_; ++z)
        for (int y = 0; y < ny_; ++y)
          for (int x = 0; x < nx_; ++x)
            h((long)(x + G) + (long)(y + G) * e_.x + (long)(z + G) * (long)e_.x * e_.y) =
                sdfInner[(std::size_t)x + (std::size_t)y * nx_ +
                         (std::size_t)z * (std::size_t)nx_ * ny_];
      Kokkos::deep_copy(sdf_, h);
      velDev_->exchange(sdf_);
    } else
#endif
    {
      // Single-rank: upload the inner SDF once and do the periodic-wrap gather on device (G4) —
      // fills the whole extended block (inner + periodic ghosts) in one kernel instead of a host
      // triple loop + a full extended-block H2D.
      CCField din("peclet::flow::sdfInner_d", static_cast<std::size_t>(nx_) * ny_ * nz_);
      Kokkos::deep_copy(
          din,
          Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
              sdfInner.data(), sdfInner.size()));
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
    rebuildStencils();
    // Staggered domain BCs bake an implicit-diffusion wall fold; the collocated grid instead uses
    // explicit reflection ghosts (refreshed each smoother sweep), so it needs no fold.
    if (hasBc_ && !Grid::collocated)
      setupBcDiffusion();
    if (useVelocityMg_) {  // velocity-MG hierarchy: IBM (staircase/upwind) or domain-BC
                           // (const-coeff) mode
      vmg_.init(nx_, ny_, nz_, vmgLevels_);
      if (hasBc_)
        vmg_.setBC(bc_);
      else {
        vmgTheta_ = CCField("vmgTheta", n_);
        vmgClean_ = CCField("vmgClean", n_);
      }
    }
    if (cutcellPressure_) {
      buildOpenness(ox_, oy_, oz_, CCConst(sdf_), e_, 1.0, 1.0, 1.0);  // on the g=2 velocity block
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
            if (t != 0 && !open)
              bcZeroOpenness(oa[a], e2, G, a, s);
          }
      }  // the MG re-derives the OPERATOR openness alpha (inflow Neumann -> closed) per level via
         // setBC.
      copyInner(ox1_, e1_, 1, CCConst(ox_), e_, G);  // bridge openness g=2 -> g=1 for the MG
      copyInner(oy1_, e1_, 1, CCConst(oy_), e_, G);
      copyInner(oz1_, e1_, 1, CCConst(oz_), e_, G);
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
        // Fragmentation guard: the binary COUPLED-face condition is stricter than aperture
        // connectivity, so tight-throat geometries (e.g. a random close packing with touching
        // spheres) fragment the fluid graph into a main component + tiny pockets at the
        // contacts. Each pocket adds its own null vector that the single global mean-removal
        // cannot handle, and BiCGStab breaks down (measured: fields to ~1e152 on the RCP
        // example). Host BFS over the coupled graph of the INNER sdf; fluid cells outside the
        // largest component are treated as SOLID for the PROJECTION ONLY (sdfGp), decoupling
        // their rows; the momentum step keeps the true sdf.
        std::vector<double> sdfGpHost = sdfInner;
        {
          const int nx = nx_, ny = ny_, nz = nz_;
          auto id = [&](int x, int y, int z) {
            return (std::size_t)((x + nx) % nx) + (std::size_t)((y + ny) % ny) * nx +
                   (std::size_t)((z + nz) % nz) * (std::size_t)nx * ny;
          };
          std::vector<int> comp(nInner, -1);
          std::vector<std::size_t> stack;
          int ncomp = 0, mainComp = -1;
          std::size_t mainSize = 0, nActive = 0;
          for (std::size_t seed = 0; seed < nInner; ++seed) {
            if (comp[seed] >= 0 || sdfInner[seed] < 0.0)
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
                if (comp[j] >= 0 || sdfInner[j] < 0.0)
                  continue;
                // COUPLED face: mean-of-centers face sdf fluid AND both centers fluid
                if (0.5 * (sdfInner[c] + sdfInner[j]) < 0.0)
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
            for (std::size_t i = 0; i < nInner; ++i)
              if (sdfInner[i] >= 0.0 && comp[i] != mainComp) {
                sdfGpHost[i] = -(std::abs(sdfInner[i]) * 1.001 + 1e-30);
                ++pockets;
              }
            printf("peclet::flow ghost projection: %d fluid components; decoupled %zu pocket "
                   "cells outside the main component (%zu of %zu fluid cells)\n",
                   ncomp, pockets, mainSize, nActive);
          }
        }
        sdfGp_ = CCField("peclet::flow::sdfGp", n_);
        CCField sdfGp = sdfGp_;  // the projection's sdf view (pockets decoupled); persisted for
                                 // the collocated gpCenterGrad predictor/correction
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
                                  hasExactCross_ ? CCConst(tEx_[2][2]) : CCConst());
        copyInner(ox1_, e1_, 1, CCConst(oxb_), e_, G);  // MG surrogate = binary openness
        copyInner(oy1_, e1_, 1, CCConst(oyb_), e_, G);
        copyInner(oz1_, e1_, 1, CCConst(ozb_), e_, G);
      }
#ifdef PECLET_FLOW_MPI
      if (distributed_)  // share the level-0 decomposition so the MG block matches this rank's
                         // block
        mg_.initMpi(gnx_, gny_, gnz_, nLevels_, comm_, dec_.get());
      else
#endif
        mg_.init(nx_, ny_, nz_,
                 nLevels_);  // geometric multigrid on the cut-cell openness (MG-PCG pressure)
      mg_.setBoundaryConditions(
          bc_);  // per-level wall openness + null-space gating (no-op if periodic)
      mg_.setOpenness(CCConst(ox1_), CCConst(oy1_), CCConst(oz1_), 1.0, 1.0, 1.0);
      mg_.setGraphAmgBottom(pressGraphAmg_);  // decomposition-agnostic algebraic coarse solve
      Kokkos::deep_copy(phi_, 0.0);
      Kokkos::deep_copy(P_, 0.0);
    }
  }

  void step() {
    // Multiphysics: refresh material properties / body forces from the current fields (frozen over
    // the step). No-op (byte-identical) when no closure is registered.
    updateProperties();
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
    if ((varProps_ || varRho_ || hasDrag_ || (porous_ && porousCons_)) && !implicitAdv())
      rebuildStencils();
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
      lastOuterIters_ = outer + 1;
      if (outerTol_ > 0)
        for (int c = 0; c < 3; ++c)
          Kokkos::deep_copy(prev_[c], C[c].u);
      if (advect_ || hasBc_ || (Grid::collocated && faceInterp_ >= 4 && faceInterp_ <= 7))
        for (int c = 0; c < 3; ++c)
          fillVelGhosts(c,
                        0);  // explicit ghosts (periodic + BC) for advect / mode-4 FV defect matvec
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
      for (int c = 0; c < 3; ++c)  // RHS from u^n base + advection lagged at u^k
        effVarRho() ? buildRhsVar(c) : (hasCellForce_ ? buildRhsForced(c) : buildRhs(c));
      // Implicit-FOU: rebuild the IBM velocity stencil = backward-Euler diffusion + rho*FOU(u^k),
      // then re-apply the cut-cell bake. Per Picard iteration (advecting velocity changes). Applies
      // to the IBM (periodic/porous) path when the user opts in, AND ALWAYS to the domain-BC
      // stencil path (inflow/outflow) -- implicitAdv() -> fully-implicit upwind advection (stable
      // at large dt). The velocity-MG BC path keeps its own FOU coarse operator.
      if (implicitAdv() && (!hasBc_ || !useVelocityMg_))
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
      if (useVelocityMg_ && implicitFou_ && advect_ && !hasBc_)
        vmg_.restrictAdvVelocities(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u));
      for (int c = 0; c < 3; ++c)
        smoothComp(c);  // per-component IBM implicit-diffusion solve
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
    // Segregated multiphysics: advance any transported scalars with the just-projected
    // divergence-free velocity (properties frozen over the step). No-op (byte-identical) when no
    // scalar is registered.
    advanceScalars();
  }

  // velocity component c (0=u,1=v,2=w) on the inner cells, flat x-fastest [nx*ny*nz].
  std::vector<double> getVelocity(int c) { return gatherInner(C[c].u); }
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
          if (bc_[2 * a + 1] == 3)
            bcNeumannGhost(fa[a], e, G, a, 1);
      }
      if (ghostProjection_ && gpNRows_ >= 0) {
        // Ghost mode: the closed point divergence of the projected face field (same kernel pair
        // as the RHS) — the mode's true residual.
        divergOpen(CCConst(uf_), CCConst(vf_), CCConst(wf_), CCConst(oxb_), CCConst(oyb_),
                   CCConst(ozb_), div_, e_, G);
        gpDivergDelta(div_, CCConst(uf_), CCConst(vf_), CCConst(wf_), gpOv_, gpNRows_,
                      C3{nx_, ny_, nz_}, e_, G);
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
                      C3{nx_, ny_, nz_}, e_, G);
      } else
        divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
                   CCConst(oz_), div_, e_, G);
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
  bool implicitAdv() const { return advect_ && (implicitFou_ || (hasBc_ && !useVelocityMg_)); }
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
  bool bcStencilPath() const {
    return hasBc_ && !useVelocityMg_ &&
           (hasSolid_ || implicitAdv() || varProps_ || varRho_ || hasDrag_);
  }
  // Fill a property field's ghosts for the face means: periodic/halo base, then zero-gradient
  // (copy) on domain-BC (wall/inflow/outflow) faces — a periodic wrap there would bring the wrong
  // layer's value to the wall face (destabilising, especially for the harmonic mean).
  void fillPropGhosts(CCField f) {
    fillGhosts(f);
    if (!distributed_)
      for (int face = 0; face < 6; ++face)
        if (bc_[face] != 0)
          applyScalarBcFace(f, face / 2, face % 2, 1, 0.0);  // type 1 = Neumann copy
  }
  void fillMuGhosts() { fillPropGhosts(muField_); }
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
    if (!distributed_)
      for (int face = 0; face < 6; ++face) {
        const int t = bc_[face];
        if (t == 0)
          continue;
        if (t == 2 || t == 3)
          applyScalarBcFace(epsField_, face / 2, face % 2, 2, 1.0);  // open face: face eps == 1
        else
          applyScalarBcFace(epsField_, face / 2, face % 2, 1, 0.0);  // wall: zero-gradient
      }
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
      fp.sc = strideOf(c);
    } else
      fp.rhoIdtC = rho_ / dt_;
    return fp;
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
                       C[c].rscale, C[c].ov, C[c].nCut, 0.0f);
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

  void buildRhs(int c) {
    CCExec space;
    const double idiag = rho_ / dt_, fc = f_[c], rho = rho_;
    C3 e = e_;
    CCField bb = C[c].b, rs = C[c].rscale, P = P_, brhs = bcBrhs_[c], inh = C[c].inhom;
    CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u), uu = CCConst(C[c].u),
            un = CCConst(old_[c]);
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
    // Mode-2 wall-aware pressure force (collocated): -grad(P) = the TRANSPOSE of the wall-aware
    // cell->face constraint interpolation, precomputed per component (the plain path's central
    // difference is the transpose of the plain 1/2-1/2 average, so this keeps the momentum/
    // constraint operators an adjoint pair on both paths).
    const bool tg = Grid::collocated && faceInterp_ >= 2 && faceInterp_ <= 5 && incr;
    // modes 6/7: openness-weighted -grad(P^n) predictor, matching the fs-weighted correction
    const bool wg = Grid::collocated && (faceInterp_ == 6 || faceInterp_ == 7) && incr;
    // ghost mode (and the mode-9/10 cutcell-ghost hybrids): directional gpCenterGrad predictor —
    // the mode-0 central difference reads the decoupled P=0 at solid-centered cells, a
    // gauge-dependent O(1) gradient error at every cut cell (measured O(1/h) in physical units,
    // ghost_collocated_apriori.py [C2]).
    const bool gg =
        Grid::collocated && (ghostProjection_ || faceInterp_ == 9 || faceInterp_ == 10) && incr;
    if constexpr (Grid::collocated) {
      if (gg) {
        gpCenterGrad(tgp_, CCConst(P_), CCConst(ghostProjection_ ? sdfGp_ : sdf_), c, e_, G);
      } else if (tg) {
        CCField xcs[3] = {xcx_, xcy_, xcz_};
        CCField oax[3] = {ox_, oy_, oz_};
        transposeGradWallAware(tgp_, CCConst(P_), CCConst(sdf_), CCConst(oax[c]), CCConst(xcs[c]),
                               faceInterp_ >= 3, c, e_, G);
      } else if (wg) {
        CCField oax[3] = {ox_, oy_, oz_};
        centerGradOpen(tgp_, CCConst(P_), CCConst(oax[c]), c, e_, G);
      }
    }
    CCConst gpw = CCConst(tgp_);  // empty view on the staggered path (tg/wg/gg false there)
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
        stencilMatvec(fvM_, CCConst(C[c].u), MConst(C[c].AC), MConst(C[c].AW), MConst(C[c].AE),
                      MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB), MConst(C[c].AT), e_, G);
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
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "rhs", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double aK = 0.0, aF = 0.0;
          if (adv) {
            sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y}, Fa{uu, e.x, e.y};
            aK = (sch == 0) ? Grid::advect_sou(c, x, y, z, Ua, Va, Wa, Fa)
                            : Grid::advect(c, x, y, z, Ua, Va, Wa, Fa);
            if (ifou)
              aF = Grid::advect_fou(c, x, y, z, Ua, Va, Wa, Fa);
          }
          // incremental predictor's -grad(P^n): central-difference cell gradient on the collocated
          // grid (or the wall-aware transpose gradient, mode 2), one-sided face gradient (P at the
          // high cell of the staggered face) on the staggered grid.
          const double gp =
              !incr ? 0.0
              : Grid::collocated
                  ? ((tg || wg || gg) ? gpw(i) : 0.5 * (P((long)i + strd) - P((long)i - strd)))
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
    CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u), uu = CCConst(C[c].u),
            un = CCConst(old_[c]);
    const long strd = (c == 0) ? 1 : (c == 1) ? e_.x : (long)e_.x * e_.y;
    const bool pureFou = implicitAdv() && !deferredCorr_;
    const bool incr = cutcellPressure_ && incremental_, adv = advect_ && !pureFou,
               bc = hasBc_ && !bcStencilPath();
    const bool ifou = implicitAdv() && deferredCorr_;
    const int sch = advScheme_;
    // Porous advection-form compensation (+rho*u_f*div(u)_f): see the step() comment.
    const bool pc = porous_ && advect_;
    CCConst dv = CCConst(divAdv_);
    const bool tg = Grid::collocated && faceInterp_ >= 2 && faceInterp_ <= 5 &&
                    incr;  // wall-aware -grad(P) (mode 2/3)
    const bool gg =
        Grid::collocated && (ghostProjection_ || faceInterp_ == 9 || faceInterp_ == 10) &&
        incr;  // directional ghost -grad(P)
    if constexpr (Grid::collocated) {
      if (gg) {
        gpCenterGrad(tgp_, CCConst(P_), CCConst(ghostProjection_ ? sdfGp_ : sdf_), c, e_, G);
      } else if (tg) {
        CCField xcs[3] = {xcx_, xcy_, xcz_};
        CCField oax[3] = {ox_, oy_, oz_};
        transposeGradWallAware(tgp_, CCConst(P_), CCConst(sdf_), CCConst(oax[c]), CCConst(xcs[c]),
                               faceInterp_ >= 3, c, e_, G);
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
            sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y}, Fa{uu, e.x, e.y};
            aK = (sch == 0) ? Grid::advect_sou(c, x, y, z, Ua, Va, Wa, Fa)
                            : Grid::advect(c, x, y, z, Ua, Va, Wa, Fa);
            if (ifou)
              aF = Grid::advect_fou(c, x, y, z, Ua, Va, Wa, Fa);
          }
          const double gp = !incr ? 0.0
                            : Grid::collocated
                                ? ((tg || gg) ? gpw(i)
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
    CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u), uu = CCConst(C[c].u),
            un = CCConst(old_[c]);
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
            sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y}, Fa{uu, e.x, e.y};
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
    CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u);
    using MD = Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "advstencil", MD(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
          double cC = AC(i), cxm = AW(i), cxp = AE(i), cym = AS(i), cyp = AN(i), czm = AB(i),
                 czp = AT(i);
          sadv::ViewAcc Ua{U, e.x, e.y}, Va{V, e.x, e.y}, Wa{W, e.x, e.y};
          Grid::fou_operator(c, x, y, z, Ua, Va, Wa, fouw, cC, cxm, cxp, cym, cyp, czm, czp);
          AC(i) = (float)cC;
          AW(i) = (float)cxm;
          AE(i) = (float)cxp;
          AS(i) = (float)cym;
          AN(i) = (float)cyp;
          AB(i) = (float)czm;
          AT(i) = (float)czp;
        });

    Kokkos::deep_copy(C[c].rscale, 1.0);
    Kokkos::deep_copy(C[c].inhom, 0.0);
    ibmModifyStencil(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, C[c].inhom,
                     C[c].rscale, C[c].ov, C[c].nCut, 0.0f);
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
    CCConst U = CCConst(C[0].u), V = CCConst(C[1].u), W = CCConst(C[2].u);
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
          AC(i) = (float)cC;
          AW(i) = (float)cxm;
          AE(i) = (float)cxp;
          AS(i) = (float)cym;
          AN(i) = (float)cyp;
          AB(i) = (float)czm;
          AT(i) = (float)czp;
        });
    Kokkos::deep_copy(C[c].rscale, 1.0);
    Kokkos::deep_copy(C[c].inhom, 0.0);
    ibmModifyStencil(C[c].AC, C[c].AW, C[c].AE, C[c].AS, C[c].AN, C[c].AB, C[c].AT, C[c].inhom,
                     C[c].rscale, C[c].ov, C[c].nCut, 0.0f);
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
      if (bc_[2 * a + s] != 3)
        continue;  // outflow faces only
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
              AC(i) += (float)(beta * rho * back);  // dissipative diagonal (u_ext = 0)
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
  void smoothComp(int c) {
    if constexpr (Grid::collocated) {
      if (hasBc_) {  // collocated domain BC: the (all-fluid) IBM diffusion stencil + cell-centered
                     // wall
        // reflection ghosts refreshed each colour (explicit no-slip; no fold). Converges to the
        // wall value.
        for (int it = 0; it < velIters_; ++it) {
          fillVelGhostsTo(C[c].u, c, 0);
          ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                              MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                              MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 0);
          fillVelGhostsTo(C[c].u, c, 0);
          ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                              MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                              MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 1);
        }
        return;
      }
    }
    if (bcStencilPath()) {
      // Domain BCs solved with the Robust-Scaled cut-cell / FOU stencil (built by setSolid /
      // buildAdvStencil) while refreshing the domain-BC ghosts each colour -- explicit walls/inflow
      // (reflection, fold=0) + outflow zero-gradient. Mirrors the collocated path above. Used for
      // an immersed solid (cut-cell no-slip in the operator) and/or implicit advection (FOU upwind
      // in the stencil -> stable at large dt). The const-coeff fold smoothers below are all-fluid,
      // diffusion-only: they ignore the solid AND run advection explicitly (CFL-limited).
      for (int it = 0; it < velIters_; ++it) {
        fillVelGhostsTo(C[c].u, c, 0);
        ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                            MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                            MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 0);
        fillVelGhostsTo(C[c].u, c, 0);
        ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                            MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                            MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 1);
      }
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
      vmg_.setBcApplyL0([this, c](CCField x) { fillVelGhostsTo(x, c, 1); });
      vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8);
      return;
    }
    if (hasBc_) {  // domain-BC (no immersed solid): CUDA's double const-coeff diff_k + dcorr fold
      const I3 e{e_.x, e_.y, e_.z}, og{0, 0, 0};
      const double beta = mu_, Ac = rho_ / dt_ + 6.0 * mu_;
      for (int it = 0; it < velIters_; ++it) {
        fillVelGhosts(c, 1);  // re-impose wall faces (fold) before each color
        diffSmoothColor(C[c].u, CCConst(C[c].b), e, og, G, beta, Ac, 0, CCConst(bcDcorr_[c]));
        fillVelGhosts(c, 1);
        diffSmoothColor(C[c].u, CCConst(C[c].b), e, og, G, beta, Ac, 1, CCConst(bcDcorr_[c]));
      }
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
      vmg_.solve(CCConst(C[c].b), C[c].u, vmgVcycles_, 2, 2, 8);
      maskVelocity(
          c);  // re-impose no-slip at solid (the masked solve leaves them at the pin value)
      return;
    }
    for (int it = 0; it < velIters_;
         ++it) {  // IBM / periodic: Robust-Scaled cut-cell stencil (float)
      fillGhostsFaces(
          C[c].u);  // 7-point smoother reads faces only -> the fused 1-kernel face fill suffices
      ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                          MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                          MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 0);
      fillGhostsFaces(C[c].u);
      ibmRbgsStencilColor(C[c].u, CCConst(C[c].b), MConst(C[c].AC), MConst(C[c].AW),
                          MConst(C[c].AE), MConst(C[c].AS), MConst(C[c].AN), MConst(C[c].AB),
                          MConst(C[c].AT), CCConst(C[c].mask), e_, og_, G, 1);
    }
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
        if (bc_[2 * a + s] == 0)
          continue;
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
  void fillVelGhosts(int comp, int fold) { fillVelGhostsTo(C[comp].u, comp, fold); }
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
          if (t == 0)
            continue;
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
        if (t == 0)
          continue;
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
                      C3{nx_, ny_, nz_}, e_, G);
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
                      C3{nx_, ny_, nz_}, e_, G);
      } else
        divergOpen(CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), CCConst(ox_), CCConst(oy_),
                   CCConst(oz_), div_, e_, G);
    }
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
      // in setSolid); the overlay delta enters the fine-level matvec only.
      lastPressureIters_ =
          mg_.solveBiCGStab(rhs1_, phi1_, r_, gpRh_, pp_, Ap_, gpT_, z_, gpZ2_, pcgMaxit_,
                            pcgRtol_, 2, 2, 12, gpOv_, gpNRows_, C3{nx_, ny_, nz_});
    } else {
      lastPressureIters_ =
          mg_.solvePCG(rhs1_, phi1_, r_, pp_, z_, Ap_, pcgMaxit_, pcgRtol_, 2, 2, 12);
    }
    copyInner(phi_, e_, G, CCConst(phi1_), e1_, 1);  // bridge phi back g=1 -> g=2
    fillGhosts(phi_);
    if (hasOutflow_) {  // hold phi=0 at the outflow ghost so grad(phi) drives the outflow face
                        // (Dirichlet p=0)
      B3 e{e_.x, e_.y, e_.z};
      for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s)
          if (bc_[2 * a + s] == 3)
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
            if (t != 0 && t != 3)
              bcNeumannGhost(phi_, e, G, a, s);
          }
      }
      // Correct the face field (-> discretely divergence-free; transient this step) and the cell
      // field (central-difference cell gradient).
      projectCorrect(uf_, vf_, wf_, CCConst(phi_), e_, G);
      fillGhosts(uf_);
      fillGhosts(vf_);
      fillGhosts(wf_);    // complete the divergence-free face field (boundary faces)
      if (hasOutflow_) {  // correct the high-side outflow face on the face field so mass leaves
                          // (phi=0 there)
        B3 e{e_.x, e_.y, e_.z};
        CCField fa[3] = {uf_, vf_, wf_};
        for (int a = 0; a < 3; ++a)
          if (bc_[2 * a + 1] == 3)
            bcCorrectOutflow(fa[a], phi_, e, G, a);
      }
      if (ghostProjection_ || faceInterp_ == 9 || faceInterp_ == 10) {
        // Ghost cell correction (also the mode-9/10 cutcell-ghost hybrids): the directional
        // gpCenterGrad gradient of phi — 2nd-order one-sided at cut cells, never reads a
        // decoupled (solid/pocket) phi. The same operator supplies the momentum's -grad(P^n)
        // predictor (buildRhs), so the pressure force the momentum feels and the correction stay
        // one operator family.
        for (int cc = 0; cc < 3; ++cc) {
          gpCenterGrad(tgp_, CCConst(phi_), CCConst(ghostProjection_ ? sdfGp_ : sdf_), cc, e_, G);
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
      else if (varRho_)  // per-face 1/rho on the gradient, matching the operator coefficient
        projectCorrectVar(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(rhoField_), rho_, e_, G);
      else
        projectCorrect(C[0].u, C[1].u, C[2].u, CCConst(phi_), e_, G);
      if (hasOutflow_) {  // correct the high-side outflow normal face that projectCorrect misses
                          // (mass leaves)
        B3 e{e_.x, e_.y, e_.z};
        for (int a = 0; a < 3; ++a)
          if (bc_[2 * a + 1] == 3)
            bcCorrectOutflow(C[a].u, phi_, e, G, a);
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
      CCExec space;
      CCField P = P_, ph = phi_, d = div_;
      // Pressure under-relaxation (MFIX §10.1): accumulate only omega_p of the increment into the
      // physical pressure P (the velocity correction still uses the full phi to satisfy
      // continuity), so the next step's incremental predictor -grad(P^n) can't overshoot for a
      // stiff drag diagonal. omega_p=1 (default) is the current behaviour; <1 only stabilizes the
      // porous+drag path.
      const double ct = pressUnderRelax_ * rho_ / dt_, mu = mu_;
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
      Kokkos::deep_copy(sc.cOld, sc.c);
      scalarFillGhosts(sc);
      scalarBuildRhs(sc.b, CCConst(sc.cOld), CCConst(Uf), CCConst(Vf), CCConst(Wf), CCConst(ox_),
                     CCConst(oy_), CCConst(oz_), idt, sc.scheme, e_, G);
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
  void setDensityMode(bool variable) {
    if constexpr (Grid::collocated) {
      if (variable)
        throw std::runtime_error("set_density_mode: variable density is staggered-only (v1)");
    }
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
      // Pressure driver: CHEBYSHEV by default under variable density. MG-PCG stalls on the
      // rho-scaled coefficient operator (the hierarchy's transfer pair was built/validated for
      // geometric openness; with scaled coefficients the V-cycle preconditioner loses the
      // SPD-preserving structure CG needs — observed: PCG 5000 iters stuck where Chebyshev
      // converges in ~20). Chebyshev only needs real spectrum bounds, which are re-estimated on
      // every coefficient rebuild (chebBoundsSet_ invalidation in project()). An explicit
      // set_pressure_pcg/set_pressure_chebyshev AFTER set_density_mode still wins (last set).
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
      // CHEBYSHEV by default (as for variable density): MG-PCG stalls on the eps-scaled coefficient
      // operator — its V-cycle preconditioner loses the SPD structure CG needs (observed: PCG 2000
      // iters stuck where Chebyshev converges in ~40). Bounds re-estimated on every coefficient
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
      mg_.setGraphAmgBottom(pressGraphAmg_);
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
          AC(i) = (float)((double)AC(i) + bd);
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
  // advection reach). Single-rank only — distributed domain-BC scalars are a later phase.
  void applyScalarBc(ScalarField& sc) {
    bool any = false;
    for (int f = 0; f < 6; ++f)
      any = any || (sc.bc[f] != 0);
    if (!any || distributed_)
      return;
    for (int f = 0; f < 6; ++f)
      if (sc.bc[f] != 0)
        applyScalarBcFace(sc.c, f / 2, f % 2, sc.bc[f], sc.bcVal[f]);
  }
  // Re-open the diffusion face at a Dirichlet domain boundary: set_domain_bc closes the boundary
  // openness (ox_=0), which correctly makes Neumann/adiabatic walls zero-flux but would also cut a
  // Dirichlet wall's heat path. For each Dirichlet face, restore the face coefficient (band = -D,
  // A_C += D); the ghost carries 2*value - inner so the row is the standard Dirichlet operator.
  void applyScalarBcStencil(ScalarField& sc) {
    if (distributed_)
      return;
    for (int f = 0; f < 6; ++f) {
      if (sc.bc[f] != 2)
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
  int pcgMaxit_ = 500;
  double pcgRtol_ = 1e-10;  // cut-cell pressure MG-PCG
  bool useChebyshev_ = false,
       chebBoundsSet_ = false;  // Chebyshev pressure driver (set_pressure_chebyshev)
  int chebMaxit_ = 120;
  double chebRtol_ = 1e-9, chebA_ = 0.0, chebB_ = 0.0;
  int nLevels_ = 4;             // multigrid depth (CUDA default; set_pressure_multigrid)
  bool pressGraphAmg_ = false;  // agglomerated GraphAMG bottom solve (decomposition-agnostic)
  long lastPressureIters_ = 0;
  CutcellMG mg_;
  // --- multi-rank (MPI) state, gated (single-GPU module never links MPI -> byte-identical when
  // off) ---
  bool distributed_ = false;
  C3 og_{0, 0, 0};  // velocity-block inner origin (global red-black parity); {0,0,0} single-rank
#ifdef PECLET_FLOW_MPI
  std::shared_ptr<GridHaloTopology<3>> velHalo_;  // g=2 velocity-block topology
  std::shared_ptr<GridHalo<double>> velDev_;      // g=2 velocity-block ghost exchange
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
  int faceInterp_ = 0;    // collocated cell->face map: 0 = plain average, 1 = wall-aware (opt-in)
  double fvRelax_ = 1.0;  // mode-4 FV defect-correction under-relaxation (setFvRelax)
  bool useVelocityMg_ = false;
  int vmgLevels_ = 4, vmgVcycles_ = 8;  // IBM velocity multigrid (staircase)
  VelocityMG vmg_;
  CCField vmgTheta_, vmgClean_;
  int outerIters_ = 1;
  double outerTol_ = 0.0;  // Picard outer iteration (CUDA set_outer_iterations)
  long lastOuterIters_ = 0;
  double lastOuterCorr_ = 0.0;
  CCField sdf_, ox_, oy_, oz_, phi_, div_, P_, ox1_, oy1_, oz1_, rhs1_, phi1_, r_, z_, pp_, Ap_;
  bool ghostProjection_ = false;  // directional ghost-cell projection (experimental 2nd IBM)
  GpOverlay gpOv_;                // its per-row overlay (built by setSolid)
  Kokkos::View<int*, CCMem> gpIdMap_;
  Kokkos::View<int, CCMem> gpCounter_;
  int gpNRows_ = -1;         // -1 = overlay not built (set_solid must run with the mode on)
  int gpMatrixOrder_ = 2, gpRhsOrder_ = 2;  // closure order: implicit phi couplings / RHS
  CCField tEx_[3][3];             // exact crossings t[c][k] (inner grid; setExactCrossings)
  bool hasExactCross_ = false;
  std::vector<double> oxOverride_, oyOverride_, ozOverride_;  // exact apertures (inner)
  bool hasOpenOverride_ = false;
  CCField oxb_, oyb_, ozb_;  // binary (COUPLED) openness on the g=2 block (ghost divergence)
  CCField sdfGp_;  // the projection's sdf (fragmentation pockets decoupled) — gpCenterGrad reads
                   // it so the collocated predictor/correction never touch a decoupled cell
  CCField gpRh_, gpT_, gpZ2_;  // extra BiCGStab scratch (g=1 block)
  CCField uf_, vf_, wf_;    // collocated: transient face (MAC) field (approx projection)
  CCField tgp_;             // collocated: transpose-gradient scratch (setFaceInterp(2/3))
  CCField wdef_;            // collocated: FV wall viscous-flux defect scratch (setFaceInterp(4))
  CCField fvM_, fvL_, cs_;  // collocated: mode-4 defect scratch (M·u, L_FV·u) + cell fluid fraction
  CCField xcx_, xcy_, xcz_;   // collocated: open-centroid wall distance per face (setFaceInterp(3))
  CCField old_[3], prev_[3];  // u^n time base + previous Picard iterate
  Comp C[3];
  peclet::core::FieldSet fields_;     // named directory of all cell fields (velocity/p/sdf + user)
  std::vector<ScalarField> scalars_;  // transported scalars (advection-diffusion)
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
