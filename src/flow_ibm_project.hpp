/// @file
/// @brief flow — IbmSolver pressure projection: step(), the project() stages, the buildRhs family,
/// and the collocated face-acceleration / applyFaceAcceleration path.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_PROJECT_HPP
#define PECLET_FLOW_FLOW_IBM_PROJECT_HPP

namespace peclet::flow {

template <class Grid>
void Solver<Grid>::step() {
  // ISSUES sweep item 1 (VOF_PLAN section 13 item 9): make `step()` ATOMIC across the two
  // explicit two-phase stability throws. Both used to fire AFTER the momentum half had already
  // advanced by the rejected dt -- the Weymouth-Yue boundedness cap from inside `advectVof`
  // (which runs at the TAIL of the step without momentum consistency) and the Brackbill
  // capillary cap from `updateVofCurvature` (which runs after `advectVofMomentum`). So the
  // obvious driver pattern -- catch, halve dt, retry -- gave the momentum equation one extra
  // over-long step the colour never saw, i.e. it DESYNCHRONISED the two fields instead of
  // retrying. Both checks are now evaluated here, at the head of `step()` and before the first
  // mutator, so a throw leaves every field bitwise as it was on entry. No-op unless VoF is on.
  vofStepPrecheck();
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
  {
    const double _t0 = vofTick();  // WO-V9
    phaseChangeStep();
    vofAdd(vt_.phaseChange, _t0);
  }
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
  {
    const double _t0 = vofTick();  // WO-V9
    updateVofCurvature();
    vofAdd(vt_.curvature, _t0);
  }
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
    if (advect_ || hasBc_ || (Grid::collocated && faceInterp_ >= 5 && faceInterp_ <= 7))
      for (int c = 0; c < 3; ++c)
        fillVelGhosts(c, 0);  // explicit ghosts (periodic + BC) for advect / embed defect matvec
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
             : (vofMomEnabled_ ? buildRhsVarMom(c)
                               : (effVarRho() ? buildRhsVar(c)
                                              : (hasCellForce_ ? buildRhsForced(c) : buildRhs(c))));
    // Rung V4 (WO-P): the balanced-force CSF, ADDED to whichever RHS the configuration built --
    // at the same place, and with the same face difference, as the incremental -grad(P^n) just
    // above it. Independent of the time term and the advection form, hence additive rather than a
    // fifth RHS kernel. Gated: byte-identical when surface tension is off.
    if (csfActive() && !coloFF) {
      const double _t0 = vofTick();  // WO-V9
      for (int c = 0; c < 3; ++c)
        vofBlockCsf() ? addCsfRhsBlocks(c)
                      : (csfMode_ == 0 ? addCsfRhs(c) : addCsfRhsCellInterp(c));
      vofAdd(vt_.csf, _t0);
    }
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
  // WO-P23: refresh k(C) / (rho c_p)(C) from the colour the energy solve is about to see. No-op
  // unless the consistent-energy transport is on.
  pcUpdateEnergyProps();
  // WO-P01: refresh the energy scalar's per-cell Dirichlet set from the colour the energy solve
  // is about to see. No-op unless the thermal mass flux is on.
  pcUpdateThermalMask();
  // Segregated multiphysics: advance any transported scalars with the just-projected
  // divergence-free velocity (properties frozen over the step). No-op (byte-identical) when no
  // scalar is registered.
  advanceScalars();
  // The UNSTABLE outflow regime, detected rather than guessed (peclet-examples ISSUES.md
  // "Inflow/outflow diverges to NaN"): reversed flow on an outflow face with the backflow
  // stabilization switched OFF. Checked only in that configuration (beta <= 0 is not the
  // default), once per solver; see outflowBackflow() for the mechanism.
  if (hasOutflow_ && backflowBeta_ <= 0.0 && !backflowWarned_) {
    const OutflowBackflow ob = outflowBackflow();
    if (ob.reversed > 0) {
      backflowWarned_ = true;
      int r = 0;
#ifdef PECLET_FLOW_MPI
      if (distributed_)
        MPI_Comm_rank(comm_, &r);
#endif
      if (r == 0)
        std::fprintf(stderr,
                     "peclet::flow WARNING: reversed flow on an OUTFLOW face (max u.n = -%.3e, "
                     "%.1f%% of the outlet area, kinetic-energy influx %.3e) with the backflow "
                     "stabilization OFF. The zero-gradient outflow is only conditionally "
                     "energy-stable under reversal (the re-entering flux carries the boundary "
                     "cell's own velocity back in with no sink); this is the configuration in "
                     "which the literature's backflow divergence occurs. Call "
                     "set_backflow_stabilization(beta) (default 0.2; beta >= 0.5 is the "
                     "unconditional bound), or lengthen the domain so the recirculation closes "
                     "before the outlet. Monitor with outflow_backflow().\n",
                     ob.maxReverse, 100.0 * ob.fraction, ob.energyInflux);
    }
  }
  tStep_ = phaseTick() - ts0;
  if (vofTiming_) {  // WO-V9: the coarse phases, summed over the profiled window
    tStepSum_ += tStep_;
    tPredSum_ += tPredictor_;
    tMomSum_ += tMomentum_;
    tProjSum_ += tProjection_;
    ++vt_.steps;
  }
}

template <class Grid>
bool Solver<Grid>::colocatedFaceForce() const {
  return Grid::collocated && (varRho_ || csfActive());
}

template <class Grid>
void Solver<Grid>::collocatedV8AutoFallback(const char* why) {
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

template <class Grid>
void Solver<Grid>::ensureFaceAcc() {
  for (int c = 0; c < 3; ++c)
    if (faceAcc_[c].extent(0) != n_)
      faceAcc_[c] = CCField("faceAcc", n_);
}

template <class Grid>
void Solver<Grid>::requireCollocatedFaceForceScope(const char* who) {
  if (!colocatedFaceForce())
    return;
  std::string m(who);
  if (hasSolid_)
    throw std::runtime_error(
        m +
        ": variable density / surface tension on SolverColocated is rung V8 and is ALL-FLUID "
        "only (set_pressure_geometry). An immersed solid needs the cut-cell face acceleration "
        "and the matching one-sided closures — not this rung.");
  if (ghostProjection_)
    throw std::runtime_error(
        m +
        ": the ghost projection (v1) does not support variable density; select "
        "set_collocated_scheme(\"gauge-exact\") (the AUTO fallback already does).");
  if (rhoFaceHarmonic_)
    throw std::runtime_error(
        m +
        ": set_rho_face_harmonic is not wired into the collocated face acceleration (the "
        "face force and the face coefficient would use different rho_f). Staggered only.");
}

template <class Grid>
void Solver<Grid>::buildRhs(int c) {
  CCExec space;
  const double idiag = rho_ / dt_, fc = f_[c], rho = rho_, wc = u_.w[c];
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
  const bool ifou = implicitAdv() &&
                    deferredCorr_;  // deferred correction: keep (HO - FOU) explicit in the RHS
                                    // (implicit on the domain-BC path by default, opt-in elsewhere)
  const int sch = advScheme_;       // 0 = SOU (default), 1 = Koren TVD
  // R0: stash the explicit advective term for the reaction-force budget (staggered scenes only).
  // The LAST Picard iteration overwrites, matching uStar_'s convention -- that is the RHS the
  // final momentum solve actually saw.
  const bool sa = ensureAdvStash(c, adv);
  CCField ar = advRhs_[c];
  // Embed rung 5: wall-aware pressure force (collocated): -grad(P) = the TRANSPOSE of the
  // wall-aware cell->face constraint interpolation, precomputed per component (the plain path's
  // central difference is the transpose of the plain 1/2-1/2 average, so this keeps the
  // momentum/constraint operators an adjoint pair on both paths).
  const bool tg = Grid::collocated && faceInterp_ == 5 && incr;
  // embed 6/7: openness-weighted -grad(P^n) predictor, matching the fs-weighted correction
  const bool wg = Grid::collocated && (faceInterp_ == 6 || faceInterp_ == 7) && incr;
  // ghost mode and the gauge-exact scheme: directional gpCenterGrad predictor — the mode-0
  // central difference reads the decoupled P=0 at solid-centered cells, a gauge-dependent O(1)
  // gradient error at every cut cell (measured O(1/h) in physical units,
  // ghost_collocated_apriori.py [C2]).
  const bool gg = Grid::collocated && (ghostProjection_ || faceInterp_ == 9) && incr;
  if constexpr (Grid::collocated) {
    // Phase 2 C2: each gradient kernel carries the metric weight w_c of ITS OWN axis on its
    // output (doc/anisotropic_metric.md §3), which is why the `gpw(i)` branch of the RHS below
    // is not scaled again.
    if (gg) {
      gpCenterGrad(tgp_, CCConst(P_), CCConst(ghostProjection_ ? sdfGp_ : sdf_), c, e_, G, u_.w[c]);
    } else if (tg) {
      CCField xcs[3] = {xcx_, xcy_, xcz_};
      CCField oax[3] = {ox_, oy_, oz_};
      transposeGradWallAware(tgp_, CCConst(P_), CCConst(sdf_), CCConst(oax[c]), CCConst(xcs[c]),
                             true, c, e_, G, u_.w[c]);
    } else if (wg) {
      CCField oax[3] = {ox_, oy_, oz_};
      centerGradOpen(tgp_, CCConst(P_), CCConst(oax[c]), c, e_, G, u_.w[c]);
    }
  }
  CCConst gpw = CCConst(tgp_);  // empty view on the staggered path (tg/wg/gg false there)
  // Embed (5/6/7) FV momentum via DEFECT CORRECTION: solve M·u^{k+1} = M·u^k − rs·L_FV(u^k) +
  // rs·b_FV so the fixed point satisfies the second-order finite-volume balance L_FV·u* = b_FV
  // exactly, with the (stable, small-cell-safe) IBM matrix M only as preconditioner. fvM_ = M·u^k
  // (stencilMatvec), fvL_ = L_FV(u^k) (embedViscousApply: o_f faces + cs time + true-normal
  // wall drag). Interior cells: M = L_FV → the defect vanishes → byte-identical to mode 0.
  // Stokes only (advection folds into the IBM matrix, not yet into L_FV).
  // Porous advection-form compensation (+rho*u_f*div(u)_f): see the step() comment. Off (and the
  // view untouched) on every non-porous path.
  const bool pc = porous_ && advect_;
  CCConst dv = CCConst(divAdv_);
  const bool wd = Grid::collocated && faceInterp_ >= 5 && faceInterp_ <= 7;
  if constexpr (Grid::collocated)
    if (wd) {
      stencilMatvec(fvM_, CCConst(C[c].u), FPC(C[c].AC), FPC(C[c].AW), FPC(C[c].AE), FPC(C[c].AS),
                    FPC(C[c].AN), FPC(C[c].AB), FPC(C[c].AT), e_, G);
      embedViscousApply(fvL_, CCConst(C[c].u), CCConst(sdf_), CCConst(cs_), CCConst(ox_),
                        CCConst(oy_), CCConst(oz_), mu_, rho_ / dt_, e_, G, u_.w[0], u_.w[1],
                        u_.w[2], u_.hp[0], u_.hp[1], u_.hp[2]);
    }
  CCConst fvM = CCConst(fvM_), fvL = CCConst(fvL_), cs = CCConst(cs_);
  // b = descale*(idiag*u^n - rho*Koren(u^k) + rho*FOU(u^k) + f - grad P^n) - inhom  (+ BC fold
  // brhs). The time base is u^n (Picard); the advecting velocity & advected field are the current
  // iterate u^k.
  ccFor3(
      "rhs", C3{G, G, G}, C3{e.x - G, e.y - G, e.z - G}, KOKKOS_LAMBDA(int x, int y, int z) {
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
        // The pressure gradient of component c carries the metric weight w_c
        // (doc/anisotropic_metric.md §1.2/§2); wc == 1.0 exactly on the isotropic path.  The
        // gpw() kernels carry their own weight (§3), so they are not scaled again here.
        const double gp =
            !incr ? 0.0
            : Grid::collocated
                ? ((tg || wg || gg) ? gpw(i) : wc * (0.5 * (P((long)i + strd) - P((long)i - strd))))
                : wc * (P(i) - P((long)i - strd));
        if (wd) {  // FV defect-correction RHS  M·u − rs·(L_FV·u − b_FV),  b_FV = idt·cs·u^n +
                   // cs·(f − grad P); the fixed point L_FV·u* = b_FV.
          const double bfv = idiag * cs(i) * un(i) + cs(i) * (fc - gp);
          bb(i) = fvM(i) - rs(i) * (fvL(i) - bfv);
        } else {
          const double comp = pc ? rho * uu(i) * 0.5 * (dv(i) + dv((long)i - strd)) : 0.0;
          bb(i) = rs(i) * (idiag * un(i) + fc - rho * aK + rho * aF + comp - gp) +
                  (bc ? brhs(i) : -inh(i));
        }
      });  // BC fold (brhs) on the domain-BC path; -inhom on the IBM path (=0 for no-slip)
}

template <class Grid>
void Solver<Grid>::buildRhsForced(int c) {
  CCExec space;
  const double idiag = rho_ / dt_, fc = f_[c], rho = rho_, wc = u_.w[c];
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
  const bool tg = Grid::collocated && faceInterp_ == 5 && incr;  // wall-aware -grad(P)
  const bool gg = Grid::collocated && (ghostProjection_ || faceInterp_ == 9) &&
                  incr;  // directional ghost -grad(P)
  if constexpr (Grid::collocated) {
    if (gg) {
      gpCenterGrad(tgp_, CCConst(P_), CCConst(ghostProjection_ ? sdfGp_ : sdf_), c, e_, G, u_.w[c]);
    } else if (tg) {
      CCField xcs[3] = {xcx_, xcy_, xcz_};
      CCField oax[3] = {ox_, oy_, oz_};
      transposeGradWallAware(tgp_, CCConst(P_), CCConst(sdf_), CCConst(oax[c]), CCConst(xcs[c]),
                             true, c, e_, G, u_.w[c]);
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
        const double gp =
            !incr ? 0.0
            : Grid::collocated
                ? ((tg || gg) ? gpw(i) : wc * (0.5 * (P((long)i + strd) - P((long)i - strd))))
                : wc * (P(i) - P((long)i - strd));
        const double comp = pc ? rho * uu(i) * 0.5 * (dv(i) + dv((long)i - strd)) : 0.0;
        bb(i) = rs(i) * (idiag * un(i) + fc + fb(i) - rho * aK + rho * aF + comp - gp) +
                (bc ? brhs(i) : -inh(i));
      });
}

template <class Grid>
void Solver<Grid>::buildRhsVar(int c) {
  CCExec space;
  const double idt = 1.0 / dt_, fc = f_[c], wc = u_.w[c];
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
                          : Grid::collocated ? wc * (0.5 * (P((long)i + strd) - P((long)i - strd)))
                                             : wc * (P(i) - P((long)i - strd));
        const double fbF = 0.5 * (fb(i) + fb(i - strd));
        const double comp = pc ? rhoF * uu(i) * 0.5 * (dv(i) + dv((long)i - strd)) : 0.0;
        bb(i) = rs(i) * (rhoF * idt * un(i) + fc + fbF - rhoF * aK + rhoF * aF + comp - gp) +
                (bc ? brhs(i) : -inh(i));
      });
}

template <class Grid>
void Solver<Grid>::buildRhsVarMom(int c) {
  CCExec space;
  const double idt = 1.0 / dt_, fc = f_[c], wc = u_.w[c];
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
                          : Grid::collocated ? wc * (0.5 * (P((long)i + strd) - P((long)i - strd)))
                                             : wc * (P(i) - P((long)i - strd));
        const double fbF = 0.5 * (fb(i) + fb(i - strd));
        bb(i) = rs(i) * (rhoF * idt * ua(i) + fc + fbF - gp) + (bc ? brhs(i) : -inh(i));
      });
}

template <class Grid>
void Solver<Grid>::buildRhsColoFF(int c) {
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

template <class Grid>
void Solver<Grid>::applyFaceAcceleration() {
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
                      CCConst(oax[c]), haveRho, rho_, f_[c], incr, dt_, sc, e_, G, u_.w[c]);
    if (csfActive())
      addFaceAccelCsf(faceAcc_[c], CCConst(cField_), CCConst(kappaField_), CCConst(kappaBranch_),
                      rho, CCConst(oax[c]), haveRho, rho_, sigmaCsf_, 1.0 / u_.w[c], dt_, sc, e_,
                      G);
    addFaceIncrement(fa[c], CCConst(faceAcc_[c]), e_, G);
  }
}

template <class Grid>
void Solver<Grid>::applyCellFaceAverageCorrection() {
  CCField oax[3] = {ox_, oy_, oz_};
  const bool haveRho = effVarRho();
  CCConst rho = CCConst(haveRho ? effRhoField() : C[0].rscale);
  for (int c = 0; c < 3; ++c) {
    const long sc = strideOf(c);
    faceAccelSubGradPhi(faceAcc_[c], CCConst(phi_), rho, CCConst(oax[c]), haveRho, rho_, sc, e_, G,
                        u_.w[c]);
    applyCellFaceAverage(C[c].u, CCConst(faceAcc_[c]), CCConst(oax[c]), sc, e_, G);
  }
}

template <class Grid>
void Solver<Grid>::filterCellField(CCField f, int axis) {
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

template <class Grid>
void Solver<Grid>::project() {
  projectAssembleDivergence();
  projectBuildCoefficients();
  projectSolve();
  projectCorrectVelocities();
  projectPressureUpdate();
}

template <class Grid>
void Solver<Grid>::projectAssembleDivergence() {
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
    if (faceInterp_ == 5 || faceInterp_ == 7)  // wall-aware flux map at solid (embed 5/7)
      centerToFaceWallAware(uf_, vf_, wf_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u),
                            CCConst(sdf_), CCConst(xcx_), CCConst(xcy_), CCConst(xcz_), true, e_,
                            G);
    else
      centerToFace(uf_, vf_, wf_, CCConst(C[0].u), CCConst(C[1].u), CCConst(C[2].u), e_, G);
    faceFieldValid_ = true;  // ISSUES sweep item 5
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
      divergOpen(CCConst(uf_), CCConst(vf_), CCConst(wf_), CCConst(ox_), CCConst(oy_), CCConst(oz_),
                 div_, e_, G);
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
          const long i2 = (long)(x + G) + (long)(y + G) * e2.x + (long)(z + G) * (long)e2.x * e2.y;
          if (sd(i2) < 0.0)
            r((long)(x + 1) + (long)(y + 1) * e1.x + (long)(z + 1) * (long)e1.x * e1.y) = 0.0;
        });
  }
}

template <class Grid>
void Solver<Grid>::projectBuildCoefficients() {
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
    // WO-R2 item 1: buildRhoCoeff covers the LOW domain face of each axis (an inner index) but
    // not the HIGH one (a ghost index). Fill the high outflow planes here, and tell the MG to
    // carry the caller's coefficient at every Dirichlet domain face instead of re-imposing the
    // literal openness 1.0 (which used to overwrite `rho0/rho_f` with 1 on BOTH sides and made
    // the low-side outlet inconsistent with projectCorrectVar by the full density ratio).
    {
      CCField cc[3] = {cx1_, cy1_, cz1_};
      for (int a = 0; a < 3; ++a)
        if (bc_[2 * a + 1] == 3 && touchesGlobalFace(2 * a + 1))
          buildRhoCoeffOutflowFace(cc[a], CCConst(rho1_), rho_, e1_, 1, a, rhoFaceHarmonic_);
    }
    mg_.setBoundaryConditions(bc_);
    mg_.setOutflowCoefficient(hasOutflow_ && outflowOpCoeff_);
    mg_.setOpenness(CCConst(cx1_), CCConst(cy1_), CCConst(cz1_), u_.w[0], u_.w[1], u_.w[2]);
    mg_.setOutflowCoefficient(false);
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
    mg_.setOpenness(CCConst(cx1_), CCConst(cy1_), CCConst(cz1_), u_.w[0], u_.w[1], u_.w[2]);
    chebBoundsSet_ = false;
  }
}

template <class Grid>
void Solver<Grid>::projectSolve() {
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
          mg_.solveBiCGStab(rhs1_, phi1_, r_, gpRh_, pp_, Ap_, gpT_, z_, gpZ2_, pcgMaxit_, pcgRtol_,
                            2, 2, 12, gpOv_, gpNRows_, C3{nx_, ny_, nz_}, gpX2_, velDev_.get(), e_);
    else
#endif
      lastPressureIters_ =
          mg_.solveBiCGStab(rhs1_, phi1_, r_, gpRh_, pp_, Ap_, gpT_, z_, gpZ2_, pcgMaxit_, pcgRtol_,
                            2, 2, 12, gpOv_, gpNRows_, C3{nx_, ny_, nz_});
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
  // ISSUES sweep item 6: a solve that gave up on a non-finite recurrence scalar reports the
  // iteration CAP (see CutcellMG::solvePCG) and raises this flag, so `pressure_solve_failed()`
  // and the usual rule-3b "no capped solve" check both catch it. It used to print one line to
  // stdout, silently zero the correction and report 0 iterations.
  lastPressureFailed_ = mg_.lastSolveFailed();
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
          const long i2 = (long)(x + G) + (long)(y + G) * e2.x + (long)(z + G) * (long)e2.x * e2.y;
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
}

template <class Grid>
void Solver<Grid>::projectCorrectVelocities() {
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
    // Phase 2 C2 (doc/anisotropic_metric.md §3): the per-axis weight w_a multiplies the whole
    // existing correction, matching the w_a on the operator row of that axis.  1.0 isotropic.
    if (varRho_)
      projectCorrectVar(uf_, vf_, wf_, CCConst(phi_), CCConst(rhoField_), rho_, e_, G, u_.w[0],
                        u_.w[1], u_.w[2]);
    else
      projectCorrect(uf_, vf_, wf_, CCConst(phi_), e_, G, u_.w[0], u_.w[1], u_.w[2]);
    if (fluidOnlyMode_ == 2)  // Design B: replace the solid side's phi=0 by phibar_s at
      starCorrectFaces(uf_, vf_, wf_, CCConst(phi_), starOv_, nStar_,  // fluid|solid faces
                       C3{nx_, ny_, nz_}, e_, G, e_, G, exactResidual_, u_.w[0], u_.w[1], u_.w[2]);
    fillGhosts(uf_);
    fillGhosts(vf_);
    fillGhosts(wf_);    // complete the divergence-free face field (boundary faces)
    if (hasOutflow_) {  // correct the high-side outflow face on the face field so mass leaves
                        // (phi=0 there)
      B3 e{e_.x, e_.y, e_.z};
      CCField fa[3] = {uf_, vf_, wf_};
      // WO-R2 item 1: same rule as the staggered path below — with the operator's outflow row
      // carrying rho0/rho_f (setOutflowCoefficient) the consistent correction carries it too.
      const bool var = varRho_ && !porous_ && outflowRhoCorr_;
      for (int a = 0; a < 3; ++a)
        if (bc_[2 * a + 1] == 3 && touchesGlobalFace(2 * a + 1)) {
          if (var)
            bcCorrectOutflowVar(fa[a], phi_, rhoField_, rho_, e, G, a, rhoFaceHarmonic_, u_.w[a]);
          else
            bcCorrectOutflow(fa[a], phi_, e, G, a, u_.w[a]);
        }
    }
    if (colocatedFaceForce()) {
      // rung V8 (WO-T): the cell sees the AVERAGE of what its two faces saw — the force
      // acceleration minus the projection's own rho-weighted face correction — through the same
      // averaging operator projectCorrectCenter applies to phi differences. A hydrostatic column
      // and a stationary droplet are exactly balanced on the faces, so the cell averages an exact
      // zero. This replaces the whole constant-density cell-correction chain below.
      applyCellFaceAverageCorrection();
    } else if (ghostProjection_ || faceInterp_ == 9) {
      // Ghost cell correction (also the gauge-exact scheme): the directional gpCenterGrad
      // gradient of phi — 2nd-order one-sided at cut cells, never reads a decoupled
      // (solid/pocket) phi. The same operator supplies the momentum's -grad(P^n) predictor
      // (buildRhs), so the pressure force the momentum feels and the correction stay one
      // operator family.
      for (int cc = 0; cc < 3; ++cc) {
        gpCenterGrad(tgp_, CCConst(phi_), CCConst(ghostProjection_ ? sdfGp_ : sdf_), cc, e_, G,
                     u_.w[cc]);
        subtractField(C[cc].u, CCConst(tgp_), e_, G);
      }
    } else if (faceInterp_ == 5) {  // embed 5: cell correction = the TRANSPOSE of the
      // wall-aware map, keeping (T, Tᵀ) an adjoint pair (transposeGradWallAware)
      CCField xcs[3] = {xcx_, xcy_, xcz_};
      CCField oax[3] = {ox_, oy_, oz_};
      for (int cc = 0; cc < 3; ++cc) {
        transposeGradWallAware(tgp_, CCConst(phi_), CCConst(sdf_), CCConst(oax[cc]),
                               CCConst(xcs[cc]), true, cc, e_, G, u_.w[cc]);
        subtractField(C[cc].u, CCConst(tgp_), e_, G);
      }
    } else if (faceInterp_ == 6 || faceInterp_ == 7) {  // embed: openness-WEIGHTED cell correction
      // (full open-face pressure force at cut cells) — Basilisk centered_grad
      projectCorrectCenterOpen(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(ox_), CCConst(oy_),
                               CCConst(oz_), e_, G, u_.w[0], u_.w[1], u_.w[2]);
    } else {
      projectCorrectCenter(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(ox_), CCConst(oy_),
                           CCConst(oz_), e_, G, u_.w[0], u_.w[1], u_.w[2]);
    }
  } else {
    if (porous_ && porousCons_)  // eps-conservative gradient rho*idt/(eps_f rho idt + beta_f),
                                 // matching buildPorousCoeffCons (see mac_pressure.hpp)
      projectCorrectPorousCons(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(epsField_),
                               CCConst(dragBeta_), hasDrag_, rho_ / dt_, e_, G, u_.w[0], u_.w[1],
                               u_.w[2]);
    else if (porous_ &&
             hasDrag_)  // drag-relaxed gradient w_f=idt/(idt+beta_f), matching buildPorousCoeffDrag
      projectCorrectPorousDrag(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(dragBeta_),
                               rho_ / dt_, e_, G, u_.w[0], u_.w[1], u_.w[2]);
    else if (varRho_ && rhoFaceHarmonic_)  // the WO-J harmonic knob: coefficient AND correction
      projectCorrectVarHarm(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(rhoField_), rho_, e_, G,
                            u_.w[0], u_.w[1], u_.w[2]);
    else if (varRho_)  // per-face 1/rho on the gradient, matching the operator coefficient
      projectCorrectVar(C[0].u, C[1].u, C[2].u, CCConst(phi_), CCConst(rhoField_), rho_, e_, G,
                        u_.w[0], u_.w[1], u_.w[2]);
    else
      projectCorrect(C[0].u, C[1].u, C[2].u, CCConst(phi_), e_, G, u_.w[0], u_.w[1], u_.w[2]);
    if (hasOutflow_) {  // correct the high-side outflow normal face that projectCorrect misses
                        // (mass leaves)
      B3 e{e_.x, e_.y, e_.z};
      // WO-R item 4: with variable density every OTHER corrected face carries the mobility
      // factor rho0/rho_f (projectCorrectVar) and this one did not — a ratio-sized error on the
      // outflow face, invisible at constant density (rho_f == rho0) which is why the channel/BFS
      // validations never saw it. The porous branches keep the plain correction: their
      // coefficient is the drag/eps relaxation, not 1/rho, and a two-phase porous outlet is not
      // this rung. `!varRho_` is byte-identical.
      // `set_outflow_rho_correction(False)` drops the 1/rho_f
      // factor. DEFAULT ON since WO-R2 fixed the operator's outflow-face coefficient — see
      // setOutflowRhoCorrection for the before/after table and the mechanism.
      const bool var = varRho_ && !porous_ && outflowRhoCorr_;
      for (int a = 0; a < 3; ++a)
        if (bc_[2 * a + 1] == 3 && touchesGlobalFace(2 * a + 1)) {
          if (var)
            bcCorrectOutflowVar(C[a].u, phi_, rhoField_, rho_, e, G, a, rhoFaceHarmonic_, u_.w[a]);
          else
            bcCorrectOutflow(C[a].u, phi_, e, G, a, u_.w[a]);
        }
      outflowCorrValid_ = true;  // the outflow face now carries the mass that leaves
    }
  }
  // the grad(phi) correction also touches solid faces; re-impose no-slip there so the decoupled
  // solid velocity cannot accumulate (matches the CUDA apply_mask/mask_k after correct_k ->
  // stability).
  for (int c = 0; c < 3; ++c)
    maskVelocity(c);
}

template <class Grid>
void Solver<Grid>::projectPressureUpdate() {
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
    const double ct = pressUnderRelax_ * rho_ / dt_, mu = rotationalP_ ? rotWeight_ * mu_ : 0.0;
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

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_PROJECT_HPP
