/// @file
/// @brief flow — IbmSolver MPI: initMpi, redistribute, rebalanceByWeights and the post-repartition
/// field-resize passes.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_MPI_HPP
#define PECLET_FLOW_FLOW_IBM_MPI_HPP

namespace peclet::flow {

#ifdef PECLET_FLOW_MPI
template <class Grid>
void Solver<Grid>::initMpi(int gnx, int gny, int gnz, MPI_Comm comm) {
  int size = 1;
  MPI_Comm_size(comm, &size);
  // Build the shared decomposition so the pressure MG can derive nested coarse levels
  // (CutcellMG::coarsened) — aligned ORB by default, or coarse-first when `set_decomposition`
  // asks for it. Depends only on the global grid and that setting, so it matches mpi_block()'s
  // sizing exactly when the same values are passed there.
  initMpi(CutcellMG::decomposition(static_cast<std::size_t>(size), gnx, gny, gnz, decompLevels_,
                                   decompMaxImbalance_),
          comm);
}

template <class Grid>
void Solver<Grid>::initMpi(const peclet::core::decomp::BlockDecomposer<3>& dec, MPI_Comm comm) {
  pcInDomain_ = CCField();  // WO-P3g: which ghosts carry a row depends on the decomposition
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
  // (rank-uniform: the decomposition is replicated), gated by `set_comm_avoiding`.
  velDevF_ = std::make_shared<GridHalo<MReal>>();
  velDevF_->init(*velHalo_);
  long minExt = std::numeric_limits<long>::max();
  for (const auto& s : dec.sizes())
    for (int k = 0; k < 3; ++k)
      minExt = std::min(minExt, (long)s[k]);
  caMomentum_ = (caMode_ & kCaMomentum) && minExt >= 4;
  for (bool& d : momStencilDirty_)
    d = true;
  dec_ = std::make_shared<peclet::core::decomp::BlockDecomposer<3>>(dec);  // remember the partition
  const auto oig = velHalo_->indexer().originInclGhost();
  og_ = {(int)oig[0] + G, (int)oig[1] + G, (int)oig[2] + G};  // block inner origin -> global parity
  // The colour field's OWN g=3 topology follows the same partition (VOF_PLAN §3 rule 1). Rebuilt
  // here rather than only in enableVof() so the order enable_vof/init_mpi does not matter — and
  // because a redistribute() re-inits through this path. The advector's block carries no state
  // between steps (C's canonical storage is the G=2 registry field "C"), so reallocating it is
  // free of consequence.
  if (vofEnabled_)
    buildVofBlock();
}

template <class Grid>
void Solver<Grid>::redistribute(const peclet::core::decomp::BlockDecomposer<3>& newDec) {
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
  //    `resizeForBlock` must run BETWEEN the two: `allocateBlock` only re-creates the buffers it
  //    owns by name, and `initMpi` -> `buildVofBlock` already writes through `cField_`, which is
  //    a registry alias. Without it the scatter below memcpy's the new padded extent into the
  //    old allocation (ASan heap-buffer-overflow -> `free(): invalid pointer`).
  allocateBlock((int)nb.size[0], (int)nb.size[1], (int)nb.size[2]);
  resizeForBlock();
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
  // 6. REFILL THE GHOSTS. `redistributeGridFields` moves the INNER cells only and says so
  //    ("the caller refills ghosts with a halo exchange afterwards"); the padded destination
  //    buffers were zero-initialised in step 1, so without this every migrated field enters the
  //    next step with a zero halo on the new partition. Measured at np = 4 on the VoF gate: one
  //    step after the move, du = 1.01e-01 against the single-rank reference (a run started
  //    directly on the SAME weighted partition reproduces it to 6.5e-17), and 0.00e+00 with the
  //    refill. np = 1/2 hid it because the new neighbour set happened to reproduce the old one.
  for (std::size_t k = 0; k < names.size(); ++k)
    fillGhosts(fields_.at(names[k]).data);
  for (auto& sc : scalars_)
    applyScalarBc(sc);  // a scalar's own domain BCs override the halo/periodic base
}

template <class Grid>
void Solver<Grid>::rebalanceByWeights(const std::vector<peclet::core::Real>& w) {
  if (!distributed_)
    return;
  int size = 1;
  MPI_Comm_size(comm_, &size);
  peclet::core::decomp::BlockDecomposer<3> newDec((std::size_t)size,
                                                  peclet::core::IVec<3>{gnx_, gny_, gnz_}, w);
  redistribute(newDec);
}
#endif

template <class Grid>
void Solver<Grid>::resizeForBlock() {
  reallocOwnedFields();
  rebindFieldAliases();
  resizeBlockScratch();
  for (int f = 0; f < 6; ++f)
    resampleBcProfile(f);  // the resampled inlet profile is indexed by LOCAL face position
}

template <class Grid>
void Solver<Grid>::reallocOwnedFields() {
  for (const auto& nm : fields_.names()) {
    auto& rec = fields_.at(nm);
    if (!rec.ownStorage || rec.data.extent(0) == n_)
      continue;
    rec.data = CCField(nm, n_);
  }
}

template <class Grid>
void Solver<Grid>::rebindFieldAliases() {
  auto bind = [this](CCField& f, const std::string& nm) {
    if (fields_.has(nm))
      f = fields_.at(nm).data;
  };
  bind(cField_, "C");          // VoF: the canonical G=2 colour mirror
  bind(kappaField_, "kappa");  // VoF: the curvature mirrors
  bind(kappaBranch_, "kappa_branch");
  bind(rhoField_, "rho");  // property closures
  bind(muField_, "mu");
  bind(epsField_, "eps");  // CFD-DEM: porosity + drag
  bind(dragBeta_, "drag_beta");
  bind(pcMdot_, "mdot");  // phase change
  bind(pcSrc_, "pc_source");
  bind(pcUser_, "div_source");
  bind(pcSink_, "div_sink");
  static const char* fn[3] = {"force_x", "force_y", "force_z"};
  for (int c = 0; c < 3; ++c)
    bind(cellForce_[c], fn[c]);
  for (auto& sc : scalars_)
    bind(sc.c, sc.name);
  for (auto& cl : closures_) {  // property_closures.hpp keeps the registry keys for exactly this
    if (fields_.has(cl.outName))
      cl.out = fields_.at(cl.outName).data;
    if (fields_.has(cl.in0Name))
      cl.in0 = CCConst(fields_.at(cl.in0Name).data);
    if (!cl.in1Name.empty() && fields_.has(cl.in1Name))
      cl.in1 = CCConst(fields_.at(cl.in1Name).data);
  }
}

template <class Grid>
void Solver<Grid>::resizeIfAllocated(CCField& f, const char* label, std::size_t n) {
  if (f.extent(0) != 0 && f.extent(0) != n)
    f = CCField(label, n);
}

template <class Grid>
void Solver<Grid>::resizeBlockScratch() {
  const std::size_t n = n_, n1 = n1_;
  // --- velocity / pressure scratch --------------------------------------------------------
  resizeIfAllocated(velRes_, "velRes", n);
  resizeIfAllocated(zp1_, "zp1", n1);
  resizeIfAllocated(vmgTheta_, "vmgTheta", n);
  resizeIfAllocated(vmgClean_, "vmgClean", n);
  for (int c = 0; c < 3; ++c) {
    resizeIfAllocated(uStar_[c], "uStar", n);
    resizeIfAllocated(advRhs_[c], "advRhs", n);
    resizeIfAllocated(uBc_[c], "uBc", n);
    resizeIfAllocated(uwCell_[c], "uwCell", n);
    resizeIfAllocated(uwAdv_[c], "uwAdv", n);
    resizeIfAllocated(faceAcc_[c], "faceAcc", n);
  }
  haveUStar_ = false;  // u* belonged to the old block's last momentum solve
  haveAdvRhs_ = false;
  // --- ghost projection (its overlay is rebuilt by setSolid) ------------------------------
  resizeIfAllocated(oxb_, "oxb", n);
  resizeIfAllocated(oyb_, "oyb", n);
  resizeIfAllocated(ozb_, "ozb", n);
  resizeIfAllocated(sdfGp_, "peclet::flow::sdfGp", n);
  resizeIfAllocated(gpRh_, "gpRh", n1);
  resizeIfAllocated(gpT_, "gpT", n1);
  resizeIfAllocated(gpZ2_, "gpZ2", n1);
  resizeIfAllocated(gpX2_, "gpXg2", n);
  // --- VoF (the g=3 block itself is rebuilt by buildVofBlock) -----------------------------
  resizeIfAllocated(vofCs_, "vofCs", n);
  resizeIfAllocated(vofSolidG2_, "vofSolidG2", n);
  for (int c = 0; c < 3; ++c) {
    resizeIfAllocated(uAdv_[c], "uAdv", n);
    resizeIfAllocated(vofDynVel_[c], "vofDynVel", n);
    resizeIfAllocated(csfBlkF_[c], "vof::blockcsf", n);
  }
  // --- phase change -----------------------------------------------------------------------
  resizeIfAllocated(pcArea_, "pc_area", n);
  for (int c = 0; c < 3; ++c) {
    resizeIfAllocated(pcNrm_[c], "pc_nrm", n);
    resizeIfAllocated(pcGn_[c], "pc_gn", n);
  }
  resizeIfAllocated(pcDep_, "pc_dep", n);
  resizeIfAllocated(pcTgt_, "pc_tgt", n);
  resizeIfAllocated(pcCnew_, "pc_cnew", n);
  resizeIfAllocated(pcDefic_, "pc_defic", n);
  resizeIfAllocated(pcKcell_, "pc_k", n);
  resizeIfAllocated(pcRcp_, "pc_rcp", n);
  resizeIfAllocated(pcClsPrev_, "pc_cls_prev", n);
  resizeIfAllocated(pcCarrySrc_, "pc_carry_src", n);
  resizeIfAllocated(pcMdotFit_, "pc_mdot_fit", n);
  resizeIfAllocated(pcKappa_, "pc_kappa", n);
  resizeIfAllocated(pcAreaCg2_, "pc_area_cascade", n);
  resizeIfAllocated(pcTgam_, "pc_tgam", n);
  resizeIfAllocated(pcGphi_, "pc_gphi", n);
  pcInDomain_ = CCField();  // which ghosts carry a row depends on the decomposition (WO-P3g)
  pcMaskFresh_ = false;
  // --- transported scalars (their operator is rebuilt; the solution rides the registry) -----
  for (auto& sc : scalars_) {
    resizeIfAllocated(sc.cOld, "scalar_old", n);
    resizeIfAllocated(sc.b, "scalar_b", n);
    resizeIfAllocated(sc.AC, "scalar_AC", n);
    resizeIfAllocated(sc.AW, "scalar_AW", n);
    resizeIfAllocated(sc.AE, "scalar_AE", n);
    resizeIfAllocated(sc.AS, "scalar_AS", n);
    resizeIfAllocated(sc.AN, "scalar_AN", n);
    resizeIfAllocated(sc.AB, "scalar_AB", n);
    resizeIfAllocated(sc.AT, "scalar_AT", n);
    resizeIfAllocated(sc.dmask, "scalar_dmask", n);
    resizeIfAllocated(sc.dval, "scalar_dval", n);
    resizeIfAllocated(sc.gfmB, "scalar_gfmb", n);
    sc.stencilBuilt = false;  // the operator is a property of the block
  }
  // `ScalarField::kcell`/`rcp` are NOT the scalar's own storage: they ALIAS `pcKcell_`/`pcRcp_`
  // (set at `set_phase_change_energy`). Resizing them independently would give the energy
  // operator a different buffer from the one `pcUpdateEnergyProps` fills every step — measured
  // as dP 3.09e-04 against the never-rebalanced control at np = 4. Re-alias instead.
  if (pcEnergy_ && !pcTName_.empty() && hasScalar(pcTName_)) {
    ScalarField& sc = scalarField(pcTName_);
    sc.kcell = pcKcell_;
    sc.rcp = pcRcp_;
  }
  // --- variable density / porous (CFD-DEM) -------------------------------------------------
  resizeIfAllocated(rho1_, "rho1", n1);
  resizeIfAllocated(cx1_, "cx1", n1);
  resizeIfAllocated(cy1_, "cy1", n1);
  resizeIfAllocated(cz1_, "cz1", n1);
  resizeIfAllocated(eps1_, "eps1", n1);
  resizeIfAllocated(beta1_, "beta1", n1);
  resizeIfAllocated(depsdt_, "depsdt", n);
  resizeIfAllocated(divAdv_, "divAdv", n);
  resizeIfAllocated(epsRho_, "epsRho", n);
  resizeIfAllocated(epsPrev_, "epsPrev", n);
  // eps^n is STATE, not scratch: a re-partition is not a time step, so the porosity must not
  // appear to have jumped across it. Seed it from the migrated eps^{n+1} => d(eps)/dt = 0 over
  // the redistribute, which is the only choice that leaves the projection RHS unchanged.
  if (epsPrev_.extent(0) == n && epsField_.extent(0) == n)
    Kokkos::deep_copy(epsPrev_, epsField_);
  for (int c = 0; c < 3; ++c)
    for (int k = 0; k < 3; ++k)
      if (tEx_[c][k].extent(0) != 0 && tEx_[c][k].extent(0) != (std::size_t)nx_ * ny_ * nz_) {
        tEx_[c][k] = CCField();  // exact crossings are single-rank only; drop the stale set
        hasExactCross_ = false;
      }
  // `cutOwner_` is deliberately NOT resized: every consumer guards on
  // `extent(0) == nx_*ny_*nz_` and skips, so a stale-sized owner map is inert, while a
  // freshly-zeroed one would silently name instance 0 as the owner of every cut cell.
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_MPI_HPP
