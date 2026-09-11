/// @file
/// @brief flow — IbmSolver domain boundary conditions: setDomainBc/profiles, pressureBcGhost,
/// fillVelGhosts*, setupBcDiffusion, backflow stabilization.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_BC_HPP
#define PECLET_FLOW_FLOW_IBM_BC_HPP

namespace peclet::flow {

template <class Grid>
void Solver<Grid>::setDomainBc(int face, int type, double vx, double vy, double vz) {
  if (face < 0 || face > 5)
    throw std::invalid_argument("set_domain_bc: face must be 0..5 (-x,+x,-y,+y,-z,+z)");
  if (type < 0 || type > 4)
    throw std::invalid_argument(
        "set_domain_bc: type must be 0 periodic / 1 wall / 2 inflow / 3 outflow / 4 free-slip");
  const bool valueUpdateOnly = geometryBuilt_ && type == bc_[face];
  if (geometryBuilt_ && !valueUpdateOnly)
    throw std::runtime_error(
        "set_domain_bc: call BEFORE the geometry (set_solid / set_pressure_geometry / "
        "set_solid_from_scene) to CHANGE a face's TYPE -- the type is folded into the "
        "operators when the geometry is built. A VALUE update (vx, vy, vz) on a face whose "
        "type is unchanged is allowed after the geometry.");
  // WO-P3g: the "does this cell carry a row" mask depends on which domain faces are periodic.
  pcInDomain_ = CCField();
  bc_[face] = type;
  // Boundary velocities are PHYSICAL; the kernels read the index velocity v_a = u_a*tRef/h_a.
  bcVelPhys_[face][0] = vx;
  bcVelPhys_[face][1] = vy;
  bcVelPhys_[face][2] = vz;
  for (int a = 0; a < 3; ++a)
    bcVel_[face][a] = bcVelPhys_[face][a] * u_.velToInt(a);
  hasBc_ = false;
  hasOutflow_ = false;
  for (int i = 0; i < 6; ++i) {
    if (bc_[i])
      hasBc_ = true;
    if (bc_[i] == 3)
      hasOutflow_ = true;
  }
  // ISSUES sweep item 3: which faces are WETTING walls is part of the colour block's geometry,
  // so a BC changed after `enable_vof` has to rebuild it. Inert (and byte-identical) without a
  // contact angle: `buildVofGeometry` then takes exactly the branch it took before.
  if (vofEnabled_)
    buildVofGeometry();
  // A VALUE update on an already-built staggered geometry: setupBcDiffusion() bakes bcVel_'s
  // TANGENTIAL component into the implicit-diffusion fold (bcDcorr_/bcBrhs_) once, at geometry
  // time -- ghost fills read bcVel_ live every step (the NORMAL component takes effect on its
  // own), but the fold does not, so a lid-driven cavity ramping its lid's tangential velocity
  // after set_solid would silently keep stepping on the OLD speed without this refresh. The
  // collocated grid uses explicit reflection ghosts (no fold) and needs none.
  if (valueUpdateOnly && hasBc_ && !Grid::collocated)
    setupBcDiffusion();
}

template <class Grid>
void Solver<Grid>::setDomainBcProfile(int face, const std::vector<double>& prof, int nb, int nc) {
  if (geometryBuilt_ && bc_[face] != 2)
    throw std::runtime_error(
        "set_domain_bc_profile: call BEFORE the geometry (set_solid / set_pressure_geometry / "
        "set_solid_from_scene) to make a face inflow -- the type is folded into the operators "
        "when the geometry is built. Updating the profile on a face that is ALREADY inflow is "
        "allowed after the geometry.");
  // The user's (nb, nc, 3) profile is KEPT so the resampling can be redone when the block
  // changes size — the resampled buffer is indexed by LOCAL face position, so a redistribute
  // that changes this rank's face grid invalidates it (see resampleBcProfile).
  bcProfRaw_[face] = prof;
  bcProfNb_[face] = nb;
  bcProfRawNc_[face] = nc;
  resampleBcProfile(face);
  bc_[face] = 2;
  hasBc_ = true;  // a profiled face is an inflow
}

template <class Grid>
void Solver<Grid>::resampleBcProfile(int face) {
  if (bcProfRaw_[face].empty())
    return;
  const std::vector<double>& prof = bcProfRaw_[face];
  const int nb = bcProfNb_[face], nc = bcProfRawNc_[face];
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
      // The raw profile is kept in the caller's PHYSICAL velocity; convert per component here,
      // so a re-resample after a redistribute (or a reference-scale change) reconverts too.
      for (int k = 0; k < 3; ++k)
        h(((long)p0 * Lc + p1) * 3 + k) =
            prof[((std::size_t)ib * nc + ic) * 3 + k] * u_.velToInt(k);
    }
  Kokkos::deep_copy(pf, h);
  bcProf_[face] = pf;
  bcProfNc_[face] = Lc;
}

template <class Grid>
void Solver<Grid>::applyBackflowStab(int c) {
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
          const double back = sgn * u(i);  // > 0 exactly where the outflow reverses (|min(u.n,0)|)
          if (back > 0.0)
            AC(i) += (MReal)(beta * rho * back);  // dissipative diagonal (u_ext = 0)
        });
  }
}

template <class Grid>
void Solver<Grid>::pressureBcGhost() {
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

template <class Grid>
void Solver<Grid>::fillVelGhosts(int comp, int fold) {
  // This is the fill that RE-IMPOSES the zero-gradient outflow face, i.e. the one that erases
  // `bcCorrectOutflow`'s correction (WO-R; see fillVelGhostsKeepOutflow). Record that so the
  // VoF bridge knows whether there is a correction left to preserve.
  outflowCorrValid_ = false;
  fillVelGhostsTo(C[comp].u, comp, fold);
}

template <class Grid>
void Solver<Grid>::applyVelocityBcComp(int comp, int fold, bool doOutflow) {
  applyVelocityBcCompTo(C[comp].u, comp, fold, doOutflow);
}

template <class Grid>
void Solver<Grid>::fillVelGhostsTo(CCField f, int comp, int fold, bool doOutflow) {
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    velDev_->exchange(f);
    applyVelocityBcCompTo(f, comp, fold, doOutflow);
    return;
  }
#endif
  for (int a = 0; a < 3; ++a)
    if (bc_[2 * a] == 0 && bc_[2 * a + 1] == 0)
      fillAxis(f, a);
  applyVelocityBcCompTo(f, comp, fold, doOutflow);
}

template <class Grid>
void Solver<Grid>::applyVelocityBcCompTo(CCField f, int comp, int fold, bool doOutflow) {
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
        if (t == 4) {  // free-slip / symmetry: the normal component is odd-reflected about the
                       // face (face value 0, as a wall), the tangential ones mirrored (zero
                       // normal derivative)
          if (comp == a)
            bcVelocityColocated(f, e, G, a, s, 0.0);
          else
            bcMirrorGhost(f, e, G, a, s);
          continue;
        }
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
      if (t == 4) {  // free-slip / symmetry (mac_bc.hpp bcSlipComp)
        bcSlipComp(f, e, G, a, s, comp, fold);
        continue;
      }
      if (bcProf_[ff].extent(0) > 0)
        bcVelocityComp(f, e, G, a, s, comp, 0.0, fold, bcProf_[ff], bcProfNc_[ff]);
      else
        bcVelocityComp(f, e, G, a, s, comp, bcVel_[ff][comp], fold);
    }
}

template <class Grid>
void Solver<Grid>::setupBcDiffusion() {
  B3 e{e_.x, e_.y, e_.z};
  for (int c = 0; c < 3; ++c) {
    Kokkos::deep_copy(bcDcorr_[c], 0.0);
    Kokkos::deep_copy(bcBrhs_[c], 0.0);
    for (int a = 0; a < 3; ++a) {
      // The fold uses the beta of the FACE's own axis (doc/anisotropic_metric.md §2/§4.1);
      // b_a == mu_ exactly on the isotropic path.
      const double beta = mu_ * u_.w[a];
      for (int s = 0; s < 2; ++s) {
        const int t = bc_[2 * a + s];
        if (!touchesGlobalFace(2 * a + s))
          continue;  // the implicit wall fold belongs to the rank owning that global face
        double dval, bval;
        if (t == 3) {
          dval = -beta;
          bval = 0.0;
        } else if (t == 4 && c != a) {  // free-slip: zero-gradient tangential (the mirror
          dval = -beta;                 // neighbour IS the cell -> the face's beta drops out);
          bval = 0.0;                   // the normal component is held at 0 like a wall
        } else if (t != 0 && c != a) {
          dval = beta;
          bval = 2.0 * beta * bcVel_[2 * a + s][c];
        } else
          continue;  // periodic, or the normal component at a wall (held directly)
        bcDiffusionFold(bcDcorr_[c], bcBrhs_[c], e, G, a, s, dval, bval);
      }
    }
    // dcorr is passed to the (double) const-coeff smoother diffSmoothColor each sweep -- matching
    // CUDA diff_k (Ac + dcorr in double), NOT baked into the float stencil.
  }
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_BC_HPP
