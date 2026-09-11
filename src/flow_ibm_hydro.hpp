/// @file
/// @brief flow — IbmSolver hydrodynamic force and torque: the reaction-budget terms and wall
/// probes.
///
/// Out-of-line member definitions of peclet::flow::Solver (QUALITY_PLAN G.1 domain split).
/// Declarations, docstrings and the class state live in flow_ibm.hpp, which includes this file at
/// the bottom; this file is never included on its own.
#ifndef PECLET_FLOW_FLOW_IBM_HYDRO_HPP
#define PECLET_FLOW_FLOW_IBM_HYDRO_HPP

namespace peclet::flow {

template <class Grid>
void Solver<Grid>::scaleForceTorque(std::vector<double>& out, std::size_t m) const {
  const double kF = u_.forceTotalToPhys(), kT = u_.torqueToPhys();
  if (kF == 1.0 && kT == 1.0)
    return;
  for (std::size_t b = 0; m > 0 && b * m < out.size(); ++b) {
    const double k = (b == 1) ? kT : kF;
    for (std::size_t i = b * m; i < (b + 1) * m && i < out.size(); ++i)
      out[i] *= k;
  }
}

template <class Grid>
std::vector<double> Solver<Grid>::hydroForceTorque() {
  std::vector<double> out((std::size_t)(nInst_ > 0 ? nInst_ : 0) * 12, 0.0);
  if (!hasScene_ || nInst_ <= 0 || cutOwner_.extent(0) != (std::size_t)nx_ * ny_ * nz_)
    return out;
  const std::size_t m = (std::size_t)nInst_ * 3;
  Kokkos::View<double*, CCMem> Fd("hydroF", m), Td("hydroT", m), Pd("hydroFp", m), Vd("hydroFv", m);
  Kokkos::deep_copy(Fd, 0.0);
  Kokkos::deep_copy(Td, 0.0);
  Kokkos::deep_copy(Pd, 0.0);
  Kokkos::deep_copy(Vd, 0.0);
  CCExec space;
  const C3 e = e_, og = og_;
  const int nx = nx_, ny = ny_;
  const double mu = mu_;
  // §4.4: kA[a] = V'/h_a' turns the index fragment normal W_a into the physical area vector; the
  // hp ratios below turn the index-velocity gradient pair into the physical strain rate.
  const double kA0 = u_.vol / u_.hp[0], kA1 = u_.vol / u_.hp[1], kA2 = u_.vol / u_.hp[2];
  const double hp0 = u_.hp[0], hp1 = u_.hp[1], hp2 = u_.hp[2];
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
  const SceneMap sm = sceneMap();
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
        const double kA[3] = {kA0, kA1, kA2};  // V'/h_a'  (exactly 1.0 isotropic)
        const double hpv[3] = {hp0, hp1, hp2};
        double dF[3], dFp[3], dFv[3];
        for (int a = 0; a < 3; ++a) {
          // A_wall is fluid-outward; the traction on the BODY takes -A_wall. Pressure and
          // viscous parts are kept apart because they fail differently: the pressure term reads
          // one cell-centred value, while the viscous term differences a velocity whose stencil
          // reaches into solid cells -- so a deficit that lives entirely in one of them says
          // immediately which.
          dFp[a] = (p * A[a]) * kA[a];
          double t = 0.0;
          for (int b = 0; b < 3; ++b) {
            const double rab = hpv[a] / hpv[b], rba = hpv[b] / hpv[a];
            t += mu * (rab * gu[a][b] + rba * gu[b][a]) * (A[b] * kA[b]);
          }
          dFv[a] = -t;
          dF[a] = dFp[a] + dFv[a];
        }
        // The instance centre and the min-image box live in the SCENE's coordinates; dF is an
        // index-unit force, so the lever arm has to come back to index units (a uniform scale, so
        // min-imaging in either system is the same shortening).
        const peclet::core::Vec3<double> rp = peclet::core::geom::minImage(
            peclet::core::Vec3<double>{
                sm.a[0] + sm.b[0] * (double)(x - G + og.x) - cen[3 * oi + 0],
                sm.a[1] + sm.b[1] * (double)(y - G + og.y) - cen[3 * oi + 1],
                sm.a[2] + sm.b[2] * (double)(z - G + og.z) - cen[3 * oi + 2]},
            box);
        const peclet::core::Vec3<double> r{rp.x * sm.dToInt, rp.y * sm.dToInt, rp.z * sm.dToInt};
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
  scaleForceTorque(out, m);
  scaleForceTorque(out, m);
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

template <class Grid>
std::vector<double> Solver<Grid>::hydroForceTorqueReaction() {
  std::vector<double> out((std::size_t)(nInst_ > 0 ? nInst_ : 0) * 6, 0.0);
  if (!hasScene_ || nInst_ <= 0)
    return out;
  if constexpr (Grid::collocated)
    throw std::runtime_error("hydro_force_torque_reaction: staggered only (v1)");
  if (implicitAdv() || porous_ || varRho_ || varProps_ || hasBc_ || ghostProjection_ || hasDrag_ ||
      fluidOnlyMode_ != 0)
    throw std::runtime_error(
        "hydro_force_torque_reaction: implicit advection / porous / variable-properties / "
        "domain-BC / ghost-projection / drag / star modes put momentum terms in the step that "
        "this budget does not carry (v2) -- a missing term is a silently mis-attributed force");
  if (!haveUStar_)
    throw std::runtime_error(
        "hydro_force_torque_reaction: call step() first (u* is stashed "
        "during the step)");
  if (advect_ && (!haveAdvRhs_ || advRhs_[0].extent(0) != n_))
    throw std::runtime_error(
        "hydro_force_torque_reaction: the advective RHS term was not "
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
  const SceneMap sm = sceneMap();
  const bool hasFb = hasCellForce_;
  // §4.4: the momentum row of component c is a force density in the component-c normalisation, so
  // the total force carries h_c'*V' (exactly 1.0 isotropic -- an identity multiplication).
  const double kR[3] = {u_.hp[0] * u_.vol, u_.hp[1] * u_.vol, u_.hp[2] * u_.vol};
  // ... while the v3 wall-torque traction below carries V'/h_b' on its AREA vector (E3).
  const double kA0 = u_.vol / u_.hp[0], kA1 = u_.vol / u_.hp[1], kA2 = u_.vol / u_.hp[2];
  for (int c = 0; c < 3; ++c) {
    CCConst un = CCConst(old_[c]), uc = CCConst(C[c].u), us = CCConst(uStar_[c]),
            mk = CCConst(C[c].mask);
    CCConst fb = hasFb ? CCConst(cellForce_[c]) : CCConst();
    CCConst av = advect_ ? CCConst(advRhs_[c]) : CCConst();
    const double fc = f_[c];
    const auto po = Grid::offset(c);
    const double offx = po.x, offy = po.y, offz = po.z;
    const int cc = c;
    const double kRc = kR[c];
    Kokkos::parallel_for(
        "peclet::flow::hydro_reaction",
        Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                       {e.x - G, e.y - G, e.z - G}),
        KOKKOS_LAMBDA(int x, int y, int z) {
          const long st[3] = {1, e.x, (long)e.x * e.y};
          const long i = (long)x + (long)y * st[1] + (long)z * st[2];
          if (mk(i) > 0.5)
            return;  // solid staggered point: no fluid momentum here
          double R =
              idt * (uc(i) - un(i)) - fc - (fb.data() ? fb(i) : 0.0) - (av.data() ? av(i) : 0.0);
          for (int a = 0; a < 3; ++a) {
            const long jp = i + st[a], jm = i - st[a];
            if (mk(jp) <= 0.5)
              R -= mu * (us(jp) - us(i));
            if (mk(jm) <= 0.5)
              R -= mu * (us(jm) - us(i));
          }
          // The scene (owner query, instance centres, min-image box) is in the CALLER's
          // coordinates; F is an index-unit force, so the lever arm comes back to index units.
          const peclet::core::Vec3<double> p{sm.a[0] + sm.b[0] * ((double)(x - G + og.x) + offx),
                                             sm.a[1] + sm.b[1] * ((double)(y - G + og.y) + offy),
                                             sm.a[2] + sm.b[2] * ((double)(z - G + og.z) + offz)};
          const int oi = q.owner(p);
          if (oi < 0)
            return;
          // force ON the body = minus the wall force on the fluid, times h_c'V' (§4.4)
          const double F = -R * kRc;
          Kokkos::atomic_add(&Fd(3 * oi + cc), F);
          const peclet::core::Vec3<double> rq = peclet::core::geom::minImage(
              peclet::core::Vec3<double>{p.x - cen(3 * oi + 0), p.y - cen(3 * oi + 1),
                                         p.z - cen(3 * oi + 2)},
              box);
          const peclet::core::Vec3<double> r{rq.x * sm.dToInt, rq.y * sm.dToInt, rq.z * sm.dToInt};
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
            return;  // not a cut cell (a sign test: the positive per-axis factors below cannot
                     // change it)
          // PHASE 2 (doc/anisotropic_metric.md §4.4, E3 reading 2): (ax, ay, az) is an INDEX
          // aperture area vector; the physical one in hRef^2 is A_b = a_b V'/h_b' -- the same
          // A the traction paragraph of §4.4 forms -- so the metric multiplies the AREA
          // component b, which is the index the cross product with Omega consumes, applied
          // OUTSIDE the existing expression so V'/h_b' = 1.0 reduces exactly.
          const double Ax = ax * kA0, Ay = ay * kA1, Az = az * kA2;
          const peclet::core::Vec3<double> p{sm.a[0] + sm.b[0] * (double)(x - G + og.x),
                                             sm.a[1] + sm.b[1] * (double)(y - G + og.y),
                                             sm.a[2] + sm.b[2] * (double)(z - G + og.z)};
          const int oi = q.owner(p);
          if (oi < 0)
            return;
          // The scene's angular velocity is 1/time in the caller's units; the index one is
          // omega*tRef (v = omega x r holds on both sides of the map).
          const double wx = ang(3 * oi + 0) * sm.angToInt, wy = ang(3 * oi + 1) * sm.angToInt,
                       wz = ang(3 * oi + 2) * sm.angToInt;
          if (wx == 0.0 && wy == 0.0 && wz == 0.0)
            return;
          // v = (n dA) x Omega  -- the missing traction integrated over this cell's wall patch
          const double vx = Ay * wz - Az * wy;
          const double vy = Az * wx - Ax * wz;
          const double vz = Ax * wy - Ay * wx;
          const peclet::core::Vec3<double> rq = peclet::core::geom::minImage(
              peclet::core::Vec3<double>{p.x - cen(3 * oi + 0), p.y - cen(3 * oi + 1),
                                         p.z - cen(3 * oi + 2)},
              box);
          const peclet::core::Vec3<double> r{rq.x * sm.dToInt, rq.y * sm.dToInt, rq.z * sm.dToInt};
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
      // PHASE 2 (§4.4, a site the note's list does not name): this correction REMOVES a term that
      // is already inside `F_a = -sum R_a h_a' V'`, so it must carry exactly the factor that term
      // carries there.  The momentum row's pressure gradient is `w_c (P(i) - P(i-s))` since C1/C2,
      // and F multiplies by `h_c' V'`, so the combined factor on `pi(i)` is
      // `w_c h_c' V' = V'/h_c'` -- the physical area of the face, as it must be for a pressure
      // force.  Exactly 1.0 isotropic (1.0/1.0), so this is an identity multiplication.
      const double kP = u_.vol / u_.hp[c];
      Kokkos::parallel_for(
          "peclet::flow::hydro_reaction_owner_flux",
          Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G},
                                                         {e.x - G, e.y - G, e.z - G}),
          KOKKOS_LAMBDA(int x, int y, int z) {
            const long i = (long)x + (long)y * e.x + (long)z * (long)e.x * e.y;
            const long j = i + strd;  // the +s neighbour: visit once
            if (mk(i) > 0.5 || mk(j) > 0.5)
              return;  // wall faces stay in the wall force
            const peclet::core::Vec3<double> pa{
                sm.a[0] + sm.b[0] * ((double)(x - G + og.x) + offx),
                sm.a[1] + sm.b[1] * ((double)(y - G + og.y) + offy),
                sm.a[2] + sm.b[2] * ((double)(z - G + og.z) + offz)};
            peclet::core::Vec3<double> pb = pa;  // one cell along cc, in scene coordinates
            (cc == 0 ? pb.x : cc == 1 ? pb.y : pb.z) += sm.b[cc];
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
            const double flux = pf(i) * kP;
            Kokkos::atomic_add(&Fd(3 * oa + cc), -flux);
            Kokkos::atomic_add(&Fd(3 * ob + cc), +flux);
            const peclet::core::Vec3<double> raq = peclet::core::geom::minImage(
                peclet::core::Vec3<double>{pa.x - cen(3 * oa + 0), pa.y - cen(3 * oa + 1),
                                           pa.z - cen(3 * oa + 2)},
                box);
            const peclet::core::Vec3<double> rbq = peclet::core::geom::minImage(
                peclet::core::Vec3<double>{pa.x - cen(3 * ob + 0), pa.y - cen(3 * ob + 1),
                                           pa.z - cen(3 * ob + 2)},
                box);
            const peclet::core::Vec3<double> ra{raq.x * sm.dToInt, raq.y * sm.dToInt,
                                                raq.z * sm.dToInt};
            const peclet::core::Vec3<double> rb{rbq.x * sm.dToInt, rbq.y * sm.dToInt,
                                                rbq.z * sm.dToInt};
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
  scaleForceTorque(out, m);
#ifdef PECLET_FLOW_MPI
  if (distributed_) {
    std::vector<double> g(out.size(), 0.0);
    MPI_Allreduce(out.data(), g.data(), (int)out.size(), MPI_DOUBLE, MPI_SUM, comm_);
    out.swap(g);
  }
#endif
  return out;
}

template <class Grid>
std::array<long, 3> Solver<Grid>::fluidMomentumCells() {
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

template <class Grid>
std::vector<double> Solver<Grid>::reactionBudgetTerms() {
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

template <class Grid>
std::array<double, 3> Solver<Grid>::wallAreaProbe() {
  std::array<double, 3> out{0, 0, 0};
  if (!hasScene_)
    return out;
  CCExec space;
  const C3 e = e_, og = og_;
  CCConst oxv = CCConst(ox_), oyv = CCConst(oy_), ozv = CCConst(oz_);
  double sx = 0, sy = 0, sz = 0;
  Kokkos::parallel_reduce(
      "peclet::flow::wall_area_probe",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
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

template <class Grid>
double Solver<Grid>::wallFluxImbalance() {
  if (!hasScene_ || !hasMotion_ || uwCell_[0].extent(0) != n_)
    return 0.0;
  CCExec space;
  const C3 e = e_;
  CCConst oxv = CCConst(ox_), oyv = CCConst(oy_), ozv = CCConst(oz_);
  CCConst wx = CCConst(uwCell_[0]), wy = CCConst(uwCell_[1]), wz = CCConst(uwCell_[2]);
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "peclet::flow::wall_flux_sum",
      Kokkos::MDRangePolicy<CCExec, Kokkos::Rank<3>>(space, {G, G, G}, {e.x - G, e.y - G, e.z - G}),
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

template <class Grid>
CCConst Solver<Grid>::wallVelView(int c) const {
  return uBc_[c].extent(0) == n_ ? CCConst(uBc_[c]) : CCConst();
}

}  // namespace peclet::flow

#endif  // PECLET_FLOW_FLOW_IBM_HYDRO_HPP
