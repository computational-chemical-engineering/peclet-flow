/// @file
/// @brief Python binding of peclet.flow: the VoF block container, open boundaries, curvature and
/// surface tension.
///
/// Binding-only: included by src/flow_bindings.cpp alone, which it was split out of verbatim.
/// Each function adds one topic's entries to the `nb::class_` that bind_solver or
/// bind_diagnostics creates; the order of those calls is the registration order, and nanobind
/// resolves overloads in it.
#pragma once

#include "flow_bindings_common.hpp"

namespace peclet::flow::bindings {

// One VoF marker block's own inner colour as a Fortran-order (nx,ny,nz) array over its block box
// (bound as `vof_block_color`, suite/docs/NAMING.md 1.6).
template <class S>
inline nb::ndarray<nb::numpy, double> vofBlockColourArray(S& s, long id) {
  std::vector<double> v = s.vofBlockColour(id);
  const auto st = s.vofBlockStats();
  std::size_t nx = 0, ny = 0, nz = 0;
  for (const auto& b : st)
    if (b.id == id) {
      nx = static_cast<std::size_t>(b.hi[0] - b.lo[0]);
      ny = static_cast<std::size_t>(b.hi[1] - b.lo[1]);
      nz = static_cast<std::size_t>(b.hi[2] - b.lo[2]);
    }
  if (v.size() != nx * ny * nz)
    return peclet::core::python::vector_to_ndarray(std::move(v), {v.size()}, {1});
  return peclet::core::python::vector_to_ndarray(
      std::move(v), {nx, ny, nz},
      {1, static_cast<std::int64_t>(nx), static_cast<std::int64_t>(nx * ny)});
}

// One marker block's own inner curvature, same layout as `vofBlockColourArray` (diagnostic).
template <class S>
inline nb::ndarray<nb::numpy, double> vofBlockKappaArray(S& s, long id) {
  std::vector<double> v = s.vofBlockKappa(id);
  const auto st = s.vofBlockStats();
  std::size_t nx = 0, ny = 0, nz = 0;
  for (const auto& b : st)
    if (b.id == id) {
      nx = static_cast<std::size_t>(b.hi[0] - b.lo[0]);
      ny = static_cast<std::size_t>(b.hi[1] - b.lo[1]);
      nz = static_cast<std::size_t>(b.hi[2] - b.lo[2]);
    }
  if (v.size() != nx * ny * nz)
    return peclet::core::python::vector_to_ndarray(std::move(v), {v.size()}, {1});
  return peclet::core::python::vector_to_ndarray(
      std::move(v), {nx, ny, nz},
      {1, static_cast<std::int64_t>(nx), static_cast<std::int64_t>(nx * ny)});
}

/// `Solver` (and its collocated twin): VoF blocks, open boundaries, curvature and surface tension.
template <class Grid, class C>
inline void bindSolverVofBlocks(C& cls) {
  using S = BoundSolver<Grid>;
  // --- Part III rung W0 (WO-W0): the per-bubble block container ----------------------------
  cls.def(
      "enable_vof_blocks",
      [](S& s, std::optional<std::vector<std::array<double, 4>>> seeds) {
        chk::vofSeeds(seeds);
        if (seeds)
          s.enableVofBlocks(*seeds);
        else
          s.disableVofBlocks();
      },
      nb::arg("seeds") = nb::none(),
      "Carry each bubble on its OWN VoF block (the TBFsolver vofBlock pattern, VOF_PLAN §10): "
      "one Weymouth-Yue advector per marker on a small moving global index box (bubble extent "
      "+ 3 cells, its own g = 3 halo on top) with a master rank of its own, and the registered "
      "'C' the closures see is the UNION C = max_blocks C_block.\n\n"
      "WHY: two bubbles that touch CANNOT coalesce numerically — colliding markers overlap in "
      "space but never merge, so coalescence becomes an explicit model decision (rung W4) "
      "instead of a numerical accident of a single global colour field. Measured on the gate: "
      "236 cells carry BOTH markers at closest approach (30.0 cells of shared liquid) while "
      "each marker's own volume is conserved to 2.6e-15, and the single-field control merges "
      "them irreversibly (neck colour 0.77 against the blocks' 0.00 after the reversal).\n\n"
      "`seeds=None` DROPS the container (the structured colour field 'C' and advect_vof are "
      "unaffected). Otherwise 'seeds' is a list of (cx, cy, cz, radius) spheres in CELL "
      "units, global indices -- these "
      "are GRID coordinates and a physical domain does not convert them. The "
      "colour is the same exact sphere fraction set_vof would take.\n\n"
      "SCOPE at W0: ALL-FLUID (an immersed solid raises — the cut-cell block is rung W12) and "
      "KINEMATIC (advect_vof_blocks(dt); NS coupling is W12). Masters are assigned round robin "
      "by block id, deliberately independent of where the bubble's cells live — the weighted-"
      "ORB assignment is rung W1; diagnostics.vof_block_imbalance() is the number to beat.");
  cls.def(
      "vof_blocks_enabled", [](S& s) { return s.vofBlocksEnabled(); },
      "True while the block container exists (enable_vof_blocks with seeds; None drops it).");
  cls.def(
      "advect_vof_blocks", [](S& s, double dt) { s.advectVofBlocks(dt); }, nb::arg("dt"),
      "Advance every block by dt with the CURRENT (projected) face velocity and union the "
      "result into 'C' — the block twin of advect_vof(dt), and it carries the same "
      "precondition: it RAISES unless the face field is discretely divergence-free to 1e-10, "
      "because Weymouth-Yue's exact conservation is conditional on it.\n\n"
      "Per step: gather the face velocity from the ranks that own each block's cells to its "
      "master (plain Isend/Irecv — the block table and the decomposition are both replicated, "
      "so every rank knows every message size and an NBX handshake would only rediscover it), "
      "run the three sweeps on the dense block, re-centre the box, replicate the table, and "
      "scatter the inner colour back with UNPACK_MAX.\n\n"
      "NOTE on the union: max() from an empty union CLIPS the negative Weymouth-Yue round-off "
      "residue to an exact 0 — measured up to 6.2e-17 on the LeVeque gate, and every measured "
      "union/global-field difference was exactly that and nothing else. "
      "dt is in the caller's time units.");
  // --- Part III rungs W1/W2 (WO-W12) --------------------------------------------------------
  cls.def(
      "enable_vof_blocks_from_field",
      [](S& s, const std::vector<std::array<int, 6>>& boxes) { s.enableVofBlocksFromField(boxes); },
      nb::arg("boxes"),
      "Seed the block container from the CURRENT colour field: one marker per given global "
      "index box (lo_x, lo_y, lo_z, hi_x, hi_y, hi_z; half-open), its colour copied out of the "
      "field set_vof installed. The general seeding path — enable_vof_blocks(spheres) is a "
      "convenience over it — and the way a marker of any shape enters (a quasi-2-D Hysing "
      "cylinder, a scanned bubble, an arbitrary blob found with a host connected-component "
      "labelling).\n\n"
      "The boxes are the BUBBLE extents; the container grows each by its own 3-cell margin. "
      "They must not share a cell: the seeding gather is a copy, not a union, so a cell inside "
      "two boxes would be handed to both markers.");
  // --- Part III rung W3: checkpoint / restart of the block container ----------------------
  cls.def("vof_block_color", &vofBlockColourArray<S>, nb::arg("id"),
          "One marker's OWN inner color as a Fortran-order (nx,ny,nz) float64 array over its "
          "block box (diagnostics.vof_block_stats()['lo'/'hi']); empty on a rank that does not "
          "master it.\n\n"
          "This is the block's ONLY state, so {box, color} per marker is a COMPLETE checkpoint "
          "of the container -- and it is the only exact one: re-seeding from the union color "
          "field (enable_vof_blocks_from_field) hands each of two TOUCHING markers a slice of the "
          "other, because that gather is a copy of the union clipped to the seed extent. "
          "enable_vof_blocks_from_colors(boxes, colors) restarts the container from these.");
  cls.def(
      "enable_vof_blocks_from_colors",
      [](S& s, const std::vector<std::array<int, 6>>& boxes, nb::list colours) {
        std::vector<std::vector<double>> c;
        c.reserve(nb::len(colours));
        for (std::size_t i = 0; i < nb::len(colours); ++i) {
          auto a = nb::cast<nb::ndarray<double, nb::f_contig>>(colours[i]);
          c.push_back(peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(a)));
        }
        s.enableVofBlocksFromColours(boxes, c);
      },
      nb::arg("boxes"), nb::arg("colors"),
      "Restart the block container from a vof_block_color() checkpoint: one marker per "
      "(lo_x, lo_y, lo_z, hi_x, hi_y, hi_z) INNER box (exactly the 'lo'/'hi' of "
      "diagnostics.vof_block_stats(), NOT grown by the margin again) with its colour written "
      "directly. "
      "Exact whatever the markers are doing -- nothing is gathered out of the union field.");
  cls.def(
      "set_vof_block_assign",
      [](S& s, const std::string& mode, long every) {
        chk::count("set_vof_block_assign", "every", every, 0);
        s.setVofBlockAssign(enum_index(mode, kBlockAssign, "set_vof_block_assign", "mode"), every);
      },
      nb::arg("mode"), nb::arg("every") = 0,
      "Master assignment of the block container (rung W1). 'round-robin' by block id (rung "
      "W0's, blind to block SIZE), 'lpt' = LONGEST-PROCESSING-TIME greedy on the block cell "
      "counts, 'orb' = weighted ORB over a 1-D block space (core's BlockDecomposer<1> with the "
      "cell counts as weights). 'every' > 0 re-runs the assignment every 'every' block steps "
      "and MIGRATES the colour of any block that changed master (nothing else in a block is "
      "state, so the migration is exact and the bitwise gates hold across it).\n\n"
      "All three are pure functions of the REPLICATED block table, so every rank computes the "
      "same assignment without an exchange -- which is what lets a re-assignment happen "
      "mid-run without breaking a bitwise gate. LPT is the measured winner (see the WO-W12 "
      "findings): the ORB's blocks must be CONTIGUOUS in block id, which LPT is free of.");
  cls.def(
      "vof_block_assign", [](S& s) { return kBlockAssign.at(s.vofBlockAssign()); },
      "The current master-assignment mode ('round-robin' | 'lpt' | 'orb').");
  cls.def(
      "enable_vof_block_csf", [](S& s) { s.enableVofBlockCsf(); },
      "Form the surface-tension force PER BLOCK (rung W2): each marker runs its own curvature "
      "cascade on its own dense box and forms the V4 balanced-force face force there "
      "(sigma kappa_f (C(i) - C(i-s))/h with the same selective face-curvature rule "
      "addCsfRhs uses), and the three face fields are scattered into the global RHS with "
      "UNPACK_SUM — TBFsolver's VOF.f90::computeSurfaceTension structure on the suite's "
      "kernels.\n\n"
      "WHY THE FORCE AND NOT THE CURVATURE IS SCATTERED: kappa is not additive and the union "
      "colour is a max, so a face between two OVERLAPPING markers has no single (kappa, dC) "
      "pair to build a force from. The force is the additive quantity, and forming it where "
      "each marker's own colour still exists is the only place the balanced-force pairing (the "
      "same face difference the projection's gradient uses) is available per marker.\n\n"
      "Requires set_surface_tension(sigma) and the block container; STAGGERED only. Once on, "
      "step() drives the whole two-phase stage through the blocks: the union C feeds the "
      "closures exactly as before, and the colour is advected by the blocks in the "
      "advect_vof slot.");
  // --- two-phase open boundaries (rung V-BC, WO-R) ------------------------------------------
  cls.def(
      "set_vof_inflow",
      [](S& s, const std::string& face, double value) {
        chk::fraction("set_vof_inflow", "value", value);
        s.setVofInflow(face_index(face, "set_vof_inflow"), value);
      },
      nb::arg("face"), nb::arg("value"),
      "Colour of the fluid ENTERING through inflow face 'face' ('-x', '+x', '-y', '+y', '-z', "
      "'+z'); 1 = liquid, 0 = gas. The face must already be an inflow (set_domain_bc(face, "
      "'inflow', ...)) "
      "or this raises.\n\n"
      "The value may be FRACTIONAL, and it then means 'this fraction of the incoming flux is "
      "liquid' — a flux statement, not a sub-cell interface position. That is exactly how it "
      "is implemented: on a face whose donor is outside the domain the geometric flux is "
      "replaced by the algebraic C_donor*a (wyFaceFluxBc), because a uniform prescribed ghost "
      "band has no usable MYC normal and the inflow colour is boundary DATA.\n\n"
      "Setting this also makes the property ghosts of that face follow the inflow colour "
      "through the registered closures (rho_ghost = rho(C_inflow)) instead of copying the "
      "interior — without it the inlet FACE density is the interior's, wrong by up to the "
      "density ratio at a liquid inlet into a gas domain. Default (nothing set): the "
      "zero-gradient copy, i.e. today's behaviour, bit for bit.");
  cls.def(
      "set_vof_inflow_profile",
      [](S& s, const std::string& face, nb::ndarray<double, nb::c_contig> prof) {
        if (prof.ndim() != 2)
          throw std::runtime_error("vof inflow profile must be (Nb,Nc)");
        const int nb_ = (int)prof.shape(0), nc = (int)prof.shape(1);
        s.setVofInflowProfile(face_index(face, "set_vof_inflow_profile"),
                              peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(prof)),
                              nb_, nc);
      },
      nb::arg("face"), nb::arg("profile"),
      "Per-position inflow colour (Nb,Nc) on the INNER grid of the face's two perpendicular "
      "axes, resampled with the same clamp rule set_domain_bc_profile uses for the velocity. "
      "This is how a liquid distributor over part of an inlet is expressed: the colour "
      "profile says WHERE liquid enters, the velocity profile says how fast.");
  cls.def(
      "set_vof_backflow",
      [](S& s, const std::string& face, double value) {
        chk::fraction("set_vof_backflow", "value", value);
        s.setVofBackflow(face_index(face, "set_vof_backflow"), value);
      },
      nb::arg("face"), nb::arg("value") = 0.0,
      "inletOutlet colour on OUTFLOW face 'face' ('-x' ... '+z'; default 0 = gas): where the "
      "boundary face "
      "velocity points back INTO the domain, the colour ghost carries this value instead of "
      "the zero-gradient copy. This is OpenFOAM's inletOutletFvPatchField (Rusche 2002 thesis "
      "section 4), the standard VoF outlet — an outlet is a place where you know what leaves "
      "(whatever is inside) but must STATE what comes back. Zero-gradient on a reversing face "
      "re-injects whatever happens to be sitting at the outlet, which for a draining film "
      "feeds the film back into the domain. Where the fluid leaves, zero-gradient is kept.");
  cls.def(
      "vof_bc_volumes", [](S& s) { return s.vofBcVolumes(); },
      "Signed liquid volume that crossed each of the six domain faces during the LAST colour "
      "advection, in cell-volume units, POSITIVE for liquid ENTERING the domain. Local to this "
      "rank. Six entries in face order (-x,+x,-y,+y,-z,+z); only faces carrying a domain BC "
      "are meaningful.");
  cls.def(
      "vof_bc_volumes_total", [](S& s) { return s.vofBcVolumesTotal(); },
      "The same, accumulated since enable_vof() or the last reset_vof_bc_volumes().");
  cls.def(
      "reset_vof_bc_volumes", [](S& s) { s.resetVofBcVolumes(); },
      "Zero the per-face boundary liquid ledger (both the per-step and the running totals).");
  cls.def(
      "compute_vof_curvature",
      [](S& s) {
        s.computeVofCurvature();
        const auto d = s.vofCurvatureStats();
        nb::dict r;
        r["interfacial"] = d.interfacial;
        r["hf"] = d.hf;
        r["hf_mixed"] = d.hfMixed;
        r["hf_fit"] = d.hfFit;
        r["pv"] = d.pv;
        r["pv_reduced"] = d.pvReduced;
        r["no_estimate"] = d.noEstimate;
        r["clipped"] = d.clipped;
        return r;
      },
      "VoF rung V3: compute the interface curvature from the CURRENT colour field and store it "
      "in the registered cell fields 'kappa' and 'kappa_branch'. Returns THIS RANK's branch "
      "census as a dict.\n\n"
      "'kappa' is the RAW registered field, kappa = 2H in 1/h (cell units); vof_curvature() "
      "returns the same thing in 1/length. POSITIVE for a convex blob of liquid: a sphere of "
      "liquid of radius R cells "
      "reads +2/R, an infinite cylinder +1/R, a plane 0.\n\n"
      "The cascade is Popinet's (2009): a standard height function on 7-cell column sums of C "
      "over the 3x3 transverse patch in the direction of the largest |n| (branch 1), the same "
      "in the other two directions (branch 2), then the PLIC-VOLUMETRIC paraboloid fit of "
      "Jibben et al. on a 5^3 Wendland-weighted stencil (branches 4/5) — the best "
      "cost/accuracy fallback per Han, Evrard & Desjardins, IJMF 174:104769 (2024). Branch 3 "
      "(the mixed height-position fit) is off by default; branch 0 is 'not an interfacial "
      "cell' and branch 6 is 'NO estimate', which must never occur.\n\n"
      "ALWAYS read 'kappa_branch' alongside 'kappa': kappa is 0 both where there is no "
      "interface (branch 0) and where no estimate could be made (branch 6).\n\n"
      "Two things the literature settles and this is NOT a bug: the fallback ALWAYS fires "
      "somewhere below ~5 cells per diameter (measured here: 100% of interfacial cells at "
      "D/dx = 2.8-4.4, ~19% at D/dx = 38-48 on a sphere), and with advection-realistic volume "
      "fractions the curvature error STOPS CONVERGING below C*dx ~ 1e-2 for every known "
      "method. Curvature order is not the thing to chase; transport fidelity is.");
  cls.def(
      "vof_curvature", [](S& s) { return field_out(s, s.getVofCurvature()); },
      "The curvature field's inner region as a Fortran-order (nx,ny,nz) float64 array, in 1/h "
      "in 1/LENGTH. Call compute_vof_curvature() first; get_field('kappa') is the RAW "
      "solver-internal field, which is 1/h.");
  cls.def(
      "vof_curvature_branch", [](S& s) { return field_out(s, s.getVofCurvatureBranch()); },
      "Which cascade branch produced each cell's curvature, as a Fortran-order (nx,ny,nz) "
      "float64 array: 0 not interfacial, 1 height function, 2 height function in a "
      "non-preferred direction, 3 mixed height-position fit, 4 PLIC-volumetric paraboloid fit, "
      "5 the same with the rank-deficient 3-parameter model, 6 NO estimate. == "
      "get_field('kappa_branch').");
  // --- VoF rung V4 (WO-P): balanced-force surface tension ----------------------------------
  cls.def(
      "set_surface_tension", [](S& s, double sigma) { s.setSurfaceTension(sigma); },
      nb::arg("sigma"),
      "VoF rung V4: turn on the BALANCED-FORCE continuum surface force with coefficient sigma "
      "(0 = off, the default). Enables VoF and the curvature cascade, which then runs once per "
      "step at the head, from the same colour field the density closure sees.\n\n"
      "The force at the staggered velocity unknown u_c(i) is\n\n"
      "    F = sigma * kappa_f * (C(i) - C(i - s_c)) / h\n\n"
      "i.e. the colour difference is the projection's OWN face difference — the identical "
      "discrete gradient operator the pressure uses (Francois et al. 2006; Popinet 2009). That "
      "is the whole content of 'balanced force': with a constant kappa the force is exactly a "
      "discrete gradient, so the projection annihilates it and a static drop stays at machine "
      "zero. Face-interpolating a cell-centred sigma*kappa*grad(C) instead — the obvious way "
      "to reuse the per-cell body force — is not a discrete gradient of anything and leaves "
      "spurious currents of order sigma*kappa/mu that no curvature accuracy removes.\n\n"
      "UNITS: sigma is a force per unit LENGTH in the caller's own units -- the system rho, mu "
      "and set_body_force are stated in (so per CELL without a physical domain, where the "
      "cell size is 1). SIGN: kappa is positive for a convex blob of liquid and C is the "
      "liquid fraction, so the equilibrium pressure is P = sigma*kappa*C + const — the "
      "Young-Laplace overpressure INSIDE the drop.\n\n"
      "Surface tension is EXPLICIT, so step() enforces the Brackbill capillary time step "
      "(see capillary_dt()).");
  cls.def(
      "surface_tension", [](S& s) { return s.surfaceTension(); },
      "The surface-tension coefficient (0 when the CSF is off).");
  cls.def(
      "capillary_dt", [](S& s) { return s.capillaryDt(); },
      "The Brackbill (1992) capillary time-step limit sqrt((rho_1+rho_2) h^3 / (4 pi sigma)), "
      "+inf when surface tension is off. Denner & van Wachem (2015) verified this IS the "
      "stability boundary of an explicit CSF — the prefactor 1/(4 pi), the h^{3/2} scaling and "
      "the SUM of the phase densities (both phases oscillate). The density sum is the declared "
      "phase pair under enable_vof_momentum, otherwise min(rho)+max(rho) over the field. "
      "In the caller's time units.");
  cls.def("set_capillary_cfl", &chk::setCapillaryCfl<S>, nb::arg("factor"),
          "Safety factor on the capillary limit: step() raises when dt > factor * capillary_dt(). "
          "Default 1.0 — the Brackbill formula was measured to be the boundary itself, so it "
          "carries no built-in margin. Set it huge to disable the check, the same escape hatch "
          "set_vof_cfl_limit is for the Weymouth-Yue cap.");
  cls.def(
      "vof_step_limits",
      [](S& s) {
        const auto L = s.vofStepLimits();
        nb::dict r;
        r["courant"] = L.courant;
        r["cfl_dt"] = L.cflDt;
        r["capillary_dt"] = L.capillaryDt;
        r["binding"] = L.binding;
        r["capillary_binds"] = L.capillaryBinds;
        return r;
      },
      "Both explicit two-phase step limits at the current state, and which one binds: the "
      "largest dt the interface-local Weymouth-Yue CFL cap admits ('cfl_dt'), the Brackbill "
      "capillary limit ('capillary_dt'), and min(cfl_dt, capillary_cfl*capillary_dt) with a "
      "'capillary_binds' flag. At pore-scale capillary numbers the capillary limit is expected "
      "to bind first.");
}

/// `SolverDiagnostics` (and its collocated twin): VoF census, block container, outflow density
/// correction, curvature and CSF instruments.
template <class Grid, class C>
inline void bindDiagnosticsVofBlocks(C& cls) {
  using D = Diagnostics<Grid>;
  cls.def(
      "vof_diagnostics",
      [](D& diag) {
        const auto d = diag.s->vofDiagnostics();
        nb::dict r;
        r["sum"] = d.sumC;
        r["min"] = d.minC;
        r["max"] = d.maxC;
        r["mixed"] = d.mixed;
        r["wisps"] = d.wisps;
        r["volume"] = d.volume;
        r["raw_volume"] = d.rawVolume;
        r["solid_sum"] = d.solidSumC;
        r["solid_fill_sum"] = d.solidFillSum;
        r["min_fluid"] = d.minCFluid;
        r["max_fluid"] = d.maxCFluid;
        r["clipped_volume"] = d.clippedVolume;
        r["clipped_signed"] = d.clippedSigned;
        r["cut_cells"] = d.cutCells;
        r["clamped_faces"] = d.clampedFaces;
        r["solid_cells"] = d.solidCells;
        // rung V-BC (WO-R): the boundary term of the exact colour budget
        const auto v = diag.s->vofBcVolumes();
        double in = 0.0, out = 0.0;
        for (int f = 0; f < 6; ++f) {
          if (v[f] > 0.0)
            in += v[f];
          else
            out -= v[f];
        }
        r["inflow_volume"] = in;
        r["outflow_volume"] = out;
        return r;
      },
      "Colour census over THIS RANK's inner cells: sum (cell-volume units), min, max, the "
      "number of mixed cells (0<C<1) and of wisps (C within 1e-8 of 0 or 1). No clipping is "
      "applied at this rung, so min/max may leave [0,1] if the CFL cap is raised.\n\n"
      "With an immersed solid (rung V5a) the dict also carries the cut-cell quantities, all "
      "zero without one: 'volume' = sum eps_eff*C over fluid cells, the functional the "
      "openness-weighted scheme conserves EXACTLY (eps_eff = max(eps, 1/64), the 4^3 "
      "subsampling resolution — see src/vof/cutcell.hpp rule 1); 'raw_volume' = sum eps*C, "
      "which differs only on eps==0 cells that still own an open face and is therefore NOT the "
      "conserved functional; 'solid_sum' = sum of C over solid cells (0 by construction — the "
      "neutral band fill lives on the working block, not on 'C'); 'min_fluid'/'max_fluid' over "
      "UNCUT fluid cells (eps==1), where Weymouth's boundedness applies verbatim; "
      "'clipped_volume' / 'clipped_signed' = the liquid volume the cut-cell clip moved during "
      "the last advection (a TRIPWIRE on the flux approximation, not a mechanism: if it is not "
      "negligible the fix is the solid-clipped flux polygon); 'cut_cells' / 'solid_cells'.\n\n"
      "'inflow_volume' / 'outflow_volume' are the liquid volume that ENTERED / LEFT through "
      "the domain faces during the LAST colour advection, in the same cell-volume units as "
      "'sum' (0 unless a VoF boundary colour is set — rung V-BC). They are the advector's OWN "
      "face fluxes, so sum(C) - sum(C_0) = integral(inflow - outflow) holds to round-off "
      "whatever the interface does; vof_bc_volumes() breaks them out per face.");
  cls.def(
      "vof_block_stats",
      [](D& diag) {
        nb::list out;
        const auto ov = diag.s->vofBlockOverlapCensus();
        for (const auto& b : diag.s->vofBlockStats()) {
          nb::dict r;
          r["id"] = b.id;
          r["master"] = b.master;
          r["lo"] = nb::make_tuple(b.lo[0], b.lo[1], b.lo[2]);
          r["hi"] = nb::make_tuple(b.hi[0], b.hi[1], b.hi[2]);
          r["cells"] = b.cells;
          r["volume"] = b.volume;
          r["centroid"] = nb::make_tuple(b.centroid[0], b.centroid[1], b.centroid[2]);
          r["velocity"] = nb::make_tuple(b.velocity[0], b.velocity[1], b.velocity[2]);
          r["moments"] = nb::make_tuple(b.moment[0], b.moment[1], b.moment[2], b.moment[3],
                                        b.moment[4], b.moment[5]);
          r["recentred"] = b.recentred;
          r["discarded"] = b.discarded;
          r["area"] = b.area;
          r["debris_cells"] = b.debrisCells;
          r["debris_volume"] = b.debrisVolume;
          r["debris_returned"] = b.debrisReturned;
          r["debris_lost"] = b.debrisLost;
          r["debris_unresolved"] = b.debrisUnresolved;
          r["full_axis"] = b.fullAxis;
          r["residue_returned"] = b.residueReturned;
          r["overlap_max_sum"] = ov.maxSum;
          r["overlap_excess"] = ov.excess;
          r["overlap_cells"] = ov.cells;
          r["overlap_bound_active"] = ov.active;
          out.append(r);
        }
        return out;
      },
      "Per-bubble Lagrangian census, one dict per block, in block-id order. 'lo'/'hi' (the "
      "global index box) and 'master' are replicated on every rank; the MEASURED entries — "
      "'volume' (sum of C over the box, cell volumes), 'centroid', 'velocity' (d(centroid)/dt "
      "of the last advection), 'moments' (the central second moments xx, yy, zz, xy, xz, yz "
      "divided by the volume, i.e. the deformation) — are filled only on the block's MASTER "
      "and are zero elsewhere.\n\n"
      "'discarded' is the colour a re-centring dropped, cumulatively: Weymouth-Yue leaves "
      "round-off residue in every cell its sweeps touch and the box tracks the BUBBLE, not "
      "that wake, so the residue falling outside the new box is discarded. Never physical "
      "liquid (measured -9.5e-17 over a 20-cell translation, against a bubble volume of 524), "
      "but it is reported rather than hidden — a container that silently loses mass is not "
      "acceptable.\n\n"
      "'debris_cells' / 'debris_volume' are this step's marker DEBRIS removal (block CSF on "
      "only; master only; see set_vof_block_debris): interfacial cells with every cell of the "
      "marker within two cells below C = 1/2, and their summed colour. 'debris_returned', "
      "'debris_lost' and 'debris_unresolved' are the cumulative ledger "
      "(doc/vof_overlap_design.md 5.3): volume + discarded + debris_lost is conserved. "
      "'full_axis' counts re-centrings whose box had to span a whole periodic axis (it is "
      "then [0, L) and copied with the wrap, losslessly; 12.3). 'residue_returned' is the "
      "cumulative SIGNED sub-wisp-threshold residue cleared after each advection and returned "
      "to the interface with the debris (13): volume + discarded + debris_lost is exact.\n\n"
      "'overlap_*' are the GLOBAL marker-overlap census of the last block advection "
      "(vof_overlap_design 5.6; block CSF or the overlap density on), the same in every dict: "
      "'overlap_max_sum' = max over cells of S = sum_k C_k, 'overlap_excess' = sum of "
      "(S - 1)^+ (cell volumes), 'overlap_cells' = cells with S > 1 + 1e-8, and "
      "'overlap_bound_active' = the phantom-aware capillary limit (2 rho_min in place of "
      "rho_l + rho_g) is in force for the next step (triggered by max S > 1 + 1e-2).");
  cls.def(
      "set_vof_phantom_capillary_bound",
      [](D& diag, bool enabled) { diag.s->setVofPhantomCapillaryBound(enabled); },
      nb::arg("enabled") = true,
      "The phantom-aware capillary limit of the block container (doc/vof_overlap_design.md "
      "5.6, 11.3), DEFAULT True: while two block markers overlap (max S > 1 + 1e-2) the "
      "capillary limit of vof_step_limits() and step() uses 2 rho_min in place of "
      "rho_l + rho_g -- the part of one marker's surface inside another is an interface with "
      "gas on both sides. False restores the pre-5.6 bound (the ablation; a fixed-dt driver "
      "at a density ratio above ~31 otherwise throws at first contact).");
  cls.def(
      "set_vof_block_debris", [](D& diag, bool enabled) { diag.s->setVofBlockDebris(enabled); },
      nb::arg("enabled"),
      "Per-step marker DEBRIS removal of the block container (doc/vof_overlap_design.md 5.3, "
      "11). Default: ON under enable_vof_block_csf, off otherwise. A debris cell is an "
      "interfacial cell of a marker (1e-8 < C < 1-1e-8) with every cell of that marker within "
      "two cells below C = 1/2: colour with no resolvable interface near it (what two "
      "overlapping markers tear off each other). Its colour is zeroed and the volume is "
      "returned to the marker's attached interfacial cells weighted C(1-C), in a fixed "
      "summation order, so the marker volume is exact and the state decomposition-"
      "independent. Ledger in vof_block_stats(): debris_cells / debris_volume (this step), "
      "debris_returned, debris_lost, debris_unresolved (cumulative, migrate with the block). "
      "enabled=False keeps the census (what WOULD be removed) and moves nothing.");
  cls.def(
      "set_vof_block_overlap_density",
      [](D& diag, bool enabled) { diag.s->setVofBlockOverlapDensity(enabled); }, nb::arg("enabled"),
      "OPT-IN MODEL, default False (the MAX union verbatim). With it the union colour that "
      "feeds rho(C) and mu(C) -- and the registered 'C' -- becomes C_eff = max_k C_k where "
      "S = sum_k C_k <= 1 + 1e-8 and max(0, 2 - S) where markers overlap: the overlap volume "
      "is treated as the sub-grid FILM liquid (the 'tent'), so the phantom interface of one "
      "marker inside another is an ordinary liquid-gas interface under the base capillary "
      "bound. Its cost: the lens holds liquid the real state does not (buoyancy and mixture "
      "weight off by 2 V_overlap during a contact), and it changes collision dynamics. See "
      "doc/vof_overlap_design.md 5.6.");
  cls.def(
      "vof_block_imbalance", [](D& diag) { return diag.s->vofBlockImbalance(); },
      "max/mean of the per-rank block-cell load under the CURRENT master assignment "
      "(round robin by block id at rung W0). 1.0 is perfect; the weighted-ORB assignment of "
      "rung W1 is what this number is here to grade.");
  cls.def(
      "vof_block_census",
      [](D& diag) {
        std::vector<long> m, c;
        diag.s->vofBlockCensus(m, c);
        nb::dict r;
        nb::list lm, lc;
        for (long v : m)
          lm.append(v);
        for (long v : c)
          lc.append(v);
        r["masters"] = lm;
        r["cells"] = lc;
        return r;
      },
      "Per-rank load census of the block container: 'masters'[r] = blocks rank r masters, "
      "'cells'[r] = the inner cells those blocks carry (the actual VoF work).");
  cls.def(
      "vof_block_imbalance_of", [](D& diag, int mode) { return diag.s->vofBlockImbalanceOf(mode); },
      nb::arg("mode"),
      "The max/mean per-rank block-cell load the given assignment mode WOULD give on the "
      "current blocks, without applying it — so a study can put the three modes side by side "
      "on one swarm without perturbing the run.");
  cls.def(
      "set_vof_block_device_staging",
      [](D& diag, bool on) { diag.s->setVofBlockDeviceStaging(on); }, nb::arg("on"),
      "Pack/unpack the block gather and scatter in the block's own MEMORY SPACE (rung W1 item "
      "b, default True), with a host staging copy only per MPI MESSAGE — and none at all for "
      "the master's own cells, which are a device-to-device copy. False selects rung W0's "
      "host-staged path (a full mirror of the local patch per step), which is what the "
      "device-vs-host measurement compares against. Every step is a copy of a double, so the "
      "two paths are BITWISE identical; the ctest gates that rather than asserting it.");
  cls.def(
      "set_vof_block_pool", [](D& diag, bool on) { diag.s->setVofBlockPool(on); }, nb::arg("on"),
      "Recycle the advectors a re-centring retires, keyed by the exact box extent (rung W1 "
      "item c, default True). A translating bubble keeps its box SIZE and only moves its "
      "origin, so the hit rate is ~100 % and the ten Views of the new box are not allocated. "
      "A recycled advector is handed back in the state a freshly initialised one is in "
      "(colour and the three face-velocity fields zeroed), so the pool is bitwise inert.");
  cls.def(
      "vof_block_pool_stats",
      [](D& diag) {
        const auto q = diag.s->vofBlockPoolStats();
        nb::dict r;
        r["hits"] = q[0];
        r["misses"] = q[1];
        return r;
      },
      "Block-pool census: 'hits' = advectors recycled, 'misses' = advectors allocated.");
  cls.def(
      "vof_block_kappa", [](D& diag, long id) { return vofBlockKappaArray(*diag.s, id); },
      nb::arg("id"),
      "One marker's OWN curvature (internal units, 1/cell) over its block box, the layout of "
      "vof_block_color; empty unless this rank masters it and the block CSF is on. A "
      "diagnostic.");
  cls.def(
      "vof_block_force",
      [](D& diag, int c) { return field_out(*diag.s, diag.s->getVofBlockForce(c)); },
      nb::arg("component"),
      "The block CSF face force of component c (0,1,2) as scattered (summed over markers) into "
      "the solver's LOW-face slot of each cell, internal units; the field the momentum RHS "
      "adds. "
      "A diagnostic.");
  cls.def(
      "vof_block_curvature_stats",
      [](D& diag) {
        const auto st = diag.s->vofBlockCurvatureStats();
        nb::dict r;
        r["interfacial"] = st.interfacial;
        r["hf"] = st.hf;
        r["hf_mixed"] = st.hfMixed;
        r["hf_fit"] = st.hfFit;
        r["pv"] = st.pv;
        r["pv_reduced"] = st.pvReduced;
        r["no_estimate"] = st.noEstimate;
        r["clipped"] = st.clipped;
        return r;
      },
      "Branch census of the last block-CSF curvature pass, SUMMED over this rank's blocks "
      "(local to the rank). 'no_estimate' must be 0 on any gated case. 'clipped' = cells whose "
      "|kappa| the admissibility clip bounded (set_vof_block_kappa_clip; a NaN counts too); 0 "
      "on a resolved interface.");
  cls.def(
      "set_outflow_rho_correction", [](D& diag, bool on) { diag.s->setOutflowRhoCorrection(on); },
      nb::arg("on") = true,
      "The 1/rho_f mobility factor on the HIGH-side outflow face correction. DEFAULT TRUE "
      "since WO-R2; pass False for the ablation.\n\n"
      "A projection correction cancels the discrete divergence only if it uses the SAME face "
      "coefficient the operator row used. Until WO-R2 the multigrid re-imposed the literal "
      "openness 1.0 at every Dirichlet domain face, overwriting buildRhoCoeff's "
      "open_f*rho0/rho_f, so the plain phi difference was the consistent partner and WO-R "
      "measured this factor making things seven orders WORSE. WO-R2 fixed the operator "
      "(CutcellMG::setOutflowCoefficient) and the verdict inverted. Stratified duct, ratio "
      "10, 5 steps, max|div(open u)| of the PROJECTED field:\n"
      "                              old operator   fixed operator\n"
      "  without the factor            8.76e-10        9.97e-05\n"
      "  with    the factor            9.24e-03        8.31e-10\n"
      "(tests/kokkos/test_vof_bc.cpp gate F2.) Bitwise inert at constant density (rho_f == "
      "rho0 makes the factor exactly 1) and gated on the variable-density path.");
  cls.def(
      "outflow_rho_correction", [](D& diag) { return diag.s->outflowRhoCorrection(); },
      "Whether the 1/rho_f factor is applied to the outflow face correction (WO-R item 4, "
      "reversed by WO-R2's operator fix; default True).");
  cls.def(
      "set_vof_curvature_weight_width",
      [](D& diag, double d) { diag.s->setVofCurvatureWeightWidth(d); }, nb::arg("width"),
      "Wendland support width of the PLIC-volumetric fallback fit, in CELL units. Default 2.5, "
      "which is Han et al.'s recommended pairing with the 5^3 stencil. Their translating "
      "droplet recovers first-order convergence of the spurious currents at 3.5 and loses "
      "convergence entirely at 4.5 (over-smoothing), so this is a real trade-off between the "
      "locality of the estimate and its robustness to transport error — not a free knob.");
  cls.def(
      "set_vof_curvature_mixed_height_fit",
      [](D& diag, bool on) { diag.s->setVofCurvatureMixedHeightFit(on); }, nb::arg("on") = true,
      "Enable cascade branch 3, the mixed height-position fit (a paraboloid through the "
      "interface positions of whichever columns closed). DEFAULT OFF and it should stay off: "
      "measured on an exact-fraction sphere at 16/32/64 it takes over the 19.5-59.6% of cells "
      "the height function cannot serve and DESTROYS the convergence of the max curvature "
      "error (order 0.00 vs 1.86 with the PLIC-volumetric fallback instead), because its data "
      "set is the columns that closed - a slope-selected, asymmetric subset whose lever-arm "
      "bias is scale invariant. Shipped as a re-measurable instrument, not a configuration.");
  cls.def(
      "csf_diagnostics",
      [](D& diag) {
        const auto d = diag.s->csfDiagnostics();
        nb::dict r;
        r["max_force"] = nb::make_tuple(d.maxForce[0], d.maxForce[1], d.maxForce[2]);
        r["orphan_faces"] = nb::make_tuple(d.orphanFaces[0], d.orphanFaces[1], d.orphanFaces[2]);
        r["forced_faces"] = nb::make_tuple(d.forcedFaces[0], d.forcedFaces[1], d.forcedFaces[2]);
        return r;
      },
      "Census of the CSF face force on THIS RANK: per component the max |F|, the number of "
      "faces that carried a force, and the number of ORPHAN faces — faces the colour jumps "
      "across but where neither cell has a curvature estimate, so the force was dropped. An "
      "orphan is a defect and must be 0; it is counted rather than hidden.");
}

}  // namespace peclet::flow::bindings
