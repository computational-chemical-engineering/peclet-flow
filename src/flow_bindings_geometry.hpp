/// @file
/// @brief Python binding of peclet.flow: domain boundary conditions, scenes, solids, moving
/// instances and hydrodynamic loads.
///
/// Binding-only: included by src/flow_bindings.cpp alone, which it was split out of verbatim.
/// Each function adds one topic's entries to the `nb::class_` that bind_solver or
/// bind_diagnostics creates; the order of those calls is the registration order, and nanobind
/// resolves overloads in it.
#pragma once

#include "flow_bindings_common.hpp"

namespace peclet::flow::bindings {

/// `Solver` (and its collocated twin): domain BCs, pressure geometry, scenes, instances,
/// hydrodynamic loads and the solid.
template <class Grid, class C>
inline void bindSolverGeometry(C& cls) {
  using S = BoundSolver<Grid>;
  cls.def(
      "set_domain_bc",
      [](S& s, const std::string& face, const std::string& type, std::array<double, 3> velocity) {
        chk::finiteEach("set_domain_bc", "velocity", velocity, "length/time");
        s.setDomainBc(face_index(face, "set_domain_bc"),
                      enum_index(type, kBcTypes, "set_domain_bc", "type"), velocity[0], velocity[1],
                      velocity[2]);
      },
      nb::arg("face"), nb::arg("type"), nb::arg("velocity") = std::array<double, 3>{0.0, 0.0, 0.0},
      "Set a per-face domain BC. `face` is one of '-x', '+x', '-y', '+y', '-z', '+z'; `type` "
      "is 'periodic' (the default on every face), 'wall' (no-slip), 'inflow' (Dirichlet "
      "`velocity` -- an inlet, or a lid with a tangential velocity), 'outflow' "
      "(zero-gradient velocity, p = 0) or 'slip' (free-slip / symmetry plane: zero normal "
      "velocity, zero normal derivative of the tangential components, pressure Neumann like a "
      "wall; `velocity` is ignored). The TYPE must be set BEFORE the geometry (set_solid / "
      "set_pressure_geometry / set_solid_from_scene): it is folded into the operators when "
      "the geometry is built, and a call that would CHANGE it raises afterwards. A VALUE "
      "update on a face whose type is unchanged (ramping an inflow jet, a lid's tangential "
      "speed) is allowed at any time and takes effect the next step. A half-domain closed by "
      "a 'slip' face reproduces the full symmetric domain pointwise "
      "(tests/kokkos/test_freeslip). Both grids (Solver and SolverColocated). "
      "`velocity` is in the caller's units (length/time).");
  cls.def(
      "set_domain_bc_profile",
      [](S& s, const std::string& face, nb::ndarray<double, nb::c_contig> prof) {
        if (prof.ndim() != 3 || prof.shape(2) != 3)
          throw std::runtime_error("profile must be (Nb,Nc,3)");
        const int nb_ = (int)prof.shape(0), nc = (int)prof.shape(1);
        s.setDomainBcProfile(face_index(face, "set_domain_bc_profile"),
                             peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(prof)),
                             nb_, nc);
      },
      nb::arg("face"), nb::arg("profile"),
      "Prescribe a per-position inlet velocity profile (Nb,Nc,3) over a face (one of '-x', "
      "'+x', '-y', '+y', '-z', '+z'; sets it to 'inflow'). Like set_domain_bc, the TYPE (making "
      "a face inflow for the first time) must be set BEFORE the geometry; updating the "
      "profile on a face that is ALREADY inflow is allowed after the geometry.");
  cls.def(
      "set_pressure_geometry",
      [](S& s, nb::ndarray<double, nb::f_contig> sdf) { s.setPressureGeometry(grid_in(sdf)); },
      nb::arg("sdf"),
      "Set an all-fluid SDF for the cut-cell pressure operator without an immersed solid (the "
      "channel/BFS domain-BC path). For a no-slip immersed BODY in an inflow/outflow domain, "
      "call "
      "set_solid(sdf, cutcell_pressure=True) instead -- do NOT also call this (a second "
      "geometry "
      "setter overwrites the SDF and wipes the solid).");
  cls.def(
      "set_scene",
      [](S& s, nb::ndarray<int, nb::c_contig> ni, nb::ndarray<double, nb::c_contig> nr,
         nb::ndarray<int, nb::c_contig> ii, nb::ndarray<double, nb::c_contig> ir, bool periodic) {
        s.setScene(std::vector<int>(ni.data(), ni.data() + ni.size()),
                   std::vector<double>(nr.data(), nr.data() + nr.size()),
                   std::vector<int>(ii.data(), ii.data() + ii.size()),
                   std::vector<double>(ir.data(), ir.data() + ir.size()), periodic);
      },
      nb::arg("node_ints"), nb::arg("node_reals"), nb::arg("instance_ints"),
      nb::arg("instance_reals"), nb::arg("periodic") = false,
      "Install an analytic geometry scene from core's flat encoding (3 ints + 16 reals per "
      "node, 2 ints + 18 reals per instance -- reals[17] is the centre-pinned flag; legacy 17-real "
      "records are accepted, reading an all-zero centre as 'follows the body'). Coordinates are "
      "the caller's own PHYSICAL ones when the solver was given an extent -- so ONE scene "
      "serves flow, dem and voro unchanged -- and CELL UNITS on the global inner grid "
      "otherwise (cell (i,j,k)'s centre at (i,j,k) -- half a cell from cell_centers(), which "
      "returns i + 1/2 in cell units). The scene is replicated on every rank, so scene-derived "
      "geometry needs no "
      "communication and -- unlike diagnostics.set_exact_crossings -- is NOT single-rank only. "
      "periodic=True treats the scene as min-image periodic over the global grid (one "
      "instance per body, no images); periodic=False leaves images to the caller.");
  cls.def("set_solid_from_scene", &S::setSolidFromScene, nb::arg("cutcell_pressure") = true,
          "Sample the installed scene onto this rank's inner grid and install it as the solid, "
          "entirely on device (no nx*ny*nz float64 host round trip). "
          "The scene's coordinates are physical (the caller's length units) under an extent.");
  cls.def("has_scene", &S::hasScene,
          "True once set_scene has installed an analytic CSG scene on this solver. False for "
          "geometry set any other way (set_solid with a sampled SDF, set_pressure_geometry), and "
          "false again if no scene has been given yet. set_solid_from_scene and the "
          "scene-derived per-body hydrodynamic force/motion bookkeeping all require this to be "
          "True first.");
  cls.def(
      "set_instance_motion",
      [](S& s, int i, std::array<double, 3> lin, std::array<double, 3> ang,
         std::optional<std::array<double, 3>> center) {
        chk::instanceMotion(lin, ang, center);
        s.setInstanceMotion(i, lin, ang, center ? center->data() : nullptr);
      },
      nb::arg("instance"), nb::arg("lin_vel") = std::array<double, 3>{0.0, 0.0, 0.0},
      nb::arg("ang_vel") = std::array<double, 3>{0.0, 0.0, 0.0}, nb::arg("center") = nb::none(),
      "Rigid-body motion of one scene instance, in the scene's own coordinates per time (the "
      "caller's physical length with an extent, CELLS without one; the angular velocity is "
      "1/time either way). Any nonzero component switches the solver onto the moving-geometry "
      "path: the momentum operator's no-slip datum becomes the local wall velocity and the "
      "cut-cell projection gains the wall's own volume flux. All-zero keeps the static path, "
      "bit for bit. Staggered grid only in v1 (no ghost projection / porous / variable rho).");
  cls.def("has_moving_instance", &S::hasMovingInstance,
          "True when at least one scene instance carries a nonzero velocity.");
  cls.def("num_scene_instances", &S::sceneInstanceCount,
          "Number of instances in the scene set by set_scene (0 before it). Instance indices "
          "run 0..num_scene_instances()-1 in set_instance_transform / set_instance_motion, "
          "scene_owner() and the hydrodynamic force/torque accessors.");
  cls.def(
      "set_instance_transform",
      [](S& s, int i, std::array<double, 3> t, std::array<double, 4> q) {
        chk::instanceTransform(t, q);
        s.setInstanceTransform(i, t, q);
      },
      nb::arg("instance"), nb::arg("translation"),
      nb::arg("quat") = std::array<double, 4>{0.0, 0.0, 0.0, 1.0},
      "Move one scene instance (quat as (x,y,z,w)). Takes effect at the next "
      "rebuild_geometry(): the SDF, the cut-cell overlay, the apertures and the pressure "
      "operator are all derived from the transforms, so moving one without rebuilding would "
      "run the solver on stale geometry.");
  cls.def("refresh_wall_velocity", &S::refreshWallVelocity,
          "Re-derive ONLY the wall-velocity fields and the momentum operator that folds them in, "
          "for a driver that changes an instance's VELOCITY every step while its transform stays "
          "put -- the linearised oscillating body, a shear cell driven by counter-moving plates. "
          "set_instance_motion alone does not reach those fields (they are built in "
          "set_solid_from_scene), so the alternative is a full rebuild_geometry() per step to "
          "update a boundary condition. Refuses if a transform has changed since the last "
          "geometry build, because it does NOT re-sample the SDF, the apertures, the ownership "
          "field or the pressure operator. Velocity and pressure are untouched. "
          "Velocities stay in the caller's units (set_instance_motion).");
  cls.def("rebuild_geometry", &S::rebuildGeometry,
          "Re-derive all geometry from the current instance transforms. Velocity and pressure are "
          "PRESERVED across the rebuild (set_solid zeroes them by design). Full rebuild: the "
          "measured cost is ~65% momentum/IBM stencils, ~35% pressure/MG, scene sampling in the "
          "noise. Cells uncovered by the motion inherit zero, not an extrapolated fluid value.");
  cls.def(
      "hydro_force_torque_reaction",
      [](S& s) {
        std::vector<double> v = s.hydroForceTorqueReaction();
        const std::size_t n = v.size() / 6;
        return peclet::core::python::vector_to_ndarray(std::move(v), {2, n, 3},
                                                       {static_cast<std::int64_t>(3 * n), 3, 1});
      },
      "THE hydrodynamic force and torque on each scene instance: the DISCRETE REACTION -- "
      "the momentum the fluid actually lost to each body, assembled from the composed step's "
      "budget (time term, body force, and the viscous fluxes at u*; the pressure telescopes "
      "inside each owner region and needs no field). Shape (2, n_instances, 3): [0] force, "
      "[1] torque about the instance centre. The force is exactly conservative (sum = f * "
      "V_fluid at steady state, to the solver residual) and as accurate as the flow solution "
      "it sustains: Zick & Homsy's drag within 0.7 % at N = 24 (tests/python/"
      "test_hydro_force_units.py). The torque carries the closed-form transposed-stress wall "
      "term; measured 2026-09-30 on a steadily spinning sphere it reads 1.019-1.027 of "
      "8 pi mu a^3 Omega / (1 - phi) at R/h 6-12 (over-reads ~2 %, falling with "
      "resolution), and on a translating sphere whose true torque is zero it is zero to "
      "1e-15 on a symmetric placement and |T| / (|F| R) = 5e-5 to 1e-3 off it. "
      "diagnostics.hydro_force_torque_traction() is the surface-traction estimate: ~30 % "
      "low, but it carries the pressure/viscous split. Staggered only; EXPLICIT advection "
      "is carried (the stashed advective RHS term is subtracted); implicit advection / "
      "porous / variable properties / domain BCs are refused loudly (v2); call after "
      "step(). Atomics: tolerance-reproducible, not bitwise.");
  cls.def(
      "hydro_force_torque",
      [](S& s) {
        // Warns from the first release that carries the canonical names, not one release
        // later as docs/NAMING.md section 0 has it for a respelling: the old spelling here
        // is not harmless, it hands out a force ~30 % low under the natural name.
        if (PyErr_WarnEx(PyExc_DeprecationWarning,
                         "hydro_force_torque() is deprecated: it returns the traction "
                         "integral, which under-reads the drag by ~30 % at any resolution. "
                         "For the force and torque on the bodies call "
                         "hydro_force_torque_reaction(); for the traction and its "
                         "pressure/viscous split, diagnostics.hydro_force_torque_traction().",
                         1) < 0)
          throw nb::python_error();
        return traction_force_torque_array<Grid>(s);
      },
      "DEPRECATED (warns) -- the canonical names are hydro_force_torque_reaction() for the "
      "force and "
      "torque on the bodies, and diagnostics.hydro_force_torque_traction() for what this "
      "returns. It is NOT the force on the bodies: it is the reconstructed surface-traction "
      "integral, which under-reads the drag by ~30 % at any resolution (the viscous part; "
      "traction / reaction 0.685-0.730 over phi 0.008-0.45, N 24-128). Returns exactly what "
      "diagnostics.hydro_force_torque_traction() returns (shape (4, n_instances, 3): force, "
      "torque, pressure part, viscous part); see that method for the details. Where the "
      "reaction refuses (SolverColocated, porous, variable properties, domain BCs, implicit "
      "advection) the traction is the only estimate there is -- call it by its diagnostic "
      "name and allow for the under-read. Kept working under the suite's rule for a shipped "
      "name (docs/NAMING.md section 0); removal needs a major release. "
      "Force and torque come back in the caller's (physical) units.");
  cls.def("instance_center", &S::instanceCenter, nb::arg("i"),
          "The resolved centre of rotation of instance i (world coordinates). "
          "In the caller's length units.");
  cls.def(
      "set_solid",
      [](S& s, nb::ndarray<double, nb::f_contig> sdf, bool cutcell_pressure) {
        s.setSolid(grid_in(sdf), cutcell_pressure);
      },
      nb::arg("sdf"), nb::arg("cutcell_pressure") = false,
      "Set the solid SDF as a Fortran-order (nx,ny,nz) float64 array sampled at cell_centers() "
      "(negative inside the solid, positive in fluid, in the solver's length unit). "
      "cutcell_pressure=True enables the open-face-weighted cut-cell pressure operator "
      "(proper no-slip); it composes with domain BCs, so this is the single call for a "
      "no-slip immersed body in an inflow/outflow domain. The multigrid coarse operators are "
      "always the rediscretized per-level cut-cell operators (the former `pressure_coarse=` "
      "mode selector is gone).");
}

/// `SolverDiagnostics` (and its collocated twin): cut-cell and moving-instance instruments,
/// zero-copy field views and the ghost exchange.
template <class Grid, class C>
inline void bindDiagnosticsGeometry(C& cls) {
  using D = Diagnostics<Grid>;
  cls.def(
      "set_exact_crossings_from_scene",
      [](D& diag) { return diag.s->setExactCrossingsFromScene(); },
      "Compute EXACT wall crossings from the installed scene, on device, on every rank -- the "
      "in-solver replacement for set_exact_crossings + scripts/exact_apertures_spheres.py. "
      "Bisection, not Newton: only SIGN correctness is guaranteed for bound-only leaves.");
  cls.def(
      "set_wall_flux_divergence",
      [](D& diag, bool on) { return diag.s->setWallFluxDivergence(on); }, nb::arg("on"),
      "Rung 3 on/off (default on = correct physics). Off leaves the moving wall's no-slip "
      "datum in the momentum operator but drops the wall's own volume flux from the cut-cell "
      "projection -- the configuration the Galilean gate uses to exhibit the failure the term "
      "fixes.");
  cls.def(
      "set_fresh_cell_seed", [](D& diag, bool on) { return diag.s->setFreshCellSeed(on); },
      nb::arg("on"),
      "Fresh-cell policy for MOVING geometry. A point a body has just uncovered inherits "
      "whatever the solid held there -- zero, or a stale masked value -- which is one of the "
      "two textbook mechanisms behind spurious force oscillations in a moving-boundary IBM. "
      "True seeds it with the LOCAL WALL VELOCITY instead, so it starts moving with the "
      "surface that released it. Bounded (no extrapolation), no new field, and exactly the "
      "old behaviour when nothing moves. DEFAULT TRUE since 2026-08-30: measured on an "
      "oscillating sphere translating through the grid, it removes a resolution-INDEPENDENT "
      "+2.6..2.9% drag bias, cuts the spurious force oscillation 20-50x to within 17% of the "
      "non-moving floor, and improves the resolved CFD-DEM loop's total-momentum "
      "conservation 95x. Pass False for the pre-2026-08-30 behaviour.");
  cls.def(
      "fluid_momentum_cells",
      [](D& diag) {
        const auto n = diag.s->fluidMomentumCells();
        return std::array<long, 3>{n[0], n[1], n[2]};
      },
      "Unmasked (fluid) staggered momentum cells per component -- the exact discrete datum of "
      "the reaction identity: at steady state, sum over bodies of F_c = f_c * N_c.");
  cls.def(
      "reaction_budget_terms",
      [](D& diag) {
        std::vector<double> v = diag.s->reactionBudgetTerms();
        return peclet::core::python::vector_to_ndarray(std::move(v), {2, std::size_t(3)}, {3, 1});
      },
      "Diagnostic decomposition of the reaction identity, shape (2, 3): [0] the unsteady sum "
      "sum_i (rho/dt)(u_i - u^n_i) and [1] the advective sum sum_i A_i, both over the FLUID "
      "momentum cells of each component. The full discrete identity is sum_bodies F_c = "
      "f_c*N_c + sum fb + [1]_c - [0]_c; the Stokes form drops both because they vanish at "
      "steady state with advection off. sum_i A_i is the advection operator's net momentum "
      "flux through the cut walls -- an O(h) property of that operator, not of the budget.");
  cls.def(
      "moving_instance_cut_cells",
      [](D& diag) {
        std::vector<long> v = diag.s->movingInstanceCutCells();
        std::vector<double> d(v.begin(), v.end());
        return peclet::core::python::vector_to_ndarray(std::move(d), {v.size()}, {1});
      },
      "Per instance: inner cells on this rank it owns that touch a fractional face aperture, "
      "recounted whenever any instance moves. ZERO for a moving instance means its surface "
      "sits on grid planes (or is sub-cell) and its wall velocity is silently inert "
      "(set_solid_from_scene warns). Empty when no instance moves.");
  cls.def(
      "moving_instance_degenerate_points",
      [](D& diag) {
        std::vector<long> v = diag.s->movingInstanceDegeneratePoints();
        std::vector<double> d(v.begin(), v.end());
        return peclet::core::python::vector_to_ndarray(std::move(d), {v.size()}, {1});
      },
      "Per instance: staggered velocity points where the sampled SDF is EXACTLY zero -- a face "
      "on a lattice plane. Those points are fluid to the mask and not ghosts to the cut-cell "
      "fold, so a moving body's datum never enters there (set_solid_from_scene warns). Empty "
      "when no instance moves.");
  cls.def(
      "instance_center_pinned", [](D& diag, int i) { return diag.s->instanceCenterPinned(i); },
      nb::arg("i"),
      "True if instance i's centre of rotation is PINNED (an explicit finite centre in the "
      "encoding, or set_instance_motion(center=...)); False if it follows the body's "
      "translation (NaN in the encoding, the builder's default; legacy all-zero raw arrays).");
  cls.def(
      "periodic_image_overlap_cells", [](D& diag) { return diag.s->periodicImageOverlapCells(); },
      "Cells on this rank whose solid/fluid sign was decided by a periodic IMAGE of an "
      "instance wider than the box (set_solid_from_scene warns when nonzero): the scene "
      "evaluates the UNION of images, so a slab wider than the box refills any cavity carved "
      "from it. 0 when no instance is that wide or the images agree.");
  cls.def(
      "hydro_force_torque_traction",
      [](D& diag) { return traction_force_torque_array<Grid>(*diag.s); },
      "DIAGNOSTIC -- not the force on the bodies; that is "
      "Solver.hydro_force_torque_reaction(). The reconstructed surface-TRACTION loads, shape "
      "(4, n_instances, 3): [0] force, [1] torque about the instance centre, [2] the pressure "
      "part, [3] the viscous part ([0] == [2] + [3]). The cut-cell surface integral of "
      "(-p I + mu(grad u + grad u^T)) against the exact aperture wall-area vector, with a "
      "central-difference velocity gradient that spans the wall. It UNDER-READS THE VISCOUS "
      "PART: traction / reaction drag = 0.685-0.730 over phi 0.008-0.45 and N 24-128 "
      "(measured 2026-09-30), i.e. ~30 % low at ANY resolution -- refinement does not fix "
      "it. The torque is low by the same mechanism: a steadily spinning sphere reads "
      "0.59-0.72 of 8 pi mu a^3 Omega / (1 - phi) at R/h 6-12 (the reaction: 1.019-1.027). "
      "What it offers that the reaction does not: the pressure/viscous split, and it runs "
      "where the reaction refuses (the collocated solver, porous, variable properties, domain "
      "BCs, implicit advection). A consistent traction (from the momentum operator's "
      "small-cell-robust wall reconstruction) is planned. Was Solver.hydro_force_torque() "
      "(deprecated). Atomics: tolerance-reproducible, not bitwise.");
  cls.def(
      "wall_area_probe", [](D& diag) { return diag.s->wallAreaProbe(); },
      "Diagnostic: [sum_c x_c*A_wall_x, sum_c y_c*A_wall_y, sum_c z_c*A_wall_z] over all cut "
      "cells. Must equal -V_solid componentwise if the aperture wall-area vectors are right, "
      "so it separates a force-integral error in the GEOMETRY from one in the traction.");
  cls.def(
      "wall_flux_imbalance", [](D& diag) { return diag.s->wallFluxImbalance(); },
      "Sum over cells of u_wall . A_wall -- the compatibility datum of the singular pressure "
      "problem. Exactly zero for a translating body in a periodic box; small but nonzero for "
      "rotation. Reported, not corrected.");
  cls.def(
      "get_cut_owner",
      [](D& diag) {
        std::vector<int> v = diag.s->getCutOwner();
        const auto nx = static_cast<std::size_t>(diag.s->nx());
        const auto ny = static_cast<std::size_t>(diag.s->ny());
        const auto nz = static_cast<std::size_t>(diag.s->nz());
        return peclet::core::python::vector_to_ndarray(
            std::move(v), {nx, ny, nz},
            {1, static_cast<std::int64_t>(nx), static_cast<std::int64_t>(nx * ny)});
      },
      "Per-inner-cell owning scene instance (the argmin behind the sampled SDF), (nx,ny,nz) "
      "int32; -1 before set_solid_from_scene. Moving geometry reads its wall velocity off the "
      "owner and resolved CFD-DEM posts the hydrodynamic force back to it.");
  cls.def(
      "field_view",
      [](D& diag, const std::string& name) {
        return field3d_out(*diag.s, diag.s->fieldView(name));
      },
      nb::arg("name"),
      "Zero-copy view of a registered field's full padded buffer as a Fortran-order "
      "(nx+2g, ny+2g, nz+2g) array (g = ghost_width); host → NumPy, device → DLPack (CuPy). "
      "step() does not device-synchronize at entry: a driver that writes a view from its own "
      "non-blocking stream (cupy/torch) must synchronize that stream before calling step() "
      "(null-stream writers stay ordered).");
  cls.def(
      "exchange_field", [](D& diag, const std::string& name) { diag.s->exchangeField(name); },
      nb::arg("name"),
      "Fill a registered field's ghost cells (cross-rank + periodic under MPI; periodic "
      "single-rank).");
  cls.def(
      "exchange_field_add",
      [](D& diag, const std::string& name) { diag.s->exchangeFieldAdd(name); }, nb::arg("name"),
      "Add-reduce halo: fold ghost-layer deposits back onto their owner (cross-rank + "
      "periodic). "
      "The particle->grid deposition primitive for MPI CFD-DEM; single-rank non-periodic "
      "no-op.");
}

}  // namespace peclet::flow::bindings
