/// @file
/// @brief Python binding of peclet.flow: scalar transport.
///
/// Binding-only: included by src/flow_bindings.cpp alone, which it was split out of verbatim.
/// Each function adds one topic's entries to the `nb::class_` that bind_solver or
/// bind_diagnostics creates; the order of those calls is the registration order, and nanobind
/// resolves overloads in it.
#pragma once

#include "flow_bindings_common.hpp"

namespace peclet::flow::bindings {

/// `Solver` (and its collocated twin): scalar transport.
template <class Grid, class C>
inline void bindSolverScalar(C& cls) {
  using S = BoundSolver<Grid>;
  // --- Scalar transport (advection-diffusion) ----------------------------------------------
  cls.def(
      "add_scalar",
      [](S& s, const std::string& name, double diffusivity, const std::string& scheme,
         std::optional<int> iters, bool cutcell) {
        chk::scalarArgs(diffusivity, iters);
        if (cutcell && iters)
          throw std::invalid_argument(
              "add_scalar: iters must be None for a cut-cell scalar (it solves to a residual "
              "tolerance, set_scalar_residual_tolerance)");
        s.addScalar(name, diffusivity, enum_index(scheme, kScalarSchemes, "add_scalar", "scheme"),
                    iters ? *iters : 50, cutcell);
      },
      nb::arg("name"), nb::arg("diffusivity") = 0.0, nb::arg("scheme") = "koren",
      nb::arg("iters") = nb::none(), nb::arg("cutcell") = false,
      "Register a transported scalar (temperature/concentration/...): constant diffusivity "
      "(length^2/time, in the solver's physical units -- cells^2 per time unit without an "
      "extent), advection scheme 'fou' (first-order upwind), 'koren' (Koren TVD, the "
      "default) or 'sou' (second-order upwind), and RB-GS diffusion sweeps (iters, default "
      "50). The scalar is a registered field (get_field/set_field/diagnostics.field_view). "
      "Requires geometry (set_solid/set_pressure_geometry) for the openness-weighted "
      "operators.\n\n"
      "cutcell=True selects the cut-cell discretization (doc/scalar_ibm_design.md): kappa "
      "(fluid fraction) storage, the scalar's own apertures, wall conditions per body "
      "(set_scalar_wall, default insulating), sources (set_scalar_source), BiCGStab to a "
      "residual tolerance (set_scalar_residual_tolerance; iters must then be None), a steady solve "
      "(solve_scalar_steady) and per-body wall fluxes (scalar_wall_flux). Cells that are not "
      "fluid unknowns hold 0. Advection is not yet part of the cut-cell operator.");
  cls.def(
      "set_scalar_bc",
      [](S& s, const std::string& name, const std::string& face, const std::string& type,
         double value) {
        chk::finite("set_scalar_bc", "value", value, "the scalar's units");
        s.setScalarBc(name, face_index(face, "set_scalar_bc"),
                      enum_index(type, kScalarBcTypes, "set_scalar_bc", "type"), value);
      },
      nb::arg("name"), nb::arg("face"), nb::arg("type"), nb::arg("value") = 0.0,
      "Scalar boundary condition on a domain face ('-x', '+x', '-y', '+y', '-z', '+z'): "
      "type 'periodic', 'neumann' (zero flux, adiabatic) or 'dirichlet' (the given value, in "
      "the scalar's own units: the solver never rescales a transported scalar). Single-rank "
      "for a legacy scalar; a cut-cell scalar takes it under MPI too, and its periodic / "
      "non-periodic faces must match set_domain_bc (checked at the first advance). For a "
      "cut-cell scalar `value` may also be a per-face profile (next overload).");
  cls.def(
      "set_scalar_bc",
      [](S& s, const std::string& name, const std::string& face, const std::string& type,
         nb::ndarray<double, nb::ndim<2>, nb::device::cpu> value) {
        if (enum_index(type, kScalarBcTypes, "set_scalar_bc", "type") != 2)
          throw std::invalid_argument("set_scalar_bc: a profile is a 'dirichlet' value");
        const int n1 = static_cast<int>(value.shape(0)), n2 = static_cast<int>(value.shape(1));
        std::vector<double> prof(static_cast<std::size_t>(n1) * n2);
        auto v = value.view();
        for (int j2 = 0; j2 < n2; ++j2)
          for (int j1 = 0; j1 < n1; ++j1)
            prof[static_cast<std::size_t>(j1) + static_cast<std::size_t>(j2) * n1] = v(j1, j2);
        s.setScalarBcProfile(name, face_index(face, "set_scalar_bc"), prof, n1, n2);
      },
      nb::arg("name"), nb::arg("face"), nb::arg("type"), nb::arg("value"),
      "Cut-cell scalars: a Dirichlet domain face with a per-face PROFILE, shape (N_t1, N_t2) "
      "over this rank's face cells in the face's tangential axes in x, y, z order (-x/+x: "
      "(ny, nz); -y/+y: (nx, nz); -z/+z: (nx, ny)), as set_domain_bc_profile. The value is "
      "the scalar at the boundary face centre.");
  cls.def(
      "set_scalar_wall",
      [](S& s, const std::string& name, const std::string& type, double value, double coefficient,
         std::optional<int> instance) {
        s.setScalarWall(name, enum_index(type, kScalarWallTypes, "set_scalar_wall", "type"), value,
                        coefficient, instance ? *instance : -1);
      },
      nb::arg("name"), nb::arg("type"), nb::arg("value") = 0.0, nb::arg("coefficient") = 0.0,
      nb::arg("instance") = nb::none(),
      "Cut-cell scalars: the condition on the immersed walls of one scene instance "
      "(instance=int, needs a scene) or of every body (instance=None, which also clears the "
      "per-instance settings). 'neumann': value = the flux INTO the fluid per area (c L/T; 0, "
      "the default for every body, is insulating). 'dirichlet': value = the wall "
      "concentration. 'robin': flux into the fluid = coefficient * (value - c_wall), "
      "coefficient = k (L/T, >= 0; a first-order surface reaction is k_r with value 0). "
      "Physical units under an extent; takes effect at the next advance or steady solve.");
  cls.def(
      "set_scalar_solid",
      [](S& s, const std::string& name, double diffusivity, double capacity, double partition,
         double contact_resistance, std::optional<int> instance) {
        s.setScalarSolid(name, diffusivity, capacity, partition, contact_resistance,
                         instance ? *instance : -1);
      },
      nb::arg("name"), nb::arg("diffusivity"), nb::arg("capacity") = 1.0,
      nb::arg("partition") = 1.0, nb::arg("contact_resistance") = 0.0,
      nb::arg("instance") = nb::none(),
      "Cut-cell scalars, conjugate transport (doc/scalar_ibm_design.md §1.1): the solid "
      "interior of one scene instance (instance=int, needs a scene) or of every body "
      "(instance=None, which also clears the per-instance settings) carries the scalar too. "
      "diffusivity = D_s (L^2/T, the solid's own, >= 0), capacity = C_s (> 0; heat: "
      "(rho c_p)_s/(rho c_p)_f, a porous particle: its porosity), partition = K (> 0; c_s = K c "
      "at equilibrium), contact_resistance = R_c (T/L, >= 0; the interface flux is "
      "(c_s/K - c)/R_c). The solid field is the registered field '<name>_solid' (it holds "
      "c_s/K, internal; get_scalar_solid returns c_s). A later set_scalar_wall for the same "
      "instance makes it a wall again (last call wins). Takes effect at the next advance or "
      "steady solve. Physical units under an extent.");
  cls.def(
      "get_scalar_solid",
      [](S& s, const std::string& name) { return field_out(s, s.getScalarSolid(name)); },
      nb::arg("name"),
      "Cut-cell scalars: a copy of the solid concentration c_s = K psi_s of a conjugate scalar, "
      "(nx, ny, nz) F-order on this rank's block, NaN outside the solid unknowns (all NaN before "
      "the first advance or steady solve with a conjugate body).");
  cls.def(
      "set_scalar_source",
      [](S& s, const std::string& name, double source) { s.setScalarSource(name, source); },
      nb::arg("name"), nb::arg("source"),
      "Cut-cell scalars: a uniform volumetric source S (c/T, physical), weighted by the fluid "
      "fraction kappa.");
  cls.def(
      "set_scalar_source",
      [](S& s, const std::string& name, nb::ndarray<double, nb::f_contig> source) {
        s.setScalarSourceField(name, grid_in(source));
      },
      nb::arg("name"), nb::arg("source"),
      "Cut-cell scalars: a per-cell volumetric source S (c/T, physical), an (nx, ny, nz) "
      "F-order array on this rank's block, weighted by kappa.");
  cls.def(
      "set_scalar_residual_tolerance",
      [](S& s, const std::string& name, double rtol) { s.setScalarResidualTolerance(name, rtol); },
      nb::arg("name"), nb::arg("rtol"),
      "Cut-cell scalars: the max-norm relative residual tolerance of the solve, "
      "max|b - A c| <= rtol * max(max|b|, max|A c0|) (default 1e-10).");
  cls.def(
      "set_scalar_mean_gradient",
      [](S& s, const std::string& name, const std::array<double, 3>& gradient) {
        s.setScalarMeanGradient(name, gradient);
      },
      nb::arg("name"), nb::arg("gradient"),
      "Cut-cell scalars: the closure (mean-gradient) mode. The field then holds the periodic "
      "fluctuation theta of c = G.x + theta, written in the frame moving with the fluid's mean "
      "velocity; G = `gradient` (c/L, physical; x the cell_centers() coordinates). Every axis "
      "with a nonzero component must be periodic (checked at the next advance or solve). "
      "(0, 0, 0) turns it off. Read the result with scalar_mean_flux.");
  cls.def(
      "scalar_mean_flux",
      [](S& s, const std::string& name) {
        const auto j = s.scalarMeanFlux(name);
        return peclet::core::python::vector_to_ndarray(std::vector<double>(j.begin(), j.end()), {3},
                                                       {1});
      },
      nb::arg("name"),
      "Cut-cell scalars: the box-averaged total flux vector (3,), c L/T (physical), of the "
      "last advance or steady solve: the diffusive face sum of -D a A (c_j - c_i) with "
      "c = theta + G.x (the face's true displacement), plus the advective (1/V_box) sum_i "
      "(U_i - kappa_i V Ubar) theta_i in the moving frame, V_box the whole box (solid "
      "included). Exact as a plane average for steady fields without wall sources. Effective "
      "diffusivity k* = -J.G/(D|G|^2); dispersion D* = -J/(phi G), phi the discrete porosity. "
      "Collective under MPI.");
  cls.def(
      "solve_scalar_steady", [](S& s, const std::string& name) { s.solveScalarSteady(name); },
      nb::arg("name"),
      "Cut-cell scalars: solve the steady problem now (no storage term) from the current "
      "field as the initial guess. Statistics in diagnostics.scalar_census.");
  cls.def(
      "scalar_wall_flux",
      [](S& s, const std::string& name) {
        std::vector<double> v = s.scalarWallFlux(name);
        const std::size_t n = v.size();
        return peclet::core::python::vector_to_ndarray(std::move(v), {n}, {1});
      },
      nb::arg("name"),
      "Cut-cell scalars: the integrated flux INTO the fluid per body over the last advance or "
      "steady solve, shape (num_bodies,) (one body without a scene), in c L^3/T (physical). "
      "Collective under MPI.");
  cls.def(
      "has_scalar", [](S& s, const std::string& name) { return s.hasScalar(name); },
      nb::arg("name"), "Whether a transported scalar of this name is registered.");
  cls.def(
      "advance_scalars", [](S& s) { s.advanceScalars(); },
      "Advance all registered scalars one dt with the current velocity (also done by step()).");
}

/// `SolverDiagnostics` (and its collocated twin): scalar geometry, census, budget and facets.
template <class Grid, class C>
inline void bindDiagnosticsScalar(C& cls) {
  using D = Diagnostics<Grid>;
  cls.def(
      "scalar_geometry",
      [](D& diag, const std::string& name) {
        diag.s->cutcellScalar(name, "scalar_geometry");
        nb::dict d;
        d["kappa"] = field_out(*diag.s, diag.s->scalarGeometryField(0));
        d["aperture_x"] = field_out(*diag.s, diag.s->scalarGeometryField(1));
        d["aperture_y"] = field_out(*diag.s, diag.s->scalarGeometryField(2));
        d["aperture_z"] = field_out(*diag.s, diag.s->scalarGeometryField(3));
        d["unknown"] = field_out(*diag.s, diag.s->scalarGeometryField(4));
        if (diag.s->cutcellScalar(name, "scalar_geometry").cut->solid.extent(0) > 0) {
          d["kappa_solid"] = field_out(*diag.s, diag.s->scalarSolidGeometryField(name, 0));
          d["solid_unknown"] = field_out(*diag.s, diag.s->scalarSolidGeometryField(name, 1));
        }
        return d;
      },
      nb::arg("name"),
      "The cut-cell scalar geometry record (doc/scalar_ibm_design.md §2) as F-order (nx, ny, "
      "nz) arrays of this rank's block: 'kappa' (fluid fraction of the fan-tetrahedron PL "
      "model), 'aperture_x/y/z' (the SNAPPED, ungated scalar aperture of the -x/-y/-z face of "
      "each cell -- the scalar's own, never the pressure openness), 'unknown' (1 where the "
      "cell is a fluid unknown: kappa > 1e-14 with an open face). Built on first use after "
      "the geometry; shared by every cut-cell scalar. A scalar with a solid field "
      "(set_scalar_solid) also gets 'kappa_solid' (1 - kappa) and 'solid_unknown' (1 where the "
      "cell is a solid unknown: kappa_solid > 0 and the cell's material conjugate, §2.7).");
  cls.def(
      "scalar_census",
      [](D& diag, const std::string& name) {
        const auto& st = *diag.s->cutcellScalar(name, "scalar_census").cut;
        const auto c = diag.s->scalarCutCensus();
        nb::dict d;
        d["num_unknowns"] = c.numUnknowns;
        d["num_cut_cells"] = c.numCutCells;
        d["num_facets"] = c.numFacets;
        d["num_two_sided"] = c.numTwoSided;
        d["num_thin_solid"] = c.numThinSolid;
        d["num_sealed"] = c.numSealed;
        d["sealed_volume"] = c.sealedVolume;
        nb::dict rungs;
        rungs["fluid"] = nb::make_tuple(c.rungs[0], c.rungs[1], c.rungs[2], c.rungs[3]);
        d["probe_rungs"] = rungs;
        d["num_solid_unknowns"] = st.conj ? st.numSolidUnknowns : 0L;
        d["solid_volume"] = st.conj ? st.solidVolume : 0.0;
        d["sealed_solid_volume"] = st.conj ? st.sealedSolidVolume : 0.0;
        if (st.conj)
          rungs["solid"] = nb::make_tuple(st.solidRungs[0], st.solidRungs[1], st.solidRungs[2],
                                          st.solidRungs[3]);
        d["krylov_iterations"] = st.iterations;
        d["krylov_residual"] = st.residual;
        d["krylov_converged"] = st.converged;
        d["num_mg_levels"] = st.mgLevels;
        d["steady_incompatibility"] = st.incompatibility;
        d["num_small_cells"] = st.numSmall;
        d["num_implicit_faces"] = st.numImplicitFaces;
        d["num_flux_faces"] = st.numFluxFaces;
        d["num_guarded_flux_faces"] = st.numGuardedFaces;
        d["num_backflow_faces"] = st.numBackflowFaces;
        d["bulk_courant"] = st.bulkCourant;
        d["max_cell_peclet"] = st.maxCellPeclet;
        return d;
      },
      nb::arg("name"),
      "Census of the cut-cell scalar (collective under MPI, summed over ranks). The solve "
      "part, of the last advance or steady solve: 'krylov_iterations', 'krylov_residual' "
      "(the true max-norm residual over the reference max(max|b|, max|A c0|)), "
      "'krylov_converged', 'num_mg_levels' (the ScalarMG levels the V-cycle used; 1: level 0 "
      "alone), "
      "'steady_incompatibility' (|sum b| / sum|b| of a singular steady problem before its "
      "projection, else 0), 'num_solid_unknowns' (0: single phase; the solid unknowns of a "
      "conjugate scalar, whose 'probe_rungs' then carries a 'solid' tuple too), 'solid_volume' "
      "/ "
      "'sealed_solid_volume' (physical: the conjugate solid kappa_s V, and its part in cells "
      "that are not solid unknowns -- kappa_s > 0 with no open solid face and no conjugate "
      "facet, ruling D-WO7-1). The advection part "
      "(design §6.3, the last advance or steady solve): 'num_small_cells' (cut cells whose "
      "explicit outflow would exceed the bulk's), 'num_implicit_faces' / 'num_flux_faces' "
      "(faces carrying flux that took implicit upwind / all faces carrying flux; steady: all "
      "implicit), 'bulk_courant' (C_bulk: max over full cells of dt Out/V; 0 steady), "
      "'num_guarded_flux_faces' (faces with projection flux toward a cell that is not a fluid "
      "unknown, whose flux the operator drops), 'num_backflow_faces' (rows of an 'outflow' "
      "domain face whose flux enters the domain: their inflow takes the zero-gradient value; "
      "a steady solve warns once), 'max_cell_peclet' (the largest "
      "aperture-weighted face Peclet number |u_a| a h_a / D over the interior faces; the "
      "steady envelope of the advective preconditioner is <= 10). The geometry "
      "part: 'num_unknowns' (fluid unknowns), 'num_cut_cells' (cells carrying a facet), "
      "'num_facets', 'num_two_sided' (cells with two facets), 'num_thin_solid' (two-facet "
      "cells with solid between two fluids -- a resolution warning), 'num_sealed' / "
      "'sealed_volume' (kappa > 0 behind closed faces; physical volume), 'probe_rungs' "
      "({'fluid': (R0, R1a, R1b, R2)} facet counts of the probe ladder).");
  cls.def(
      "scalar_budget",
      [](D& diag, const std::string& name) {
        const auto b = diag.s->scalarBudget(name);
        nb::dict d;
        d["mass"] = b.mass;
        d["mass_solid"] = b.massSolid;
        d["d_mass"] = b.dMass;
        d["wall_in"] = b.wallIn;
        d["boundary_in"] = b.boundaryIn;
        d["source_in"] = b.sourceIn;
        d["defect"] = b.defect;
        d["identity_error"] = b.identityError;
        return d;
      },
      nb::arg("name"),
      "Budget of a cut-cell scalar over the last advance (collective under MPI), physical: "
      "'mass' = sum kappa V c, 'mass_solid' (= sum C_s kappa_s V c_s of a conjugate scalar; 0 "
      "single phase), 'd_mass' = the change of mass + mass_solid over the advance, 'wall_in' / "
      "'boundary_in' / 'source_in' = the rates INTO the fluid from the "
      "immersed walls (the conjugate interfaces are internal: not in it), the domain faces "
      "(the Dirichlet faces' diffusive flux plus the advective "
      "flux through inflow/outflow faces) and the source, 'defect' = dt sum V r (r the "
      "linear-solver residual), and 'identity_error' = d_mass - dt (wall_in + boundary_in + "
      "source_in) + defect, which is round-off for a correct discretization. After a steady "
      "solve it is the rate balance: d_mass = 0, defect = sum V r and identity_error = "
      "defect - (wall_in + boundary_in + source_in). Computed on request from the stored "
      "state.");
  cls.def(
      "scalar_facets",
      [](D& diag, const std::string& name) {
        auto f = diag.s->scalarFacets(name);
        const std::size_t n = f.area.size();
        nb::dict d;
        d["centroid"] =
            peclet::core::python::vector_to_ndarray(std::move(f.centroid), {n, 3}, {3, 1});
        d["normal"] = peclet::core::python::vector_to_ndarray(std::move(f.normal), {n, 3}, {3, 1});
        d["area"] = peclet::core::python::vector_to_ndarray(std::move(f.area), {n}, {1});
        d["instance"] = peclet::core::python::vector_to_ndarray(std::move(f.instance), {n}, {1});
        d["wall_value"] = peclet::core::python::vector_to_ndarray(std::move(f.wallValue), {n}, {1});
        d["flux"] = peclet::core::python::vector_to_ndarray(std::move(f.flux), {n}, {1});
        d["probe_rung"] = peclet::core::python::vector_to_ndarray(std::move(f.rung), {n}, {1});
        return d;
      },
      nb::arg("name"),
      "Per-facet arrays of a cut-cell scalar on this rank (facet order = x-fastest cell "
      "order): 'centroid' (n, 3) physical coordinates, 'normal' (n, 3) into the fluid, 'area' "
      "(physical), 'instance' (scene body; 0 for raw-SDF geometry), 'wall_value' (c at the "
      "wall from the elimination: robin (D u_p + k s g)/(D + k s), dirichlet g, neumann "
      "u_p + q s/D; NaN on a facet whose cell is not an unknown), 'flux' (into the fluid per "
      "area, c L/T physical, after the last advance or steady solve; local Nu/Sh maps, first "
      "order in the max norm) and 'probe_rung' (0..3 = R0, R1a, R1b, R2).");
  cls.def(
      "set_scalar_max_iterations",
      [](D& diag, const std::string& name, int max_iter) {
        diag.s->setScalarMaxIterations(name, max_iter);
      },
      nb::arg("name"), nb::arg("max_iter"),
      "Cut-cell scalars: the BiCGStab iteration cap (default 200). On non-convergence the "
      "iterate is kept, diagnostics.scalar_census reports it and a one-time warning is "
      "printed.");
}

}  // namespace peclet::flow::bindings
