/// @file
/// @brief Python binding of peclet.flow: the sharp viscous interface stress of the VoF momentum
/// (setter, read-out, census, fit).
///
/// Binding-only: included by src/flow_bindings.cpp alone, which it was split out of verbatim.
/// Each function adds one topic's entries to the `nb::class_` that bind_solver or
/// bind_diagnostics creates; the order of those calls is the registration order, and nanobind
/// resolves overloads in it.
#pragma once

#include "flow_bindings_common.hpp"

namespace peclet::flow::bindings {

/// `Solver` (and its collocated twin): sharp viscous interface setter and read-out.
template <class Grid, class C>
inline void bindSolverSharpViscous(C& cls) {
  using S = BoundSolver<Grid>;
  cls.def("set_vof_viscous_interface", &S::setVofViscousInterface, nb::arg("mode"),
          "The viscous stress at the VoF interface (doc/vof_sharp_viscous_stress.md): 'auto' "
          "(default), 'sharp' or 'averaged'. 'averaged' = the cell mu(C) and the 4-cell mean on "
          "the MAC edges, first order across an oblique interface (the operator before the "
          "sharp interface; it reproduces earlier numbers bitwise). 'sharp' = on every band edge "
          "the row's own PLIC face-fraction viscosity (implicit band and explicit transpose), a "
          "per-link interface-traction correction from a kinked least-squares fit of the "
          "interface state, and the cell midsection force of the normal-stress jump 2[mu]D_nn "
          "(explicit on u^n): second order. 'auto' resolves at each step to 'sharp' when VoF is "
          "on, mu = LinearMix(C) (set_property_model('mu', 'linear', 'C', ...)) varies and the "
          "full viscous stress is active (set_viscous_stress), with no immersed solid, porous "
          "momentum, moving scene instances or phase change; else to 'averaged' with a one-time "
          "stderr notice. SolverColocated always resolves 'auto' to 'averaged' and raises on "
          "'sharp'. An explicit 'sharp' raises at step() outside that scope and with the "
          "harmonic face viscosity. Inert for a constant mu. A cell is interfacial iff eps < C "
          "< 1 - eps with eps = diagnostics.vof_interface_eps() (the wisp guard, default 1e-8). "
          "Read-outs: diagnostics.vof_viscous_interface_census() / vof_viscous_interface_fit(). "
          "Any other mode raises ValueError.");
  cls.def("vof_viscous_interface", &S::vofViscousInterface,
          "The selected VoF viscous interface mode, 'auto', 'sharp' or 'averaged' (see "
          "set_vof_viscous_interface; what 'auto' resolved to shows in "
          "diagnostics.vof_viscous_interface_census()).");
}

/// `SolverDiagnostics` (and its collocated twin): sharp viscous interface census and fit.
template <class Grid, class C>
inline void bindDiagnosticsSharpViscous(C& cls) {
  using D = Diagnostics<Grid>;
  cls.def(
      "vof_viscous_interface_census",
      [](D& diag) {
        const auto c = diag.s->vofViscousInterfaceCensus();
        nb::dict d;
        d["band_cells"] = c.band;
        d["virtual_cells"] = c.virt;
        d["film_cells"] = c.film;
        d["fit_fallbacks"] = c.fallback;
        d["tied_edges"] = c.tied;
        d["max_tie_spread"] = c.tieSpread;
        d["min_untied_gap"] = c.untiedGap;
        d["blended_edges"] = c.blended;
        return d;
      },
      "The sharp interface band of the last step, LOCAL to this rank (sum over ranks "
      "yourself): 'band_cells' (interfacial cells, PLIC and virtual planes), "
      "'virtual_cells' (pure cells carrying a face plane), 'film_cells' (virtual cells "
      "whose two neighbours along the plane axis are both of the other phase) and "
      "'fit_fallbacks' (cells whose interface fit was singular: tau = D_nn = 0, first order "
      "locally). The edge-owner ties (§5.8; every in-domain edge of the owned cells, an edge "
      "on a non-periodic high domain face not counted): 'tied_edges' (band edges whose owner "
      "set's plateau, the candidates within 1e-6 cells of the closest plane, has two or more "
      "members), 'max_tie_spread' (the largest distance spread within a plateau, index "
      "units; 0.0 if none), 'min_untied_gap' (the smallest distance from the closest plane "
      "to a candidate outside the plateau; inf if none) and 'blended_edges' (band edges with "
      "a candidate on the owner-weight ramp, 1e-6 to 1e-5 cells beyond the closest, which "
      "enters the edge's weighted mean with a partial weight) -- INFO, not a defect signal; "
      "take max/min over ranks yourself. All zero (gap inf) while the sharp interface is "
      "inactive.");
  cls.def(
      "vof_viscous_interface_fit",
      [](D& diag) {
        auto f = diag.s->vofViscousInterfaceFit();
        const char* key[5] = {"tau_x", "tau_y", "tau_z", "dnn", "fallback"};
        nb::dict d;
        for (int k = 0; k < 5; ++k)
          if (!f[k].empty())
            d[key[k]] = field_out(*diag.s, std::move(f[k]));
        return d;
      },
      "The sharp interface fit of the last step (doc/vof_sharp_viscous_stress.md §5.3), "
      "LOCAL to this rank, as (nx, ny, nz) arrays: 'tau_x', 'tau_y', 'tau_z' (the interface "
      "traction tau_hat) and 'dnn' (D_nn_hat), both in internal units, zero off the band, "
      "and 'fallback' (1 where the fit was singular). Empty while the sharp interface is "
      "inactive. A read-out for verification; it changes nothing.");
}

}  // namespace peclet::flow::bindings
