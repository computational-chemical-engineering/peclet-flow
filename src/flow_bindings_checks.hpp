/// @file
/// @brief Input validation of the `peclet.flow` Python binding (src/flow_bindings.cpp only).
///
/// The C++ solver API (flow_ibm.hpp) trusts its caller; the public setters refuse a NaN/Inf or a
/// physically inadmissible value at the binding, as ValueError (std::invalid_argument), before it
/// reaches the solver. A valid value takes exactly the path it took before. The message names the
/// setter, the parameter, the admissible range with its dimension (flow computes in the caller's
/// consistent units, or in cells without an extent) and the offending value.
///
/// Two kinds of entries: the check helpers (`finite`, `positive`, ...), called from the binding's
/// own lambdas, and the CHECKED SETTERS (`set_rho<S>`, ...), free functions bound in place of a
/// member pointer or a lambda: they keep the Python parameter names, defaults and return type, and
/// keep the validation branches out of bind_solver. Binding-only: no solver header includes this.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace peclet::flow::binding_checks {

inline std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%g", v);
  return b;
}
inline void require(bool ok, const char* who, const std::string& name, const std::string& rule,
                    double v) {
  if (!ok)
    throw std::invalid_argument(std::string(who) + ": " + name + " must be " + rule + ", got " +
                                num(v));
}
inline void finite(const char* who, const std::string& name, double v, const char* dim) {
  require(std::isfinite(v), who, name, std::string("finite (") + dim + ")", v);
}
inline void positive(const char* who, const std::string& name, double v, const char* dim) {
  require(std::isfinite(v) && v > 0.0, who, name, std::string("finite and > 0 (") + dim + ")", v);
}
inline void nonneg(const char* who, const std::string& name, double v, const char* dim) {
  require(std::isfinite(v) && v >= 0.0, who, name, std::string("finite and >= 0 (") + dim + ")", v);
}
/// A volume fraction or a dimensionless coefficient bounded by one: [0, 1].
inline void fraction(const char* who, const std::string& name, double v) {
  require(std::isfinite(v) && v >= 0.0 && v <= 1.0, who, name,
          "finite and in [0, 1] (dimensionless)", v);
}
/// A count (cells, levels, iterations): an integer >= `lo`.
inline void count(const char* who, const std::string& name, long v, long lo) {
  require(v >= lo, who, name, "an integer >= " + std::to_string(lo), static_cast<double>(v));
}
/// Per-component forms for a triple or a quaternion: the parameter is named `name[i]`.
inline std::string component(const char* name, std::size_t i) {
  return std::string(name) + "[" + std::to_string(i) + "]";
}
template <class T, std::size_t N>
inline void finite_each(const char* who, const char* name, const std::array<T, N>& a,
                        const char* dim) {
  for (std::size_t i = 0; i < N; ++i)
    finite(who, component(name, i), a[i], dim);
}
inline void finite_each(const char* who, const char* name, const std::vector<double>& a,
                        const char* dim) {
  for (std::size_t i = 0; i < a.size(); ++i)
    finite(who, component(name, i), a[i], dim);
}
template <std::size_t N>
inline void count_each(const char* who, const char* name, const std::array<long, N>& a, long lo) {
  for (std::size_t i = 0; i < N; ++i)
    count(who, component(name, i), a[i], lo);
}

// --- composite checks of one binding lambda ------------------------------------------------

/// set_instance_motion: finite velocities and, when given, a finite rotation center.
inline void instance_motion(const std::array<double, 3>& lin, const std::array<double, 3>& ang,
                            const std::optional<std::array<double, 3>>& center) {
  finite_each("set_instance_motion", "lin_vel", lin, "length/time");
  finite_each("set_instance_motion", "ang_vel", ang, "1/time");
  if (center)
    finite_each("set_instance_motion", "center", *center, "length");
}
/// set_instance_transform: a finite translation and a finite quaternion.
inline void instance_transform(const std::array<double, 3>& t, const std::array<double, 4>& q) {
  finite_each("set_instance_transform", "translation", t, "length");
  finite_each("set_instance_transform", "quat", q, "dimensionless");
}
/// enable_vof_blocks: each seed sphere (cx, cy, cz, radius) in cell units, global indices.
inline void vof_seeds(const std::optional<std::vector<std::array<double, 4>>>& seeds) {
  if (!seeds)
    return;
  for (std::size_t k = 0; k < seeds->size(); ++k) {
    const auto& c = (*seeds)[k];
    const std::string s = "seeds[" + std::to_string(k) + "]";
    for (std::size_t i = 0; i < 3; ++i)
      finite("enable_vof_blocks", s + "[" + std::to_string(i) + "]", c[i], "cells");
    positive("enable_vof_blocks", s + "[3]", c[3], "radius, cells");
  }
}
/// set_property_table: finite abscissae and values.
inline void property_table(const std::vector<double>& x, const std::vector<double>& y) {
  finite_each("set_property_table", "x", x, "the field's units");
  finite_each("set_property_table", "y", y, "the target property's units");
}
/// init_mpi / the global grid: at least one cell per axis.
inline void global_dims(const char* who, int gnx, int gny, int gnz) {
  count(who, "gnx", gnx, 1);
  count(who, "gny", gny, 1);
  count(who, "gnz", gnz, 1);
}
/// add_scalar: a non-negative diffusivity and, when given, at least one sweep.
inline void scalar_args(double diffusivity, const std::optional<int>& iters) {
  nonneg("add_scalar", "diffusivity", diffusivity, "length^2/time");
  if (iters)
    count("add_scalar", "iters", *iters, 1);
}

// --- checked setters (bound in place of a member pointer) -----------------------------------

/// Solver(nx, ny, nz): at least one cell per axis (a negative count asked Kokkos for 16 EiB).
template <class S>
void init_cells(S* self, int nx, int ny, int nz) {
  count("Solver", "nx", nx, 1);
  count("Solver", "ny", ny, 1);
  count("Solver", "nz", nz, 1);
  new (self) S(nx, ny, nz);
}
/// Solver(cells, extent=None, origin=(0, 0, 0), global_cells=None). The extent is checked by the
/// solver (setPhysicalDomain); the cell-count message and type are the ones it always had.
template <class S>
void init_physical(S* self, std::array<long, 3> cells, std::optional<std::array<double, 3>> extent,
                   std::array<double, 3> origin, std::optional<std::array<long, 3>> global_cells) {
  for (int a = 0; a < 3; ++a)
    if (cells[a] <= 0)
      throw std::runtime_error("Solver: every cell count must be positive");
  finite_each("Solver", "origin", origin, "length");
  if (global_cells)
    count_each("Solver", "global_cells", *global_cells, 1);
  new (self) S((int)cells[0], (int)cells[1], (int)cells[2]);
  if (extent)
    self->setPhysicalDomain(*extent, origin, global_cells ? *global_cells : cells);
}
template <class S>
void set_rho(S& s, double rho) {
  positive("set_rho", "rho", rho, "density");
  s.setRho(rho);
}
/// mu = 0 is admitted: the inviscid limit (scripts/pressure_h_independence.py and others use it).
template <class S>
void set_mu(S& s, double mu) {
  nonneg("set_mu", "mu", mu, "dynamic viscosity");
  s.setMu(mu);
}
template <class S>
void set_dt(S& s, double dt) {
  positive("set_dt", "dt", dt, "time");
  s.setDt(dt);
}
/// rtol < 0 follows the pressure solver's rtol and 0 is the legacy criterion: finite only.
template <class S>
void set_velocity_residual_tolerance(S& s, double rtol) {
  finite("set_velocity_residual_tolerance", "rtol", rtol, "dimensionless");
  s.setVelocityResidualTolerance(rtol);
}
template <class S>
void set_backflow_stabilization(S& s, double beta) {
  fraction("set_backflow_stabilization", "beta", beta);
  s.setBackflowStab(beta);
}
template <class S>
void set_pressure_solver_params(S& s, int iters) {
  count("set_pressure_solver_params", "iters", iters, 1);
  s.setPressureIterations(iters);
}
/// levels = 0 is the automatic depth.
template <class S>
void set_pressure_multigrid(S& s, bool, int levels) {
  count("set_pressure_multigrid", "levels", levels, 0);
  s.setPressureLevels(levels);
}
/// The three pressure drivers share the cap/tolerance pair: >= 1 iteration, rtol >= 0 (0 runs
/// to the cap).
inline void krylov_args(const char* who, int max_iter, double rtol) {
  count(who, "max_iter", max_iter, 1);
  nonneg(who, "rtol", rtol, "dimensionless");
}
template <class S>
void set_pressure_chebyshev(S& s, bool on, int max_iter, double rtol) {
  krylov_args("set_pressure_chebyshev", max_iter, rtol);
  s.setPressureChebyshev(on, max_iter, rtol);
}
template <class S>
void set_pressure_pcg(S& s, bool on, int max_iter, double rtol) {
  krylov_args("set_pressure_pcg", max_iter, rtol);
  s.setPressurePcg(on, max_iter, rtol);
}
template <class S>
void set_pressure_fcg(S& s, bool on, int max_iter, double rtol) {
  krylov_args("set_pressure_fcg", max_iter, rtol);
  s.setPressureFcg(on, max_iter, rtol);
}
/// The solver used to clamp levels and vcycles below 1 up to 1 silently.
template <class S>
void set_velocity_multigrid(S& s, bool on, int levels, int vcycles) {
  count("set_velocity_multigrid", "levels", levels, 1);
  count("set_velocity_multigrid", "vcycles", vcycles, 1);
  s.setVelocityMultigrid(on, levels, vcycles);
}
template <class S>
void set_body_force(S& s, std::array<double, 3> force) {
  finite_each("set_body_force", "force", force, "force/volume");
  s.setBodyForce(force[0], force[1], force[2]);
}
template <class S>
void set_vof_cfl_limit(S& s, double value) {
  positive("set_vof_cfl_limit", "value", value, "Courant number, dimensionless");
  s.setVofCflLimit(value);
}
template <class S>
void set_capillary_cfl(S& s, double factor) {
  positive("set_capillary_cfl", "factor", factor, "dimensionless");
  s.setCapillaryCfl(factor);
}
/// The sign is the direction (positive = evaporation): finite only.
template <class S>
void set_mass_flux_uniform(S& s, double mdot) {
  finite("set_mass_flux_uniform", "mdot", mdot, "mass/(area time)");
  s.setMassFluxUniform(mdot);
}
template <class S>
void set_property_table(S& s, const std::string& target, const std::string& field,
                        const std::vector<double>& x, const std::vector<double>& y) {
  property_table(x, y);
  s.setPropertyTable(target, field, x, y);
}

}  // namespace peclet::flow::binding_checks
