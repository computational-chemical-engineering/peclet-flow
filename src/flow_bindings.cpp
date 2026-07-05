/// @file
/// @brief nanobind module `flow` — the Kokkos cut-cell IBM Navier-Stokes solver
/// (`peclet.flow.Solver`).
///
/// Exposes peclet::flow::IbmSolver to Python: set rho/mu/dt, a body force, an SDF solid (cut-cell
/// IBM no-slip
/// + optional cut-cell pressure projection), step, read back the velocity/pressure, and query the
/// cut-cell flux divergence. Exercised by verify_poiseuille_sdflow (IBM channel) and
/// verify_periodic_spheres_sdflow (cut-cell Stokes through a sphere packing). Kokkos is initialized
/// at import and finalized via Python atexit (the solver holds Kokkos Views, so callers must
/// release the Solver before exit -- del + gc.collect()). rank()/bcast_from_root() are single-rank
/// stubs (the multi-rank path lives in tests/kokkos_mpi).
///
/// Arrays cross the boundary through the shared zero-copy bridge (peclet::core::python, in core):
/// fields come back as Fortran-order (nx,ny,nz) float64 NumPy arrays referencing the field buffer,
/// and inputs are read as flat x-fastest buffers. See tpx/python/ndarray_interop.hpp.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#ifdef PECLET_FLOW_MPI
#include <mpi.h>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#endif

#include "flow_ibm.hpp"
#include "peclet/core/python/ndarray_interop.hpp"

#ifdef PECLET_FLOW_MPI
// Ensure MPI_Init has been called (mirrors the dem init_mpi idiom); safe to call repeatedly.
static void ensure_mpi_init() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
  }
}
#endif

namespace nb = nanobind;

// A solver field (flat x-fastest, ghost-stripped) -> Fortran-order (nx,ny,nz) float64 NumPy array
// for [x,y,z] indexing. The vector is moved into the array's backing store (no extra copy vs the
// old to_xyz).
template <class S>
static nb::ndarray<nb::numpy, double> field_out(S& s, std::vector<double>&& v) {
  const auto nx = static_cast<std::size_t>(s.nx());
  const auto ny = static_cast<std::size_t>(s.ny());
  const auto nz = static_cast<std::size_t>(s.nz());
  return peclet::core::python::vector_to_ndarray(
      std::move(v), {nx, ny, nz},
      {1, static_cast<std::int64_t>(nx), static_cast<std::int64_t>(nx * ny)});
}

// A Fortran-order (nx,ny,nz) float64 array -> flat x-fastest host vector (F-contiguous data() is
// already x-fastest). nanobind casts/copies the input to f_contig double if needed.
static std::vector<double> grid_in(nb::ndarray<double, nb::f_contig> a) {
  return peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(a));
}

// Zero-copy export of a registered field's padded device buffer as a Fortran-order 3-D array of the
// full block shape (ex,ey,ez) = (nx+2G, ny+2G, nz+2G), x-fastest strides {1,ex,ex*ey}. Includes the
// ghost band (the flat buffer is contiguous; a ghost-stripped view would not be). The capsule owns a
// copy of the managed CCField, so the allocation outlives the array — host → NumPy referencing the
// buffer, device → DLPack for CuPy/torch. Mirrors peclet::core::python::view_to_ndarray but with an
// explicit 3-D reshape of the flat 1-D field.
template <class S>
static auto field3d_out(S& s, peclet::flow::CCField f) {
  namespace pcp = peclet::core::python;
  using Mem = peclet::flow::CCMem;
  const auto bs = s.blockShape();
  std::array<std::size_t, 3> shape{static_cast<std::size_t>(bs[0]), static_cast<std::size_t>(bs[1]),
                                   static_cast<std::size_t>(bs[2])};
  std::array<std::int64_t, 3> strides{1, static_cast<std::int64_t>(bs[0]),
                                      static_cast<std::int64_t>(bs[0]) * bs[1]};
  auto* held = new peclet::flow::CCField(f);
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<peclet::flow::CCField*>(p); });
  double* data = f.data();
  if constexpr (pcp::is_host_space_v<Mem>) {
    return nb::ndarray<nb::numpy, double>(data, 3, shape.data(), owner, strides.data(),
                                          nb::dtype<double>(), nb::device::cpu::value, 0);
  } else {
    auto dev = pcp::dlpack_device<Mem>();
    return nb::ndarray<double>(data, 3, shape.data(), owner, strides.data(), nb::dtype<double>(),
                               dev.first, dev.second);
  }
}

// Register a solver class for the given GridLayout policy (Staggered -> "Solver", Colocated ->
// "SolverColocated"). The Python API is identical across grids; only the velocity-unknown placement
// and the advection control volume differ inside Solver<Grid>.
template <class Grid>
static void bind_solver(nb::module_& m, const char* name) {
  using S = peclet::flow::Solver<Grid>;
  nb::class_<S>(m, name)
      .def(nb::init<int, int, int>(), nb::arg("nx"), nb::arg("ny"), nb::arg("nz"),
           "Create a solver on an nx x ny x nz unit-spacing grid (x-fastest, I = x + y*nx + "
           "z*nx*ny). "
           "Set physical parameters (rho/mu/dt) and any domain BCs before the geometry / first "
           "step.")
      .def("set_rho", &S::setRho, nb::arg("rho"),
           "Set fluid density rho (physical units). Set before geometry/first step.")
      .def("set_mu", &S::setMu, nb::arg("mu"), "Set dynamic viscosity mu (physical units).")
      .def("set_dt", &S::setDt, nb::arg("dt"),
           "Set the time step dt; the momentum solve is scaled by 1/dt (well-conditioned at large "
           "dt).")
      .def("set_body_force", &S::setBodyForce, nb::arg("fx"), nb::arg("fy"), nb::arg("fz"),
           "Set the body force per unit volume (fx, fy, fz) — e.g. a mean pressure gradient.")
      .def("set_advection", &S::setAdvection, nb::arg("on"),
           "Enable/disable explicit high-order momentum advection (default scheme SOU). Off ⇒ "
           "Stokes.")
      .def("set_advection_scheme", &S::setAdvectionScheme, nb::arg("scheme"),
           "High-order advection scheme: 0 = second-order upwind (SOU, default), 1 = Koren TVD.")
      .def("set_incremental_pressure", &S::setIncrementalPressure, nb::arg("on"),
           "Toggle the rotational incremental-pressure projection.")
      .def("set_pressure_warmstart", &S::setPressureWarmstart, nb::arg("on"),
           "Seed each pressure solve from the previous step's phi (default off).")
      .def("set_face_interp", &S::setFaceInterp, nb::arg("mode"),
           "Collocated cut-cell projection treatment: 0 = plain averaging + central-difference "
           "grad(P) (default), 1 = wall-aware cell->face map only (ablation), 2 = wall-aware map + "
           "its transpose (face-centre; ablation), 3 = mode 2 at the open-face-centroid (FV "
           "constraint, FD momentum), 4 = fully-FV (mode-3 projection + second-order wall "
           "viscous-flux deferred correction on the momentum; targets 2nd-order drag). No effect on "
           "the staggered solver.")
      .def("set_fv_relax", &S::setFvRelax, nb::arg("w"),
           "Mode-4 FV wall-flux defect-correction under-relaxation (1=full; <1 damps the stiff "
           "explicit-lagged wall term). Steady state is independent of w.")
      .def("set_velocity_streams", &S::setVelocityStreams, nb::arg("on"),
           "Toggle overlapped per-component velocity solves.")
      .def("set_implicit_advection", &S::setImplicitAdvection, nb::arg("on"),
           "Use implicit-FOU advection with deferred-correction TVD.")
      .def("set_outer_iterations", &S::setOuterIterations, nb::arg("n"),
           "Set the number of Picard/outer iterations per step.")
      .def("set_outer_tolerance", &S::setOuterTolerance, nb::arg("tol"),
           "Set the outer (Picard) convergence tolerance.")
      .def("last_outer_iterations", &S::lastOuterIterations,
           "Return the outer-iteration count from the last step().")
      .def("set_velocity_solver_params", &S::setVelocityIterations, nb::arg("iters"),
           "Set the velocity (diffusion) smoother iteration count.")
      .def("set_deferred_correction", &S::setDeferredCorrection, nb::arg("on"),
           "Deferred-correction advection: True (default) = 2nd order (implicit FOU + explicit "
           "high-order correction, the high-order scheme being SOU by default or Koren TVD via "
           "set_advection_scheme); False = pure implicit FOU (1st-order upwind, more dissipative but "
           "unconditionally stable at sharp shear layers).")
      .def("set_backflow_stabilization", &S::setBackflowStab, nb::arg("beta"),
           "Outflow backflow-stabilization coefficient (Bazilevs 2009 / Esmaily-Moghadam 2011): beta "
           "in [0,1] scales the dissipative outflow term that prevents backflow divergence when flow "
           "reverses at the outlet (e.g. a separated wake / BFS recirculation). Default 0.2; 0 = off. "
           "Inert where the outlet is purely outgoing.")
      .def("set_pressure_solver_params", &S::setPressureIterations, nb::arg("iters"),
           "Set the pressure smoother iteration count.")
      .def(
          "set_pressure_multigrid", [](S& s, bool, int levels) { s.setPressureLevels(levels); },
          nb::arg("on"), nb::arg("levels") = 4,
          "Set the pressure multigrid depth (levels=1 => pure RB-GS, no coarse grid).")
      .def("set_pressure_chebyshev", &S::setPressureChebyshev, nb::arg("on"),
           nb::arg("max_iter") = 120, nb::arg("rtol") = 1e-9,
           "Use the communication-light Chebyshev pressure accelerator (exclusive with PCG).")
      .def("set_pressure_pcg", &S::setPressurePcg, nb::arg("on"), nb::arg("max_iter") = 200,
           nb::arg("rtol") = 1e-8,
           "Use the MG-PCG pressure accelerator (single-GPU default; exclusive with Chebyshev).")
      .def("set_velocity_multigrid", &S::setVelocityMultigrid, nb::arg("on"), nb::arg("levels") = 4,
           nb::arg("vcycles") = 8,
           "Enable velocity (momentum) multigrid for the implicit diffusion solve.")
      .def("last_pressure_iterations", &S::lastPressureIterations,
           "Return the pressure-solver iteration count from the last step().")
      .def("set_domain_bc", &S::setDomainBc, nb::arg("face"), nb::arg("type"), nb::arg("vx") = 0.0,
           nb::arg("vy") = 0.0, nb::arg("vz") = 0.0,
           "Set a per-face domain BC (face 0..5 = -x,+x,-y,+y,-z,+z; type 0 periodic/1 wall/2 "
           "inflow/3 outflow).")
      .def(
          "set_domain_bc_profile",
          [](S& s, int face, nb::ndarray<double, nb::c_contig> prof) {
            if (prof.ndim() != 3 || prof.shape(2) != 3)
              throw std::runtime_error("profile must be (Nb,Nc,3)");
            const int nb_ = (int)prof.shape(0), nc = (int)prof.shape(1);
            s.setDomainBcProfile(
                face, peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(prof)), nb_,
                nc);
          },
          nb::arg("face"), nb::arg("profile"),
          "Prescribe a per-position inlet velocity profile (Nb,Nc,3) over a face (sets it to "
          "inflow).")
      .def(
          "set_pressure_geometry",
          [](S& s, nb::ndarray<double, nb::f_contig> sdf) { s.setPressureGeometry(grid_in(sdf)); },
          nb::arg("sdf"),
          "Set an all-fluid SDF for the cut-cell pressure operator without an immersed solid (the "
          "channel/BFS domain-BC path). For a no-slip immersed BODY in an inflow/outflow domain, call "
          "set_solid(sdf, cutcell_pressure=True) instead -- do NOT also call this (a second geometry "
          "setter overwrites the SDF and wipes the solid).")
      .def(
          "set_solid",
          [](S& s, nb::ndarray<double, nb::f_contig> sdf, bool cutcell_pressure,
             const std::string& /*pressure_coarse*/) {
            s.setSolid(grid_in(sdf), cutcell_pressure);
          },
          nb::arg("sdf"), nb::arg("cutcell_pressure") = false, nb::arg("pressure_coarse") = "const",
          "Set the solid SDF as a Fortran-order (nx,ny,nz) float64 array (negative inside the "
          "solid, positive in fluid). cutcell_pressure=True enables the open-face-weighted cut-cell "
          "pressure operator (proper no-slip); it composes with domain BCs, so this is the single "
          "call for a no-slip immersed body in an inflow/outflow domain.")
      .def(
          "set_state",
          [](S& s, nb::ndarray<double, nb::f_contig> u, nb::ndarray<double, nb::f_contig> v,
             nb::ndarray<double, nb::f_contig> w) {
            s.uploadVelocity(grid_in(u), grid_in(v), grid_in(w));
          },
          nb::arg("u"), nb::arg("v"), nb::arg("w"),
          "Upload an initial velocity field (u,v,w each a Fortran-order (nx,ny,nz) float64 array).")
      .def("step", &S::step,
           "Advance the solver one time step (semi-implicit: diffusion + projection).")
      .def(
          "get_u", [](S& s) { return field_out(s, s.getVelocity(0)); },
          "Return the x-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_v", [](S& s) { return field_out(s, s.getVelocity(1)); },
          "Return the y-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_w", [](S& s) { return field_out(s, s.getVelocity(2)); },
          "Return the z-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_p", [](S& s) { return field_out(s, s.getPressure()); },
          "Return the physical pressure as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_ox", [](S& s) { return field_out(s, s.getOpenness(0)); },
          "TEMP: -x face openness (fluid area fraction) per inner cell, (nx,ny,nz).")
      .def(
          "get_oy", [](S& s) { return field_out(s, s.getOpenness(1)); },
          "TEMP: -y face openness per inner cell, (nx,ny,nz).")
      .def(
          "get_oz", [](S& s) { return field_out(s, s.getOpenness(2)); },
          "TEMP: -z face openness per inner cell, (nx,ny,nz).")
      .def(
          "get_uf", [](S& s) { return field_out(s, s.getFaceVelocity(0)); },
          "Return the divergence-free FACE x-velocity (collocated: projected MAC field; staggered: "
          "== get_u).")
      .def(
          "get_vf", [](S& s) { return field_out(s, s.getFaceVelocity(1)); },
          "Return the divergence-free FACE y-velocity (collocated: projected MAC field; staggered: "
          "== get_v).")
      .def(
          "get_wf", [](S& s) { return field_out(s, s.getFaceVelocity(2)); },
          "Return the divergence-free FACE z-velocity (collocated: projected MAC field; staggered: "
          "== get_w).")
      // --- Named field registry (multiphysics field container) ---------------------------------
      .def(
          "add_field", [](S& s, const std::string& name) { s.addField(name); }, nb::arg("name"),
          "Register a new zero-initialised cell-centred field on the grid (for transported scalars "
          "or material properties). Idempotent.")
      .def(
          "has_field", [](S& s, const std::string& name) { return s.hasField(name); },
          nb::arg("name"), "Whether a field of this name is registered.")
      .def(
          "field_names", [](S& s) { return s.fieldNames(); },
          "Names of all registered fields (velocity u/v/w, p, sdf, plus any added), sorted.")
      .def(
          "get_field", [](S& s, const std::string& name) { return field_out(s, s.getField(name)); },
          nb::arg("name"),
          "Return a registered field's inner region as a Fortran-order (nx,ny,nz) float64 array.")
      .def(
          "set_field",
          [](S& s, const std::string& name, nb::ndarray<double, nb::f_contig> a) {
            s.setField(name, grid_in(a));
          },
          nb::arg("name"), nb::arg("array"),
          "Write a Fortran-order (nx,ny,nz) float64 array into a registered field's inner region "
          "(ghosts refilled on the next exchange_field/step).")
      .def(
          "field_view", [](S& s, const std::string& name) { return field3d_out(s, s.fieldView(name)); },
          nb::arg("name"),
          "Zero-copy view of a registered field's full padded buffer as a Fortran-order "
          "(nx+2g, ny+2g, nz+2g) array (g = ghost_width); host → NumPy, device → DLPack (CuPy).")
      .def(
          "exchange_field", [](S& s, const std::string& name) { s.exchangeField(name); },
          nb::arg("name"),
          "Fill a registered field's ghost cells (cross-rank + periodic under MPI; periodic "
          "single-rank).")
      .def(
          "exchange_field_add", [](S& s, const std::string& name) { s.exchangeFieldAdd(name); },
          nb::arg("name"),
          "Add-reduce halo: fold ghost-layer deposits back onto their owner (cross-rank + periodic). "
          "The particle->grid deposition primitive for MPI CFD-DEM; single-rank non-periodic no-op.")
      // --- Scalar transport (advection-diffusion) ----------------------------------------------
      .def(
          "add_scalar",
          [](S& s, const std::string& name, double diffusivity, int scheme, int iters) {
            s.addScalar(name, diffusivity, scheme, iters);
          },
          nb::arg("name"), nb::arg("diffusivity") = 0.0, nb::arg("scheme") = 1,
          nb::arg("iters") = 50,
          "Register a transported scalar (temperature/concentration/…): constant diffusivity (grid "
          "units), advection scheme 0=FOU/1=Koren TVD/2=SOU, and RB-GS diffusion sweeps. The scalar "
          "is a registered field (get_field/set_field/field_view). Requires geometry "
          "(set_solid/set_pressure_geometry) for the openness-weighted operators.")
      .def(
          "set_scalar_bc",
          [](S& s, const std::string& name, int face, int type, double value) {
            s.setScalarBc(name, face, type, value);
          },
          nb::arg("name"), nb::arg("face"), nb::arg("type"), nb::arg("value") = 0.0,
          "Scalar boundary condition on a domain face (0..5 = -x,+x,-y,+y,-z,+z): type 0 periodic, "
          "1 Neumann zero-flux (adiabatic), 2 Dirichlet value. Single-rank.")
      .def(
          "has_scalar", [](S& s, const std::string& name) { return s.hasScalar(name); },
          nb::arg("name"), "Whether a transported scalar of this name is registered.")
      .def(
          "advance_scalars", [](S& s) { s.advanceScalars(); },
          "Advance all registered scalars one dt with the current velocity (also done by step()).")
      // --- Property closures + Boussinesq body force -------------------------------------------
      .def(
          "set_property_model",
          [](S& s, const std::string& target, const std::string& kind, const std::string& in0,
             const std::vector<double>& params, const std::string& in1) {
            peclet::flow::ClosureKind k;
            if (kind == "linear")
              k = peclet::flow::ClosureKind::LinearMix;
            else if (kind == "boussinesq")
              k = peclet::flow::ClosureKind::BoussinesqForce;
            else if (kind == "arrhenius")
              k = peclet::flow::ClosureKind::ArrheniusMu;
            else
              throw std::runtime_error("set_property_model: unknown kind '" + kind + "'");
            s.setPropertyModel(target, k, in0, in1, params);
          },
          nb::arg("target"), nb::arg("kind"), nb::arg("field"),
          nb::arg("params") = std::vector<double>{}, nb::arg("field2") = std::string{},
          "Register a device closure writing a property/body-force field from input field(s). "
          "target: a registered field (a property 'mu'/'rho'/… or a body-force component "
          "'force_x'/'force_y'/'force_z'). kind: 'linear' (params [p0,p1,p2]: p0+p1*field+p2*field2), "
          "'boussinesq' (params [rho0,g,beta,T0]: rho0*g*beta*(field-T0) buoyancy), 'arrhenius' "
          "(params [mu_ref,B,Tref]: mu_ref*exp(B*(1/field-1/Tref))). Applied at the top of step().")
      .def(
          "set_property_table",
          [](S& s, const std::string& target, const std::string& field,
             const std::vector<double>& x, const std::vector<double>& y) {
            s.setPropertyTable(target, field, x, y);
          },
          nb::arg("target"), nb::arg("field"), nb::arg("x"), nb::arg("y"),
          "Register a tabulated property: target = piecewise-linear interpolation of (x, y) at the "
          "input field value (x ascending, clamped at the ends).")
      .def(
          "update_properties", [](S& s) { s.updateProperties(); },
          "Apply all registered property/force closures now (also done at the top of step()).")
      .def(
          "enable_cell_force", [](S& s) { s.enableCellForce(); },
          "Allocate + register the per-cell body-force fields force_x/force_y/force_z and route them "
          "into the momentum RHS, for an external writer (e.g. CFD-DEM drag feedback) to fill "
          "directly via field_view('force_z'). They persist across steps until overwritten.")
      .def(
          "enable_drag", [](S& s) { s.enableDrag(); },
          "Enable implicit (semi-implicit) linear drag for CFD-DEM: allocate the per-cell 'drag_beta' "
          "field (added to the momentum diagonal so a -beta*(u-u_p) source is treated implicitly -> "
          "unconditionally stable for the stiff beta of a dense bed) plus force_x/y/z (which carry "
          "beta*u_p, the RHS target). Fill 'drag_beta' and 'force_*' via field_view each step.")
      .def(
          "set_property_mode",
          [](S& s, const std::string& mode, bool harmonic) {
            s.setPropertyMode(mode == "variable", harmonic);
          },
          nb::arg("mode") = "variable", nb::arg("harmonic") = false,
          "Enable variable-coefficient momentum (variable viscosity): mode 'variable' binds the 'mu' "
          "field (get/set_field('mu')) into the diffusion operator; 'constant' reverts. harmonic = "
          "harmonic face-viscosity mean (continuous shear stress across a jump) vs arithmetic. A "
          "closure targeting 'mu' enables this automatically. The incremental-rotational pressure "
          "scheme (large-dt / steady-Stokes) stays active — see set_variable_rotational.")
      .def(
          "set_variable_rotational",
          [](S& s, const std::string& mode, double chi) {
            int m = 0;
            if (mode == "min")
              m = 0;
            else if (mode == "full")
              m = 1;
            else if (mode == "off")
              m = 2;
            else
              throw std::runtime_error("set_variable_rotational: mode must be min/full/off");
            s.setVariableRotational(m, chi);
          },
          nb::arg("mode") = "min", nb::arg("chi") = 1.0,
          "Rotational-pressure term under variable viscosity (the constant-mu Timmermans term "
          "-mu*div(u*) is only valid for homogeneous viscosity — Deteix & Yakoubi 2018). 'min' "
          "(default): constant coefficient chi*mu_min — provably stable at any contrast, exact "
          "fallback to the constant-mu scheme for uniform mu. 'full': pointwise chi*mu(i) — better "
          "pressure consistency at MILD contrast only. 'off': plain incremental (no rotational "
          "term). All modes keep the incremental predictor (large-dt / steady-Stokes capability).")
      .def(
          "set_density_mode",
          [](S& s, const std::string& mode) { s.setDensityMode(mode == "variable"); },
          nb::arg("mode") = "variable",
          "Enable variable density (staggered solver only): binds the 'rho' field "
          "(get/set_field('rho'), created seeded with set_rho's value if absent) into the momentum "
          "time term, the advection weight, the per-cell body force (face-interpolated), and the "
          "pressure projection (face coefficient openness*rho0/rho_f with the matching 1/rho_f "
          "velocity correction; rho0 = set_rho's value, so a uniform field reduces exactly to the "
          "constant solver). A closure targeting 'rho' (e.g. a linear mixture of a transported "
          "phase fraction) enables this automatically. For gravity, register a closure "
          "force_z = linear(rho, params=[0, -g]).")
      .def(
          "ghost_width", [](S& s) { return s.ghostWidth(); },
          "Ghost-layer width g of the velocity block (field_view returns an (n+2g) buffer).")
      .def("max_open_divergence", &S::maxOpenDivergence,
           "Return the max cut-cell flux divergence (the incompressibility residual; ~0 when "
           "converged).")
      .def(
          "get_resolution", [](S& s) { return std::vector<int>{s.nx(), s.ny(), s.nz()}; },
          "Return the LOCAL grid resolution [nx, ny, nz] (this rank's block under MPI).")
      .def(
          "global_resolution",
          [](S& s) {
            auto g = s.globalResolution();
            return std::vector<int>{g[0], g[1], g[2]};
          },
          "Return the GLOBAL grid resolution [gnx, gny, gnz] (== local single-rank). For the "
          "CFD-DEM co-decomposition weight field.")
      .def(
          "block_origin",
          [](S& s) {
            auto o = s.blockOrigin();
            return std::vector<int>{o[0], o[1], o[2]};
          },
          "This rank's inner-block origin in GLOBAL cells ([0,0,0] single-rank). Shift the coupling "
          "deposit origin by this so particles in global coordinates land in the local block.")
      .def(
          "get_spacing", [](S&) { return std::vector<double>{1.0, 1.0, 1.0}; },
          "Return the grid spacing [dx, dy, dz] (always unit on this grid).")
#ifdef PECLET_FLOW_MPI
      // Distributed path (built with -DPECLET_FLOW_MPI): construct the Solver with this rank's
      // LOCAL block dims (see the module-level mpi_block()), then init_mpi with the GLOBAL grid
      // dims. step() then does the g=2 velocity-block halo exchange + the distributed cut-cell
      // pressure MG. Bit-exact to single-rank.
      .def(
          "init_mpi",
          [](S& s, int gnx, int gny, int gnz) {
            ensure_mpi_init();
            s.initMpi(gnx, gny, gnz, MPI_COMM_WORLD);
          },
          nb::arg("gnx"), nb::arg("gny"), nb::arg("gnz"),
          "Wire the multi-rank step: pass the GLOBAL grid dims (gnx,gny,gnz). The Solver must have "
          "been "
          "constructed with this rank's LOCAL block dims (from mpi_block). MPI_Init is called if "
          "needed.")
      .def(
          "rebalance_by_weights",
          [](S& s, const std::vector<double>& w) { s.rebalanceByWeights(w); }, nb::arg("weights"),
          "Dynamic load balancing: redistribute the solver's state onto the weighted ORB of per-cell "
          "weights (global x-fastest, gnx*gny*gnz). Pass fluid work + gamma*particle_count and the "
          "coupled dem migrates onto the SAME partition from the same array. State-preserving "
          "(bit-exact at np=1, reduction floor at np>1).")
      .def(
          "rank",
          [](S&) {
            ensure_mpi_init();
            int r = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &r);
            return r;
          },
          "This rank's index in MPI_COMM_WORLD.")
      .def(
          "size",
          [](S&) {
            ensure_mpi_init();
            int n = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &n);
            return n;
          },
          "The number of ranks in MPI_COMM_WORLD.")
#else
      .def(
          "rank", [](S&) { return 0; },
          "MPI rank (always 0 in the single-rank Python module; the multi-rank path is the "
          "tests/kokkos_mpi suite).")
      .def(
          "size", [](S&) { return 1; }, "MPI size (1 in the single-rank Python module).")
#endif
      .def(
          "bcast_from_root", [](S&, nb::object v) { return v; }, nb::arg("value"),
          "Broadcast a value from rank 0 (identity in the single-rank module; mirrors the MPI "
          "API).");
}

NB_MODULE(_flow, m) {
  m.attr("__doc__") =
      "flow — Kokkos cut-cell IBM incompressible Navier-Stokes solver for porous media.\n\n"
      "Two solver classes share an identical API (only the velocity-unknown placement differs):\n"
      "  Solver          — staggered MAC grid (THE flow solver; permeability/drag accuracy "
      "default)\n"
      "  SolverColocated — collocated / cell-centered velocities (ABC approximate projection)\n\n"
      "Conventions: physical units throughout (density rho, viscosity mu, physical pressure p); "
      "SDFs\n"
      "are negative inside the solid; fields are Fortran-order (nx,ny,nz) float64 (x-fastest). "
      "This is\n"
      "the single-rank module — the multi-rank MPI path is exercised by the tests/kokkos_mpi "
      "suite.\n\n"
      "Kokkos is initialized at import and finalized via a Python atexit hook. Release every "
      "Solver "
      "before interpreter exit (it goes out of scope, or `del s; gc.collect()`) so no Kokkos View "
      "outlives finalize.";
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
  // Register Kokkos::finalize via Python atexit. This is REQUIRED on CUDA: without it, Kokkos's
  // internal device state is torn down by static destructors AFTER the CUDA runtime unloads,
  // aborting with cudaErrorCudartUnloading at every exit. atexit runs the hook while the driver is
  // still up. (Returned fields are backed by host std::vectors, not device Views, so they never
  // block finalize; a live Solver still holding Views at exit must be released first — hence the
  // docstring note.)
  nb::module_::import_("atexit").attr("register")(nb::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
  }));
  // The active Kokkos backend ("OpenMP", "Cuda", "HIP"), chosen by the build's install prefix.
  m.attr("execution_space") = nb::str(Kokkos::DefaultExecutionSpace::name());

  // Staggered MAC grid (THE flow solver) + the collocated/cell-centered variant. Same Python API.
  bind_solver<peclet::flow::Staggered>(m, "Solver");
  bind_solver<peclet::flow::Colocated>(m, "SolverColocated");

#ifdef PECLET_FLOW_MPI
  // Module-level: this rank's ORB block of the global (gnx,gny,gnz) grid, matching the
  // deterministic BlockDecomposer the Solver's initMpi re-derives internally (and the C++
  // tests/kokkos_mpi template). Returns (origin=[ox,oy,oz], size=[lnx,lny,lnz]); slice the global
  // SDF with these to build the local block, then Solver(*size) + init_mpi(gnx,gny,gnz). MPI_Init
  // is called if needed.
  m.def(
      "mpi_block",
      [](int gnx, int gny, int gnz) {
        ensure_mpi_init();
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size),
                                                     peclet::core::IVec<3>{gnx, gny, gnz});
        auto blk = dec.block(static_cast<std::size_t>(rank));
        std::vector<int> origin{(int)blk.origin[0], (int)blk.origin[1], (int)blk.origin[2]};
        std::vector<int> bsize{(int)blk.size[0], (int)blk.size[1], (int)blk.size[2]};
        return std::make_pair(origin, bsize);
      },
      nb::arg("gnx"), nb::arg("gny"), nb::arg("gnz"),
      "Return this MPI rank's ORB block of the global (gnx,gny,gnz) grid as (origin, size), each a "
      "length-3 list [x,y,z]. Use it to slice the global SDF into this rank's local block for a "
      "distributed Solver (see Solver.init_mpi). MPI_Init is called if needed.");

  m.attr("has_mpi") = true;
#else
  m.attr("has_mpi") = false;
#endif
}
