/// @file
/// @brief pybind11 module `sdflow` — the Kokkos cut-cell IBM Navier-Stokes solver (`sdflow.Solver`).
///
/// Exposes sdflow::SdflowIbm to Python: set rho/mu/dt, a body force, an SDF solid (cut-cell IBM no-slip
/// + optional cut-cell pressure projection), step, read back the velocity/pressure, and query the
/// cut-cell flux divergence. Exercised by verify_poiseuille_sdflow (IBM channel) and
/// verify_periodic_spheres_sdflow (cut-cell Stokes through a sphere packing). Kokkos is initialized at
/// import and finalized via Python atexit (the solver holds Kokkos Views, so callers must release the
/// Solver before exit -- del + gc.collect()). rank()/bcast_from_root() are single-rank stubs (the
/// multi-rank path lives in tests/kokkos_mpi).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Kokkos_Core.hpp>
#include <cstring>

#include "sdflow_ibm.hpp"

namespace py = pybind11;
using sdflow::SdflowIbm;

// flat x-fastest vector -> (nx,ny,nz) Fortran-strided numpy array for [x,y,z] indexing
static py::array_t<double> to_xyz(const std::vector<double>& v, int nx, int ny, int nz) {
  py::array_t<double> out(std::vector<py::ssize_t>{nx, ny, nz},
                          std::vector<py::ssize_t>{(py::ssize_t)sizeof(double),
                                                   (py::ssize_t)(sizeof(double) * nx),
                                                   (py::ssize_t)(sizeof(double) * nx * ny)});
  std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
  return out;
}

PYBIND11_MODULE(sdflow, m) {
  m.doc() = "sdflow (Kokkos) cut-cell IBM incompressible Navier-Stokes solver";
  if (!Kokkos::is_initialized()) Kokkos::initialize();
  py::module_::import("atexit").attr("register")(py::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
  }));
  m.attr("execution_space") = py::str(Kokkos::DefaultExecutionSpace::name());

  py::class_<SdflowIbm>(m, "Solver")
      .def(py::init<int, int, int>(), py::arg("nx"), py::arg("ny"), py::arg("nz"))
      .def("set_rho", &SdflowIbm::setRho, "Set fluid density rho (physical units). Set before geometry/first step.")
      .def("set_mu", &SdflowIbm::setMu, "Set dynamic viscosity mu (physical units).")
      .def("set_dt", &SdflowIbm::setDt, "Set the time step dt; the momentum solve is scaled by 1/dt (well-conditioned at large dt).")
      .def("set_body_force", &SdflowIbm::setBodyForce, "Set the body force per unit volume (fx, fy, fz).")
      .def("set_advection", &SdflowIbm::setAdvection, "Enable/disable explicit Koren-TVD momentum advection.")
      .def("set_incremental_pressure", &SdflowIbm::setIncrementalPressure, py::arg("on"), "Toggle the rotational incremental-pressure projection.")
      .def("set_pressure_warmstart", &SdflowIbm::setPressureWarmstart, py::arg("on"), "Seed each pressure solve from the previous step's phi (default off).")
      .def("set_velocity_streams", &SdflowIbm::setVelocityStreams, py::arg("on"), "Toggle overlapped per-component velocity solves.")
      .def("set_implicit_advection", &SdflowIbm::setImplicitAdvection, py::arg("on"), "Use implicit-FOU advection with deferred-correction TVD.")
      .def("set_outer_iterations", &SdflowIbm::setOuterIterations, py::arg("n"), "Set the number of Picard/outer iterations per step.")
      .def("set_outer_tolerance", &SdflowIbm::setOuterTolerance, py::arg("tol"), "Set the outer (Picard) convergence tolerance.")
      .def("last_outer_iterations", &SdflowIbm::lastOuterIterations, "Return the outer-iteration count from the last step().")
      .def("set_velocity_solver_params", &SdflowIbm::setVelocityIterations, py::arg("iters"), "Set the velocity (diffusion) smoother iteration count.")
      .def("set_pressure_solver_params", &SdflowIbm::setPressureIterations, py::arg("iters"), "Set the pressure smoother iteration count.")
      .def("set_pressure_multigrid",
           [](SdflowIbm& s, bool, int levels) { s.setPressureLevels(levels); },
           py::arg("on"), py::arg("levels") = 4,
           "Set the pressure multigrid depth (levels=1 => pure RB-GS, no coarse grid).")
      .def("set_pressure_chebyshev", &SdflowIbm::setPressureChebyshev,
           py::arg("on"), py::arg("max_iter") = 120, py::arg("rtol") = 1e-9,
           "Use the communication-light Chebyshev pressure accelerator (exclusive with PCG).")
      .def("set_pressure_pcg", &SdflowIbm::setPressurePcg,
           py::arg("on"), py::arg("max_iter") = 200, py::arg("rtol") = 1e-8,
           "Use the MG-PCG pressure accelerator (single-GPU default; exclusive with Chebyshev).")
      .def("set_velocity_multigrid", &SdflowIbm::setVelocityMultigrid,
           py::arg("on"), py::arg("levels") = 4, py::arg("vcycles") = 8,
           "Enable velocity (momentum) multigrid for the implicit diffusion solve.")
      .def("last_pressure_iterations", &SdflowIbm::lastPressureIterations, "Return the pressure-solver iteration count from the last step().")
      .def("set_domain_bc", &SdflowIbm::setDomainBc,
           py::arg("face"), py::arg("type"), py::arg("vx") = 0.0, py::arg("vy") = 0.0, py::arg("vz") = 0.0,
           "Set a per-face domain BC (face 0..5 = -x,+x,-y,+y,-z,+z; type 0 periodic/1 wall/2 inflow/3 outflow).")
      .def("set_domain_bc_profile",
           [](SdflowIbm& s, int face, py::array_t<double, py::array::c_style | py::array::forcecast> prof) {
             auto b = prof.request();
             if (b.ndim != 3 || b.shape[2] != 3) throw std::runtime_error("profile must be (Nb,Nc,3)");
             const int nb = (int)b.shape[0], nc = (int)b.shape[1];
             std::vector<double> v((size_t)nb * nc * 3);
             std::memcpy(v.data(), b.ptr, v.size() * sizeof(double));
             s.setDomainBcProfile(face, v, nb, nc);
           },
           "Prescribe a per-position inlet velocity profile (Nb,Nc,3) over a face (sets it to inflow).")
      .def("set_pressure_geometry",
           [](SdflowIbm& s, py::array_t<double, py::array::f_style | py::array::forcecast> sdf) {
             std::vector<double> v(static_cast<size_t>(sdf.size()));
             std::memcpy(v.data(), sdf.data(), v.size() * sizeof(double));
             s.setPressureGeometry(v);
           },
           "Set an all-fluid SDF for the cut-cell pressure operator without an immersed solid.")
      .def("set_solid",
           [](SdflowIbm& s, py::array_t<double, py::array::f_style | py::array::forcecast> sdf,
              bool cutcell_pressure, const std::string& /*pressure_coarse*/) {
             std::vector<double> v(static_cast<size_t>(sdf.size()));
             std::memcpy(v.data(), sdf.data(), v.size() * sizeof(double));
             s.setSolid(v, cutcell_pressure);
           },
           py::arg("sdf"), py::arg("cutcell_pressure") = false, py::arg("pressure_coarse") = "const",
           "Set the solid SDF [x,y,z] (negative inside); optionally enable the cut-cell pressure operator.")
      .def("set_state",
           [](SdflowIbm& s,
              py::array_t<double, py::array::f_style | py::array::forcecast> u,
              py::array_t<double, py::array::f_style | py::array::forcecast> v,
              py::array_t<double, py::array::f_style | py::array::forcecast> w) {
             auto vec = [](const auto& a) {
               std::vector<double> o(static_cast<size_t>(a.size()));
               std::memcpy(o.data(), a.data(), o.size() * sizeof(double));
               return o;
             };
             s.uploadVelocity(vec(u), vec(v), vec(w));
           },
           py::arg("u"), py::arg("v"), py::arg("w"),
           "Upload an initial velocity field (u,v,w as Fortran-order [x,y,z] arrays).")
      .def("step", &SdflowIbm::step, "Advance the solver one time step.")
      .def("get_u", [](SdflowIbm& s) { return to_xyz(s.getVelocity(0), s.nx(), s.ny(), s.nz()); }, "Return the x-velocity component as a 3-D [x,y,z] numpy array.")
      .def("get_v", [](SdflowIbm& s) { return to_xyz(s.getVelocity(1), s.nx(), s.ny(), s.nz()); }, "Return the y-velocity component as a 3-D [x,y,z] numpy array.")
      .def("get_w", [](SdflowIbm& s) { return to_xyz(s.getVelocity(2), s.nx(), s.ny(), s.nz()); }, "Return the z-velocity component as a 3-D [x,y,z] numpy array.")
      .def("get_p", [](SdflowIbm& s) { return to_xyz(s.getPressure(), s.nx(), s.ny(), s.nz()); }, "Return the physical pressure as a 3-D [x,y,z] numpy array.")
      .def("max_open_divergence", &SdflowIbm::maxOpenDivergence, "Return the max cut-cell flux divergence (the incompressibility residual).")
      .def("get_resolution", [](SdflowIbm& s) { return std::vector<int>{s.nx(), s.ny(), s.nz()}; }, "Return the grid resolution [nx, ny, nz].")
      .def("get_spacing", [](SdflowIbm&) { return std::vector<double>{1.0, 1.0, 1.0}; })  // unit grid
      .def("rank", [](SdflowIbm&) { return 0; })            // single-rank Python module (MPI = kokkos_mpi tests, "MPI rank (0 in the single-rank Python module).")
      .def("size", [](SdflowIbm&) { return 1; }, "MPI size (1 in the single-rank Python module).")
      .def("bcast_from_root", [](SdflowIbm&, py::object v) { return v; });
}
