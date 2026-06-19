// cfd-gpu — pybind11 module for the Kokkos IBM Navier-Stokes solver (drop-in for the sdflow Solver API).
//
// A Kokkos+Blackwell drop-in for the sdflow Solver exercised by verify_poiseuille_sdflow (IBM channel)
// and verify_periodic_spheres_sdflow (cut-cell Stokes through a sphere packing): set rho/mu/dt, a body
// force, an SDF solid (cut-cell IBM no-slip + optional cut-cell pressure projection), step, read back the
// velocity/pressure, and query the cut-cell flux divergence. Kokkos is initialized at import and finalized
// via Python atexit (the solver holds Kokkos Views, so callers must release the Solver before exit -- del
// + gc.collect()). rank()/bcast_from_root() are single-rank stubs (MPI is a follow-up).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Kokkos_Core.hpp>
#include <cstring>

#include "sdflow_ibm_kokkos.hpp"

namespace py = pybind11;
using cfdk::SdflowIbm;

// flat x-fastest vector -> (nx,ny,nz) Fortran-strided numpy array for [x,y,z] indexing
static py::array_t<double> to_xyz(const std::vector<double>& v, int nx, int ny, int nz) {
  py::array_t<double> out(std::vector<py::ssize_t>{nx, ny, nz},
                          std::vector<py::ssize_t>{(py::ssize_t)sizeof(double),
                                                   (py::ssize_t)(sizeof(double) * nx),
                                                   (py::ssize_t)(sizeof(double) * nx * ny)});
  std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
  return out;
}

PYBIND11_MODULE(sdflow_kokkos, m) {
  m.doc() = "cfd-gpu sdflow (Kokkos) IBM Navier-Stokes solver";
  if (!Kokkos::is_initialized()) Kokkos::initialize();
  py::module_::import("atexit").attr("register")(py::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
  }));
  m.attr("execution_space") = py::str(Kokkos::DefaultExecutionSpace::name());

  py::class_<SdflowIbm>(m, "Solver")
      .def(py::init<int, int, int>(), py::arg("nx"), py::arg("ny"), py::arg("nz"))
      .def("set_rho", &SdflowIbm::setRho)
      .def("set_mu", &SdflowIbm::setMu)
      .def("set_dt", &SdflowIbm::setDt)
      .def("set_body_force", &SdflowIbm::setBodyForce)
      .def("set_advection", &SdflowIbm::setAdvection)
      .def("set_velocity_solver_params", &SdflowIbm::setVelocityIterations, py::arg("iters"))
      .def("set_pressure_solver_params", &SdflowIbm::setPressureIterations, py::arg("iters"))
      .def("set_pressure_multigrid",
           [](SdflowIbm& s, bool, int levels) { s.setPressureLevels(levels); },
           py::arg("on"), py::arg("levels") = 4)
      .def("last_pressure_iterations", &SdflowIbm::lastPressureIterations)
      .def("set_domain_bc", &SdflowIbm::setDomainBc,
           py::arg("face"), py::arg("type"), py::arg("vx") = 0.0, py::arg("vy") = 0.0, py::arg("vz") = 0.0)
      .def("set_domain_bc_profile",
           [](SdflowIbm& s, int face, py::array_t<double, py::array::c_style | py::array::forcecast> prof) {
             auto b = prof.request();
             if (b.ndim != 3 || b.shape[2] != 3) throw std::runtime_error("profile must be (Nb,Nc,3)");
             const int nb = (int)b.shape[0], nc = (int)b.shape[1];
             std::vector<double> v((size_t)nb * nc * 3);
             std::memcpy(v.data(), b.ptr, v.size() * sizeof(double));
             s.setDomainBcProfile(face, v, nb, nc);
           })
      .def("set_pressure_geometry",
           [](SdflowIbm& s, py::array_t<double, py::array::f_style | py::array::forcecast> sdf) {
             std::vector<double> v(static_cast<size_t>(sdf.size()));
             std::memcpy(v.data(), sdf.data(), v.size() * sizeof(double));
             s.setPressureGeometry(v);
           })
      .def("set_solid",
           [](SdflowIbm& s, py::array_t<double, py::array::f_style | py::array::forcecast> sdf,
              bool cutcell_pressure, const std::string& /*pressure_coarse*/) {
             std::vector<double> v(static_cast<size_t>(sdf.size()));
             std::memcpy(v.data(), sdf.data(), v.size() * sizeof(double));
             s.setSolid(v, cutcell_pressure);
           },
           py::arg("sdf"), py::arg("cutcell_pressure") = false, py::arg("pressure_coarse") = "const")
      .def("step", &SdflowIbm::step)
      .def("get_u", [](SdflowIbm& s) { return to_xyz(s.getVelocity(0), s.nx(), s.ny(), s.nz()); })
      .def("get_v", [](SdflowIbm& s) { return to_xyz(s.getVelocity(1), s.nx(), s.ny(), s.nz()); })
      .def("get_w", [](SdflowIbm& s) { return to_xyz(s.getVelocity(2), s.nx(), s.ny(), s.nz()); })
      .def("get_p", [](SdflowIbm& s) { return to_xyz(s.getPressure(), s.nx(), s.ny(), s.nz()); })
      .def("max_open_divergence", &SdflowIbm::maxOpenDivergence)
      .def("rank", [](SdflowIbm&) { return 0; })
      .def("bcast_from_root", [](SdflowIbm&, py::object v) { return v; });
}
