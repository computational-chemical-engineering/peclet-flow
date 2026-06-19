// cfd-gpu — pybind11 module for the Kokkos IBM velocity solver (sdflow_kokkos).
//
// A Kokkos+Blackwell drop-in for the sdflow Solver API exercised by verify_poiseuille_sdflow: set
// rho/mu/dt, a body force, an SDF solid (cut-cell IBM no-slip), step, and read back the velocity.
// Kokkos is initialized at import and finalized via Python atexit (the solver holds Kokkos Views, so
// callers must release the Solver before exit — del + gc.collect()).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Kokkos_Core.hpp>

#include "sdflow_ibm_kokkos.hpp"

namespace py = pybind11;
using cfdk::SdflowIbm;

PYBIND11_MODULE(sdflow_kokkos, m) {
  m.doc() = "cfd-gpu sdflow (Kokkos + ArborX-free) IBM velocity solver";
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
      .def("set_velocity_solver_params", &SdflowIbm::setVelocityIterations, py::arg("iters"))
      .def("set_solid",
           [](SdflowIbm& s, py::array_t<double, py::array::f_style | py::array::forcecast> sdf) {
             // sdf is (nx,ny,nz) Fortran-order == x-fastest flat
             std::vector<double> v(static_cast<size_t>(sdf.size()));
             std::memcpy(v.data(), sdf.data(), v.size() * sizeof(double));
             s.setSolid(v);
           })
      .def("step", &SdflowIbm::step)
      .def("get_u", [](SdflowIbm& s) {
        auto v = s.getVelocity(0);
        py::array_t<double> a({s.nx(), s.ny(), s.nz()});  // C-order shape; we fill x-fastest below
        // return as flat x-fastest reshaped to (nx,ny,nz) Fortran for [x,y,z] indexing
        py::array_t<double> out(std::vector<py::ssize_t>{s.nx(), s.ny(), s.nz()},
                                std::vector<py::ssize_t>{(py::ssize_t)sizeof(double),
                                                         (py::ssize_t)(sizeof(double) * s.nx()),
                                                         (py::ssize_t)(sizeof(double) * s.nx() * s.ny())});
        std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
        (void)a;
        return out;
      })
      .def("get_v", [](SdflowIbm& s) {
        auto v = s.getVelocity(1);
        py::array_t<double> out(std::vector<py::ssize_t>{s.nx(), s.ny(), s.nz()},
                                std::vector<py::ssize_t>{(py::ssize_t)sizeof(double),
                                                         (py::ssize_t)(sizeof(double) * s.nx()),
                                                         (py::ssize_t)(sizeof(double) * s.nx() * s.ny())});
        std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
        return out;
      });
}
