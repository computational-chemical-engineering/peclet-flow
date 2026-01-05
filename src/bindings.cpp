#include "cfd_solver.cuh"
#include "pore_extraction.cuh"
#include "sdf_reader.h"
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pnm_backend, m) {
  m.doc() = "PNM Extraction Backend reading VTI files";

  py::class_<SDFData>(m, "SDFData")
      .def(py::init([](std::vector<float> sdf_values, int3 res, float3 origin,
                       float3 spacing) {
             return SDFData{sdf_values,
                            {res.x, res.y, res.z},
                            {origin.x, origin.y, origin.z},
                            {spacing.x, spacing.y, spacing.z}};
           }),
           py::arg("sdf_values"), py::arg("resolution"), py::arg("origin"),
           py::arg("spacing"))
      .def_readonly("resolution", &SDFData::resolution)
      .def_readonly("origin", &SDFData::origin)
      .def_readonly("spacing", &SDFData::spacing)
      .def_property_readonly("sdf_values", [](SDFData &d) {
        return py::array_t<float>({d.resolution[2], d.resolution[1],
                                   d.resolution[0]}, // Shape (Z, Y, X)
                                  d.sdf_values.data());
      });

  py::class_<SDFReader>(m, "SDFReader")
      .def_static("read_vti", &SDFReader::read_vti);

  py::class_<Pore>(m, "Pore")
      .def_readwrite("x", &Pore::x)
      .def_readwrite("y", &Pore::y)
      .def_readwrite("z", &Pore::z)
      .def_readwrite("radius", &Pore::radius)
      .def("__repr__", [](const Pore &p) {
        return "<Pore (" + std::to_string(p.x) + ", " + std::to_string(p.y) +
               ", " + std::to_string(p.z) + ") r=" + std::to_string(p.radius) +
               ">";
      });

  m.def("extract_pores", &extract_pores_gpu,
        "Extract pore centers from SDF data on GPU");

  m.def("segment_volume", &segment_volume_gpu,
        "Partition volume into Pores (+ID) and Solids (-ID)");

  m.def("extract_topology_gpu", &extract_topology_gpu, "Extract Topology (GPU)",
        py::arg("segmentation"), py::arg("resolution"));

  // --------------------------------------------------------
  // CFD Solver Bindings
  // --------------------------------------------------------
  py::class_<int3>(m, "int3")
      .def(py::init([](int x, int y, int z) { return make_int3(x, y, z); }))
      .def_readwrite("x", &int3::x)
      .def_readwrite("y", &int3::y)
      .def_readwrite("z", &int3::z);

  py::class_<float3>(m, "float3")
      .def(py::init(
          [](float x, float y, float z) { return make_float3(x, y, z); }))
      .def_readwrite("x", &float3::x)
      .def_readwrite("y", &float3::y)
      .def_readwrite("z", &float3::z);

  py::class_<CFDSolver>(m, "CFDSolver")
      .def(py::init<int3, float3>(), py::arg("res"), py::arg("spacing"))
      .def("initialize", &CFDSolver::initialize, py::arg("sdf_data"))
      .def("set_body_force", &CFDSolver::set_body_force, py::arg("force"))
      .def("set_diffusion_theta", &CFDSolver::set_diffusion_theta,
           py::arg("theta"))
      .def("set_cfl", &CFDSolver::set_cfl, py::arg("cfl"))
      .def("get_cfl", &CFDSolver::get_cfl)
      .def("get_dt", &CFDSolver::get_dt)
      .def("set_rho", &CFDSolver::set_rho, py::arg("rho"))
      .def("set_mu", &CFDSolver::set_mu, py::arg("mu"))
      .def("set_pressure_solver_params", &CFDSolver::set_pressure_solver_params,
           py::arg("max_iter"), py::arg("tol"))
      .def("set_velocity_solver_params", &CFDSolver::set_velocity_solver_params,
           py::arg("max_iter"), py::arg("tol"))
      .def("step", &CFDSolver::step, py::arg("dt"))
      .def("get_u", &CFDSolver::get_u)
      .def("get_v", &CFDSolver::get_v)
      .def("get_w", &CFDSolver::get_w)
      .def("get_p", &CFDSolver::get_p)
      .def("get_fluid_fraction", &CFDSolver::get_fluid_fraction,
           py::arg("type"), py::arg("offset"))
      .def("set_u", &CFDSolver::set_u, py::arg("u"))
      .def("set_v", &CFDSolver::set_v, py::arg("v"))
      .def("set_w", &CFDSolver::set_w, py::arg("w"))
      .def("project", &CFDSolver::project, py::arg("dt"),
           py::arg("incremental") = false);
}
