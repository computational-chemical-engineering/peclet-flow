#include "cfd_solver.cuh"
#include "pore_extraction.cuh"
#include "sdf_reader.h"
#include <array>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(pnm_backend, m) {
  m.doc() = "PNM Extraction Backend reading VTI files";

  // -----------------------------------------------------------------------
  // SDFReader
  // -----------------------------------------------------------------------
  py::class_<SDFReader>(m, "SDFReader")
      .def_static(
          "read_vti",
          [](const std::string &filename) {
            auto *data = new SDFData(SDFReader::read_vti(filename));

            // Map resolution {nx, ny, nz} -> shape {nz, ny, nx}
            std::vector<ssize_t> py_shape = {
                static_cast<ssize_t>(data->resolution[2]),
                static_cast<ssize_t>(data->resolution[1]),
                static_cast<ssize_t>(data->resolution[0])};

            // Map origin/spacing {x, y, z} -> {z, y, x}
            std::vector<double> py_org = {static_cast<double>(data->origin[2]),
                                          static_cast<double>(data->origin[1]),
                                          static_cast<double>(data->origin[0])};
            std::vector<double> py_spc = {
                static_cast<double>(data->spacing[2]),
                static_cast<double>(data->spacing[1]),
                static_cast<double>(data->spacing[0])};

            py::capsule free_when_done(
                data, [](void *f) { delete reinterpret_cast<SDFData *>(f); });

            auto sdf_3d = py::array_t<float>(py_shape, data->sdf_values.data(),
                                             free_when_done);

            return py::make_tuple(sdf_3d, py_org, py_spc);
          },
          "Reads VTI; returns (sdf_3d[nz,ny,nx], origin_zyx, spacing_zyx)");

  // -----------------------------------------------------------------------
  // Pore Struct
  // -----------------------------------------------------------------------
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

  // -----------------------------------------------------------------------
  // Extract Pores
  // -----------------------------------------------------------------------
  m.def(
      "extract_pores",
      [](py::array_t<float> sdf_array, std::vector<double> origin_zyx,
         std::vector<double> spacing_zyx) {
        py::buffer_info buf = sdf_array.request();
        if (buf.ndim != 3) {
          throw std::runtime_error("SDF array must be 3D (Nz, Ny, Nx)");
        }

        // Reconstruct SDFData using std::array
        SDFData temp_sdf;

        // Shape (Nz, Ny, Nx) -> Resolution (Nx, Ny, Nz)
        temp_sdf.resolution = {static_cast<int>(buf.shape[2]),
                               static_cast<int>(buf.shape[1]),
                               static_cast<int>(buf.shape[0])};

        // ZYX -> XYZ and explicit double->float cast
        temp_sdf.origin = {static_cast<float>(origin_zyx[2]),
                           static_cast<float>(origin_zyx[1]),
                           static_cast<float>(origin_zyx[0])};

        temp_sdf.spacing = {static_cast<float>(spacing_zyx[2]),
                            static_cast<float>(spacing_zyx[1]),
                            static_cast<float>(spacing_zyx[0])};

        float *ptr = static_cast<float *>(buf.ptr);
        temp_sdf.sdf_values.assign(
            ptr, ptr + (buf.shape[0] * buf.shape[1] * buf.shape[2]));

        return extract_pores_gpu(temp_sdf);
      },
      "Extract pore centers. Inputs: (sdf_3d, origin_zyx, spacing_zyx)");

  // -----------------------------------------------------------------------
  // Segment Volume
  // -----------------------------------------------------------------------
  m.def(
      "segment_volume",
      [](py::array_t<float> sdf_array, std::vector<double> spacing_zyx) {
        py::buffer_info buf = sdf_array.request();
        if (buf.ndim != 3)
          throw std::runtime_error("SDF array must be 3D");

        SDFData temp_sdf;
        // Construct std::array directly
        temp_sdf.resolution = {static_cast<int>(buf.shape[2]),
                               static_cast<int>(buf.shape[1]),
                               static_cast<int>(buf.shape[0])};

        temp_sdf.spacing = {static_cast<float>(spacing_zyx[2]),
                            static_cast<float>(spacing_zyx[1]),
                            static_cast<float>(spacing_zyx[0])};

        float *ptr = static_cast<float *>(buf.ptr);
        temp_sdf.sdf_values.assign(
            ptr, ptr + (buf.shape[0] * buf.shape[1] * buf.shape[2]));

        return segment_volume_gpu(temp_sdf);
      },
      "Partition volume. Inputs: (sdf_3d, spacing_zyx)");

  // -----------------------------------------------------------------------
  // Extract Topology
  // -----------------------------------------------------------------------
  m.def(
      "extract_topology_gpu",
      [](std::vector<int> segmentation, std::vector<int> shape_zyx) {
        // Fix: Use std::array<int, 3> because the C++ signature requires it
        std::array<int, 3> res_xyz = {shape_zyx[2], shape_zyx[1], shape_zyx[0]};
        return extract_topology_gpu(segmentation, res_xyz);
      },
      py::arg("segmentation"), py::arg("shape"));

  // -----------------------------------------------------------------------
  // CFD Solver
  // -----------------------------------------------------------------------
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
      .def(py::init(
               [](std::vector<int> shape_zyx, std::vector<double> spacing_zyx) {
                 int3 res = {shape_zyx[2], shape_zyx[1], shape_zyx[0]};
                 float3 spc = {static_cast<float>(spacing_zyx[2]),
                               static_cast<float>(spacing_zyx[1]),
                               static_cast<float>(spacing_zyx[0])};
                 return new CFDSolver(res, spc);
               }),
           py::arg("shape"), py::arg("spacing"))

      .def(
          "initialize",
          [](CFDSolver &self, py::array_t<float> sdf_array,
             std::vector<double> origin_zyx, std::vector<double> spacing_zyx) {
            py::buffer_info buf = sdf_array.request();

            SDFData temp_sdf;
            // Explicit std::array construction
            temp_sdf.resolution = {static_cast<int>(buf.shape[2]),
                                   static_cast<int>(buf.shape[1]),
                                   static_cast<int>(buf.shape[0])};

            temp_sdf.origin = {static_cast<float>(origin_zyx[2]),
                               static_cast<float>(origin_zyx[1]),
                               static_cast<float>(origin_zyx[0])};

            temp_sdf.spacing = {static_cast<float>(spacing_zyx[2]),
                                static_cast<float>(spacing_zyx[1]),
                                static_cast<float>(spacing_zyx[0])};

            float *ptr = static_cast<float *>(buf.ptr);
            temp_sdf.sdf_values.assign(
                ptr, ptr + (buf.shape[0] * buf.shape[1] * buf.shape[2]));

            self.initialize(temp_sdf);
          },
          py::arg("sdf_array"), py::arg("origin"), py::arg("spacing"))

      // --- Getters returning 3D arrays ---
      .def("get_u",
           [](CFDSolver &self) {
             auto data = self.get_u();
             int3 res = self.get_resolution(); // Using PUBLIC getter
             std::vector<ssize_t> shape = {(ssize_t)res.z, (ssize_t)res.y,
                                           (ssize_t)res.x};
             return py::array_t<double>(shape, data.data());
           })
      .def("get_v",
           [](CFDSolver &self) {
             auto data = self.get_v();
             int3 res = self.get_resolution();
             std::vector<ssize_t> shape = {(ssize_t)res.z, (ssize_t)res.y,
                                           (ssize_t)res.x};
             return py::array_t<double>(shape, data.data());
           })
      .def("get_w",
           [](CFDSolver &self) {
             auto data = self.get_w();
             int3 res = self.get_resolution();
             std::vector<ssize_t> shape = {(ssize_t)res.z, (ssize_t)res.y,
                                           (ssize_t)res.x};
             return py::array_t<double>(shape, data.data());
           })
      .def("get_p",
           [](CFDSolver &self) {
             auto data = self.get_p();
             int3 res = self.get_resolution();
             std::vector<ssize_t> shape = {(ssize_t)res.z, (ssize_t)res.y,
                                           (ssize_t)res.x};
             return py::array_t<double>(shape, data.data());
           })

      .def("set_body_force", &CFDSolver::set_body_force, py::arg("force"))
      .def("set_theta_", &CFDSolver::set_theta_, py::arg("theta"))
      .def("set_rho", &CFDSolver::set_rho, py::arg("rho"))
      .def("set_mu", &CFDSolver::set_mu, py::arg("mu"))
      .def("set_ibm_scheme", &CFDSolver::set_ibm_scheme, py::arg("scheme"))
      .def("set_boundary_velocity", &CFDSolver::set_boundary_velocity,
           py::arg("u_bc"))
      .def("set_pressure_solver_params", &CFDSolver::set_pressure_solver_params,
           py::arg("iter"))
      .def("set_velocity_solver_params", &CFDSolver::set_velocity_solver_params,
           py::arg("iter"))
      .def("set_outer_iterations", &CFDSolver::set_outer_iterations,
           py::arg("iterations"))
      .def("set_outer_tolerance", &CFDSolver::set_outer_tolerance,
           py::arg("tol"))
      .def("step", &CFDSolver::step, py::arg("dt"))
      .def("get_fluid_fraction", &CFDSolver::get_fluid_fraction,
           py::arg("type"), py::arg("offset"))
      .def("set_u", &CFDSolver::set_u, py::arg("u"))
      .def("set_v", &CFDSolver::set_v, py::arg("v"))
      .def("set_w", &CFDSolver::set_w, py::arg("w"))
      .def("get_ibm_scaling", &CFDSolver::get_ibm_scaling,
           py::arg("component_idx"));
}