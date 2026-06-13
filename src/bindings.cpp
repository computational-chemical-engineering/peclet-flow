/// @file
/// @brief pnm_backend Python module: pybind11 bindings for the cut-cell NS solver.
#include "cfd_solver.cuh"
#include "pore_extraction.cuh"
#include "sdf_reader.h"
#include <array>
#include <cstring>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace {

py::array_t<double> make_owned_array(const std::vector<double> &data, int3 res) {
  py::array_t<double> array({static_cast<ssize_t>(res.z), static_cast<ssize_t>(res.y),
                             static_cast<ssize_t>(res.x)});
  std::memcpy(array.mutable_data(), data.data(), data.size() * sizeof(double));
  return array;
}

std::vector<double> copy_field_from_numpy(const py::array_t<double> &field,
                                          int3 res,
                                          const char *name) {
  py::buffer_info buf = field.request();
  const ssize_t expected_size =
      static_cast<ssize_t>(res.x) * res.y * res.z;

  if (buf.size != expected_size) {
    throw std::runtime_error(std::string(name) + ": expected " +
                             std::to_string(expected_size) +
                             " values, got " + std::to_string(buf.size));
  }

  if (buf.ndim != 1 &&
      !(buf.ndim == 3 && buf.shape[0] == res.z && buf.shape[1] == res.y &&
        buf.shape[2] == res.x)) {
    throw std::runtime_error(std::string(name) +
                             ": expected flat array or shape (nz, ny, nx)");
  }

  const double *ptr = static_cast<const double *>(buf.ptr);
  return std::vector<double>(ptr, ptr + expected_size);
}

} // namespace

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
      .def("get_resolution",
           [](const CFDSolver &self) {
             int3 res = self.get_resolution();
             return py::make_tuple(res.z, res.y, res.x);
           })
      .def("get_spacing",
           [](const CFDSolver &self) {
             float3 spc = self.get_spacing();
             return py::make_tuple(spc.z, spc.y, spc.x);
           })
      .def("update_ibm_geometry", &CFDSolver::update_ibm_geometry)

      // --- Getters returning 3D arrays ---
      .def("get_u",
            [](CFDSolver &self) {
              auto data = self.get_u();
              int3 res = self.get_resolution();
              return make_owned_array(data, res);
            })
      .def("get_v",
            [](CFDSolver &self) {
              auto data = self.get_v();
              int3 res = self.get_resolution();
              return make_owned_array(data, res);
            })
      .def("get_w",
            [](CFDSolver &self) {
              auto data = self.get_w();
              int3 res = self.get_resolution();
              return make_owned_array(data, res);
            })
      .def("get_p",
            [](CFDSolver &self) {
              auto data = self.get_p();
              int3 res = self.get_resolution();
              return make_owned_array(data, res);
             })
       .def("get_last_outer_iterations", &CFDSolver::get_last_outer_iterations)
       .def("get_momentum_residual_max", &CFDSolver::get_momentum_residual_max,
            py::arg("fluid_only") = false)
       .def("set_debug_stats", &CFDSolver::set_debug_stats, py::arg("enabled"))
       .def("get_debug_stats", &CFDSolver::get_debug_stats)
      .def("get_debug_fields",
           [](CFDSolver &self) {
             auto fields = self.get_debug_fields();
             int3 res = self.get_resolution();
             py::list out;
             for (const auto &field : fields) {
               py::array_t<float> array(
                   {static_cast<ssize_t>(res.z), static_cast<ssize_t>(res.y),
                    static_cast<ssize_t>(res.x)});
               std::memcpy(array.mutable_data(), field.data(),
                           field.size() * sizeof(float));
               out.append(array);
             }
             return out;
           })
      .def("set_debug_cell", &CFDSolver::set_debug_cell, py::arg("cell"))
      .def("get_debug_cell_info", &CFDSolver::get_debug_cell_info)

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
      .def("set_pressure_multigrid_enabled",
           &CFDSolver::set_pressure_multigrid_enabled, py::arg("enabled"))
      .def("set_pressure_multigrid_params",
           &CFDSolver::set_pressure_multigrid_params, py::arg("max_levels"),
           py::arg("pre_sweeps"), py::arg("post_sweeps"),
           py::arg("bottom_sweeps"), py::arg("v_cycles"))
      .def("set_velocity_multigrid_enabled",
           &CFDSolver::set_velocity_multigrid_enabled, py::arg("enabled"))
      .def("set_velocity_multigrid_params",
           &CFDSolver::set_velocity_multigrid_params, py::arg("max_levels"),
           py::arg("pre_sweeps"), py::arg("post_sweeps"),
           py::arg("bottom_sweeps"), py::arg("v_cycles"))
      .def("set_outer_iterations", &CFDSolver::set_outer_iterations,
           py::arg("iterations"))
       .def("set_outer_tolerance", &CFDSolver::set_outer_tolerance,
            py::arg("tol"))
       .def("set_outer_convergence_mode",
            &CFDSolver::set_outer_convergence_mode, py::arg("mode"))
       .def("step", &CFDSolver::step, py::arg("dt"))
       .def("get_fluid_fraction", &CFDSolver::get_fluid_fraction,
            py::arg("type"), py::arg("offset"))
       .def("set_u",
            [](CFDSolver &self,
               py::array_t<double, py::array::c_style | py::array::forcecast>
                  u_array) {
             self.set_u(copy_field_from_numpy(u_array, self.get_resolution(),
                                              "set_u"));
           },
           py::arg("u"))
      .def("set_v",
           [](CFDSolver &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  v_array) {
             self.set_v(copy_field_from_numpy(v_array, self.get_resolution(),
                                              "set_v"));
           },
           py::arg("v"))
      .def("set_w",
           [](CFDSolver &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  w_array) {
             self.set_w(copy_field_from_numpy(w_array, self.get_resolution(),
                                              "set_w"));
           },
           py::arg("w"))
      .def("set_p",
           [](CFDSolver &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  p_array) {
             self.set_p(copy_field_from_numpy(p_array, self.get_resolution(),
                                              "set_p"));
           },
           py::arg("p"))
      .def("set_state",
           [](CFDSolver &self,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  u_array,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  v_array,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  w_array,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  p_array) {
             const int3 res = self.get_resolution();
             self.set_u(copy_field_from_numpy(u_array, res, "set_state(u)"));
             self.set_v(copy_field_from_numpy(v_array, res, "set_state(v)"));
             self.set_w(copy_field_from_numpy(w_array, res, "set_state(w)"));
             self.set_p(copy_field_from_numpy(p_array, res, "set_state(p)"));
           },
           py::arg("u"), py::arg("v"), py::arg("w"), py::arg("p"))
      // Scale continuation states on the GPU to avoid unnecessary host copies.
      .def("scale_state", &CFDSolver::scale_state, py::arg("velocity_scale"),
           py::arg("pressure_scale"))
      .def("get_ibm_scaling", &CFDSolver::get_ibm_scaling,
           py::arg("component_idx"));
}
