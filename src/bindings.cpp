/// @file
/// @brief pnm_backend Python module: pybind11 bindings for SDF pore-network extraction.
#include "pore_extraction.cuh"
#include "sdf_reader.h"
#include <cuda_runtime.h>
#include <array>
#include <cstring>
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
  // Helper types (int3 / float3)
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
}
