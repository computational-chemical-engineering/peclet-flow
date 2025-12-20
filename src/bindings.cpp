#include "pore_extraction.cuh"
#include "sdf_reader.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pnm_backend, m) {
  m.doc() = "PNM Extraction Backend reading VTI files";

  py::class_<SDFData>(m, "SDFData")
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

  m.def("extract_topology", &extract_topology_gpu,
        "Extract adjacent pairs (ID_A, ID_B) from segmentation");
}
