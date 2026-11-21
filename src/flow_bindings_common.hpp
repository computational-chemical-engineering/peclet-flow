/// @file
/// @brief Shared helpers of the peclet.flow Python binding: array conversion, the string-enum
/// tables, the bound Solver / Anderson accelerator classes and the diagnostics view.
///
/// Binding-only: included by src/flow_bindings.cpp and its topic headers
/// (flow_bindings_<topic>.hpp) alone; moved out of flow_bindings.cpp verbatim.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <limits>
#include <string>
#include <vector>

#ifdef PECLET_FLOW_MPI
#include <mpi.h>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#endif

#include "anderson_accelerator.hpp"
#include "flow_bindings_checks.hpp"
#include "flow_ibm.hpp"
#include "peclet/core/python/kokkos_teardown.hpp"
#include "peclet/core/python/ndarray_interop.hpp"

namespace peclet::flow::bindings {

namespace nb = nanobind;
// Input validation and the checked setters bound in place of a member (flow_bindings_checks.hpp).
namespace chk = peclet::flow::binding_checks;

#ifdef PECLET_FLOW_MPI
// Ensure MPI_Init has been called (mirrors the dem init_mpi idiom); safe to call repeatedly.
inline void ensure_mpi_init() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
  }
}
#endif

// A solver field (flat x-fastest, ghost-stripped) -> Fortran-order (nx,ny,nz) float64 NumPy array
// for [x,y,z] indexing. The vector is moved into the array's backing store (no extra copy vs the
// old to_xyz).
template <class S>
inline nb::ndarray<nb::numpy, double> field_out(S& s, std::vector<double>&& v) {
  const auto nx = static_cast<std::size_t>(s.nx());
  const auto ny = static_cast<std::size_t>(s.ny());
  const auto nz = static_cast<std::size_t>(s.nz());
  return peclet::core::python::vector_to_ndarray(
      std::move(v), {nx, ny, nz},
      {1, static_cast<std::int64_t>(nx), static_cast<std::int64_t>(nx * ny)});
}

// A Fortran-order (nx,ny,nz) float64 array -> flat x-fastest host vector (F-contiguous data() is
// already x-fastest). nanobind casts/copies the input to f_contig double if needed.
inline std::vector<double> grid_in(nb::ndarray<double, nb::f_contig> a) {
  return peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(a));
}

// String enums of the public API (suite/docs/NAMING.md; QUALITY_PLAN F): every integer mode code
// became a name, and a name outside the accepted set raises listing that set.
inline int enum_index(const std::string& v, const std::vector<std::string>& names, const char* who,
                      const char* what) {
  for (std::size_t i = 0; i < names.size(); ++i)
    if (v == names[i])
      return static_cast<int>(i);
  std::string all;
  for (std::size_t i = 0; i < names.size(); ++i)
    all += (i ? " | '" : "'") + names[i] + "'";
  throw std::invalid_argument(std::string(who) + ": " + what + " must be one of " + all +
                              ", got '" + v + "'");
}
inline const std::vector<std::string> kFaces{"-x", "+x", "-y", "+y", "-z", "+z"};
inline int face_index(const std::string& f, const char* who) {
  return enum_index(f, kFaces, who, "face");
}
inline const std::vector<std::string> kAxes{"x", "y", "z"};
inline const std::vector<std::string> kBcTypes{"periodic", "wall", "inflow", "outflow", "slip"};
inline const std::vector<std::string> kScalarBcTypes{"periodic", "neumann", "dirichlet"};
inline const std::vector<std::string> kAdvSchemes{"sou", "koren"};
inline const std::vector<std::string> kScalarSchemes{"fou", "koren", "sou"};
inline const std::vector<std::string> kScalarWallTypes{"neumann", "dirichlet", "robin"};
inline const std::vector<std::string> kFluidOnly{"off", "filter", "star"};
inline const std::vector<std::string> kCsfModes{"face", "cell"};
inline const std::vector<std::string> kPivots{"volume", "centroid", "projected-centroid",
                                              "contact-line"};
inline const std::vector<std::string> kAreaModes{"plic",
                                                 "cascade-metric",
                                                 "cascade-normal",
                                                 "cascade-footprint",
                                                 "sheet-color-centroid",
                                                 "sheet-color-split",
                                                 "sheet-plic-centroid",
                                                 "sheet-plic-split"};
inline const std::vector<std::string> kBlockAssign{"round-robin", "lpt", "orb"};

// Zero-copy export of a registered field's padded device buffer as a Fortran-order 3-D array of the
// full block shape (ex,ey,ez) = (nx+2G, ny+2G, nz+2G), x-fastest strides {1,ex,ex*ey}. Includes the
// ghost band (the flat buffer is contiguous; a ghost-stripped view would not be). The shared bridge
// makes the capsule a Releasable holding the managed CCField, so the allocation outlives the array
// (host → NumPy referencing the buffer, device → DLPack for CuPy/torch) and is dropped by the
// module's shutdown before Kokkos::finalize.
template <class S>
inline auto field3d_out(S& s, peclet::flow::CCField f) {
  const auto bs = s.blockShape();
  const std::array<std::size_t, 3> shape{static_cast<std::size_t>(bs[0]),
                                         static_cast<std::size_t>(bs[1]),
                                         static_cast<std::size_t>(bs[2])};
  const std::array<std::int64_t, 3> strides{1, static_cast<std::int64_t>(bs[0]),
                                            static_cast<std::int64_t>(bs[0]) * bs[1]};
  return peclet::core::python::view_to_ndarray(f, shape, strides);
}

// The bound solver: Solver<Grid> plus the suite's teardown registration. Solver<Grid> owns its
// Kokkos Views all over (fields, MG hierarchy, IBM overlays, ...) with no piecemeal reset, so
// release() destroys the bound instance outright (kokkos_teardown.hpp: nanobind marks the Python
// object uninitialized; later calls raise, dealloc skips the destructor). Only reached for a Solver
// still alive when the module shuts down.
template <class Grid>
struct BoundSolver final : peclet::flow::Solver<Grid>, peclet::core::python::Releasable {
  using peclet::flow::Solver<Grid>::Solver;
  void release() noexcept override { peclet::core::python::destruct_bound_instance(this); }
};

// The bound Anderson accelerator (doc/steady_acceleration.md §5.2): it owns Kokkos Views (the
// history), so it is a Releasable like the solver and is destroyed by the module's atexit hook
// before Kokkos::finalize if it is still alive then.
template <class Grid>
struct BoundAccelerator final : peclet::flow::AndersonAccelerator<Grid>,
                                peclet::core::python::Releasable {
  using peclet::flow::AndersonAccelerator<Grid>::AndersonAccelerator;
  void release() noexcept override { peclet::core::python::destruct_bound_instance(this); }
};

/// The reconstructed-traction loads as the (4, n_instances, 3) array that both of its Python names
/// return: `diagnostics.hydro_force_torque_traction()` (canonical) and the deprecated
/// `hydro_force_torque()`. One helper, so the two stay bit-identical.
template <class Grid>
inline auto traction_force_torque_array(peclet::flow::Solver<Grid>& s) {
  std::vector<double> v = s.hydroForceTorque();
  const std::size_t n = v.size() / 12;
  return peclet::core::python::vector_to_ndarray(std::move(v), {4, n, 3},
                                                 {static_cast<std::int64_t>(3 * n), 3, 1});
}

/// The developer tier: a view onto one Solver<Grid> (no state of its own). Bound as
/// `Solver.diagnostics` / `SolverColocated.diagnostics`; the property keeps the solver alive.
template <class Grid>
struct Diagnostics {
  BoundSolver<Grid>* s;
};

}  // namespace peclet::flow::bindings
