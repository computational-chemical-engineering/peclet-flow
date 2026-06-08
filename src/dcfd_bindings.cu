// Python bindings for the distributed Navier-Stokes solver (DistributedStokes). Exposes a single-GPU-
// friendly API: run as plain `python script.py` (one rank, whole grid on one GPU) or under
// `mpirun -np N python script.py` (multi-rank). The module auto-initialises MPI. Global fields (SDF,
// velocity) are passed as flat x-fastest numpy arrays of size nx*ny*nz; the wrapper scatters them to
// each rank's extended block (periodic wrap) and gathers results back to the root.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mpi.h>

#include <cstring>
#include <vector>

#include "distributed_stokes.cuh"

namespace py = pybind11;
using dstokes::DistributedStokes;

// Wrap an x-fastest host buffer (flat[x + y*nx + z*nx*ny]) as a 3-D numpy array indexed u[x,y,z], so
// callers never touch reshape/order. The x-fastest buffer is exactly the F-contiguous layout of shape
// (nx,ny,nz): strides (8, 8*nx, 8*nx*ny). Empty input -> empty array (non-root ranks).
static py::array_t<double> to_numpy3d(const std::vector<double>& v, int3 res) {
  if (v.empty()) return py::array_t<double>(std::vector<py::ssize_t>{0});
  std::vector<py::ssize_t> shape{res.x, res.y, res.z};
  std::vector<py::ssize_t> strides{(py::ssize_t)sizeof(double),
                                   (py::ssize_t)(sizeof(double) * res.x),
                                   (py::ssize_t)(sizeof(double) * (size_t)res.x * res.y)};
  py::array_t<double> a(shape, strides);
  std::memcpy(a.request().ptr, v.data(), v.size() * sizeof(double));
  return a;
}

static void ensure_mpi() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
  }
}

class DCfdSolver {
 public:
  DCfdSolver(int nx, int ny, int nz, double nu, double dt) {
    ensure_mpi();
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    res_ = make_int3(nx, ny, nz);
    s_.init(res_, rank, size, nu, dt, MPI_COMM_WORLD);
  }

  int rank() const {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
  }
  int size() const {
    int s = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
  }

  void set_body_force(double fx, double fy, double fz) { s_.set_body_force(fx, fy, fz); }
  void set_advection(bool on) { s_.set_advection(on); }
  void set_implicit_advection(bool on) { s_.set_implicit_advection(on); }
  void set_outer_iterations(int n) { s_.set_outer_iterations(n); }
  void set_outer_tolerance(double t) { s_.set_outer_tolerance(t); }
  void set_pressure_multigrid(bool on, int levels) { s_.set_pressure_multigrid(on, levels); }
  void set_pressure_pcg(bool on, int max_iter, double rtol) { s_.set_pressure_pcg(on, max_iter, rtol); }
  void set_velocity_multigrid(bool on, int levels, int v_cycles) {
    s_.set_velocity_multigrid(on, levels, v_cycles);
  }
  using Arr = py::array_t<double, py::array::forcecast>;  // double, any memory order (strides respected)
  void set_cutcell_pressure_operator(Arr sdf, bool galerkin) {
    s_.set_cutcell_pressure_operator(to_block(sdf), galerkin);
  }
  void set_ibm_solid(Arr sdf, double ubx, double uby, double ubz) {
    s_.set_ibm_solid(to_block(sdf), make_float3((float)ubx, (float)uby, (float)ubz));
  }
  void set_velocity(Arr u, Arr v, Arr w) {
    auto bu = to_block(u), bv = to_block(v), bw = to_block(w);
    s_.upload_velocity(bu.data(), bv.data(), bw.data());
  }

  void step(int n_diff, int n_pois) { s_.step(n_diff, n_pois); }
  int last_outer_iterations() const { return s_.last_outer_iterations(); }
  double max_open_divergence() { return s_.max_open_divergence(); }

  // gather a velocity component to the root rank as a 3-D numpy array u[x,y,z] (empty on other ranks).
  // The array is F-contiguous, so x is fastest in memory (the VTK/ParaView convention, as in PyVista).
  py::array_t<double> get_u() { return gather(s_.u()); }
  py::array_t<double> get_v() { return gather(s_.v()); }
  py::array_t<double> get_w() { return gather(s_.w()); }

  // x-fastest flat buffer (idx = x + y*nx + z*nx*ny), ready to write into a VTI / vtkImageData point
  // array with no reshape or order= (equivalent to get_u().ravel(order='F'), without the footgun).
  py::array_t<double> get_u_flat() { return flat(s_.u()); }
  py::array_t<double> get_v_flat() { return flat(s_.v()); }
  py::array_t<double> get_w_flat() { return flat(s_.w()); }

 private:
  // a 3-D global field g[x,y,z] (shape (nx,ny,nz), any memory order) -> this rank's extended block via
  // periodic wrap. Reads through numpy strides, so the caller never has to flatten or pick an order.
  std::vector<double> to_block(const Arr& g) {
    if (g.ndim() != 3)
      throw std::runtime_error("dcfd: field must be a 3-D (nx,ny,nz) array indexed [x,y,z]");
    auto r = g.unchecked<3>();
    if (r.shape(0) != res_.x || r.shape(1) != res_.y || r.shape(2) != res_.z)
      throw std::runtime_error("dcfd: field shape must be (nx,ny,nz)");
    int3 e = s_.ext(), og = s_.origin_incl_ghost();
    std::vector<double> out(s_.num_cells());
    auto wrap = [](int v, int m) { return ((v % m) + m) % m; };
    for (int lz = 0; lz < e.z; ++lz)
      for (int ly = 0; ly < e.y; ++ly)
        for (int lx = 0; lx < e.x; ++lx) {
          int gx = wrap(og.x + lx, res_.x), gy = wrap(og.y + ly, res_.y), gz = wrap(og.z + lz, res_.z);
          out[(std::size_t)lx + (std::size_t)ly * e.x + (std::size_t)lz * e.x * e.y] = r(gx, gy, gz);
        }
    return out;
  }

  py::array_t<double> gather(double* comp) {
    return to_numpy3d(s_.gather_to_root(comp), res_);  // u[x,y,z] on root, empty elsewhere
  }
  py::array_t<double> flat(double* comp) {  // x-fastest 1-D buffer on root, empty elsewhere
    std::vector<double> g = s_.gather_to_root(comp);
    py::array_t<double> a(std::vector<py::ssize_t>{(py::ssize_t)g.size()});
    if (!g.empty()) std::memcpy(a.request().ptr, g.data(), g.size() * sizeof(double));
    return a;
  }

  DistributedStokes s_;
  int3 res_{};
};

PYBIND11_MODULE(dcfd, m) {
  m.doc() = "Distributed GPU incompressible Navier-Stokes solver (DistributedStokes) Python API";
  py::class_<DCfdSolver>(m, "Solver")
      .def(py::init<int, int, int, double, double>(), py::arg("nx"), py::arg("ny"), py::arg("nz"),
           py::arg("nu"), py::arg("dt"))
      .def("rank", &DCfdSolver::rank)
      .def("size", &DCfdSolver::size)
      .def("set_body_force", &DCfdSolver::set_body_force, py::arg("fx"), py::arg("fy"), py::arg("fz"))
      .def("set_advection", &DCfdSolver::set_advection, py::arg("on"))
      .def("set_implicit_advection", &DCfdSolver::set_implicit_advection, py::arg("on"))
      .def("set_outer_iterations", &DCfdSolver::set_outer_iterations, py::arg("n"))
      .def("set_outer_tolerance", &DCfdSolver::set_outer_tolerance, py::arg("tol"))
      .def("set_pressure_multigrid", &DCfdSolver::set_pressure_multigrid, py::arg("on"),
           py::arg("levels") = 4)
      .def("set_pressure_pcg", &DCfdSolver::set_pressure_pcg, py::arg("on"), py::arg("max_iter") = 60,
           py::arg("rtol") = 1e-8)
      .def("set_velocity_multigrid", &DCfdSolver::set_velocity_multigrid, py::arg("on"),
           py::arg("levels") = 3, py::arg("v_cycles") = 4)
      .def("set_cutcell_pressure_operator", &DCfdSolver::set_cutcell_pressure_operator, py::arg("sdf"),
           py::arg("galerkin") = true)
      .def("set_ibm_solid", &DCfdSolver::set_ibm_solid, py::arg("sdf"), py::arg("ubx") = 0.0,
           py::arg("uby") = 0.0, py::arg("ubz") = 0.0)
      .def("set_velocity", &DCfdSolver::set_velocity, py::arg("u"), py::arg("v"), py::arg("w"))
      .def("step", &DCfdSolver::step, py::arg("n_diff"), py::arg("n_pois"))
      .def("last_outer_iterations", &DCfdSolver::last_outer_iterations)
      .def("max_open_divergence", &DCfdSolver::max_open_divergence)
      .def("get_u", &DCfdSolver::get_u)
      .def("get_v", &DCfdSolver::get_v)
      .def("get_w", &DCfdSolver::get_w)
      .def("get_u_flat", &DCfdSolver::get_u_flat)
      .def("get_v_flat", &DCfdSolver::get_v_flat)
      .def("get_w_flat", &DCfdSolver::get_w_flat);
}
