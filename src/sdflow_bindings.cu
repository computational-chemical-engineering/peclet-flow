// sdflow -- GPU incompressible Navier-Stokes flow through arbitrary SDF geometry (cut-cell immersed
// boundary on a staggered MAC grid), optionally MPI-distributed. Python API for the canonical solver
// (dns::DistributedNS). Run as plain `python script.py` (one rank, whole grid on one GPU) or under
// `mpirun -np N python script.py` (multi-rank); MPI auto-initialises. Global fields (SDF, velocity) are
// passed as 3-D numpy arrays indexed [x,y,z], shape (nx,ny,nz); the wrapper scatters them to each rank's
// extended block (periodic wrap) and gathers results to the root.
//
// Units are physical: set_rho(rho) + set_mu(mu) (internally nu = mu/rho); body force is a force per unit
// volume (e.g. a pressure gradient) -> internal acceleration f/rho; pressure p = rho/dt * phi. Grid
// spacing is unit (dx=1): work in grid units and scale results afterwards (physical dx is a follow-up).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tpx/common/mpi.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "distributed_ns.cuh"

namespace py = pybind11;
using dns::DistributedNS;

// Wrap an x-fastest host buffer (flat[x + y*nx + z*nx*ny]) as a 3-D numpy array indexed u[x,y,z]. The
// x-fastest buffer is exactly the F-contiguous layout of shape (nx,ny,nz). Empty input -> empty array.
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

// The solver bakes beta = nu*dt into the IBM stencil at set_solid time, so rho/mu/dt must be fixed
// before the geometry is installed. The wrapper therefore stores the physical parameters and lazily
// initialises dns::DistributedNS (which needs nu, dt) on first use.
class Solver {
 public:
  Solver(int nx, int ny, int nz) {
    ensure_mpi();
    res_ = make_int3(nx, ny, nz);
  }

  int rank() const { int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }
  int size() const { int s = 1; MPI_Comm_size(MPI_COMM_WORLD, &s); return s; }
  // Broadcast a flag from the root rank to all ranks (so a driver's convergence/early-stop decision,
  // taken on root from gathered fields, is agreed by every rank -- collectives stay matched at np>1).
  bool bcast_from_root(bool b) {
    int flag = b ? 1 : 0;
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return flag != 0;
  }
  std::vector<int> get_resolution() const { return {res_.x, res_.y, res_.z}; }
  std::vector<double> get_spacing() const { return {dx_, dx_, dx_}; }  // unit grid for now

  // --- physical parameters (set before geometry; they fix the baked diffusion stencil) ---
  void set_rho(double rho) { require_pre_init("set_rho"); rho_ = rho; }
  void set_mu(double mu) { require_pre_init("set_mu"); mu_ = mu; }
  void set_dt(double dt) { require_pre_init("set_dt"); dt_ = dt; }
  // body force per unit volume (e.g. -dp/dx); stored, applied as acceleration f/rho once rho is known.
  void set_body_force(double fx, double fy, double fz) {
    fx_ = fx; fy_ = fy; fz_ = fz;
    if (inited_) s_.set_body_force(fx_ / rho_, fy_ / rho_, fz_ / rho_);
  }

  // --- scheme flags (safe before or after init; just toggle members) ---
  void set_advection(bool on) { s_.set_advection(on); }
  // Incremental-rotational pressure correction (default ON): more accurate transient + near-wall
  // pressure than classical Chorin; same steady velocity. Off => classical non-incremental Chorin.
  void set_incremental_pressure(bool on) { incremental_ = on; if (inited_) s_.set_incremental_pressure(on); }
  void set_implicit_advection(bool on) { s_.set_implicit_advection(on); }
  void set_outer_iterations(int n) { s_.set_outer_iterations(n); }
  void set_outer_tolerance(double t) { s_.set_outer_tolerance(t); }
  void set_velocity_streams(bool on) { s_.set_velocity_streams(on); }
  void set_pressure_multigrid(bool on, int levels) { s_.set_pressure_multigrid(on, levels); }
  void set_pressure_pcg(bool on, int max_iter, double rtol) { s_.set_pressure_pcg(on, max_iter, rtol); }
  void set_velocity_multigrid(bool on, int levels, int v_cycles) {
    s_.set_velocity_multigrid(on, levels, v_cycles);
  }
  // inner-iteration counts used by step() (persistent; step() takes no per-call counts).
  void set_velocity_solver_params(int n_diff) { n_diff_ = n_diff < 1 ? 1 : n_diff; }
  void set_pressure_solver_params(int n_pois) { n_pois_ = n_pois < 1 ? 1 : n_pois; }

  using Arr = py::array_t<double, py::array::forcecast>;  // double, any memory order (strides respected)

  // Install an SDF solid (negative inside) -> Robust-Scaled cut-cell IBM no-slip (or moving wall u_bc),
  // AND the matching cut-cell pressure operator. SDF is a 3-D [x,y,z] array. Fixes the geometry.
  void set_solid(Arr sdf, double ubx, double uby, double ubz, bool cutcell_pressure,
                 const std::string& pressure_coarse) {
    ensure_init();
    auto block = to_block(sdf);
    s_.set_ibm_solid(block, make_float3((float)ubx, (float)uby, (float)ubz));
    if (cutcell_pressure) {
      int mode = 0;  // rediscretized (default, recommended)
      if (pressure_coarse == "galerkin") mode = 1;
      else if (pressure_coarse == "const") mode = 2;
      else if (pressure_coarse != "rediscretized")
        throw std::runtime_error("sdflow: pressure_coarse must be 'rediscretized', 'galerkin' or 'const'");
      s_.set_cutcell_pressure_operator(block, mode);
    }
  }
  void set_state(Arr u, Arr v, Arr w) {  // restore/seed the velocity state
    ensure_init();
    auto bu = to_block(u), bv = to_block(v), bw = to_block(w);
    s_.upload_velocity(bu.data(), bv.data(), bw.data());
  }

  void step() { ensure_init(); s_.step(n_diff_, n_pois_); }
  int last_outer_iterations() const { return s_.last_outer_iterations(); }
  double max_open_divergence() { ensure_init(); return s_.max_open_divergence(); }

  // gathered global fields on root (empty elsewhere), 3-D [x,y,z] (F-contiguous: x fastest, VTK order).
  py::array_t<double> get_u() { ensure_init(); return gather(s_.u(), 1.0); }
  py::array_t<double> get_v() { ensure_init(); return gather(s_.v(), 1.0); }
  py::array_t<double> get_w() { ensure_init(); return gather(s_.w(), 1.0); }
  py::array_t<double> get_p() {  // p = rho/dt * potential (accumulated under the incremental scheme)
    ensure_init();
    return gather(s_.pressure_potential(), rho_ / dt_);
  }

 private:
  void require_pre_init(const char* what) const {
    if (inited_)
      throw std::runtime_error(std::string("sdflow: ") + what +
                               "() must be called before the geometry/first step (it fixes the "
                               "baked diffusion stencil)");
  }
  void ensure_init() {
    if (inited_) return;
    if (!(rho_ > 0.0) || !(mu_ >= 0.0) || !(dt_ > 0.0))
      throw std::runtime_error("sdflow: set_rho(>0), set_mu(>=0) and set_dt(>0) before use");
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    s_.init(res_, rank, size, /*nu=*/mu_ / rho_, dt_, MPI_COMM_WORLD);
    s_.set_body_force(fx_ / rho_, fy_ / rho_, fz_ / rho_);
    s_.set_incremental_pressure(incremental_);
    inited_ = true;
  }

  // a 3-D global field g[x,y,z] (any memory order) -> this rank's extended block via periodic wrap.
  std::vector<double> to_block(const Arr& g) {
    if (g.ndim() != 3)
      throw std::runtime_error("sdflow: field must be a 3-D (nx,ny,nz) array indexed [x,y,z]");
    auto r = g.unchecked<3>();
    if (r.shape(0) != res_.x || r.shape(1) != res_.y || r.shape(2) != res_.z)
      throw std::runtime_error("sdflow: field shape must be (nx,ny,nz)");
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

  py::array_t<double> gather(double* comp, double scale) {
    std::vector<double> g = s_.gather_to_root(comp);  // global on root, empty elsewhere
    if (scale != 1.0)
      for (double& x : g) x *= scale;
    return to_numpy3d(g, res_);
  }

  DistributedNS s_;
  int3 res_{};
  double dx_ = 1.0;
  double rho_ = -1.0, mu_ = -1.0, dt_ = -1.0;   // physical params (must be set before use)
  double fx_ = 0.0, fy_ = 0.0, fz_ = 0.0;       // body force per unit volume
  int n_diff_ = 30, n_pois_ = 50;               // inner-iteration counts for step()
  bool incremental_ = true;                     // incremental-rotational pressure (default on)
  bool inited_ = false;
};

PYBIND11_MODULE(sdflow, m) {
  m.doc() =
      "sdflow -- GPU incompressible Navier-Stokes flow through SDF geometry (cut-cell IBM), MPI-optional";
  py::class_<Solver>(m, "Solver")
      .def(py::init<int, int, int>(), py::arg("nx"), py::arg("ny"), py::arg("nz"))
      .def("rank", &Solver::rank)
      .def("size", &Solver::size)
      .def("bcast_from_root", &Solver::bcast_from_root, py::arg("flag"))
      .def("get_resolution", &Solver::get_resolution)
      .def("get_spacing", &Solver::get_spacing)
      .def("set_rho", &Solver::set_rho, py::arg("rho"))
      .def("set_mu", &Solver::set_mu, py::arg("mu"))
      .def("set_dt", &Solver::set_dt, py::arg("dt"))
      .def("set_body_force", &Solver::set_body_force, py::arg("fx"), py::arg("fy"), py::arg("fz"))
      .def("set_advection", &Solver::set_advection, py::arg("on"))
      .def("set_incremental_pressure", &Solver::set_incremental_pressure, py::arg("on"))
      .def("set_implicit_advection", &Solver::set_implicit_advection, py::arg("on"))
      .def("set_outer_iterations", &Solver::set_outer_iterations, py::arg("n"))
      .def("set_outer_tolerance", &Solver::set_outer_tolerance, py::arg("tol"))
      .def("set_velocity_streams", &Solver::set_velocity_streams, py::arg("on"))
      .def("set_pressure_multigrid", &Solver::set_pressure_multigrid, py::arg("on"),
           py::arg("levels") = 4)
      .def("set_pressure_pcg", &Solver::set_pressure_pcg, py::arg("on"), py::arg("max_iter") = 60,
           py::arg("rtol") = 1e-8)
      .def("set_velocity_multigrid", &Solver::set_velocity_multigrid, py::arg("on"),
           py::arg("levels") = 3, py::arg("v_cycles") = 4)
      .def("set_velocity_solver_params", &Solver::set_velocity_solver_params, py::arg("n_diff"))
      .def("set_pressure_solver_params", &Solver::set_pressure_solver_params, py::arg("n_pois"))
      .def("set_solid", &Solver::set_solid, py::arg("sdf"), py::arg("ubx") = 0.0, py::arg("uby") = 0.0,
           py::arg("ubz") = 0.0, py::arg("cutcell_pressure") = true,
           py::arg("pressure_coarse") = "rediscretized")
      .def("set_state", &Solver::set_state, py::arg("u"), py::arg("v"), py::arg("w"))
      .def("step", &Solver::step)
      .def("last_outer_iterations", &Solver::last_outer_iterations)
      .def("max_open_divergence", &Solver::max_open_divergence)
      .def("get_u", &Solver::get_u)
      .def("get_v", &Solver::get_v)
      .def("get_w", &Solver::get_w)
      .def("get_p", &Solver::get_p);
}
