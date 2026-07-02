/// @file
/// @brief nanobind module `flow` — the Kokkos cut-cell IBM Navier-Stokes solver
/// (`peclet.flow.Solver`).
///
/// Exposes peclet::flow::IbmSolver to Python: set rho/mu/dt, a body force, an SDF solid (cut-cell
/// IBM no-slip
/// + optional cut-cell pressure projection), step, read back the velocity/pressure, and query the
/// cut-cell flux divergence. Exercised by verify_poiseuille_sdflow (IBM channel) and
/// verify_periodic_spheres_sdflow (cut-cell Stokes through a sphere packing). Kokkos is initialized
/// at import and finalized via Python atexit (the solver holds Kokkos Views, so callers must
/// release the Solver before exit -- del + gc.collect()). rank()/bcast_from_root() are single-rank
/// stubs (the multi-rank path lives in tests/kokkos_mpi).
///
/// Arrays cross the boundary through the shared zero-copy bridge (peclet::core::python, in core):
/// fields come back as Fortran-order (nx,ny,nz) float64 NumPy arrays referencing the field buffer,
/// and inputs are read as flat x-fastest buffers. See tpx/python/ndarray_interop.hpp.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <Kokkos_Core.hpp>
#include <vector>

#ifdef PECLET_FLOW_MPI
#include <mpi.h>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#endif

#include "flow_ibm.hpp"
#include "peclet/core/python/ndarray_interop.hpp"

#ifdef PECLET_FLOW_MPI
// Ensure MPI_Init has been called (mirrors the dem init_mpi idiom); safe to call repeatedly.
static void ensure_mpi_init() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
  }
}
#endif

namespace nb = nanobind;

// A solver field (flat x-fastest, ghost-stripped) -> Fortran-order (nx,ny,nz) float64 NumPy array
// for [x,y,z] indexing. The vector is moved into the array's backing store (no extra copy vs the
// old to_xyz).
template <class S>
static nb::ndarray<nb::numpy, double> field_out(S& s, std::vector<double>&& v) {
  const auto nx = static_cast<std::size_t>(s.nx());
  const auto ny = static_cast<std::size_t>(s.ny());
  const auto nz = static_cast<std::size_t>(s.nz());
  return peclet::core::python::vector_to_ndarray(
      std::move(v), {nx, ny, nz},
      {1, static_cast<std::int64_t>(nx), static_cast<std::int64_t>(nx * ny)});
}

// A Fortran-order (nx,ny,nz) float64 array -> flat x-fastest host vector (F-contiguous data() is
// already x-fastest). nanobind casts/copies the input to f_contig double if needed.
static std::vector<double> grid_in(nb::ndarray<double, nb::f_contig> a) {
  return peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(a));
}

// Register a solver class for the given GridLayout policy (Staggered -> "Solver", Colocated ->
// "SolverColocated"). The Python API is identical across grids; only the velocity-unknown placement
// and the advection control volume differ inside Solver<Grid>.
template <class Grid>
static void bind_solver(nb::module_& m, const char* name) {
  using S = peclet::flow::Solver<Grid>;
  nb::class_<S>(m, name)
      .def(nb::init<int, int, int>(), nb::arg("nx"), nb::arg("ny"), nb::arg("nz"),
           "Create a solver on an nx x ny x nz unit-spacing grid (x-fastest, I = x + y*nx + "
           "z*nx*ny). "
           "Set physical parameters (rho/mu/dt) and any domain BCs before the geometry / first "
           "step.")
      .def("set_rho", &S::setRho, nb::arg("rho"),
           "Set fluid density rho (physical units). Set before geometry/first step.")
      .def("set_mu", &S::setMu, nb::arg("mu"), "Set dynamic viscosity mu (physical units).")
      .def("set_dt", &S::setDt, nb::arg("dt"),
           "Set the time step dt; the momentum solve is scaled by 1/dt (well-conditioned at large "
           "dt).")
      .def("set_body_force", &S::setBodyForce, nb::arg("fx"), nb::arg("fy"), nb::arg("fz"),
           "Set the body force per unit volume (fx, fy, fz) — e.g. a mean pressure gradient.")
      .def("set_advection", &S::setAdvection, nb::arg("on"),
           "Enable/disable explicit high-order momentum advection (default scheme SOU). Off ⇒ "
           "Stokes.")
      .def("set_advection_scheme", &S::setAdvectionScheme, nb::arg("scheme"),
           "High-order advection scheme: 0 = second-order upwind (SOU, default), 1 = Koren TVD.")
      .def("set_incremental_pressure", &S::setIncrementalPressure, nb::arg("on"),
           "Toggle the rotational incremental-pressure projection.")
      .def("set_pressure_warmstart", &S::setPressureWarmstart, nb::arg("on"),
           "Seed each pressure solve from the previous step's phi (default off).")
      .def("set_velocity_streams", &S::setVelocityStreams, nb::arg("on"),
           "Toggle overlapped per-component velocity solves.")
      .def("set_implicit_advection", &S::setImplicitAdvection, nb::arg("on"),
           "Use implicit-FOU advection with deferred-correction TVD.")
      .def("set_outer_iterations", &S::setOuterIterations, nb::arg("n"),
           "Set the number of Picard/outer iterations per step.")
      .def("set_outer_tolerance", &S::setOuterTolerance, nb::arg("tol"),
           "Set the outer (Picard) convergence tolerance.")
      .def("last_outer_iterations", &S::lastOuterIterations,
           "Return the outer-iteration count from the last step().")
      .def("set_velocity_solver_params", &S::setVelocityIterations, nb::arg("iters"),
           "Set the velocity (diffusion) smoother iteration count.")
      .def("set_pressure_solver_params", &S::setPressureIterations, nb::arg("iters"),
           "Set the pressure smoother iteration count.")
      .def(
          "set_pressure_multigrid", [](S& s, bool, int levels) { s.setPressureLevels(levels); },
          nb::arg("on"), nb::arg("levels") = 4,
          "Set the pressure multigrid depth (levels=1 => pure RB-GS, no coarse grid).")
      .def("set_pressure_chebyshev", &S::setPressureChebyshev, nb::arg("on"),
           nb::arg("max_iter") = 120, nb::arg("rtol") = 1e-9,
           "Use the communication-light Chebyshev pressure accelerator (exclusive with PCG).")
      .def("set_pressure_pcg", &S::setPressurePcg, nb::arg("on"), nb::arg("max_iter") = 200,
           nb::arg("rtol") = 1e-8,
           "Use the MG-PCG pressure accelerator (single-GPU default; exclusive with Chebyshev).")
      .def("set_velocity_multigrid", &S::setVelocityMultigrid, nb::arg("on"), nb::arg("levels") = 4,
           nb::arg("vcycles") = 8,
           "Enable velocity (momentum) multigrid for the implicit diffusion solve.")
      .def("last_pressure_iterations", &S::lastPressureIterations,
           "Return the pressure-solver iteration count from the last step().")
      .def("set_domain_bc", &S::setDomainBc, nb::arg("face"), nb::arg("type"), nb::arg("vx") = 0.0,
           nb::arg("vy") = 0.0, nb::arg("vz") = 0.0,
           "Set a per-face domain BC (face 0..5 = -x,+x,-y,+y,-z,+z; type 0 periodic/1 wall/2 "
           "inflow/3 outflow).")
      .def(
          "set_domain_bc_profile",
          [](S& s, int face, nb::ndarray<double, nb::c_contig> prof) {
            if (prof.ndim() != 3 || prof.shape(2) != 3)
              throw std::runtime_error("profile must be (Nb,Nc,3)");
            const int nb_ = (int)prof.shape(0), nc = (int)prof.shape(1);
            s.setDomainBcProfile(
                face, peclet::core::python::ndarray_to_vector<double>(nb::ndarray<>(prof)), nb_,
                nc);
          },
          nb::arg("face"), nb::arg("profile"),
          "Prescribe a per-position inlet velocity profile (Nb,Nc,3) over a face (sets it to "
          "inflow).")
      .def(
          "set_pressure_geometry",
          [](S& s, nb::ndarray<double, nb::f_contig> sdf) { s.setPressureGeometry(grid_in(sdf)); },
          nb::arg("sdf"),
          "Set an all-fluid SDF for the cut-cell pressure operator without an immersed solid.")
      .def(
          "set_solid",
          [](S& s, nb::ndarray<double, nb::f_contig> sdf, bool cutcell_pressure,
             const std::string& /*pressure_coarse*/) {
            s.setSolid(grid_in(sdf), cutcell_pressure);
          },
          nb::arg("sdf"), nb::arg("cutcell_pressure") = false, nb::arg("pressure_coarse") = "const",
          "Set the solid SDF as a Fortran-order (nx,ny,nz) float64 array (negative inside the "
          "solid, "
          "positive in fluid); optionally enable the cut-cell pressure operator.")
      .def(
          "set_state",
          [](S& s, nb::ndarray<double, nb::f_contig> u, nb::ndarray<double, nb::f_contig> v,
             nb::ndarray<double, nb::f_contig> w) {
            s.uploadVelocity(grid_in(u), grid_in(v), grid_in(w));
          },
          nb::arg("u"), nb::arg("v"), nb::arg("w"),
          "Upload an initial velocity field (u,v,w each a Fortran-order (nx,ny,nz) float64 array).")
      .def("step", &S::step,
           "Advance the solver one time step (semi-implicit: diffusion + projection).")
      .def(
          "get_u", [](S& s) { return field_out(s, s.getVelocity(0)); },
          "Return the x-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_v", [](S& s) { return field_out(s, s.getVelocity(1)); },
          "Return the y-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_w", [](S& s) { return field_out(s, s.getVelocity(2)); },
          "Return the z-velocity component as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_p", [](S& s) { return field_out(s, s.getPressure()); },
          "Return the physical pressure as a Fortran-order (nx,ny,nz) float64 array (index "
          "[x,y,z]).")
      .def(
          "get_uf", [](S& s) { return field_out(s, s.getFaceVelocity(0)); },
          "Return the divergence-free FACE x-velocity (collocated: projected MAC field; staggered: "
          "== get_u).")
      .def(
          "get_vf", [](S& s) { return field_out(s, s.getFaceVelocity(1)); },
          "Return the divergence-free FACE y-velocity (collocated: projected MAC field; staggered: "
          "== get_v).")
      .def(
          "get_wf", [](S& s) { return field_out(s, s.getFaceVelocity(2)); },
          "Return the divergence-free FACE z-velocity (collocated: projected MAC field; staggered: "
          "== get_w).")
      .def("max_open_divergence", &S::maxOpenDivergence,
           "Return the max cut-cell flux divergence (the incompressibility residual; ~0 when "
           "converged).")
      .def(
          "get_resolution", [](S& s) { return std::vector<int>{s.nx(), s.ny(), s.nz()}; },
          "Return the grid resolution [nx, ny, nz].")
      .def(
          "get_spacing", [](S&) { return std::vector<double>{1.0, 1.0, 1.0}; },
          "Return the grid spacing [dx, dy, dz] (always unit on this grid).")
#ifdef PECLET_FLOW_MPI
      // Distributed path (built with -DPECLET_FLOW_MPI): construct the Solver with this rank's
      // LOCAL block dims (see the module-level mpi_block()), then init_mpi with the GLOBAL grid
      // dims. step() then does the g=2 velocity-block halo exchange + the distributed cut-cell
      // pressure MG. Bit-exact to single-rank.
      .def(
          "init_mpi",
          [](S& s, int gnx, int gny, int gnz) {
            ensure_mpi_init();
            s.initMpi(gnx, gny, gnz, MPI_COMM_WORLD);
          },
          nb::arg("gnx"), nb::arg("gny"), nb::arg("gnz"),
          "Wire the multi-rank step: pass the GLOBAL grid dims (gnx,gny,gnz). The Solver must have "
          "been "
          "constructed with this rank's LOCAL block dims (from mpi_block). MPI_Init is called if "
          "needed.")
      .def(
          "rank",
          [](S&) {
            ensure_mpi_init();
            int r = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &r);
            return r;
          },
          "This rank's index in MPI_COMM_WORLD.")
      .def(
          "size",
          [](S&) {
            ensure_mpi_init();
            int n = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &n);
            return n;
          },
          "The number of ranks in MPI_COMM_WORLD.")
#else
      .def(
          "rank", [](S&) { return 0; },
          "MPI rank (always 0 in the single-rank Python module; the multi-rank path is the "
          "tests/kokkos_mpi suite).")
      .def(
          "size", [](S&) { return 1; }, "MPI size (1 in the single-rank Python module).")
#endif
      .def(
          "bcast_from_root", [](S&, nb::object v) { return v; }, nb::arg("value"),
          "Broadcast a value from rank 0 (identity in the single-rank module; mirrors the MPI "
          "API).");
}

NB_MODULE(_flow, m) {
  m.attr("__doc__") =
      "flow — Kokkos cut-cell IBM incompressible Navier-Stokes solver for porous media.\n\n"
      "Two solver classes share an identical API (only the velocity-unknown placement differs):\n"
      "  Solver          — staggered MAC grid (THE flow solver; permeability/drag accuracy "
      "default)\n"
      "  SolverColocated — collocated / cell-centered velocities (ABC approximate projection)\n\n"
      "Conventions: physical units throughout (density rho, viscosity mu, physical pressure p); "
      "SDFs\n"
      "are negative inside the solid; fields are Fortran-order (nx,ny,nz) float64 (x-fastest). "
      "This is\n"
      "the single-rank module — the multi-rank MPI path is exercised by the tests/kokkos_mpi "
      "suite.\n\n"
      "Kokkos is initialized at import and finalized via a Python atexit hook. Release every "
      "Solver "
      "before interpreter exit (it goes out of scope, or `del s; gc.collect()`) so no Kokkos View "
      "outlives finalize.";
  if (!Kokkos::is_initialized())
    Kokkos::initialize();
  // Register Kokkos::finalize via Python atexit. This is REQUIRED on CUDA: without it, Kokkos's
  // internal device state is torn down by static destructors AFTER the CUDA runtime unloads,
  // aborting with cudaErrorCudartUnloading at every exit. atexit runs the hook while the driver is
  // still up. (Returned fields are backed by host std::vectors, not device Views, so they never
  // block finalize; a live Solver still holding Views at exit must be released first — hence the
  // docstring note.)
  nb::module_::import_("atexit").attr("register")(nb::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
  }));
  // The active Kokkos backend ("OpenMP", "Cuda", "HIP"), chosen by the build's install prefix.
  m.attr("execution_space") = nb::str(Kokkos::DefaultExecutionSpace::name());

  // Staggered MAC grid (THE flow solver) + the collocated/cell-centered variant. Same Python API.
  bind_solver<peclet::flow::Staggered>(m, "Solver");
  bind_solver<peclet::flow::Colocated>(m, "SolverColocated");

#ifdef PECLET_FLOW_MPI
  // Module-level: this rank's ORB block of the global (gnx,gny,gnz) grid, matching the
  // deterministic BlockDecomposer the Solver's initMpi re-derives internally (and the C++
  // tests/kokkos_mpi template). Returns (origin=[ox,oy,oz], size=[lnx,lny,lnz]); slice the global
  // SDF with these to build the local block, then Solver(*size) + init_mpi(gnx,gny,gnz). MPI_Init
  // is called if needed.
  m.def(
      "mpi_block",
      [](int gnx, int gny, int gnz) {
        ensure_mpi_init();
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size),
                                                     peclet::core::IVec<3>{gnx, gny, gnz});
        auto blk = dec.block(static_cast<std::size_t>(rank));
        std::vector<int> origin{(int)blk.origin[0], (int)blk.origin[1], (int)blk.origin[2]};
        std::vector<int> bsize{(int)blk.size[0], (int)blk.size[1], (int)blk.size[2]};
        return std::make_pair(origin, bsize);
      },
      nb::arg("gnx"), nb::arg("gny"), nb::arg("gnz"),
      "Return this MPI rank's ORB block of the global (gnx,gny,gnz) grid as (origin, size), each a "
      "length-3 list [x,y,z]. Use it to slice the global SDF into this rank's local block for a "
      "distributed Solver (see Solver.init_mpi). MPI_Init is called if needed.");

  m.attr("has_mpi") = true;
#else
  m.attr("has_mpi") = false;
#endif
}
