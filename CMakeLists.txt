cmake_minimum_required(VERSION 3.24)

# One version source: pyproject.toml (`version = "x.y.z"`); CMake derives its project version from it
# (suite/docs/QUALITY_PLAN.md D4) so the two can never disagree.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _peclet_flow_version_line REGEX "^version = \"")
string(REGEX REPLACE "^version = \"([0-9]+\\.[0-9]+\\.[0-9]+)[^\"]*\".*$" "\\1" PECLET_FLOW_VERSION
       "${_peclet_flow_version_line}")
if(NOT PECLET_FLOW_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "flow: could not read `version = \"x.y.z\"` from pyproject.toml (got '${_peclet_flow_version_line}')")
endif()
project(peclet_flow VERSION ${PECLET_FLOW_VERSION} LANGUAGES CXX)

# Canonical build: the Kokkos `peclet.flow` cut-cell IBM Navier-Stokes solver, as an importable
# Python module. Kokkos provides the backend (CUDA / HIP / OpenMP), selected by the install prefix
# (extern/install/<backend>, built by ../tools/bootstrap_deps.sh -- a HARD build dependency).
# (Pore-network extraction moved to its own suite project: ../pnm, peclet.pnm.)
#
# Build (single-rank Python module); nanobind is found via the active interpreter (SuiteNanobind):
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
#   cmake --build build -j        ->  build/peclet/flow/_flow.*.so
#
# Tests: -DPECLET_FLOW_BUILD_TESTS=ON adds tests/kokkos (single-rank kernel ctests) and, with
# PECLET_FLOW_MPI, tests/kokkos_mpi (np = 1, 2, 4) to THIS tree, plus the Python regression suite and
# the canonical verify scripts as ctests on the built module. One tree per backend. See CLAUDE.md.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Multi-rank: -DPECLET_FLOW_MPI=ON links MPI + exposes Solver.init_mpi / rank / size + the module-level
# mpi_block() in the flow module (the distributed IbmSolver step, bit-exact to single-rank). OFF (default)
# leaves the single-rank module byte-identical.
option(PECLET_FLOW_MPI "Build the flow module with the distributed (MPI) step exposed" OFF)

# Redistributable single-GPU CUDA wheel: libcudart is provided by the `nvidia-cuda-runtime` PyPI
# dependency (installed to site-packages/nvidia/cu13/lib), not the system. ON sets each module's RPATH
# to reach that wheel from its install location so `import peclet.flow` works with no system CUDA. Used
# by the peclet-flow-cu13 packaging (packaging/pyproject-cuda.toml + the release workflow's cuda job).
option(PECLET_CUDA_RUNTIME_WHEEL "RPATH the modules to the nvidia-cuda-runtime wheel's libcudart" OFF)

# Operator storage precision (QUALITY_PLAN G.6, suite/docs/SCALING_ISSUES.md #1): OFF (default)
# keeps MReal (mac_cutcell_mg.hpp) = float -- the pressure hierarchy AND, via IbmSolver::FV /
# IbmOverlay, the momentum + cut-cell overlay storage. ON makes the whole operator double end to
# end (+~12% step time); float rounding breaks the singular row-sum identity A.1 = 0 and a dense
# cut-cell bed's residual floors then REBOUNDS -- the run is invalid, not merely degraded. This
# used to be reachable only via a raw, untested `-DCMAKE_CXX_FLAGS=-DPECLET_FLOW_OPERATOR_DOUBLE`
# (renamed from PECLET_FLOW_MREAL_DOUBLE); it is now `pip install . -C
# cmake.define.PECLET_FLOW_OPERATOR_DOUBLE=ON` like every other option here. Directory-scoped so
# every target in this tree (the module, tests/kokkos, tests/kokkos_mpi) sees the same macro.
option(PECLET_FLOW_OPERATOR_DOUBLE "Operator (pressure + momentum + cut-cell overlay) storage in double, not float" OFF)
if(PECLET_FLOW_OPERATOR_DOUBLE)
  add_compile_definitions(PECLET_FLOW_OPERATOR_DOUBLE=1)
  message(STATUS "flow: operator storage precision = DOUBLE (+~12% step time; see CLAUDE.md)")
endif()

# Test suites in this tree (suite/docs/QUALITY_PLAN.md §3.D): OFF by default so a wheel build is
# unaffected; ON in CI and in the CLAUDE.md dev recipe. tests/kokkos and tests/kokkos_mpi remain
# configurable standalone as well (their project() is guarded).
option(PECLET_FLOW_BUILD_TESTS "Build the ctest suites (tests/kokkos; + tests/kokkos_mpi with PECLET_FLOW_MPI; regression + verify scripts)" OFF)

# Dependencies via the vendored PecletDeps helper: an installed Kokkos prefix + sibling checkout for the
# dev/suite build, or FetchContent-built Kokkos + fetched core headers for a self-contained
# sdist/wheel (cibuildwheel). See cmake/PecletDeps.cmake.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PecletDeps)
peclet_require_kokkos()
peclet_require_nanobind()
peclet_sibling_include(peclet-core "${PECLET_CORE_TAG}" "../core" PECLET_CORE_INCLUDE)

# The ONE compiled instantiation of Solver<Staggered> / Solver<Colocated> (QUALITY_PLAN G.8): the
# module and every test link it instead of instantiating the ~12 k-line class template themselves.
# PECLET_FLOW_MPI changes the class, so there is one library per configuration and each consumer
# links the matching one -- PUBLIC on the library, so the macro cannot disagree with the
# instantiation. The MPI variant is built when the option is on (the module and tests/kokkos_mpi
# need it); the plain variant when the module is single-rank or when tests/kokkos is configured.
include(PecletFlowSolver)

# ccache (QUALITY_PLAN G.8) is opt-in, via the standard -DCMAKE_CXX_COMPILER_LAUNCHER=ccache; nothing
# here turns it on. It works on the host backends. It CANNOT work on a Kokkos CUDA/HIP prefix, and
# the failure is otherwise baffling, so fail here with the reason instead: Kokkos routes device
# compilation by putting `kokkos_launch_compiler <nvcc_wrapper> <c++>` in the global
# RULE_LAUNCH_COMPILE, and that script redirects only when the executable that FOLLOWS it is the
# very compiler it was given. A compiler launcher is expanded between the two, so the launcher --
# not the compiler -- is what the script sees, it declines to redirect, and plain `c++` is handed
# nvcc's flags ("unrecognized command-line option '-arch=sm_120'") after a long build.
get_property(_peclet_flow_rule_launch GLOBAL PROPERTY RULE_LAUNCH_COMPILE)
if(CMAKE_CXX_COMPILER_LAUNCHER AND _peclet_flow_rule_launch MATCHES "kokkos_launch_compiler")
  message(FATAL_ERROR
    "flow: CMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER} cannot be combined with this "
    "Kokkos prefix -- Kokkos already owns the compile rule's launcher slot with kokkos_launch_compiler "
    "(device backend), and a second launcher breaks its redirect. Configure this tree without the "
    "launcher; ccache is available on the host-openmp / host-serial prefixes.")
endif()

if(PECLET_FLOW_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  peclet_flow_add_solver_library(peclet_flow_solver_mpi MPI)
endif()
if(PECLET_FLOW_BUILD_TESTS OR NOT PECLET_FLOW_MPI)
  peclet_flow_add_solver_library(peclet_flow_solver)
endif()

# peclet_flow -- the cut-cell IBM Navier-Stokes solver module (target; imports as peclet.flow).
# NB_STATIC: bundle nanobind's runtime into the module (no shared libnanobind to ship); the Kokkos
# device path is routed by the launch compiler regardless.
# NOMINSIZE: nanobind's default -Os size optimization is rejected by nvcc ("'s': expected a number")
# since the Kokkos device sources compile as CXX through the launch compiler.
# The extension is assembled into the PEP-420 peclet namespace as `peclet.flow` (the private `_flow`
# extension re-exported by peclet/flow/__init__.py). The package __init__.py is kept as a plain file
# under packaging/ (OUTSIDE any importable peclet/ dir, so an incomplete source package can never
# shadow the installed one) and staged into <build>/peclet/... so `PYTHONPATH=<build> python …`
# (`import peclet.flow`) works in the dev loop too. The SKBUILD install rules (guarded below) place
# it into the wheel.
nanobind_add_module(peclet_flow NB_STATIC NOMINSIZE src/flow_bindings.cpp)
set_target_properties(peclet_flow PROPERTIES OUTPUT_NAME _flow   # -> peclet.flow._flow (NB_MODULE(_flow))
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/flow")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/flow_init.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/flow/__init__.py COPYONLY)
target_include_directories(peclet_flow PRIVATE src "${PECLET_CORE_INCLUDE}")
target_link_libraries(peclet_flow PRIVATE Kokkos::kokkos)
# G.8: the solver instantiation. STATIC, so its objects land INSIDE _flow.so -- nothing extra to
# ship in the wheel. PIC and hidden visibility on the library match what nanobind_add_module puts
# on this target, so the module's code generation is what it was before the split.
if(PECLET_FLOW_MPI)
  target_link_libraries(peclet_flow PRIVATE peclet_flow_solver_mpi)
else()
  target_link_libraries(peclet_flow PRIVATE peclet_flow_solver)
endif()
# HIP/lld strictness: nanobind's --gc-sections can drop Kokkos SharedAllocationRecord<HIPSpace> vtables
# it only later finds referenced -> "undefined hidden symbol: vtable". Keep sections on HIP (nvcc/ld is lenient).
if(Kokkos_ENABLE_HIP)
  target_link_options(peclet_flow PRIVATE -Wl,--no-gc-sections)
  # HIP/lld experiment 1 (docs/RELEASE.md §8): nanobind sets CXX_VISIBILITY_PRESET hidden, and under hipcc
  # the host objects then reference the Kokkos SharedAllocationRecord / shared_ptr control-block vtables
  # as hidden symbols that nothing defines ("undefined hidden symbol: vtable for ..."). Default visibility
  # on the HIP path only; CUDA/OpenMP builds are untouched.
  set_target_properties(peclet_flow PROPERTIES CXX_VISIBILITY_PRESET default)
endif()

# Distributed (MPI) flow: compile the binding TU with the halo-gating flag + link MPI (the core grid
# halo lives in the header-only PECLET_CORE_INCLUDE). Mirrors dem's PECLET_DEM_MPI wiring; OFF keeps the module untouched.
if(PECLET_FLOW_MPI)
  # PECLET_FLOW_MPI=1 and MPI::MPI_CXX arrive PUBLIC from peclet_flow_solver_mpi (G.8), which is
  # what guarantees the module's binding TU sees the same class the instantiation compiled.
  message(STATUS "flow: distributed step ENABLED (MPI)")
endif()

# Redistributable CUDA wheel: point the module at the nvidia-cuda-runtime wheel's libcudart via a
# relative $ORIGIN RPATH (up to site-packages, then into nvidia/cu13/lib).
if(PECLET_CUDA_RUNTIME_WHEEL)
  set_target_properties(peclet_flow PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib" INSTALL_RPATH_USE_LINK_PATH OFF)
endif()

# --- test suites (PECLET_FLOW_BUILD_TESTS) -------------------------------------------------------
if(PECLET_FLOW_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/kokkos)          # 44 kernel ctests (+ the `bench`-labelled instruments)
  if(PECLET_FLOW_MPI)
    add_subdirectory(tests/kokkos_mpi)    # 103 distributed ctests, np = 1, 2, 4[, 8]
  endif()

  # Python gates on the module built HERE (PYTHONPATH = this tree). Run from the repo root so the
  # scripts' relative data paths resolve. `python_*` tests exit 77 -> ctest SKIPPED, never green-by-no-op.
  # (the verify scripts also insert $SDFLOW_BUILD -- default `build_mpi`/`build` -- at the FRONT of
  # sys.path, ahead of PYTHONPATH: point it here too, or a stale sibling tree gets tested instead)
  set(_flow_py_env "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};SDFLOW_BUILD=${CMAKE_CURRENT_BINARY_DIR}")
  # The single-GPU accuracy + efficiency regression suite (tests/regression/README.md): Z&H sphere,
  # random-sphere bed, hollow-ring bed on the grid ladder, checked against the recorded baseline.
  add_test(NAME regression_staggered
           COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/sdflow_regression.py
                   --build ${CMAKE_CURRENT_BINARY_DIR}
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(regression_staggered PROPERTIES ENVIRONMENT "${_flow_py_env}" LABELS regression
                       SKIP_RETURN_CODE 77)
  # The three smallest canonical verify scripts (CLAUDE.md "Running Tests and Verification").
  foreach(v poiseuille_flow lid_cavity_sdflow colocated_taylor_green)
    add_test(NAME verify_${v}
             COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_${v}.py
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(verify_${v} PROPERTIES ENVIRONMENT "${_flow_py_env}" LABELS verify
                         SKIP_RETURN_CODE 77)
  endforeach()
  # QUALITY_PLAN D3: `src/` carries no environment variable that can change a result (package E).
  add_test(NAME no_env_knobs
           COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_no_env_knobs.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(no_env_knobs PROPERTIES LABELS quality)
  # QUALITY_PLAN G.6: no hard (float) cast / float-typed view on the operator-storage path outside
  # the allow-list (mreal/MReal must be used, or the line carries a PRECISION-EXEMPT reason).
  add_test(NAME no_float_operator_casts
           COMMAND ${Python_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_no_float_operator_casts.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(no_float_operator_casts PROPERTIES LABELS quality)
endif()

# --- pip / scikit-build-core install rule -------------------------------------------------------
if(DEFINED SKBUILD)
  install(TARGETS peclet_flow LIBRARY DESTINATION peclet/flow COMPONENT python)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/flow_init.py
          DESTINATION peclet/flow RENAME __init__.py COMPONENT python)
endif()
