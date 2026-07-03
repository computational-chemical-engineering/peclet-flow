cmake_minimum_required(VERSION 3.24)
project(peclet_flow LANGUAGES CXX)

# Canonical cfd-gpu build: the Kokkos `sdflow` cut-cell IBM Navier-Stokes solver + the `pnm`
# pore-network extraction, as importable Python modules. Kokkos provides the backend (CUDA / HIP / OpenMP),
# selected by the install prefix (extern/install/<backend>, built by ../tools/bootstrap_deps.sh -- a HARD
# build dependency).
#
# Build (single-rank Python modules); nanobind is found via the active interpreter (SuiteNanobind):
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda"
#   cmake --build build -j        ->  build/sdflow.*.so + build/pnm.*.so
#
# Multi-rank MPI is exercised by the tests/kokkos_mpi ctest suite (its own find_package project); the
# single-rank kernel unit tests live in tests/kokkos. See CLAUDE.md.

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

# Dependencies via the vendored PecletDeps helper: an installed Kokkos prefix + sibling checkout for the
# dev/suite build, or FetchContent-built Kokkos + fetched core headers for a self-contained
# sdist/wheel (cibuildwheel). See cmake/PecletDeps.cmake.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PecletDeps)
peclet_require_kokkos()
peclet_require_nanobind()
peclet_sibling_include(peclet-core "${PECLET_TPX_TAG}" "../core" TPX_INC)

# sdflow -- the cut-cell IBM Navier-Stokes solver (single-rank Python module).
# NB_STATIC: bundle nanobind's runtime into the module (no shared libnanobind to ship); the Kokkos
# device path is routed by the launch compiler regardless.
# NOMINSIZE: nanobind's default -Os size optimization is rejected by nvcc ("'s': expected a number")
# since the Kokkos device sources compile as CXX through the launch compiler.
# The two extensions are assembled into the PEP-420 peclet namespace: the solver as `peclet.flow` (the
# private `_flow` extension re-exported by peclet/flow/__init__.py) and pore extraction as
# `peclet.flow.pnm`. The package __init__.py files are kept as plain files under packaging/ (OUTSIDE any
# importable peclet/ dir, so an incomplete source package can never shadow the installed one) and staged
# into <build>/peclet/... so `PYTHONPATH=<build> python …` (`import peclet.flow`) works in the dev loop
# too. The SKBUILD install rules (guarded below) place them into the wheel.
nanobind_add_module(sdflow NB_STATIC NOMINSIZE src/flow_bindings.cpp)
set_target_properties(sdflow PROPERTIES OUTPUT_NAME _flow   # -> peclet.flow._flow (NB_MODULE(_flow))
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/flow")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/flow_init.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/flow/__init__.py COPYONLY)
target_include_directories(sdflow PRIVATE src "${TPX_INC}")
target_link_libraries(sdflow PRIVATE Kokkos::kokkos)
# HIP/lld strictness: nanobind's --gc-sections can drop Kokkos SharedAllocationRecord<HIPSpace> vtables
# it only later finds referenced -> "undefined hidden symbol: vtable". Keep sections on HIP (nvcc/ld is lenient).
if(Kokkos_ENABLE_HIP)
  target_link_options(sdflow PRIVATE -Wl,--no-gc-sections)
endif()

# Distributed (MPI) flow: compile the binding TU with the halo-gating flag + link MPI (the core grid
# halo lives in the header-only TPX_INC). Mirrors dem's PECLET_DEM_MPI wiring; OFF keeps sdflow untouched.
if(PECLET_FLOW_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_compile_definitions(sdflow PRIVATE PECLET_FLOW_MPI=1)
  target_link_libraries(sdflow PRIVATE MPI::MPI_CXX)
  message(STATUS "flow: distributed step ENABLED (MPI)")
endif()

# pnm -- pore-network extraction (Kokkos GPU compute + pure-C++ SDFReader VTI reader).
nanobind_add_module(pnm NB_STATIC NOMINSIZE src/pnm_bindings.cpp src/sdf_reader.cpp)
set_target_properties(pnm PROPERTIES OUTPUT_NAME _pnm       # -> peclet.flow.pnm._pnm (NB_MODULE(_pnm))
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/peclet/flow/pnm")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/flow_pnm_init.py
               ${CMAKE_CURRENT_BINARY_DIR}/peclet/flow/pnm/__init__.py COPYONLY)
target_include_directories(pnm PRIVATE src "${TPX_INC}")
target_link_libraries(pnm PRIVATE Kokkos::kokkos)
if(Kokkos_ENABLE_HIP)
  target_link_options(pnm PRIVATE -Wl,--no-gc-sections)
endif()

# Redistributable CUDA wheel: point each module at the nvidia-cuda-runtime wheel's libcudart via a
# relative $ORIGIN RPATH (up to site-packages, then into nvidia/cu13/lib). sdflow lives at peclet/flow,
# pnm one level deeper at peclet/flow/pnm.
if(PECLET_CUDA_RUNTIME_WHEEL)
  set_target_properties(sdflow PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib" INSTALL_RPATH_USE_LINK_PATH OFF)
  set_target_properties(pnm PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../../nvidia/cu13/lib" INSTALL_RPATH_USE_LINK_PATH OFF)
endif()

# --- pip / scikit-build-core install rule -------------------------------------------------------
if(DEFINED SKBUILD)
  install(TARGETS sdflow LIBRARY DESTINATION peclet/flow COMPONENT python)
  install(TARGETS pnm    LIBRARY DESTINATION peclet/flow/pnm COMPONENT python)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/flow_init.py
          DESTINATION peclet/flow RENAME __init__.py COMPONENT python)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packaging/flow_pnm_init.py
          DESTINATION peclet/flow/pnm RENAME __init__.py COMPONENT python)
endif()
