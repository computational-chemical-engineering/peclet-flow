cmake_minimum_required(VERSION 3.24)
project(cfd_gpu_sdflow LANGUAGES CXX)

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

find_package(Kokkos CONFIG REQUIRED)
message(STATUS "cfd-gpu: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES})")

# nanobind (shared suite helper) + the header-only View<->ndarray zero-copy bridge in transport-core.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(SuiteNanobind)
suite_require_nanobind()
set(TPX_INC "${CMAKE_CURRENT_SOURCE_DIR}/../transport-core/include")

# sdflow -- the cut-cell IBM Navier-Stokes solver (single-rank Python module).
# NB_STATIC: bundle nanobind's runtime into the module (no shared libnanobind to ship); the Kokkos
# device path is routed by the launch compiler regardless.
nanobind_add_module(sdflow NB_STATIC src/sdflow_bindings.cpp)
target_include_directories(sdflow PRIVATE src "${TPX_INC}")
target_link_libraries(sdflow PRIVATE Kokkos::kokkos)

# pnm -- pore-network extraction (Kokkos GPU compute + pure-C++ SDFReader VTI reader).
nanobind_add_module(pnm NB_STATIC src/pnm_bindings.cpp src/sdf_reader.cpp)
target_include_directories(pnm PRIVATE src "${TPX_INC}")
target_link_libraries(pnm PRIVATE Kokkos::kokkos)

# --- pip / scikit-build-core install rule -------------------------------------------------------
# `pip install .` (driven by pyproject.toml) builds both modules against a Kokkos prefix on
# CMAKE_PREFIX_PATH and installs them as top-level `sdflow` + `pnm` modules. Inert for a plain
# `cmake --build` (no `cmake --install`), so the developer workflow is unchanged.
if(DEFINED SKBUILD)
  install(TARGETS sdflow pnm LIBRARY DESTINATION . COMPONENT python)
endif()
