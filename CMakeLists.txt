cmake_minimum_required(VERSION 3.24)
project(cfd_gpu_sdflow LANGUAGES CXX)

# Canonical cfd-gpu build: the Kokkos `sdflow` cut-cell IBM Navier-Stokes solver + the `pnm_backend`
# pore-network extraction, as importable Python modules. Kokkos provides the backend (CUDA / HIP / OpenMP),
# selected by the install prefix (extern/install/<backend>, built by ../tools/bootstrap_deps.sh -- a HARD
# build dependency).
#
# Build (single-rank Python modules):
#   cmake -S . -B build \
#     -DCMAKE_PREFIX_PATH="$PWD/../extern/install/nvidia-cuda;$(python -m pybind11 --cmakedir)"
#   cmake --build build -j        ->  build/sdflow.*.so + build/pnm_backend.*.so
#
# Multi-rank MPI is exercised by the tests/kokkos_mpi ctest suite (its own find_package project); the
# single-rank kernel unit tests live in tests/kokkos. See CLAUDE.md.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "cfd-gpu: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES})")

# sdflow -- the cut-cell IBM Navier-Stokes solver (single-rank Python module).
# NO_EXTRAS: pybind LTO conflicts with the nvcc launch-compiler objects.
pybind11_add_module(sdflow NO_EXTRAS src/sdflow_bindings.cpp)
target_include_directories(sdflow PRIVATE src)
target_link_libraries(sdflow PRIVATE Kokkos::kokkos)

# pnm_backend -- pore-network extraction (Kokkos GPU compute + pure-C++ SDFReader VTI reader).
pybind11_add_module(pnm_backend NO_EXTRAS src/pnm_bindings.cpp src/sdf_reader.cpp)
target_include_directories(pnm_backend PRIVATE src)
target_link_libraries(pnm_backend PRIVATE Kokkos::kokkos)
