# sdflow_kokkos — Kokkos pybind11 module (drop-in for the sdflow Solver API, IBM velocity solve).
# Standalone: needs only Kokkos + pybind11. Build:
#   cmake -S kokkos_module -B build_module \
#     -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>;$(python -m pybind11 --cmakedir)"
#   cmake --build build_module -j   ->  sdflow_kokkos.*.so
cmake_minimum_required(VERSION 3.24)
project(sdflow_kokkos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "sdflow_kokkos: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES})")

# NO_EXTRAS: pybind LTO conflicts with the nvcc launch-compiler objects.
pybind11_add_module(sdflow_kokkos NO_EXTRAS binding.cpp)
target_include_directories(sdflow_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/kokkos)
target_link_libraries(sdflow_kokkos PRIVATE Kokkos::kokkos)

# pore-network extraction (the canonical pnm_backend: Kokkos compute + pure-C++ SDFReader VTI reader)
pybind11_add_module(pnm_kokkos NO_EXTRAS pnm_binding.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../src/sdf_reader.cpp)
target_include_directories(pnm_kokkos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/kokkos
                                              ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(pnm_kokkos PRIVATE Kokkos::kokkos)
