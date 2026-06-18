# cfd-gpu portable-kernel unit tests (Phase 3, Kokkos). Standalone (Kokkos only): each ported kernel
# is validated in isolation before sdflow is migrated. Build:
#   cmake -S tests/kokkos -B build_kokkos -DCMAKE_PREFIX_PATH=<suite>/extern/install/<backend>
#   cmake --build build_kokkos -j && ctest --test-dir build_kokkos --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(cfd_kokkos_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)
message(STATUS "cfd/kokkos: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES})")

enable_testing()
add_executable(test_reductions test_reductions.cpp)
target_include_directories(test_reductions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_reductions PRIVATE Kokkos::kokkos)
add_test(NAME reductions COMMAND test_reductions)

add_executable(test_advection test_advection.cpp)
target_include_directories(test_advection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_advection PRIVATE Kokkos::kokkos)
add_test(NAME advection COMMAND test_advection)
