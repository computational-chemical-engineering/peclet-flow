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

add_executable(test_stencils test_stencils.cpp)
target_include_directories(test_stencils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_stencils PRIVATE Kokkos::kokkos)
add_test(NAME stencils COMMAND test_stencils)

add_executable(test_transfer test_transfer.cpp)
target_include_directories(test_transfer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_transfer PRIVATE Kokkos::kokkos)
add_test(NAME transfer COMMAND test_transfer)

add_executable(test_cutcell test_cutcell.cpp)
target_include_directories(test_cutcell PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_cutcell PRIVATE Kokkos::kokkos)
add_test(NAME cutcell COMMAND test_cutcell)

add_executable(test_bc test_bc.cpp)
target_include_directories(test_bc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_bc PRIVATE Kokkos::kokkos)
add_test(NAME bc COMMAND test_bc)

add_executable(test_ibm test_ibm.cpp)
target_include_directories(test_ibm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_ibm PRIVATE Kokkos::kokkos)
add_test(NAME ibm COMMAND test_ibm)

add_executable(test_ibm_overlay test_ibm_overlay.cpp)
target_include_directories(test_ibm_overlay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_ibm_overlay PRIVATE Kokkos::kokkos)
add_test(NAME ibm_overlay COMMAND test_ibm_overlay)

add_executable(test_ibm_apply test_ibm_apply.cpp)
target_include_directories(test_ibm_apply PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_ibm_apply PRIVATE Kokkos::kokkos)
add_test(NAME ibm_apply COMMAND test_ibm_apply)

add_executable(test_sdflow_tg test_sdflow_tg.cpp)
target_include_directories(test_sdflow_tg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_sdflow_tg PRIVATE Kokkos::kokkos)
add_test(NAME sdflow_tg COMMAND test_sdflow_tg)

add_executable(bench_rbgs bench_rbgs.cpp)
target_include_directories(bench_rbgs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(bench_rbgs PRIVATE Kokkos::kokkos)

add_executable(test_mg test_mg.cpp)
target_include_directories(test_mg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kokkos)
target_link_libraries(test_mg PRIVATE Kokkos::kokkos)
add_test(NAME mg COMMAND test_mg)
