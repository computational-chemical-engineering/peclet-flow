# WO-M — dense assembly of the multigrid V-cycle preconditioner M, so that its symmetry and its
# DEFINITENESS can be measured instead of inferred. Standalone (Kokkos only), built twice:
#
#   cmake -S tests/study/mg_precond -B build_wom_probe_f \
#         -DCMAKE_PREFIX_PATH=<suite>/extern/install/host-openmp
#   cmake -S tests/study/mg_precond -B build_wom_probe_d \
#         -DCMAKE_PREFIX_PATH=<suite>/extern/install/host-openmp \
#         -DCMAKE_CXX_FLAGS=-DPECLET_FLOW_MREAL_DOUBLE
#
# The two builds differ ONLY in the operator/level storage precision (mac_cutcell_mg.hpp MReal), so
# the A/B isolates float rounding from the coarsening approximation. Analysis (skew, LDL pivots,
# spectrum) is done in Python by mg_precond_analyze.py on the matrices this writes.
cmake_minimum_required(VERSION 3.24)
project(peclet_flow_mg_precond LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Kokkos CONFIG REQUIRED)
message(STATUS "flow/mg_precond: Kokkos ${Kokkos_VERSION} (${Kokkos_DEVICES})")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
include(PecletDeps)
peclet_sibling_include(peclet-core "${PECLET_TPX_TAG}" "../../../../core" CORE_INC)

add_executable(mg_dense_precond mg_dense_precond.cpp)
target_include_directories(mg_dense_precond PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../src
                           ${CORE_INC})
target_link_libraries(mg_dense_precond PRIVATE Kokkos::kokkos)
