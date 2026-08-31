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

# core headers (field registry / GraphAMG bottom / halo types included by the solver headers):
# the sibling checkout in the suite, or FetchContent at the release pin on a standalone CI checkout.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(PecletDeps)
peclet_sibling_include(peclet-core "${PECLET_TPX_TAG}" "../../../core" CORE_INC)

enable_testing()
add_executable(test_reductions test_reductions.cpp)
target_include_directories(test_reductions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_reductions PRIVATE Kokkos::kokkos)
add_test(NAME reductions COMMAND test_reductions)

# Named field registry on the Solver (includes flow_ibm.hpp -> needs the core headers).
add_executable(test_field_registry test_field_registry.cpp)
target_include_directories(test_field_registry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                           ${CORE_INC})
target_link_libraries(test_field_registry PRIVATE Kokkos::kokkos)
add_test(NAME field_registry COMMAND test_field_registry)

# Scalar transport (advection-diffusion): 2nd-order diffusion operator + advection conservation.
add_executable(test_scalar_transport test_scalar_transport.cpp)
target_include_directories(test_scalar_transport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                           ${CORE_INC})
target_link_libraries(test_scalar_transport PRIVATE Kokkos::kokkos)
add_test(NAME scalar_transport COMMAND test_scalar_transport)

# Property closures + per-cell body force (LinearMix / Boussinesq / Arrhenius vs host oracle).
add_executable(test_closures test_closures.cpp)
target_include_directories(test_closures PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                           ${CORE_INC})
target_link_libraries(test_closures PRIVATE Kokkos::kokkos)
add_test(NAME closures COMMAND test_closures)

# Variable-viscosity diffusion stencil (ibmBuildDiffusionVar + FaceProps means vs oracle).
add_executable(test_variable_mu test_variable_mu.cpp)
target_include_directories(test_variable_mu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                           ${CORE_INC})
target_link_libraries(test_variable_mu PRIVATE Kokkos::kokkos)
add_test(NAME variable_mu COMMAND test_variable_mu)

# Variable-density projection: hydrostatic acid test (ratio 3 + 1e3) + uniform-rho reduction.
add_executable(test_vardensity_projection test_vardensity_projection.cpp)
target_include_directories(test_vardensity_projection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                           ${CORE_INC})
target_link_libraries(test_vardensity_projection PRIVATE Kokkos::kokkos)
add_test(NAME vardensity_projection COMMAND test_vardensity_projection)

# WO-H: 3-D wall-bounded (nz >= 8) pressure convergence for all three Krylov drivers, including the
# stratified hydrostatic column and the small-coefficient-at-an-inflow-face configurations.
# PECLET_FLOW_MG_BCGHOST=0 reinstates the pre-repair coarse ghost and makes every PCG row fail.
add_executable(test_pressure_wallbounded test_pressure_wallbounded.cpp)
target_include_directories(test_pressure_wallbounded PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                           ${CORE_INC})
target_link_libraries(test_pressure_wallbounded PRIVATE Kokkos::kokkos)
add_test(NAME pressure_wallbounded COMMAND test_pressure_wallbounded)

# VoF rung V0: PLIC toolbox (SZ/Lehmann-Gekle plane<->volume, MYC/Youngs normals, slab flux).
add_executable(test_vof_plic test_vof_plic.cpp)
target_include_directories(test_vof_plic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_vof_plic PRIVATE Kokkos::kokkos)
add_test(NAME vof_plic COMMAND test_vof_plic)

# VoF rung V1: Weymouth-Yue split geometric advection on prescribed velocity fields.
add_executable(test_vof_advect test_vof_advect.cpp)
target_include_directories(test_vof_advect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                           ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_vof_advect PRIVATE Kokkos::kokkos)
add_test(NAME vof_advect COMMAND test_vof_advect)

add_executable(test_advection test_advection.cpp)
target_include_directories(test_advection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_advection PRIVATE Kokkos::kokkos)
add_test(NAME advection COMMAND test_advection)

add_executable(test_stencils test_stencils.cpp)
target_include_directories(test_stencils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_stencils PRIVATE Kokkos::kokkos)
add_test(NAME stencils COMMAND test_stencils)

add_executable(test_transfer test_transfer.cpp)
target_include_directories(test_transfer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_transfer PRIVATE Kokkos::kokkos)
add_test(NAME transfer COMMAND test_transfer)

add_executable(test_cutcell test_cutcell.cpp)
target_include_directories(test_cutcell PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_cutcell PRIVATE Kokkos::kokkos)
add_test(NAME cutcell COMMAND test_cutcell)

add_executable(test_bc test_bc.cpp)
target_include_directories(test_bc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_bc PRIVATE Kokkos::kokkos)
add_test(NAME bc COMMAND test_bc)

add_executable(test_ibm test_ibm.cpp)
target_include_directories(test_ibm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_ibm PRIVATE Kokkos::kokkos)
add_test(NAME ibm COMMAND test_ibm)

add_executable(test_ibm_overlay test_ibm_overlay.cpp)
target_include_directories(test_ibm_overlay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_ibm_overlay PRIVATE Kokkos::kokkos)
add_test(NAME ibm_overlay COMMAND test_ibm_overlay)

add_executable(test_ibm_apply test_ibm_apply.cpp)
target_include_directories(test_ibm_apply PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_ibm_apply PRIVATE Kokkos::kokkos)
add_test(NAME ibm_apply COMMAND test_ibm_apply)

add_executable(test_sdflow_tg test_sdflow_tg.cpp)
target_include_directories(test_sdflow_tg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_sdflow_tg PRIVATE Kokkos::kokkos)
add_test(NAME sdflow_tg COMMAND test_sdflow_tg)

add_executable(bench_rbgs bench_rbgs.cpp)
target_include_directories(bench_rbgs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(bench_rbgs PRIVATE Kokkos::kokkos)

add_executable(test_mg test_mg.cpp)
target_include_directories(test_mg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_mg PRIVATE Kokkos::kokkos)
add_test(NAME mg COMMAND test_mg)

add_executable(test_fou_operator test_fou_operator.cpp)
target_include_directories(test_fou_operator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_fou_operator PRIVATE Kokkos::kokkos)
add_test(NAME fou_operator COMMAND test_fou_operator)

add_executable(test_poiseuille test_poiseuille.cpp)
target_include_directories(test_poiseuille PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_poiseuille PRIVATE Kokkos::kokkos)
add_test(NAME poiseuille COMMAND test_poiseuille)

# VoF rung V2a (WO-J): the colour field wired into the NS solver — bridge, hydrostatic acid test
# through C, conservation, C==const reductions, harmonic rho_f ablation, interface-local CFL.
# PECLET_VOF_TWOPHASE_LONG=1 runs the conservation gate for 1000 steps instead of 200.
add_executable(test_vof_twophase test_vof_twophase.cpp)
target_include_directories(test_vof_twophase PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                           ${CORE_INC})
target_link_libraries(test_vof_twophase PRIVATE Kokkos::kokkos)
add_test(NAME vof_twophase COMMAND test_vof_twophase)

# VoF rung V2b (WO-K): momentum-consistent transport of rho^c u_c on the half-shifted MAC CVs.
add_executable(test_vof_momentum test_vof_momentum.cpp)
target_include_directories(test_vof_momentum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src
                           ${CORE_INC})
target_link_libraries(test_vof_momentum PRIVATE Kokkos::kokkos)
add_test(NAME vof_momentum COMMAND test_vof_momentum)

# VoF rung V3 (WO-O): the Popinet height-function curvature cascade + the Jibben/Han PLIC-volumetric
# paraboloid fallback (src/vof/curvature.hpp, src/vof/curvature_field.hpp). Standalone — no Solver.
add_executable(test_vof_curvature test_vof_curvature.cpp)
target_include_directories(test_vof_curvature PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_vof_curvature PRIVATE Kokkos::kokkos)
add_test(NAME vof_curvature COMMAND test_vof_curvature)

add_executable(test_poiseuille_ibm test_poiseuille_ibm.cpp)
target_include_directories(test_poiseuille_ibm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(test_poiseuille_ibm PRIVATE Kokkos::kokkos)
add_test(NAME poiseuille_ibm COMMAND test_poiseuille_ibm)
