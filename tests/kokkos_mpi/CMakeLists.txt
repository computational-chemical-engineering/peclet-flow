# flow distributed (MPI) Kokkos grid-halo integration tests (the multi-rank flow foundation;
# 103 ctests at np = 1, 2, 4[, 8]). Built by the module's CMake under `-DPECLET_FLOW_BUILD_TESTS=ON
# -DPECLET_FLOW_MPI=ON` (one tree per backend; suite/docs/QUALITY_PLAN.md §3.D), which supplies
# Kokkos, MPI and the core include path already resolved for the module; the directory also still
# configures STANDALONE (Kokkos + MPI + the header-only core + the flow operator headers):
#   cmake -S tests/kokkos_mpi -B build_kmpi \
#         -DCMAKE_PREFIX_PATH="<suite>/extern/install/<backend>" \
#         -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun -DPECLET_CORE_DIR=<suite>/core
#   cmake --build build_kmpi -j && ctest --test-dir build_kmpi --output-on-failure
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # ---- standalone project -----------------------------------------------------------------------
  project(peclet_flow_mpi_tests LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()

  find_package(Kokkos CONFIG REQUIRED)
  find_package(MPI REQUIRED)

  set(PECLET_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../core" CACHE PATH "core repo")
  set(PECLET_CORE_INCLUDE "${PECLET_CORE_DIR}/include")
  enable_testing()
elseif(NOT PECLET_CORE_INCLUDE)
  message(FATAL_ERROR "flow/tests/kokkos_mpi: add_subdirectory() needs PECLET_CORE_INCLUDE from the module's CMake")
endif()
# Force -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun on a host where FindMPI could pick a bundled mpiexec
# (ParaView's) that launches the OpenMPI-linked binaries as singletons (`*_np4` = 4 x np1).
if(NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE /usr/bin/mpirun)
endif()
# Standalone distributed primitives (own halo/reduction harness; do not need the gated CutcellMG MPI path).
foreach(t distributed_diffusion distributed_pressure distributed_mg)
  add_executable(test_${t} test_${t}.cpp)
  target_include_directories(test_${t} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${PECLET_CORE_INCLUDE})
  target_link_libraries(test_${t} PRIVATE Kokkos::kokkos MPI::MPI_CXX)
  foreach(np 1 2 4)
    add_test(NAME ${t}_np${np} COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:test_${t}>)
  endforeach()
endforeach()

# The production CutcellMG / VelocityMG / assembled SdflowIbm distributed (gated behind PECLET_FLOW_MPI).
foreach(t cutcellmg_mpi telescope_mpi telescope_varrho_mpi velocitymg_mpi velocitymg_bc_mpi sdflow_mpi sdflow_colocated_mpi redistribute_mpi graphamg_mpi
        multiphysics_mpi ghost_projection_mpi vardensity_mpi varmu_mpi bodyforce_ghost_mpi
        dragbeta_ghost_mpi cutcellmg_aniso_mpi vof_twophase_mpi vof_momentum_mpi vof_curvature_mpi
        vof_surface_tension_mpi vof_cutcell_mpi vof_wetting_mpi vof_wetting_dynamic_mpi
        vof_collocated_mpi vof_bc_mpi vof_phase_change_mpi movingscene_advect_mpi wall_slip_mpi
        vof_blocks_ns_mpi vof_redistribute_mpi)
  add_executable(test_${t} test_${t}.cpp)
  target_include_directories(test_${t} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src ${PECLET_CORE_INCLUDE})
  target_compile_definitions(test_${t} PRIVATE PECLET_FLOW_MPI)
  target_link_libraries(test_${t} PRIVATE Kokkos::kokkos MPI::MPI_CXX)
  foreach(np 1 2 4)
    add_test(NAME ${t}_np${np} COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:test_${t}>)
  endforeach()
endforeach()

# VoF rung V1: distributed Weymouth-Yue advection. Standalone (its own g=3 colour halo, no
# CutcellMG), so it does NOT need PECLET_FLOW_MPI; it shares the scene builders with the
# single-rank battery in tests/kokkos.
# VoF Part III rung W0 (WO-W0): the distributed per-bubble BLOCK container. Standalone (its own
# block advectors + the block gather/scatter over the core BlockDecomposer), so it does NOT need
# PECLET_FLOW_MPI; it shares the scene builders with the single-rank battery in tests/kokkos.
add_executable(test_vof_blocks_mpi test_vof_blocks_mpi.cpp)
target_include_directories(test_vof_blocks_mpi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src ${CMAKE_CURRENT_SOURCE_DIR}/../kokkos ${PECLET_CORE_INCLUDE})
target_compile_definitions(test_vof_blocks_mpi PRIVATE PECLET_FLOW_MPI)
target_link_libraries(test_vof_blocks_mpi PRIVATE Kokkos::kokkos MPI::MPI_CXX)
# np 8 as well since rung W1 (WO-W12): the 64-bubble swarm's LPT assignment and its migration are
# only exercised once there are more ranks than the few blocks the W0 scenes carry.
foreach(np 1 2 4 8)
  add_test(NAME vof_blocks_mpi_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:test_vof_blocks_mpi>)
endforeach()

add_executable(test_vof_advect_mpi test_vof_advect_mpi.cpp)
target_include_directories(test_vof_advect_mpi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src ${CMAKE_CURRENT_SOURCE_DIR}/../kokkos ${PECLET_CORE_INCLUDE})
target_link_libraries(test_vof_advect_mpi PRIVATE Kokkos::kokkos MPI::MPI_CXX)
foreach(np 1 2 4)
  add_test(NAME vof_advect_mpi_np${np}
           COMMAND ${MPIEXEC_EXECUTABLE} -np ${np} $<TARGET_FILE:test_vof_advect_mpi>)
endforeach()
