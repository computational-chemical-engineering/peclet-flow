cmake_minimum_required(VERSION 3.18)
project(pnm_extraction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 120)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Add source files
set(SOURCES
    src/bindings.cpp
    src/sdf_reader.cpp
    src/pore_extraction.cu
    src/cfd_solver.cu
    src/cfd_solver_ibm.cu
    src/cfd_solver_multigrid.cu
)

# Define the Python extension module
pybind11_add_module(pnm_backend ${SOURCES})

# target include directories
target_include_directories(pnm_backend PRIVATE src)

# Link against CUDA Runtime
target_link_libraries(pnm_backend PRIVATE CUDA::cudart)

# ---------------------------------------------------------------------------
# Optional: MPI distributed-halo capability (transport-core integration).
# Off by default so the standard module build is unchanged. Enable with
#   cmake -S . -B build_mpi -DCFD_BUILD_MPI=ON
# transport-core is header-only and picked up from the sibling checkout.
# ---------------------------------------------------------------------------
option(CFD_BUILD_MPI "Build MPI distributed-halo tests (needs transport-core + MPI)" OFF)
if(CFD_BUILD_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  set(TPX_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../transport-core/include")
  if(NOT EXISTS "${TPX_INCLUDE}/tpx/halo/grid_halo.hpp")
    message(FATAL_ERROR "transport-core not found at ${TPX_INCLUDE} (clone it as a sibling repo)")
  endif()

  enable_testing()
  # transport-core's halo headers are C++17-clean; the RTX 5080 is sm_120.
  foreach(t test_mac_halo test_advdiff_mpi test_gs_implicit_mpi test_projection_mpi test_stokes_mpi
            test_stokes_solid_mpi test_poiseuille_mpi test_distributed_stokes test_advection_mpi
            test_navier_stokes_mpi test_ns_solid_mpi test_gather_vti_mpi test_reductions_mpi
            test_multigrid_mpi)
    add_executable(${t} tests/${t}.cu)
    target_include_directories(${t} PRIVATE src ${TPX_INCLUDE})
    target_link_libraries(${t} PRIVATE CUDA::cudart MPI::MPI_CXX)
    set_target_properties(${t} PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES "120")
  endforeach()

  foreach(np IN ITEMS 1 2 4)
    add_test(NAME mac_halo_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mac_halo> ${MPIEXEC_POSTFLAGS})
    add_test(NAME advdiff_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_advdiff_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME gs_implicit_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_gs_implicit_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME projection_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_projection_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME stokes_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_stokes_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME stokes_solid_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_stokes_solid_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME poiseuille_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_poiseuille_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME distributed_stokes_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_distributed_stokes> ${MPIEXEC_POSTFLAGS})
    add_test(NAME advection_op_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_advection_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME navier_stokes_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_navier_stokes_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME ns_solid_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_ns_solid_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME gather_vti_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_gather_vti_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME reductions_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_reductions_mpi> ${MPIEXEC_POSTFLAGS})
    add_test(NAME multigrid_np${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_multigrid_mpi> ${MPIEXEC_POSTFLAGS})
  endforeach()

  # Runnable demo (writes a ParaView .vti); CI runs only a tiny smoke case.
  add_executable(demo_flow_sphere tests/demo_flow_sphere.cu)
  target_include_directories(demo_flow_sphere PRIVATE src ${TPX_INCLUDE})
  target_link_libraries(demo_flow_sphere PRIVATE CUDA::cudart MPI::MPI_CXX)
  set_target_properties(demo_flow_sphere PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES "120")
  add_test(NAME demo_flow_sphere_smoke
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                   $<TARGET_FILE:demo_flow_sphere> 16 3 ${CMAKE_CURRENT_BINARY_DIR}/smoke.vti
                   ${MPIEXEC_POSTFLAGS})
endif()
