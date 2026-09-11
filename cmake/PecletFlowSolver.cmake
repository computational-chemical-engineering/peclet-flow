# The ONE compiled instantiation of `peclet::flow::Solver<Grid>` (suite/docs/QUALITY_PLAN.md §3.G.8).
#
# `Solver<Grid>` is a ~12 k-line class template (517 out-of-line members across the twelve
# `src/flow_ibm_*.hpp` domain headers).  Every consumer used to instantiate all of it at -O3 for
# itself -- 45 test executables plus the nanobind module, the latter for both grids -- so a full
# rebuild cost ~45-50 CPU-minutes and an edit to any domain header invalidated all 45.  The library
# built here compiles it once per grid; `flow_ibm.hpp`'s `extern template` declaration suppresses
# the implicit instantiation in every consumer, which then links against this.
#
# STATIC, not OBJECT, on purpose: the link is inert for a target that never names `Solver` (the
# linker pulls no member in), so the whole test directory can link it without bloating the kernel
# unit tests.
#
# `PECLET_FLOW_MPI` changes the class definition, so it must agree between this library and its
# consumers.  It is a PUBLIC compile definition of the MPI variant, which makes the agreement
# structural rather than a convention: a target that links `peclet_flow_solver_mpi` gets the macro,
# a target that links `peclet_flow_solver` does not, and a target cannot link both.
#
# Code generation is preserved exactly for the module, which is what the no-performance-loss
# condition is judged on: PIC and hidden visibility match what `nanobind_add_module` puts on
# `peclet_flow`, so the solver's calls and globals are resolved as directly as before, and the build
# carries no LTO and no -march, so there was never any cross-TU optimization to lose.

get_filename_component(PECLET_FLOW_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../src" ABSOLUTE)

# peclet_flow_add_solver_library(<target> [MPI])
#   Defines <target> if it does not exist yet (idempotent: the root project defines them before
#   add_subdirectory(), and each test directory defines what it needs when configured standalone).
#   Requires Kokkos::kokkos, PECLET_CORE_INCLUDE, and MPI::MPI_CXX for the MPI variant.
function(peclet_flow_add_solver_library target)
  cmake_parse_arguments(PARSE_ARGV 1 _pfs "MPI" "" "")
  if(TARGET ${target})
    return()
  endif()
  if(NOT PECLET_CORE_INCLUDE)
    message(FATAL_ERROR "peclet_flow_add_solver_library(${target}): PECLET_CORE_INCLUDE is not set")
  endif()
  add_library(${target} STATIC
    "${PECLET_FLOW_SRC_DIR}/flow_solver_staggered.cpp"
    "${PECLET_FLOW_SRC_DIR}/flow_solver_colocated.cpp")
  # PUBLIC: a consumer sees the same headers, the same macro and the same Kokkos as the
  # instantiation did -- an ODR mismatch between the two is then not expressible.
  target_include_directories(${target} PUBLIC "${PECLET_FLOW_SRC_DIR}" "${PECLET_CORE_INCLUDE}")
  target_link_libraries(${target} PUBLIC Kokkos::kokkos)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON
                        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  # HIP/lld: nanobind's hidden visibility leaves the Kokkos SharedAllocationRecord vtables
  # undefined-hidden under hipcc (see the module's CMakeLists); follow the module there.
  if(Kokkos_ENABLE_HIP)
    set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET default)
  endif()
  if(_pfs_MPI)
    target_compile_definitions(${target} PUBLIC PECLET_FLOW_MPI=1)
    target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
  endif()
endfunction()
