# Q6 diagnosis (case A, validation/Q6_DIAG.md): replay ONE momentum-consistent VoF advection on
# a dumped solver state and write the half-shifted colour C^e it ends with (not exposed to Python).
# cmake -S . -B <build> -DCMAKE_PREFIX_PATH=<suite>/extern/install/host-openmp && cmake --build <build>
cmake_minimum_required(VERSION 3.24)
project(vofmom_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
find_package(Kokkos CONFIG REQUIRED)
set(FLOW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
add_executable(vofmom_probe vofmom_probe.cpp)
target_include_directories(vofmom_probe PRIVATE ${FLOW_ROOT}/src ${FLOW_ROOT}/../core/include)
target_link_libraries(vofmom_probe PRIVATE Kokkos::kokkos)
