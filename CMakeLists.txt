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
