# Native-CUDA RB-GS benchmark (the reference for the Kokkos-vs-CUDA efficiency comparison). Plain
# CUDA, no Kokkos. Build: cmake -S tests/cuda_bench -B build_cuda_bench -DCMAKE_CUDA_ARCHITECTURES=120
cmake_minimum_required(VERSION 3.24)
project(cfd_cuda_bench LANGUAGES CXX CUDA)
add_executable(bench_rbgs_cuda bench_rbgs.cu)
set_target_properties(bench_rbgs_cuda PROPERTIES CUDA_ARCHITECTURES "120")
target_compile_options(bench_rbgs_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
