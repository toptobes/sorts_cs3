cmake_minimum_required(VERSION 3.22)

set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")

project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda main.cu main.cuh)

set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
