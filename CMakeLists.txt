cmake_minimum_required(VERSION 3.19)

project(bvhcd CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(include/base)
include_directories(src/common)
add_subdirectory(src)