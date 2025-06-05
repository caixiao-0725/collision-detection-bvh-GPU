cmake_minimum_required(VERSION 3.19)

project(bvhcd CXX CUDA)
# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(include/base)
include_directories(src/common)
include_directories(src/LBVH)

add_compile_definitions(ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/")

add_subdirectory(src)
add_subdirectory(example)