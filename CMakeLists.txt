cmake_minimum_required(VERSION 3.7)

project(bvhcd CXX CUDA)
# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if (WIN32)
  add_definitions(-DNOMINMAX)
endif()

find_program(BIN2C bin2c
  DOC "Path to the cuda-sdk bin2c executable.")

set(PTX_DEST_DIR "${CMAKE_SOURCE_DIR}/assets/ptx/")
file(MAKE_DIRECTORY ${PTX_DEST_DIR}) 

macro(cuda_compile_and_embed output_var cuda_file)
  set(c_var_name ${output_var})
  cuda_compile_ptx(ptx_files ${cuda_file} )
  list(GET ptx_files 0 ptx_file)
  set(embedded_file ${ptx_file}_embedded.c)
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file}
    COMMENT "compiling (and embedding ptx from) ${cuda_file}"
    )
  set(${output_var} ${embedded_file})
  file(COPY ${ptx_files} DESTINATION ${PTX_DEST_DIR})
endmacro()

include_directories(include/base)
include_directories(include/optix9)
include_directories(src/common)
include_directories(src/LBVH)
include_directories(src/optixLauncher)

add_compile_definitions(ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/")

add_subdirectory(src)
add_subdirectory(example)