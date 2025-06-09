cmake_minimum_required(VERSION 3.7)
 
add_executable(optixTest main.cu)

target_link_libraries(optixTest PRIVATE optixLauncher Vector)