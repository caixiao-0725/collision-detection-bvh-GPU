cmake_minimum_required(VERSION 3.5)
 
add_executable(simData main.cu)

target_link_libraries(simData PRIVATE LBVH)