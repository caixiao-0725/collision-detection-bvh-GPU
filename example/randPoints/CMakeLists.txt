cmake_minimum_required(VERSION 3.5)
 
add_executable(randPoints main.cu)

target_link_libraries(randPoints PRIVATE LBVH)