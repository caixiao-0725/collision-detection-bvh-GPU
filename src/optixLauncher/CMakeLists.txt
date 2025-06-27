
set(NAME optixLauncher)
cuda_compile_and_embed(embedded_ptx_code edgeTriangles.cu)

add_library(${NAME} ${embedded_ptx_code} 
	optixLauncher.h 
	optixLauncher.cpp
	optixLauncher.cu
	edgeTriangles.h
)

target_link_libraries(${NAME} Vector)
