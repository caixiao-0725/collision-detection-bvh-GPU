#include "optixLauncher.h"
#include "device_launch_parameters.h"




__global__ void setVertexBufferKernel(float3* vertexBuffer,
	const void* verts,
	uint strideInBytes,
	uint posOffsetInBytes,
	uint numVertices)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numVertices)
		return;

	const float* position = (const float*)((char*)verts + i * strideInBytes + posOffsetInBytes);
	vertexBuffer[i] = make_float3(position[0], position[1], position[2]);
}

void setVertexBuffer(float3* gpuVertexBuffer, const void* verts, uint strideInBytes, uint posOffsetInBytes, uint numVertices)
{
	int block = 256;
	int grid = (numVertices + block - 1) / block;
	setVertexBufferKernel << <grid, block >> > (gpuVertexBuffer, verts, strideInBytes, posOffsetInBytes, numVertices);
}
