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

__global__ void setAABBBufferKernel(AABB* aabbBuffer,
	const vec3f* verts,
	const vec3i* indexs, 
	const float thickness, 
	uint numAABB)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numAABB)
		return;
	vec3f v0 = verts[indexs[i].x];
	vec3f v1 = verts[indexs[i].y];
	vec3f v2 = verts[indexs[i].z];

	AABB aabb;
	aabb.combines(v0);
	aabb.combines(v1);
	aabb.combines(v2);
	aabb.enlarges(thickness);

	aabbBuffer[i] = aabb;
}

void setVertexBuffer(float3* gpuVertexBuffer, const void* verts, uint strideInBytes, uint posOffsetInBytes, uint numVertices)
{
	int block = 256;
	int grid = (numVertices + block - 1) / block;
	setVertexBufferKernel << <grid, block >> > (gpuVertexBuffer, verts, strideInBytes, posOffsetInBytes, numVertices);
}


void setAABBBuffer(AABB* AABBBuffer, const vec3f* verts, const vec3i* indexs, const float thickness,uint numAABB)
{
	int block = 256;
	int grid = (numAABB + block - 1) / block;
	setAABBBufferKernel << <grid, block >> > (AABBBuffer, verts, indexs, thickness, numAABB);
}
