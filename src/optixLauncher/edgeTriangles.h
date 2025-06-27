#ifndef OPTIX_EDGE_TRIANGLE_H
#define OPTIX_EDGE_TRIANGLE_H

#include "optix_types.h"
#include <cuda_runtime.h>
#include "bv.h"

struct Params
{
    const float3* vertexs;
	const int2* edgeIndex;
	const vec3f* vertexBuffer;
	const vec3i* indexBuffer;
	float thickness;
	int* cdIndex;
	int* cdBuffer;
    OptixTraversableHandle handle;
};

struct RayGenData
{
	//float rayCollisionRadius;
};

struct MissData
{
	//
};

struct HitGroupData
{
	//float rayCollisionRadius;
};


#endif