#ifndef OPTIX_EDGE_TRIANGLE_H
#define OPTIX_EDGE_TRIANGLE_H

#include "optix_types.h"
#include <cuda_runtime.h>
__align__(8) struct HitResult
{
    int edgeIndex;
    int faceIndex;
};

struct Params
{
    const float3* vertexs;
	const int2* edgeIndex;
    HitResult* hitResults;
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