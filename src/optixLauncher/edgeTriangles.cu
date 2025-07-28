#include <assert.h>
#include <vector_types.h>

#include "optix.h"
#include "optix_types.h"

#include "edgeTriangles.h"

#define float3_as_uints(u) __float_as_uint(u.x), __float_as_uint(u.y), __float_as_uint(u.z)

extern "C"
{
	__constant__ Params params;
}


extern "C" __global__ void __miss__ms()
{
	//optixSetPayload_4(0);  // hit
}

static __forceinline__ __device__ vec3f optixGetObjectRayO()
{
	float f0, f1, f2;
	asm("call (%0), _optix_get_object_ray_origin_x, ();" : "=f"(f0) : );
	asm("call (%0), _optix_get_object_ray_origin_y, ();" : "=f"(f1) : );
	asm("call (%0), _optix_get_object_ray_origin_z, ();" : "=f"(f2) : );
	return vec3f(f0, f1, f2);
}

static __forceinline__ __device__ vec3f optixGetObjectRayD()
{
	float f0, f1, f2;
	asm("call (%0), _optix_get_object_ray_direction_x, ();" : "=f"(f0) : );
	asm("call (%0), _optix_get_object_ray_direction_y, ();" : "=f"(f1) : );
	asm("call (%0), _optix_get_object_ray_direction_z, ();" : "=f"(f2) : );
	return vec3f(f0, f1, f2);
}

static __forceinline__ __device__ void trace_edge(OptixTraversableHandle handle,
	float3 ray_origin,
	float3 ray_direction,
	float tmin,
	float tmax)
{
	unsigned int count = 0;
	optixTrace(handle,
		ray_origin,
		ray_direction,
		tmin,
		tmax,
		0.0f,  // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_ENFORCE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,//OPTIX_RAY_FLAG_NONE,
		0,  // SBT offset
		0,  // SBT stride
		0,  // missSBTIndex
		count
	);
	uint3 idx = optixGetLaunchIndex();
	params.cdIndex[idx.x] = count;

}

static __forceinline__ __device__ void trace_point(OptixTraversableHandle handle,
	float3 ray_origin)
{
	unsigned int count = 0;
	optixTrace(handle,
		ray_origin,
		float3{-1.0,0,0},
		0.0f,
		1e-6f,
		0.0f,  // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_ENFORCE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,//OPTIX_RAY_FLAG_NONE,
		0,  // SBT offset
		0,  // SBT stride
		0,  // missSBTIndex
		count);
	uint3 idx = optixGetLaunchIndex();
	params.cdIndex[idx.x] = count;
}

extern "C" __global__ void __raygen__rg_edge()
{
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	unsigned int index = idx.x;

	float3 v0 = params.vertexs[params.edgeIndex[index].x];
	float3 v1 = params.vertexs[params.edgeIndex[index].y];

	const float3 d = float3{v1.x-v0.x,v1.y-v0.y,v1.z-v0.z};
	const float length = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
	trace_edge(params.handle,
		v0,
		d,
		0.00f,           // tmin
		length          // tmax  //default 1.2
		);
}

extern "C" __global__ void __raygen__rg_point()
{
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	unsigned int index = idx.x;

	float3 v0 = params.vertexs[index];

	trace_point(params.handle,
		v0
		);
}


extern "C" __global__ void __anyhit__ch()
{
	//uint3 idx = optixGetLaunchIndex();  // ray's index
	//const uint3 dim = optixGetLaunchDimensions();
	//unsigned int ray_idx = idx.x;
	//optixSetPayload_0(ray_idx);
	//
	//const unsigned int prim_idx = optixGetPrimitiveIndex();
	//optixSetPayload_1(prim_idx);
	
	optixIgnoreIntersection();
}


extern "C" __global__ void __closesthit__ch()
{
	uint3 idx = optixGetLaunchIndex();  // ray's index
	const uint3 dim = optixGetLaunchDimensions();
	unsigned int ray_idx = idx.x;
	optixSetPayload_0(ray_idx);

	const unsigned int prim_idx = optixGetPrimitiveIndex();
	optixSetPayload_1(prim_idx);

	
	//printf("%d   %d\n", ray_idx, prim_idx);

}


extern "C" __global__ void __intersection__is()
{
	//const vec3f e0 = optixGetObjectRayO();
	//const vec3f e1 = optixGetObjectRayD() + e0;
	uint3 idx = optixGetLaunchIndex();  // ray's index
	const uint3 dim = optixGetLaunchDimensions();
	unsigned int ray_idx = idx.x;
	const unsigned int prim_idx = optixGetPrimitiveIndex();

	uint count = optixGetPayload_0();
	vec3i face = params.indexBuffer[prim_idx];
	int2 edge = params.edgeIndex[ray_idx];
	for (int i = 0;i < 3;i++) {
		if (face.raw[i] == edge.x)
			return;
		if (face.raw[i] == edge.y)
			return;
	}
	params.cdBuffer[ray_idx * MAX_CD_NUM_PER_VERT + count] = prim_idx;
	optixSetPayload_0(count + 1);

	//if (face.x != edge.x && face.x != edge.y &&
	//	face.y != edge.x && face.y != edge.y &&
	//	face.z != edge.x && face.z != edge.y)
	//{ 
	//	params.cdBuffer[ray_idx * MAX_CD_NUM_PER_VERT + count] = prim_idx;
	//	optixSetPayload_0(count+1);
	//}
}

extern "C" __global__ void __intersection__pointAABB()
{
	//const vec3f e0 = optixGetObjectRayO();
	//const vec3f e1 = optixGetObjectRayD() + e0;
	uint3 idx = optixGetLaunchIndex();  // ray's index
	const uint3 dim = optixGetLaunchDimensions();
	unsigned int ray_idx = idx.x;
	const unsigned int prim_idx = optixGetPrimitiveIndex();
	uint count = optixGetPayload_0();

	if (ray_idx > prim_idx) {
		float3 v0 = optixGetObjectRayOrigin();
		float3 v1 = params.vertexs[prim_idx];
		float d0 = v0.x - v1.x;
		float d1 = v0.y - v1.y;
		float d2 = v0.z - v1.z;
		if (fabsf(d0) < params.thickness &&
			fabsf(d1) < params.thickness &&
			fabsf(d2) < params.thickness)
		{
			params.cdBuffer[ray_idx * MAX_CD_NUM_PER_VERT + count] = prim_idx;
			optixSetPayload_0(count + 1);
		}

		
	}


}
