#ifndef ATOMICFUNCTIONS_CUH
#define ATOMICFUNCTIONS_CUH

#include <cuda_runtime_api.h>
#include "origin.h"
__device__ __forceinline__ float atomicMinf(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

__device__ __forceinline__ float atomicMaxf(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

__device__ __forceinline__ float3 atomicAddFloat3(float3* addr, float3 value) {
    float* x = &(addr[0].x);
    float* y = &(addr[0].y);
    float* z = &(addr[0].z);
    float a = atomicAdd(x, value.x);
    float b = atomicAdd(y, value.y);
    float c = atomicAdd(z, value.z);
    return make_float3(a, b, c);
}

__device__ __forceinline__ vec3f atomicAddVec3f(vec3f* addr, vec3f value) {
    float* x = &(addr[0].x);
    float* y = &(addr[0].y);
    float* z = &(addr[0].z);
    float a = atomicAdd(x, value.x);
    float b = atomicAdd(y, value.y);
    float c = atomicAdd(z, value.z);
    return vec3f(a, b, c);
}

__device__ __forceinline__ void gridSync(int SyncCount, volatile unsigned int* LoopCount)
{
    // Grid sync implementation
    bool cta_master = (threadIdx.x == 0);
    bool gpu_master = (blockIdx.x == 0);

    __syncthreads();

    if (cta_master) {
        unsigned int nb = 1;
        if (gpu_master) {
            nb = 0x80000000 - (SyncCount - 1);
        }

        __threadfence();

        unsigned int oldArrive;
        oldArrive = atomicAdd((unsigned int*)LoopCount, nb);

        while ((((oldArrive ^ LoopCount[0]) & 0x80000000) == 0));

        __threadfence();
    }

    __syncthreads();
}

#endif
