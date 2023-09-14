#ifndef __UTILS__H
#define __UTILS__H

#include "origin.h"
using  uint = unsigned int;
__device__ uint expandBits(uint v) {					///< Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

__device__ uint morton3D(float x, float y, float z) {	///< Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
	x = ::fmin(::fmax(x * 1024.0f, 0.0f), 1023.0f);
	y = ::fmin(::fmax(y * 1024.0f, 0.0f), 1023.0f);
	z = ::fmin(::fmax(z * 1024.0f, 0.0f), 1023.0f);
	uint xx = expandBits((uint)x);
	uint yy = expandBits((uint)y);
	uint zz = expandBits((uint)z);
	return (xx * 4 + yy * 2 + zz);
}

__host__ __device__ __forceinline__ void FuncGetContinuousAABB(vec3f a0Curr, vec3f a1Curr, vec3f a2Curr,
    vec3f a0Prev, vec3f a1Prev, vec3f a2Prev, float thickness, vec3f& aMin, vec3f& aMax)
{
    float aMaxX = fmaxf(a0Curr.x, fmaxf(a1Curr.x, a2Curr.x));
    float aMaxY = fmaxf(a0Curr.y, fmaxf(a1Curr.y, a2Curr.y));
    float aMaxZ = fmaxf(a0Curr.z, fmaxf(a1Curr.z, a2Curr.z));
    aMaxX = fmaxf(fmaxf(aMaxX, a0Prev.x), fmaxf(a1Prev.x, a2Prev.x));
    aMaxY = fmaxf(fmaxf(aMaxY, a0Prev.y), fmaxf(a1Prev.y, a2Prev.y));
    aMaxZ = fmaxf(fmaxf(aMaxZ, a0Prev.z), fmaxf(a1Prev.z, a2Prev.z));

    float aMinX = fminf(a0Curr.x, fminf(a1Curr.x, a2Curr.x));
    float aMinY = fminf(a0Curr.y, fminf(a1Curr.y, a2Curr.y));
    float aMinZ = fminf(a0Curr.z, fminf(a1Curr.z, a2Curr.z));
    aMinX = fminf(fminf(aMinX, a0Prev.x), fminf(a1Prev.x, a2Prev.x));
    aMinY = fminf(fminf(aMinY, a0Prev.y), fminf(a1Prev.y, a2Prev.y));
    aMinZ = fminf(fminf(aMinZ, a0Prev.z), fminf(a1Prev.z, a2Prev.z));

    aMin = vec3f(aMinX - thickness, aMinY - thickness, aMinZ - thickness);
    aMax = vec3f(aMaxX + thickness, aMaxY + thickness, aMaxZ + thickness);
}

#endif