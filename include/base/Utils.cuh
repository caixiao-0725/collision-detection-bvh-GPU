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

#endif