#ifndef _BVHUTILS_H_
#define _BVHUTILS_H_

#include "origin.h"
#include "bv.h"
#include "device_launch_parameters.h"
#include "atomicFunctions.cuh"
namespace CXE {
	namespace BvhUtils {
		using  uint = unsigned int;
		int const K_THREADS = 256;
		int const K_REDUCTION_LAYER = 5;
		int const K_REDUCTION_NUM = 1 << K_REDUCTION_LAYER;
		int const K_REDUCTION_MODULO = K_REDUCTION_NUM - 1;
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

		__global__ void calcMaxBV(int size, const vec3i* _faces, const vec3f* _vertices, BOX* _bv) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= size) return;
			//for (; idx < size; idx += gridDim.x * blockDim.x) {
			__shared__ vec3f sMemAABBMin[K_THREADS];
			__shared__ vec3f sMemAABBMax[K_THREADS];

			BOX bv{};
			auto v = _vertices[_faces[idx].x];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].y];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].z];
			bv.combines(v.x, v.y, v.z);
			

			sMemAABBMin[threadIdx.x] = bv._min;
			sMemAABBMax[threadIdx.x] = bv._max;
			//atomicMinf(&_bv->_min.x, bv._min.x);
			//atomicMinf(&_bv->_min.y, bv._min.y);
			//atomicMinf(&_bv->_min.z, bv._min.z);
			//atomicMinf(&_bv->_max.x, bv._max.x);
			//atomicMinf(&_bv->_max.y, bv._max.y);
			//atomicMinf(&_bv->_max.z, bv._max.z);
			__syncthreads();

			if ((threadIdx.x & K_REDUCTION_MODULO) == 0)
			{
				vec3f lowerBound = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
				vec3f upperBound = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
				for (int i = 0; i < K_REDUCTION_NUM; i++)
				{
					const unsigned int newIdx = idx + i;
					if (newIdx >= size) break;
					vec3f aMin = sMemAABBMin[threadIdx.x + i];
					vec3f aMax = sMemAABBMax[threadIdx.x + i];
					lowerBound.x = fminf(lowerBound.x, aMin.x);
					lowerBound.y = fminf(lowerBound.y, aMin.y);
					lowerBound.z = fminf(lowerBound.z, aMin.z);
					upperBound.x = fmaxf(upperBound.x, aMax.x);
					upperBound.y = fmaxf(upperBound.y, aMax.y);
					upperBound.z = fmaxf(upperBound.z, aMax.z);
				}
				// minmax
				float* ptrLowerBound = reinterpret_cast<float*>(&(_bv[0]._min));
				float* ptrUpperBound = reinterpret_cast<float*>(&(_bv[0]._max));

				atomicMinf(ptrLowerBound, lowerBound.x);
				atomicMinf(ptrLowerBound + 1, lowerBound.y);
				atomicMinf(ptrLowerBound + 2, lowerBound.z);
				atomicMaxf(ptrUpperBound, upperBound.x);
				atomicMaxf(ptrUpperBound + 1, upperBound.y);
				atomicMaxf(ptrUpperBound + 2, upperBound.z);

			}
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

		__global__ void calcMCs(int size,const vec3i* _faces,const vec3f* _vertices, BOX* scene, uint* codes) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= size) return;
			//for (; idx < size; idx += gridDim.x * blockDim.x) {
			BOX bv{};
			auto v = _vertices[_faces[idx].x];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].y];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].z];
			bv.combines(v.x, v.y, v.z);
			const vec3f c = bv.center();
			const vec3f offset = c - scene[0]._min;
			codes[idx] = morton3D(offset.x / scene[0].width(), offset.y / scene[0].height(), offset.z / scene[0].depth());
			//}
			if (idx == 0) printf("%f    %f    %f\n ", scene[0]._min.x, scene[0]._min.y, scene[0]._min.z);
		}

        /// incoherent access, thus poor performance
        __global__ void calcInverseMapping(int size, int* map, int* invMap) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= size) return;
            invMap[map[idx]] = idx;
        }

	}

}


#endif