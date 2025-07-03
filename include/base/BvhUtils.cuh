#ifndef _BVHUTILS_H_
#define _BVHUTILS_H_

#include "origin.h"
#include "bv.h"
#include "device_launch_parameters.h"
#include "atomicFunctions.cuh"
#include "lbvh.h"
#include <cuda_runtime.h>
namespace Lczx {
	namespace BvhUtils {
		using  uint = unsigned int;
		int const K_THREADS = 256;
		int const K_REDUCTION_LAYER = 5;
		int const K_REDUCTION_NUM = 1 << K_REDUCTION_LAYER;
		int const K_REDUCTION_MODULO = K_REDUCTION_NUM - 1;

		int const aabbBits = 14;
		int const indexBits = 64-3*aabbBits;
		int const offset3 = aabbBits * 3;
		int const offset2 = aabbBits * 2;
		int const offset1 = aabbBits * 1;
		ullint const indexMask = 0xFFFFFFFFFFFFFFFFu << offset3;
		uint const aabbMask = 0xFFFFFFFFu >> (32 - aabbBits);
		uint const MaxIndex = 0xFFFFFFFFFFFFFFFFu >> offset3;
		
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

		__global__ void calcMaxBVFromBox(int size, const BOX* box, BOX* _bv) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			int warpTid = threadIdx.x % 32;
			int warpId = (threadIdx.x >> 5);
			int warpNum = ((blockDim.x) >> 5);
			//if (idx >= size) return;

			__shared__ AABB aabbData[K_THREADS >> 5];

			BOX temp;
			if (idx < size) {
				temp = box[idx];
			}
			else {
				temp = BOX(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
			}
			__syncthreads();
			
			
			//if (blockIdx.x == gridDim.x - 1)
			//{
			//	//tidNum = numbers - idof;
			//	warpNum = ((size - blockIdx.x * blockDim.x + 31) >> 5);
			//}
			//else
			//{
			//	warpNum = ((blockDim.x) >> 5);
			//}

			for (int i = 1; i < 32; i = (i << 1))
			{
				float tempMinX = __shfl_down_sync(0xffffffff, temp._min.x, i);
				float tempMinY = __shfl_down_sync(0xffffffff, temp._min.y, i);
				float tempMinZ = __shfl_down_sync(0xffffffff, temp._min.z, i);
				float tempMaxX = __shfl_down_sync(0xffffffff, temp._max.x, i);
				float tempMaxY = __shfl_down_sync(0xffffffff, temp._max.y, i);
				float tempMaxZ = __shfl_down_sync(0xffffffff, temp._max.z, i);

				//printf("%d  %f   %f \n", idx, temp._min.x, tempMinX);
				temp._min.x = __mm_min(temp._min.x, tempMinX);
				temp._min.y = __mm_min(temp._min.y, tempMinY);
				temp._min.z = __mm_min(temp._min.z, tempMinZ);
				temp._max.x = __mm_max(temp._max.x, tempMaxX);
				temp._max.y = __mm_max(temp._max.y, tempMaxY);
				temp._max.z = __mm_max(temp._max.z, tempMaxZ);

			}
			//if (idx >= size) return;
			if (warpTid == 0)
			{
				aabbData[warpId] = temp;
			}
			__syncthreads();
			if (threadIdx.x >= warpNum)
				return;

			if (warpNum > 1)
			{
				//	tidNum = warpNum;
				temp = aabbData[threadIdx.x];

				//	warpNum = ((tidNum + 31) >> 5);
				for (int i = 1; i < warpNum; i = (i << 1))
				{
					float tempMinX = __shfl_down_sync(0xffffffff, temp._min.x, i);
					float tempMinY = __shfl_down_sync(0xffffffff, temp._min.y, i);
					float tempMinZ = __shfl_down_sync(0xffffffff, temp._min.z, i);
					float tempMaxX = __shfl_down_sync(0xffffffff, temp._max.x, i);
					float tempMaxY = __shfl_down_sync(0xffffffff, temp._max.y, i);
					float tempMaxZ = __shfl_down_sync(0xffffffff, temp._max.z, i);
					temp._min.x = __mm_min(temp._min.x, tempMinX);
					temp._min.y = __mm_min(temp._min.y, tempMinY);
					temp._min.z = __mm_min(temp._min.z, tempMinZ);
					temp._max.x = __mm_max(temp._max.x, tempMaxX);
					temp._max.y = __mm_max(temp._max.y, tempMaxY);
					temp._max.z = __mm_max(temp._max.z, tempMaxZ);
				}
			}

			if (threadIdx.x == 0) {
				float* ptrLowerBound = reinterpret_cast<float*>(&(_bv[0]._min));
				float* ptrUpperBound = reinterpret_cast<float*>(&(_bv[0]._max));
				atomicMinf(ptrLowerBound, temp._min.x);
				atomicMinf(ptrLowerBound + 1, temp._min.y);
				atomicMinf(ptrLowerBound + 2, temp._min.z);
				atomicMaxf(ptrUpperBound, temp._max.x);
				atomicMaxf(ptrUpperBound + 1, temp._max.y);
				atomicMaxf(ptrUpperBound + 2, temp._max.z);
			}
		}

		__global__ void calcMaxBV(int size, const vec3i* _faces, const vec3f* _vertices, BOX* _bv) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= size) return;
			if (idx == 0) {
				_bv[0] = BOX(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
			}
			//for (; idx < size; idx += gridDim.x * blockDim.x) {
			__shared__ vec3f sMemAABBMin[K_THREADS];
			__shared__ vec3f sMemAABBMax[K_THREADS];
			BOX bv{};
			vec3i pair = _faces[idx];

			vec3f a1 = _vertices[pair.x];
			vec3f a2 = _vertices[pair.y];
			vec3f a3 = _vertices[pair.z];

			bv.combines(a1.x, a1.y, a1.z);
			bv.combines(a2.x, a2.y, a2.z);
			bv.combines(a3.x, a3.y, a3.z);
			

			sMemAABBMin[threadIdx.x] = bv._min;
			sMemAABBMax[threadIdx.x] = bv._max;
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

		__global__ void calcMaxBVWarpShuffle(int size, const vec3i* _faces, const vec3f* _vertices, BOX* _bv) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			int warpTid = threadIdx.x % 32;
			int warpId = (threadIdx.x >> 5);
			int warpNum;
			if (idx >= size) return;
			if (idx == 0) {
				_bv[0] = BOX(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
			}

			__shared__ AABB aabbData[K_THREADS>>5];

			BOX temp;
			vec3i pair = _faces[idx];

			vec3f a1 = _vertices[pair.x];
			vec3f a2 = _vertices[pair.y];
			vec3f a3 = _vertices[pair.z];

			temp.combines(a1.x, a1.y, a1.z);
			temp.combines(a2.x, a2.y, a2.z);
			temp.combines(a3.x, a3.y, a3.z);
			
			__syncthreads();

			for (int i = 1; i < 32; i = (i << 1))
			{
				float tempMinX = __shfl_down_sync(0xffffffff, temp._min.x, i);
				float tempMinY = __shfl_down_sync(0xffffffff, temp._min.y, i);
				float tempMinZ = __shfl_down_sync(0xffffffff, temp._min.z, i);
				float tempMaxX = __shfl_down_sync(0xffffffff, temp._max.x, i);
				float tempMaxY = __shfl_down_sync(0xffffffff, temp._max.y, i);
				float tempMaxZ = __shfl_down_sync(0xffffffff, temp._max.z, i);
				temp._min.x = __mm_min(temp._min.x, tempMinX);
				temp._min.y = __mm_min(temp._min.y, tempMinY);
				temp._min.z = __mm_min(temp._min.z, tempMinZ);
				temp._max.x = __mm_max(temp._max.x, tempMaxX);
				temp._max.y = __mm_max(temp._max.y, tempMaxY);
				temp._max.z = __mm_max(temp._max.z, tempMaxZ);
			}

			if (blockIdx.x == gridDim.x - 1)
			{
				//tidNum = numbers - idof;
				warpNum = ((size - blockIdx.x * blockDim.x + 31) >> 5);
			}
			else
			{
				warpNum = ((blockDim.x) >> 5);
			}

			if (warpTid == 0)
			{
				aabbData[warpId] = temp;
			}
			__syncthreads();
			if (threadIdx.x >= warpNum)
				return;

			if (warpNum > 1)
			{
				//	tidNum = warpNum;
				temp = aabbData[threadIdx.x];

				//	warpNum = ((tidNum + 31) >> 5);
				for (int i = 1; i < warpNum; i = (i << 1))
				{
					float tempMinX = __shfl_down_sync(0xffffffff, temp._min.x, i);
					float tempMinY = __shfl_down_sync(0xffffffff, temp._min.y, i);
					float tempMinZ = __shfl_down_sync(0xffffffff, temp._min.z, i);
					float tempMaxX = __shfl_down_sync(0xffffffff, temp._max.x, i);
					float tempMaxY = __shfl_down_sync(0xffffffff, temp._max.y, i);
					float tempMaxZ = __shfl_down_sync(0xffffffff, temp._max.z, i);
					temp._min.x = __mm_min(temp._min.x, tempMinX);
					temp._min.y = __mm_min(temp._min.y, tempMinY);
					temp._min.z = __mm_min(temp._min.z, tempMinZ);
					temp._max.x = __mm_max(temp._max.x, tempMaxX);
					temp._max.y = __mm_max(temp._max.y, tempMaxY);
					temp._max.z = __mm_max(temp._max.z, tempMaxZ);
				}
			}

			if (threadIdx.x == 0) {
				float* ptrLowerBound = reinterpret_cast<float*>(&(_bv[0]._min));
				float* ptrUpperBound = reinterpret_cast<float*>(&(_bv[0]._max));
				atomicMinf(ptrLowerBound, temp._min.x);
				atomicMinf(ptrLowerBound + 1, temp._min.y);
				atomicMinf(ptrLowerBound + 2, temp._min.z);
				atomicMaxf(ptrUpperBound, temp._max.x);
				atomicMaxf(ptrUpperBound + 1, temp._max.y);
				atomicMaxf(ptrUpperBound + 2, temp._max.z);
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
			vec3i pair = _faces[idx];
			vec3f a1 = _vertices[pair.x];
			vec3f a2 = _vertices[pair.y];
			vec3f a3 = _vertices[pair.z];
			
			bv.combines(a1.x, a1.y, a1.z);
			bv.combines(a2.x, a2.y, a2.z);
			bv.combines(a3.x, a3.y, a3.z);

			const vec3f c = bv.center();
			const vec3f offset = c - scene[0]._min;
			codes[idx] = morton3D(offset.x / scene[0].width(), offset.y / scene[0].height(), offset.z / scene[0].depth());
		}

		__global__ void calcMCsFromBox(int size, const BOX* box, BOX* scene, uint* codes) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= size) return;
			BOX bv = box[idx];

			const vec3f c = bv.center();
			const vec3f offset = c - scene[0]._min;
			codes[idx] = morton3D(offset.x / scene[0].width(), offset.y / scene[0].height(), offset.z / scene[0].depth());
		}

        /// incoherent access, thus poor performance
        __global__ void calcInverseMapping(int size, int* map, int* invMap) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= size) return;
            invMap[map[idx]] = idx;
        }

		__global__ void buildPrimitives(int size,int* _primIdx,BOX* _primBox, int* _primMap,const vec3i* _faces,const vec3f* _vertices) {	///< update idx-th _bxs to idx-th leaf
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= size) return;
			//for (; idx < size; idx += gridDim.x * blockDim.x) {
			int newIdx = _primMap[idx];
			BOX bv;
			auto v = _vertices[_faces[idx].x];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].y];
			bv.combines(v.x, v.y, v.z);
			v = _vertices[_faces[idx].z];
			bv.combines(v.x, v.y, v.z);
			//_prims.vida(newIdx) = _faces[idx].x;
			//_prims.vidb(newIdx) = _faces[idx].y;
			//_prims.vidc(newIdx) = _faces[idx].z;
			_primIdx[newIdx] = idx;
			_primBox[newIdx] = bv;
		}

		__global__ void buildPrimitivesFromBox(int size, int* _primIdx, BOX* _primBox, int* _primMap, const BOX* box) {	///< update idx-th _bxs to idx-th leaf
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= size) return;
			//for (; idx < size; idx += gridDim.x * blockDim.x) {
			int newIdx = _primMap[idx];
			BOX bv = box[idx];
			//_prims.vida(newIdx) = _faces[idx].x;
			//_prims.vidb(newIdx) = _faces[idx].y;
			//_prims.vidc(newIdx) = _faces[idx].z;
			_primIdx[newIdx] = idx;
			_primBox[newIdx] = bv;
		}

		__device__ __host__ __forceinline__ void UnzipQuantilizedNode(ulonglong2& qaabb, bvhNodeV1& aabb, vec3f& origin, vec3f& delta) {

			ullint temp = (qaabb.x >> 45) & 0x7FFFF;
			if (temp == 0x7FFFF) {
				aabb.lc = -1;
			}
			else {
			aabb.lc = static_cast<int>(temp);
			}

			temp = (qaabb.x >> 26) & 0x7FFFF;
			if (temp == 0x7FFFF) {
				aabb.escape = -1;
			}
			else {
				aabb.escape = static_cast<int>(temp);
			}
			aabb.bound._min.x = origin.x + static_cast<float>((qaabb.x >> 11) & 0x7FFF) * delta.x;
			temp = (qaabb.x & 0x7FF) << 4;
			temp |= (qaabb.y >> 60);
			aabb.bound._min.y = origin.y + static_cast<float>(temp) * delta.y;
			aabb.bound._min.z = origin.z + static_cast<float>((qaabb.y >> 45) & 0x7FFF) * delta.z;
			aabb.bound._max.x = origin.x + static_cast<float>((qaabb.y >> 30) & 0x7FFF) * delta.x;
			aabb.bound._max.y = origin.y + static_cast<float>((qaabb.y >> 15) & 0x7FFF) * delta.y;
			aabb.bound._max.z = origin.z + static_cast<float>(qaabb.y & 0x7FFF) * delta.z;
		}

		__global__ void Unzip(bvhNodeV1* node, ulonglong2* qNode,AABB* scene, int num) {
			const int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= num) return;
			vec3f origin = scene[0]._min;
			vec3f delta = scene[0]._max;
			ulonglong2 qaabb = qNode[idx];
			/*if (idx == 18) {
				printf("%f  %f  %f\n", origin.x, delta.x, static_cast<float>((qaabb.y >> 30) & 0x7FFF));
			}*/
			UnzipQuantilizedNode(qaabb, node[idx], origin, delta);
		}


		__global__ void buildIntNodes(int size, uint* _depths,
			int* _lvs_lca,int* _lvs_metric,uint* _lvs_par,uint* _lvs_mark, BOX* _lvs_box,
			int* _tks_rc,int*_tks_lc,int* _tks_range_y, int* _tks_range_x,uint* _tks_mark, BOX* _tks_box, uint* _flag,int* _tks_par) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= size) return;
			//_tks_range_x[idx] = -1;
			//__syncthreads();
			_lvs_lca[idx] = -1, _depths[idx] = 0;
			int		l = idx - 1, r = idx;	///< (l, r]
			bool	mark;
			
			if (l >= 0)	mark = _lvs_metric[l] < _lvs_metric[r]; //determine direction
			else		mark = false;
			int		cur = mark ? l : r;

			//if (cur == 254)printf("%d  %d  %d  %d  %d\n", idx, mark, _lvs_metric[l], _lvs_metric[r], cur);
			_lvs_par[idx] = cur;
			if (mark)	_tks_rc[cur] = idx, _tks_range_y[cur] = idx, atomicOr(&_tks_mark[cur], 0x00000002), _lvs_mark[idx] = 0x00000007;
			else		_tks_lc[cur] = idx, _tks_range_x[cur] = idx, atomicOr(&_tks_mark[cur], 0x00000001), _lvs_mark[idx] = 0x00000003;
			//__threadfence();
			while (atomicAdd(&_flag[cur], 1) == 1) {
				//_tks.update(cur, _lvs);	/// Update
				//_tks.refit(cur, _lvs);	/// Refit
				int chl = _tks_lc[cur];
				int chr = _tks_rc[cur];
				uint temp_mark = _tks_mark[cur];
				if (temp_mark & 1) {
					_tks_box[cur] = _lvs_box[chl];
				}
				else {
					_tks_box[cur] = _tks_box[chl];
				}
				if (temp_mark & 2) {
					_tks_box[cur].combines(_lvs_box[chr]);
				}
				else {
					_tks_box[cur].combines(_tks_box[chr]);
				}

				_tks_mark[cur] &= 0x00000007;
				//if (idx < 10) printf("%d   %d\n", idx, _tks_mark[cur]);
				//if (_tks_range_x[cur] == 0) printf("cur:%d  %d  %d   %d   %d\n", cur, _tks_range_x[252], _tks_range_y[252], _tks_lc[252], _tks_rc[252]);
				l = _tks_range_x[cur] - 1, r = _tks_range_y[cur];
				_lvs_lca[l + 1] = cur/*, _tks.rcd(cur) = ++_lvs.rcl(r)*/, _depths[l + 1]++;
				if (l >= 0)	mark = _lvs_metric[l] < _lvs_metric[r];	///< true when right child, false otherwise
				else		mark = false;

				if (l + 1 == 0 && r == size - 1) {
					_tks_par[cur] = -1;
					_tks_mark[cur] &= 0xFFFFFFFB;
					break;
				}

				int par = mark ? l : r;
				_tks_par[cur] = par;
				if (mark)	_tks_rc[par] = cur, _tks_range_y[par] = r, atomicAnd(&_tks_mark[par], 0xFFFFFFFD), _tks_mark[cur] |= 0x00000004;
				else		_tks_lc[par] = cur, _tks_range_x[par] = l + 1, atomicAnd(&_tks_mark[par], 0xFFFFFFFE), _tks_mark[cur] &= 0xFFFFFFFB;
				__threadfence();
				cur = par;
			}
		}

		__global__ void calcIntNodeOrders(int size, int* _tks_lc, int* _lcas, uint* _depths, uint* _offsets, int* _tkMap) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= size) return;
			//for (; idx < size; idx += gridDim.x * blockDim.x) {
			int node = _lcas[idx], depth = _depths[idx], id = _offsets[idx];
			//if (node == 874)printf("%d\n", idx);
			if (node != -1) {
				for (; depth--; node = _tks_lc[node]) {
					_tkMap[node] = id++;
				}
			}
			//}
		}

		__global__ void updateBvhExtNodeLinks(int size, const int* _mapTable, int* _lcas, uint* _pars) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= size) return;
			int ori;
			_pars[idx] = _mapTable[_pars[idx]];
			if ((ori = _lcas[idx]) != -1)
				_lcas[idx] = _mapTable[ori] << 1;
			else
				_lcas[idx] = idx << 1 | 1;
			//if (_lvs.getrca(idx - (size - 1)) != -1)
			//	_lvs.rca(idx - (size - 1)) = _mapTable[_lvs.getrca(idx - (size - 1))] << 1;
			//else
			//	_lvs.rca(idx - (size - 1)) = idx - (size - 1) << 1 | 1;
		}

		__global__ void reorderIntNode(int intSize, const int* _tkMap,
		int* _unorderedTks_lc,int* _unorderedTks_rc,uint* _unorderedTks_mark,int* _unorderedTks_par,int* _unorderedTks_rangex,int* _unorderedTks_rangey,AABB* _unorderedTks_box,
		int* _tks_lc, int* _tks_rc, uint* _tks_mark, int* _tks_par, int* _tks_rangex, int* _tks_rangey, AABB* _tks_box
		) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= intSize) return;
			int newId = _tkMap[idx];
			uint mark = _unorderedTks_mark[idx];

			_tks_lc[newId] = mark & 1 ? _unorderedTks_lc[idx] : _tkMap[_unorderedTks_lc[idx]];
			_tks_rc[newId] = mark & 2 ? _unorderedTks_rc[idx] : _tkMap[_unorderedTks_rc[idx]];
			_tks_mark[newId] = mark;
			int mark_ = _unorderedTks_par[idx];
			
			_tks_par[newId] = mark_ != -1 ? _tkMap[mark_] : -1;
			_tks_rangex[newId] = _unorderedTks_rangex[idx];
			_tks_rangey[newId] = _unorderedTks_rangey[idx];
			_tks_box[newId] = _unorderedTks_box[idx];
		}

		__global__ void reorderIntNodeV1(int intSize, const int* _tkMap,int* _lvs_lca, AABB* _lvs_box,
			int* _unorderedTks_lc, uint* _unorderedTks_mark,  int* _unorderedTks_rangey, AABB* _unorderedTks_box,
			bvhNodeV1* _nodes
		) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= intSize + 1) return;

			bvhNodeV1 node;
			node.lc = -1;
			int escape = _lvs_lca[idx + 1];
			
			if (escape == -1) {
				node.escape = -1;
			}
			else {
				int bLeaf = escape & 1;
				escape >>= 1;
				node.escape = escape + (bLeaf ? intSize : 0);
			}
			node.bound = _lvs_box[idx];
			

			_nodes[idx + intSize] = node;

			if (idx >= intSize) return;

			bvhNodeV1 internalNode;
			int newId = _tkMap[idx];
			uint mark = _unorderedTks_mark[idx];

			internalNode.lc = mark & 1 ? _unorderedTks_lc[idx] + intSize: _tkMap[_unorderedTks_lc[idx]];
			internalNode.bound = _unorderedTks_box[idx];

			int internalEscape = _lvs_lca[_unorderedTks_rangey[idx] + 1];

			if (internalEscape == -1) {
				internalNode.escape = -1;
			}
			else {
				int bLeaf = internalEscape & 1;
				internalEscape >>= 1;
				internalNode.escape = internalEscape + (bLeaf ? intSize : 0);
			}
			_nodes[newId] = internalNode;
		}


		__device__ __host__ __forceinline__ void quantilizeAABB(ulonglong2& qaabb, AABB& aabb, vec3f& origin, vec3f& delta) {			
			//ullint a = 0;
			//printf("%llu  %f  %f  %f\n", static_cast<ullint>((aabb._max.x - origin.x) / delta.x) << offset2, aabb._max.x, ceilf((aabb._max.x - origin.x) / delta.x),(aabb._max.x - origin.x) / delta.x);
			qaabb.x |= static_cast<ullint>((aabb._min.x - origin.x) / delta.x) << offset2;
			qaabb.x |= static_cast<ullint>((aabb._min.y - origin.y) / delta.y) << offset1;
			qaabb.x |= static_cast<ullint>((aabb._min.z - origin.z) / delta.z);

			qaabb.y |= static_cast<ullint>(ceilf((aabb._max.x - origin.x) / delta.x)) << offset2;
			//printf("%llu %llu\n", qaabb.y , static_cast<ullint>(ceilf((aabb._max.x - origin.x) / delta.x)) << offset2);
			qaabb.y |= static_cast<ullint>(ceilf((aabb._max.y - origin.y) / delta.y)) << offset1;
			qaabb.y |= static_cast<ullint>(ceilf((aabb._max.z - origin.z) / delta.z));
			
		}

		__global__ void reorderQuantilizedNode(int intSize, const int* _tkMap, int* _lvs_lca, AABB* _lvs_box,
			int* _unorderedTks_lc, uint* _unorderedTks_mark, int* _unorderedTks_rangey, AABB* _unorderedTks_box,
			AABB* _scene_box,ulonglong2* _nodes
		) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= intSize + 1) return;
			vec3f origin = _scene_box[0]._min;
			vec3f delta = _scene_box[0]._max - origin;
			//if (idx == 0) {
			//	printf("scene x :%f  %f \n", _scene_box[0]._min.x, _scene_box[0]._min.y);
			//	printf("de :%f  %f \n", delta.x, delta.y);
			//}
			delta /= ((1 << aabbBits) - 2);

			ulonglong2 node{0,0};
			quantilizeAABB(node, _lvs_box[idx], origin, delta);


			node.x |= indexMask;
			int escape = _lvs_lca[idx + 1];
			
			if (escape == -1) {
				node.y |= indexMask;
			}
			else {
				int bLeaf = escape & 1;
				escape >>= 1;
				escape += (bLeaf ? intSize : 0);
				node.y |= *reinterpret_cast<ullint*>(&escape) << offset3;
			}
			
			//printf("%llu  %llu\n", node.x,node.y);

			_nodes[idx + intSize] = node;

			if (idx >= intSize) return;

			int newId = _tkMap[idx];
			uint mark = _unorderedTks_mark[idx];

			ulonglong2 internalNode{0,0};
			//if (newId == 0) {
			//	printf("%llu  %f  %f  %f %f\n", static_cast<ullint>((_unorderedTks_box[idx]._max.x - origin.x) / delta.x) << offset2, _unorderedTks_box[idx]._max.x, ceilf((_unorderedTks_box[idx]._max.x - origin.x) / delta.x), (_unorderedTks_box[idx]._max.x - origin.x) , delta.x);
			//	printf("%f  %f\n", _unorderedTks_box[idx]._min.x, _unorderedTks_box[idx]._max.x);
			//}
			quantilizeAABB(internalNode, _unorderedTks_box[idx], origin, delta);
			//internalNode.x &= 0x00001FFFFFFFFFFF; // clear the escape bits
			//internalNode.y &= 0x00001FFFFFFFFFFF;

			//printf("%d  %llu\n", newId, internalNode.y);
			int lc = mark & 1 ? _unorderedTks_lc[idx] + intSize : _tkMap[_unorderedTks_lc[idx]];
			internalNode.x |= ((ullint)lc << offset3);

			int internalEscape = _lvs_lca[_unorderedTks_rangey[idx] + 1];

			if (internalEscape == -1) {
				internalNode.y |= indexMask;
			}
			else {
				int bLeaf = internalEscape & 1;
				internalEscape >>= 1;
				internalEscape += (bLeaf ? intSize : 0);
				internalNode.y |= *reinterpret_cast<ullint*>(&internalEscape) << offset3;
			}
			
			_nodes[newId] = internalNode;
			if (idx == intSize-1) {
				_scene_box[0]._max = delta;
			}
		}

		__device__ __host__ __forceinline__ void AABB2AABBhalf(const AABB& aabb,AABBhalf& aabb_half) {

			aabb_half.x.x = __float2half_rd(aabb._min.x);
			aabb_half.x.y = __float2half_ru(aabb._max.x);
			aabb_half.y.x = __float2half_rd(aabb._min.y);
			aabb_half.y.y = __float2half_ru(aabb._max.y);
			aabb_half.z.x = __float2half_rd(aabb._min.z);
			aabb_half.z.y = __float2half_ru(aabb._max.z);
			//AABB temp;
			//temp._min.x = __half2float(aabb_half.x.x);
			//temp._max.x = __half2float(aabb_half.x.y);
			//temp._min.y = __half2float(aabb_half.y.x);
			//temp._max.y = __half2float(aabb_half.y.y);
			//temp._min.z = __half2float(aabb_half.z.x);
			//temp._max.z = __half2float(aabb_half.z.y);
			//
			//printf("min: %f %f %f  max: %f %f %f\n", aabb._min.x, aabb._min.y, aabb._min.z, aabb._max.x, aabb._max.y, aabb._max.z);
			//printf("min: %f %f %f  max: %f %f %f\n", temp._min.x, temp._min.y, temp._min.z, temp._max.x, temp._max.y, temp._max.z);

			return;
		}

		__global__ void reorderQNodeV1(int intSize, const int* _tkMap, int* _lvs_lca, AABB* _lvs_box,
			int* _unorderedTks_lc, uint* _unorderedTks_mark, int* _unorderedTks_rangey, AABB* _unorderedTks_box,
			int* _escape ,qNode* _nodes
		) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= intSize + 1) return;

			int escape = _lvs_lca[idx + 1];

			if (escape != -1) 
			{
				int bLeaf = escape & 1;
				escape >>= 1;
				escape += (bLeaf ? intSize : 0);
			}
			//_escape[idx + intSize] = escape;
			
			qNode node;
			node.lc = -1;
			node.escape = escape;
			AABB2AABBhalf(_lvs_box[idx], node.bound);

			_nodes[idx + intSize] = node;

			if (idx >= intSize) return;
			
			int newId = _tkMap[idx];
			
			int internalEscape = _lvs_lca[_unorderedTks_rangey[idx] + 1];
			
			if (internalEscape != -1) {
				int bLeaf = internalEscape & 1;
				internalEscape >>= 1;
				internalEscape += (bLeaf ? intSize : 0);
			}
			//_escape[newId] = internalEscape;
			
			qNode internalNode;
			
			internalNode.lc = _unorderedTks_mark[idx] & 1 ? _unorderedTks_lc[idx] + intSize : _tkMap[_unorderedTks_lc[idx]];
			AABB2AABBhalf(_unorderedTks_box[idx], internalNode.bound);
			internalNode.escape = internalEscape;
			_nodes[newId] = internalNode;
		}

		__device__ __host__ __forceinline__ void AABB2AABBull(AABB& aabb, ulonglong2& ll) {
			__half min_x;
			ullint min_x_uint;
			min_x = __float2half_rd(aabb._min.x);
			min_x_uint = *reinterpret_cast<ullint*>(&min_x);
			ll.x |= (min_x_uint & 0x8000) << 44;
			ll.x |= (min_x_uint & 0x3FFF) << 30;

			__half min_y;
			ullint min_y_uint;
			min_y = __float2half_rd(aabb._min.y);
			min_y_uint = *reinterpret_cast<ullint*>(&min_y);
			ll.x |= (min_y_uint & 0x8000) << 29;
			ll.x |= (min_y_uint & 0x3FFF) << 15;

			__half min_z;
			ullint min_z_uint;
			min_z = __float2half_rd(aabb._min.z);
			min_z_uint = *reinterpret_cast<ullint*>(&min_z);
			ll.x |= (min_z_uint & 0x8000) << 14;
			ll.x |= (min_z_uint & 0x3FFF);

			__half max_x;
			ullint max_x_uint;
			max_x = __float2half_ru(aabb._max.x);
			max_x_uint = *reinterpret_cast<ullint*>(&max_x);
			ll.y |= (max_x_uint & 0x8000) << 44;
			ll.y |= (max_x_uint & 0x3FFF) << 30;

			__half max_y;
			ullint max_y_uint;
			max_y = __float2half_ru(aabb._max.y);
			max_y_uint = *reinterpret_cast<ullint*>(&max_y);
			ll.y |= (max_y_uint & 0x8000) << 29;
			ll.y |= (max_y_uint & 0x3FFF) << 15;

			__half max_z;
			ullint max_z_uint;
			max_z = __float2half_ru(aabb._max.z);
			max_z_uint = *reinterpret_cast<ullint*>(&max_z);
			ll.y |= (max_z_uint & 0x8000) << 14;
			ll.y |= (max_z_uint & 0x3FFF);

		}

		__global__ void reorderUllintNode(int intSize, const int* _tkMap, int* _lvs_lca, AABB* _lvs_box,
			int* _unorderedTks_lc, uint* _unorderedTks_mark, int* _unorderedTks_rangey, AABB* _unorderedTks_box,
			ulonglong2* _nodes
		) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= intSize + 1) return;
			ulonglong2 node{0,0};
			AABB2AABBull(_lvs_box[idx], node);
			int escape = _lvs_lca[idx + 1];			
			node.x &= 0x00001FFFFFFFFFFF; // clear the escape bits
			node.y &= 0x00001FFFFFFFFFFF;
			if (escape == -1) {
				node.y |= (ullint)0x7FFFF << 45;
				
			}else
			{
				int bLeaf = escape & 1;
				escape >>= 1;
				escape += (bLeaf ? intSize : 0);
				node.y |= static_cast<ullint>(escape) << 45;				
			}
			
			node.x |= (ullint)0x7FFFF << 45;
			

			_nodes[idx + intSize] = node;

			if (idx >= intSize) return;

			ulonglong2 internalNode{ 0,0 };

			AABB2AABBull(_unorderedTks_box[idx], internalNode);
			internalNode.x &= 0x00001FFFFFFFFFFF; // clear the escape bits
			internalNode.y &= 0x00001FFFFFFFFFFF;
			int newId = _tkMap[idx];

			int internalEscape = _lvs_lca[_unorderedTks_rangey[idx] + 1];

			if (internalEscape == -1) {
				internalNode.y |= (ullint)0x7FFFF << 45;
			} 
			else{
				int bLeaf = internalEscape & 1;
				internalEscape >>= 1;
				internalEscape += (bLeaf ? intSize : 0);
				internalNode.y |= static_cast<ullint>(internalEscape) << 45;
			}
			
			int lc = _unorderedTks_mark[idx] & 1 ? _unorderedTks_lc[idx] + intSize : _tkMap[_unorderedTks_lc[idx]];

			if (lc == -1) {
				internalNode.x |= (ullint)0x7FFFF << 45;
			}
			else {
				internalNode.x |= static_cast<ullint>(lc) << 45;
			}
			_nodes[newId] = internalNode;
		}

		__device__ __host__ __forceinline__ void AABB2AABBullV1(AABB& aabb, ulonglong2& ll) {
			__half min_x;
			ullint min_x_uint;
			min_x = __float2half_rd(aabb._min.x);
			min_x_uint = *reinterpret_cast<ullint*>(&min_x);
			ll.x |= (min_x_uint & 0x7FFF) << 30;

			__half min_y;
			ullint min_y_uint;
			min_y = __float2half_rd(aabb._min.y);
			min_y_uint = *reinterpret_cast<ullint*>(&min_y);
			ll.x |= (min_y_uint & 0x7FFF) << 15;

			__half min_z;
			ullint min_z_uint;
			min_z = __float2half_rd(aabb._min.z);
			min_z_uint = *reinterpret_cast<ullint*>(&min_z);
			ll.x |= (min_z_uint & 0x7FFF);

			__half max_x;
			ullint max_x_uint;
			max_x = __float2half_ru(aabb._max.x);
			max_x_uint = *reinterpret_cast<ullint*>(&max_x);
			ll.y |= (max_x_uint & 0x7FFF) << 30;

			__half max_y;
			ullint max_y_uint;
			max_y = __float2half_ru(aabb._max.y);
			max_y_uint = *reinterpret_cast<ullint*>(&max_y);
			ll.y |= (max_y_uint & 0x7FFF) << 15;

			__half max_z;
			ullint max_z_uint;
			max_z = __float2half_ru(aabb._max.z);
			max_z_uint = *reinterpret_cast<ullint*>(&max_z);
			ll.y |= (max_z_uint & 0x7FFF);

		}

		__global__ void reorderUllintNodeV1(int intSize, const int* _tkMap, int* _lvs_lca, AABB* _lvs_box,
			int* _unorderedTks_lc, uint* _unorderedTks_mark, int* _unorderedTks_rangey, AABB* _unorderedTks_box,
			ulonglong2* _nodes
		) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= intSize + 1) return;
			ulonglong2 node{ 0,0 };
			AABB2AABBullV1(_lvs_box[idx], node);
			int escape = _lvs_lca[idx + 1];
			node.x &= 0x00001FFFFFFFFFFF; // clear the escape bits
			node.y &= 0x00001FFFFFFFFFFF;
			if (escape == -1) {
				node.y |= (ullint)0x7FFFF << 45;

			}
			else
			{
				int bLeaf = escape & 1;
				escape >>= 1;
				escape += (bLeaf ? intSize : 0);
				node.y |= static_cast<ullint>(escape) << 45;
			}

			node.x |= (ullint)0x7FFFF << 45;


			_nodes[idx + intSize] = node;

			if (idx >= intSize) return;

			ulonglong2 internalNode{ 0,0 };

			AABB2AABBullV1(_unorderedTks_box[idx], internalNode);
			internalNode.x &= 0x00001FFFFFFFFFFF; // clear the escape bits
			internalNode.y &= 0x00001FFFFFFFFFFF;
			int newId = _tkMap[idx];

			int internalEscape = _lvs_lca[_unorderedTks_rangey[idx] + 1];

			if (internalEscape == -1) {
				internalNode.y |= (ullint)0x7FFFF << 45;
			}
			else {
				int bLeaf = internalEscape & 1;
				internalEscape >>= 1;
				internalEscape += (bLeaf ? intSize : 0);
				internalNode.y |= static_cast<ullint>(internalEscape) << 45;
			}

			int lc = _unorderedTks_mark[idx] & 1 ? _unorderedTks_lc[idx] + intSize : _tkMap[_unorderedTks_lc[idx]];

			if (lc == -1) {
				internalNode.x |= (ullint)0x7FFFF << 45;
			}
			else {
				internalNode.x |= static_cast<ullint>(lc) << 45;
			}
			_nodes[newId] = internalNode;
		}

		__global__ void reorderIntNodeSoa(int intSize, const int* _tkMap,
			int* _unorderedTks_lc, int* _unorderedTks_rc, uint* _unorderedTks_mark, int* _unorderedTks_par, int* _unorderedTks_rangex, int* _unorderedTks_rangey, AABB* _unorderedTks_box,
			int* _tks_lc, int* _tks_rc, uint* _tks_mark, int* _tks_par, bvhNodeV2* _nodes) 
		{
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= intSize) return;
			int newId = _tkMap[idx];
			uint mark = _unorderedTks_mark[idx];

			_tks_lc[newId] = mark & 1 ? _unorderedTks_lc[idx] : _tkMap[_unorderedTks_lc[idx]];
			_tks_rc[newId] = mark & 2 ? _unorderedTks_rc[idx] : _tkMap[_unorderedTks_rc[idx]];
			_tks_mark[newId] = mark;
			int mark_ = _unorderedTks_par[idx];
			
			_tks_par[newId] = mark_ != -1 ? _tkMap[mark_] : -1;

			bvhNodeV2 node;
			node.rangex = _unorderedTks_rangex[idx];
			node.rangey = _unorderedTks_rangey[idx];
			node.bound = _unorderedTks_box[idx];
			
			_nodes[newId] = node;
		}

		__global__ void reorderMergeNode(int intSize, const int* _tkMap,
			bvhNode* _mergeNode,BOX* _extBox,BOX* _intBox,
			int* _unorderedTks_lc, int* _unorderedTks_rc, 
			uint* _unorderedTks_mark,int* _unorderedTks_par) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= intSize) return;
			int newId = _tkMap[idx];
			uint mark = _unorderedTks_mark[idx];
			bvhNode temp;
			//temp.mark = mark;
			temp.lc = mark & 1 ? ((uint)_unorderedTks_lc[idx]) | 0x80000000 : _tkMap[_unorderedTks_lc[idx]];
			temp.rc = mark & 2 ? ((uint)_unorderedTks_rc[idx]) | 0x80000000 : _tkMap[_unorderedTks_rc[idx]];
			temp.bounds[0] = mark & 1 ? _extBox[_unorderedTks_lc[idx]] : _intBox[_unorderedTks_lc[idx]];
			temp.bounds[1] = mark & 2 ? _extBox[_unorderedTks_rc[idx]] : _intBox[_unorderedTks_rc[idx]];
	
			int mark_ = _unorderedTks_par[idx];
			temp.par = mark_ != -1 ? _tkMap[mark_] : 0;
			
			_mergeNode[newId] = temp;
		}

		template<bool SELF>
		__global__ void pureBvhStacklessCD(uint travPrimSize, const BOX* _box, 
			const uint* _lvs_par, const int* _lvs_idx, const BOX* _lvs_box,const int* _lvs_lca,
			const BOX* _tks_box, const int* _tks_rangex, const int* _tks_rangey,
			int* _cpNum, int* _cpRes) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= travPrimSize) return;

			int	lbd;
			int st = 0;
			int count = 0;
			
			const BOX bv = _box[idx];
			do {
				int t = st & 1;
				st >>= 1;
				if (!t)	for (t = _lvs_par[lbd = _tks_rangex[st]]; st <= t && _tks_box[st].overlaps(bv); st++);
				else	t = st - 1, lbd = st;

				if (st > t) {
					if (_lvs_box[lbd].overlaps(bv)){
						int temp_idx = _lvs_idx[lbd];
						if (SELF) {
							if (temp_idx < idx) {
								_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
								count++;
							}
						}
						else {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}
					}
					st = _lvs_lca[lbd + 1];
				}
				else {
					st = _lvs_lca[_tks_rangey[st] + 1];
				}
			} while (st != -1);
			_cpNum[idx] = count;
		}

		template<bool SELF>
		__global__ void AosBvhStacklessCD(uint travPrimSize, const BOX* _box,
			const uint* _lvs_par, const int* _lvs_idx, const BOX* _lvs_box, const int* _lvs_lca,
			const bvhNodeV2* _nodes,
			int* _cpNum, int* _cpRes) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= travPrimSize) return;

			int	lbd;
			int st = 0;
			int count = 0;
			bvhNodeV2 node;
			const BOX bv = _box[idx];
			do {
				int t = st & 1;
				st >>= 1;
				if (!t)
				{
					node = _nodes[st];
					lbd = node.rangex;
					for (t = _lvs_par[lbd]; st <= t && node.bound.overlaps(bv); node = _nodes[st++]);
				}	
				else	t = st - 1, lbd = st;
				if (st > t) {
					if (_lvs_box[lbd].overlaps(bv)) {
						int temp_idx = _lvs_idx[lbd];
						if (SELF) {
							if (temp_idx < idx) {
								_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
								count++;
							}
						}
						else {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}
					}
					st = _lvs_lca[lbd + 1];
				}
				else {
					st = _lvs_lca[node.rangey + 1];
				}
			} while (st != -1);
			_cpNum[idx] = count;
		}

		template<bool SELF>
		__global__ void AosBvhStacklessCDV1(uint Size, const BOX* _box,
			const int intSize, const int* _lvs_idx,
			const bvhNodeV1* _nodes,
			int* _cpNum, int* _cpRes) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= Size) return;
			idx = _lvs_idx[idx];
			bvhNodeV1 node;
			int st = 0;
			int count = 0;
			const BOX bv = _box[idx];
			do {
				node = _nodes[st];
				if (node.bound.overlaps(bv)) {
					if (node.lc == -1) {
						int temp_idx = _lvs_idx[st - intSize];
						if (SELF) {
							if (temp_idx < idx){
								_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
								count++;
							}
						}
						else {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}
					
						st = node.escape;
					}
					else {
						st = node.lc;
					}
				}
				else {
					st = node.escape;
				}
			} while (st != -1);
			_cpNum[idx] = count;
		}

		template<bool SELF>
		__global__ void qNodeCD(uint Size, const BOX* _box,
			const int intSize, const int* _lvs_idx,
			const int* _escape,const qNode* _nodes,
			int* _cpNum, int* _cpRes) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= Size) return;
			qNode node;
			//int escape;
			int st = 0;
			int count = 0;
			const BOX bv = _box[idx];
			do {
				node = _nodes[st];
				//escape = _escape[st];
				if (node.bound.overlaps(bv)) {
					if (node.lc == -1) {
						int temp_idx = _lvs_idx[st - intSize];
						if (SELF) {
							if (temp_idx < idx) {
								_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
								count++;
							}
						}
						else {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}

						st = node.escape;
					}
					else {
						st = node.lc;
					}
				}
				else {
					st = node.escape;
				}
			} while (st != -1);
			_cpNum[idx] = count;
		}
		
		__device__ __host__ __forceinline__ bool overlapsLonglong(const ulonglong2& a,const AABB& b) {
			uint temp_u;
			__half2 temp;
			float2 data;
			temp_u =   (uint)(a.x >> 30) & 0x3fff;
			temp_u |= ((uint)(a.x >> 44) & 0x1) << 15;
			temp_u |= ((uint)(a.y >> 30) & 0x3fff) << 16;
			temp_u |= ((uint)(a.y >> 44) & 0x1) << 31;
			memcpy(&temp, &temp_u, sizeof(temp_u));
			
			data = __half22float2(temp);
			if (b._min.x > data.y || b._max.x < data.x) return false;
			
			temp_u =   (uint)(a.x >> 15) & 0x3fff;
			temp_u |= ((uint)(a.x >> 29) & 0x1) << 15;
			temp_u |= ((uint)(a.y >> 15) & 0x3fff) << 16;
			temp_u |= ((uint)(a.y >> 29) & 0x1) << 31;
			memcpy(&temp, &temp_u, sizeof(temp_u));
			
			data = __half22float2(temp);
			if (b._min.y > data.y || b._max.y < data.x) return false;
			
			temp_u = (uint)(a.x) & 0x3fff;
			temp_u |= ((uint)(a.x >> 14) & 0x1) << 15;
			temp_u |= ((uint)(a.y) & 0x3fff) << 16;
			temp_u |= ((uint)(a.y >> 14) & 0x1) << 31;
			memcpy(&temp, &temp_u, sizeof(temp_u));
			
			data = __half22float2(temp);
			if (b._min.z > data.y || b._max.z < data.x) return false;

			return true;
		}

		__global__ void AABB2half(const AABB* aabb, AABBhalf* aabb_half,int num) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= num) return;
			AABB2AABBhalf(aabb[idx], aabb_half[idx]);
			return;
		}

		template<bool SELF>
		__global__ void UllintCD(uint Size, const BOX* _box,
			const int intSize, const int* _lvs_idx,
			const ulonglong2* _nodes,
			int* _cpNum, int* _cpRes) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= Size) return;
			ulonglong2 node;
			uint st = 0;
			int count = 0;
			uint lc;
			uint escape;
			const BOX bv = _box[idx];
			do {
				node = _nodes[st];
				lc = node.x >> 45;
				escape = node.y >> 45;
				//printf("%u %u\n", lc,escape);
				if (overlapsLonglong(node,bv)) {
					if (lc == 0x7FFFFu) {
						int temp_idx = _lvs_idx[st - intSize];
						if (SELF) {
							if (temp_idx < idx) {
								_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
								count++;
							}
						}
						else {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}
		
						st = escape;
					}
					else {
						st = lc;
					}
				}
				else {
					st = escape;
				}
			} while (st != 0x7FFFFu);
			_cpNum[idx] = count;
		}

		__device__ __host__ __forceinline__ bool overlapsLonglong2half(const ulonglong2& a, const AABBhalf& b) {
			uint temp_a,temp_b;
			temp_a = (a.x >> 30) & 0x7fff;
			temp_b = *reinterpret_cast<const unsigned short*>(&(b.x.y));
			//printf("%u  %hu\n", temp_b, *(reinterpret_cast<const unsigned short*>(&(b.x.y))));
			if (temp_a > temp_b) return false;

			temp_a = (a.y >> 30) & 0x7fff;
			temp_b = *reinterpret_cast<const unsigned short*>(&(b.x.x));
			if (temp_a < temp_b) return false;

			temp_a = (a.x >> 15) & 0x7fff;
			temp_b = *reinterpret_cast<const unsigned short*>(&(b.y.y));
			if (temp_a > temp_b) return false;

			temp_a = (a.y >> 15) & 0x7fff; 
			temp_b = *reinterpret_cast<const unsigned short*>(&(b.y.x));
			if (temp_a < temp_b) return false;

			temp_a = a.x & 0x7fff;
			temp_b = *reinterpret_cast<const unsigned short*>(&(b.z.y));
			if (temp_a > temp_b) return false;

			temp_a = a.y & 0x7fff;
			temp_b = *reinterpret_cast<const unsigned short*>(&(b.z.x));
			if (temp_a < temp_b) return false;

			return true;
		}


		template<bool SELF>
		__global__ void UllintCDV1(uint Size, const AABBhalf* _box,
			const int intSize, const int* _lvs_idx,
			const ulonglong2* _nodes,
			int* _cpNum, int* _cpRes) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx >= Size) return;
			ulonglong2 node;
			uint st = 0;
			int count = 0;
			uint lc;
			uint escape;
			const AABBhalf bv = _box[idx];
			do {
				node = _nodes[st];
				lc = node.x >> 45;
				escape = node.y >> 45;
				//printf("%u %u\n", lc,escape);
				if (overlapsLonglong2half(node, bv)) {
					if (lc == 0x7FFFFu) {
						int temp_idx = _lvs_idx[st - intSize];
						if (SELF) {
							if (temp_idx < idx) {
								_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
								count++;
							}
						}
						else {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}

						st = escape;
					}
					else {
						st = lc;
					}
				}
				else {
					st = escape;
				}
			} while (st != 0x7FFFFu);
			_cpNum[idx] = count;
		}

		__device__ __host__ __forceinline__ bool overlapsLonglong2int(const ulonglong2& a, const intAABB& b) {
			int temp_a, temp_b;
			temp_a = (a.x >> offset2) & aabbMask;
			temp_b = b._max.x;
			if (temp_a > temp_b) return false;

			temp_a = (a.y >> offset2) & aabbMask;
			temp_b = b._min.x;
			if (temp_a < temp_b) return false;

			temp_a = (a.x >> offset1) & aabbMask;
			temp_b = b._max.y;
			if (temp_a > temp_b) return false;

			temp_a = (a.y >> offset1) & aabbMask;
			temp_b = b._min.y;
			if (temp_a < temp_b) return false;

			temp_a = a.x & aabbMask;
			temp_b = b._max.z;
			if (temp_a > temp_b) return false;

			temp_a = a.y & aabbMask;
			temp_b = b._min.z;
			if (temp_a < temp_b) return false;

			return true;
		}

		template<bool SELF>
		__global__ void quantilizedStacklessCD(uint Size, const BOX* _box,
			const int intSize, const int* _lvs_idx,
			const AABB* scene,const ulonglong2* _nodes,
			int* _cpNum, int* _cpRes) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= Size) return;
			idx = _lvs_idx[idx];
			vec3f origin = scene[0]._min;
			vec3f delta = scene[0]._max;
			const AABB bv_ = _box[idx];
			intAABB bv;
			bv.convertFrom(bv_, origin,delta);
			ulonglong2 node;
			int st = 0;
			int count = 0;
			uint lc;
			uint escape;
			do {
				node = _nodes[st];
				lc = node.x >> offset3;
				escape = node.y >> offset3;
				if (overlapsLonglong2int(node,bv)) {
					if (lc == MaxIndex) {
						int temp_idx = _lvs_idx[st - intSize];
						if (SELF) {
							if (temp_idx < idx) {
								_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
								count++;
							}
						}
						else {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}

						st = escape;
					}
					else {
						st = lc;
					}
				}
				else {
					st = escape;
				}
			} while (st != MaxIndex);
			_cpNum[idx] = count;
		}

		template<bool SELF>
		__global__ void pureBvhStackCD(uint travPrimSize, const BOX* _box,
			const uint* _lvs_par, const int* _lvs_idx, const BOX* _lvs_box, const int* _lvs_lca,
			const BOX* _tks_box, const int* _tks_lc, const int* _tks_rc, const uint* _tks_mark,
			int* _cpNum, int* _cpRes) {
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= travPrimSize) return;
			const BOX bv = _box[idx];

			uint stack[32];			// This is dynamically sized through templating
			uint* stackPtr = stack;
			*(stackPtr++) = 0;					// Push
			int idxL, idxR,count;
			uint mark;
			while (stackPtr != stack) {
				uint nodeIdx = *(--stackPtr);	// Pop
				
				idxL = _tks_lc[nodeIdx];
				idxR = _tks_rc[nodeIdx];
				mark = _tks_mark[nodeIdx];
				
				if (mark & 1) {					
					if (_lvs_box[idxL].overlaps(bv)) {
						int temp_idx = _lvs_idx[idxL];
						if (SELF) {
							if (temp_idx < idx) {
								_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
								count++;
							}
						}
						else {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}
					}
				}
				else {
					if (bv.overlaps(_tks_box[idxL])) {
						*(stackPtr++) = idxL;
					}
				}

				if (mark & 2) {
					if (_lvs_box[idxR].overlaps(bv)){
						int temp_idx = _lvs_idx[idxR];
						if (SELF) {
							if (temp_idx < idx) {
								_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
								count++;
							}
						}
						else {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}
					}
				}
				else {
					if (bv.overlaps(_tks_box[idxR])) {
						*(stackPtr++) = idxR;
					}
				}

			}
			_cpNum[idx] = count;
		}

		template<bool SELF>
		__global__ void pureMergeBvhStackCD(uint travPrimSize, const BOX* _box,
			bvhNode* _nodes, int* _lvs_idx,
			int* _cpNum, int* _cpRes) {
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= travPrimSize) return;
			const BOX bv = _box[idx];
			int count = 0;

			int stack[32];			// This is dynamically sized through templating
			int* stackPtr = stack;
			*(stackPtr++) = 0;					// Push		
			while (stackPtr != stack) {

				uint32_t nodeIdx = *(--stackPtr);	// Pop
				bool isLeaf = nodeIdx & 0x80000000;
				nodeIdx = nodeIdx & 0x7FFFFFFF;

				if (isLeaf) {
					int temp_idx = _lvs_idx[nodeIdx];
					if (SELF) {
						if (temp_idx < idx) {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}
					}
					else {
						_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
						count++;
					}
					continue;
				}

				auto node = _nodes[nodeIdx];

				if (bv.overlaps(node.bounds[0])) {
					*(stackPtr++) = node.lc;
				}

				if (bv.overlaps(node.bounds[1])) {
					*(stackPtr++) = node.rc;
				}

			}
			_cpNum[idx] = count;
		}

		template<bool SELF>
		__global__ void pureMergeBvhStackSortElementCD(uint travPrimSize, const BOX* _box,
			bvhNode* _nodes, int* _lvs_idx,
			int* _cpNum, int* _cpRes) {
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= travPrimSize) return;
			idx = _lvs_idx[idx];
			const BOX bv = _box[idx];
			int count = 0;

			int stack[32];			// This is dynamically sized through templating
			int* stackPtr = stack;
			*(stackPtr++) = 0;					// Push
			while (stackPtr != stack) {

				uint32_t nodeIdx = *(--stackPtr);	// Pop
				bool isLeaf = nodeIdx & 0x80000000;
				nodeIdx = nodeIdx & 0x7FFFFFFF;

				if (isLeaf) {
					int temp_idx = _lvs_idx[nodeIdx];
					if (SELF) {
						if (temp_idx < idx) {
							_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
							count++;
						}
					}
					else {
						_cpRes[count + idx * MAX_CD_NUM_PER_VERT] = temp_idx;
						count++;
					}
					continue;
				}

				auto node = _nodes[nodeIdx];

				if (bv.overlaps(node.bounds[0])) {
					*(stackPtr++) = node.lc;
				}

				if (bv.overlaps(node.bounds[1])) {
					*(stackPtr++) = node.rc;
				}

			}
			_cpNum[idx] = count;
		}

		__global__ void refitIntNode(int size, 
			int* _lvs_par, int* _tks_par,uint* _tks_flag,int* _tks_lc,int* _tks_rc,uint* _tks_mark,
			BOX* _lvs_box,BOX* _tks_box) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= size) return;
			//_tks_box[idx] = BOX(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
			//__syncthreads();
			//for (; idx < size; idx += gridDim.x * blockDim.x) {
			int par = _lvs_par[idx];
			while (atomicAdd(&_tks_flag[par], 1) == 1) {
				int chl = _tks_lc[par];
				int chr = _tks_rc[par];
				if (_tks_mark[par] & 1) {
					_tks_box[par]=_lvs_box[chl];
				}
				else {
					_tks_box[par]=_tks_box[chl];
				}
				if (_tks_mark[par] & 2) {
					_tks_box[par].combines(_lvs_box[chr]);
				}
				else {
					_tks_box[par].combines(_tks_box[chr]);
				}
				__threadfence();
				//_tks_box[par].combines(_lvs_box[chl]);
				//_tks_box[par].combines(_lvs_box[chr]);
				if (par == 0) break;
				par = _tks_par[par];
			}
			//}
		}

	

		// Uses CUDA intrinsics for counting leading zeros
		__device__ inline int commonUpperBits(const uint64_t lhs, const uint64_t rhs) {
			return ::__clzll(lhs ^ rhs);
		}

		// Merges morton code with its index to output a sorted unique 64-bit key.
		__device__ inline uint64_t mergeIdx(const uint code, const int idx) {
			return ((uint64_t)code << 32ul) | (uint64_t)idx;
		}

		__device__ inline vec2i determineRange(uint const* mortonCodes,
			const uint numObjs, uint idx) {

			// This is the root node
			if (idx == 0)
				return vec2i(0, numObjs - 1);

			// Determine direction of the range
			const uint64_t selfCode = mergeIdx(mortonCodes[idx], idx);
			const int lDelta = commonUpperBits(selfCode, mergeIdx(mortonCodes[idx - 1], idx - 1));
			const int rDelta = commonUpperBits(selfCode, mergeIdx(mortonCodes[idx + 1], idx + 1));
			const int d = (rDelta > lDelta) ? 1 : -1;

			// Compute upper bound for the length of the range
			const int minDelta = thrust::min(lDelta, rDelta);
			int lMax = 2;
			int i;
			while ((i = idx + d * lMax) >= 0 && i < numObjs) {
				if (commonUpperBits(selfCode, mergeIdx(mortonCodes[i], i)) <= minDelta) break;
				lMax <<= 1;
			}

			// Find the exact range by binary search
			int t = lMax >> 1;
			int l = 0;
			while (t > 0) {
				i = idx + (l + t) * d;
				if (0 <= i && i < numObjs)
					if (commonUpperBits(selfCode, mergeIdx(mortonCodes[i], i)) > minDelta)
						l += t;
				t >>= 1;
			}

			unsigned int jdx = idx + l * d;
			if (d < 0) thrust::swap(idx, jdx); // Make sure that idx < jdx
			return vec2i(idx, jdx);
		}

		__device__ inline uint findSplit(uint const* mortonCodes,
			const uint first, const uint last) {

			const uint64_t firstCode = mergeIdx(mortonCodes[first], first);
			const uint64_t lastCode = mergeIdx(mortonCodes[last], last);
			const int deltaNode = commonUpperBits(firstCode, lastCode);

			// Binary search for split position
			int split = first;
			int stride = last - first;
			do {
				stride = (stride + 1) >> 1;
				const int middle = split + stride;
				if (middle < last)
					if (commonUpperBits(firstCode, mergeIdx(mortonCodes[middle], middle)) > deltaNode)
						split = middle;
			} while (stride > 1);

			return split;
		}

		// Builds out the internal nodes of the LBVH
		__global__ void lbvhBuildInternalKernel(bvhNode* nodes,
			uint* leafParents, uint const* mortonCodes, int const* objIDs, int numObjs) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs - 1) return;

			vec2i range = determineRange(mortonCodes, numObjs, tid);
			//nodes[tid].fence = (tid == range.x) ? range.y : range.x;

			const int gamma = findSplit(mortonCodes, range.x, range.y);

			// Left and right children are neighbors to the split point
			// Check if there are leaf nodes, which are indexed behind the (numObj - 1) internal nodes
			if (range.x == gamma) {
				leafParents[gamma] = (uint32_t)tid;
				range.x = gamma | 0x80000000;
			}
			else {
				range.x = gamma;
				nodes[range.x].par = (uint32_t)tid;
			}

			if (range.y == gamma + 1) {
				leafParents[gamma + 1] = (uint32_t)tid | 0x80000000;
				range.y = (gamma + 1) | 0x80000000;
			}
			else {
				range.y = gamma + 1;
				nodes[range.y].par = (uint32_t)tid | 0x80000000;
			}

			nodes[tid].lc = range.x;
			nodes[tid].rc = range.y;
		}

		// Refits the AABBs of the internal nodes
		__global__ void mergeNodeRefit(bvhNode* nodes,
			uint* leafParents, AABB* aabbs, int* objIDs, int* flags, int size) {
		
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= size) return;

			AABB last = aabbs[tid];
			uint parent = leafParents[tid];
			while (true) {
				int isRight = (parent & 0x80000000) != 0;
				parent = parent & 0x7FFFFFFF;
				nodes[parent].bounds[isRight] = last;

				// Exit if we are the first thread here
				int flag = atomicAdd(flags + parent, 1);
				if (!flag) return;

				// Ensure memory coherency before we read.
				__threadfence();

				if (parent == 0) break;
				last.combines(nodes[parent].bounds[1 - isRight]);
				parent = nodes[parent].par;
			}
		}
	}
}


#endif