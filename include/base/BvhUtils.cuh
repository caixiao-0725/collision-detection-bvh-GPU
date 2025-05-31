#ifndef _BVHUTILS_H_
#define _BVHUTILS_H_

#include "origin.h"
#include "bv.h"
#include "device_launch_parameters.h"
#include "atomicFunctions.cuh"
#include "lbvh.h"
#include <cuda_runtime.h>
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

		__global__ void calcMaxBVFromBox(int size, const BOX* box, BOX* _bv) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			int warpTid = threadIdx.x % 32;
			int warpId = (threadIdx.x >> 5);
			int warpNum;
			if (idx >= size) return;
			if (idx == 0) {
				_bv[0] = BOX(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
			}

			__shared__ AABB aabbData[K_THREADS >> 5];

			BOX temp = box[idx];
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


		__global__ void buildIntNodes(int size, uint* _depths,
			int* _lvs_lca,int* _lvs_metric,int* _lvs_par,uint* _lvs_mark, BOX* _lvs_box,
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

		__global__ void updateBvhExtNodeLinks(int size, const int* _mapTable, int* _lcas, int* _pars) {
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
			//if (newId == 0) printf("%d  %d\n", mark_, idx);
			//_tks.rcd(newId) = _rcls[mark] - _unorderedTks.getrcd(idx);
			_tks_box[newId] = _unorderedTks_box[idx];
		}


		__global__ void pureBvhInterCD(uint travPrimSize, const BOX* _box, 
			const int* _lvs_par, const int* _lvs_idx, const BOX* _lvs_box,const int* _lvs_lca,
			const BOX* _tks_box, const int* _tks_rangex, const int* _tks_rangey,
			int* _cpNum, int2* _cpRes) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= travPrimSize) return;

			int	lbd;
			int st = 0;
			
			const BOX bv = _box[idx];
			do {
				int t = st & 1;
				st >>= 1;
				if (!t)	for (t = _lvs_par[lbd = _tks_rangex[st]]; st <= t && _tks_box[st].overlaps(bv); st++);
				else	t = st - 1, lbd = st;
				if (st > t) {
					if (_lvs_box[lbd].overlaps(bv)){
						//int temp_idx = _lvs_idx[lbd];
						//if(temp_idx < idx)
							//_cpRes[atomicAdd(_cpNum, 1)] = make_int2(temp_idx, idx);
					}
					st = _lvs_lca[lbd + 1];
				}
				else {
					st = _lvs_lca[_tks_rangey[st] + 1];
				}
			} while (st != -1);
		}

		
		__global__ void pureBvhStackCD(uint travPrimSize, const BOX* _box,
			const int* _lvs_par, const int* _lvs_idx, const BOX* _lvs_box, const int* _lvs_lca,
			const BOX* _tks_box, const int* _tks_lc, const int* _tks_rc, const uint* _tks_mark,
			int* _cpNum, int2* _cpRes) {
			int idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (idx >= travPrimSize) return;
			const BOX bv = _box[idx];

			uint32_t stack[32];			// This is dynamically sized through templating
			uint32_t* stackPtr = stack;
			*(stackPtr++) = 0;					// Push
			int idxL, idxR, temp_idx;
			uint mark;
			while (stackPtr != stack) {
				uint32_t nodeIdx = *(--stackPtr);	// Pop
				
				idxL = _tks_lc[nodeIdx];
				idxR = _tks_rc[nodeIdx];
				mark = _tks_mark[nodeIdx];
				
				if (mark & 1) {
					temp_idx = _lvs_idx[idxL];
					if (_lvs_box[idxL].overlaps(bv))
						if (temp_idx < idx)
							_cpRes[atomicAdd(_cpNum, 1)] = make_int2(temp_idx, idx);
				}
				else {
					if (bv.overlaps(_tks_box[idxL])) {
						*(stackPtr++) = idxL;
					}
				}

				if (mark & 2) {
					temp_idx = _lvs_idx[idxR];
					if (_lvs_box[idxR].overlaps(bv))
						if (temp_idx < idx)
							_cpRes[atomicAdd(_cpNum, 1)] = make_int2(temp_idx, idx);
				}
				else {
					if (bv.overlaps(_tks_box[idxR])) {
						*(stackPtr++) = idxR;
					}
				}

			}
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

	}

}


#endif