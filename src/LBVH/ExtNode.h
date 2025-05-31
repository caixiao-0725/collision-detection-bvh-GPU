#ifndef _EXTNODE_H_
#define _EXTNODE_H_

#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "../common/DeviceHostVector.h"
#include "../common/HelperCuda.h"
#include "../../include/base/origin.h"
#include "../../include/base/bv.h"
#include "../../include/base/CudaThrustUtils.hpp"
#include "../common/HelperMath.h"

namespace CXE {
	class ExtNodeArray
	{
	public:
		void setup(const int Prims,const int ExtSize);
		int buildExtNodes(int primsize);
		void clearExtNodes(int size);
		void calcSplitMetrics(int extsize);

		unsigned int* getMtCodes() { return _mtcode; }
		int* getIdx() { return _idx; }
		BOX* getBox() { return _box; }
		unsigned int* getMarks() { return _mark; }
		int* getExtIds() { return _extId; }
		int* getMetrics() { return _metric; }
		int* getLCA() { return _lca; }

		__device__ int getmetric(int i) const { return _metric.ptr[i]; }
		__device__ int& idx(int i) { return ((int*)_idx.ptr)[i]; }
		__device__ unsigned int& mtcode(int i) { return _mtcode.ptr[i]; }
		__device__ BOX& box(int i) { return _box.ptr[i]; }
		__device__ int& lca(int i) { return _lca.ptr[i]; }
		__device__ int& metric(int i) { return _metric.ptr[i]; }
		__device__ int& par(int i) { return _par.ptr[i]; }
		__device__ unsigned int& mark(int i) { return _mark.ptr[i]; }

		int _prims;
		int _extSize;
		DeviceHostVector<unsigned int> _mtcode;
		DeviceHostVector<int> _idx;
		DeviceHostVector<BOX> _box;
		DeviceHostVector<int> _lca;
		DeviceHostVector<int> _metric;
		DeviceHostVector<int> _par;   //parent node idx
		DeviceHostVector<int> _extId;
		DeviceHostVector<unsigned int> _mark;
	};
}

#endif // !_EXTNODE_H_

