#ifndef _EXTNODE_H_
#define _EXTNODE_H_

#include "DeviceHostVector.h"

#include "bv.h"

namespace CXE {
	class ExtNodeArray
	{
	public:
		void setup(const int Prims,const int ExtSize);
		int buildExtNodes(int primsize);
		void clearExtNodes(int size);
		void calcSplitMetrics(int extsize);

		uint* getMtCodes() { return static_cast<uint*>(_attribs[MTCODE]); }
		int* getIdx() { return static_cast<int*>(_attribs[IDX]); }
		BOX* getBox() { return static_cast<BOX*>(_attribs[AABB]); }
		uint* getMarks() { return static_cast<uint*>(_attribs[MARK]); }
		int* getExtIds() { return static_cast<int*>(_attribs[EXT_NO]); }
		int* getMetrics() { return static_cast<int*>(_attribs[SPLIT_METRIC]); }

		__device__ int getmetric(int i) const { return ((int*)_attribs[SPLIT_METRIC])[i]; }

		__device__ int& idx(int i) { return ((int*)_attribs[IDX])[i]; }
		__device__ uint& mtcode(int i) { return ((uint*)_attribs[MTCODE])[i]; }
		__device__ BOX& box(int i) { return ((BOX*)_attribs[AABB])[i]; }
		__device__ int& lca(int i) { return ((int*)_attribs[LCA])[i]; }
		__device__ int& metric(int i) { return ((int*)_attribs[SPLIT_METRIC])[i]; }
		__device__ int& par(int i) { return ((int*)_attribs[PAR])[i]; }
		__device__ uint& mark(int i) { return ((uint*)_attribs[MARK])[i]; }

		enum { MTCODE, IDX , AABB , MARK, EXT_NO, PAR, SPLIT_METRIC,LCA};
		void* _attribs[10];
		int _prims;
		int _extSize;
	};
}

#endif // !_EXTNODE_H_

