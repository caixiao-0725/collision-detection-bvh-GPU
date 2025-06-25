#ifndef _LBVH_H_
#define _LBVH_H_

#include "bv.h"
#include "IntNode.h"
#include "ExtNode.h"
#include "MergeNode.h"

#include "thrust/sequence.h"
#include "thrust/sort.h"
#include "thrust/device_vector.h"
#include "CudaThrustUtils.hpp"
#include "DeviceHostVector.h"

#define MAX_CD_NUM_PER_VERT 64

namespace CXE {

	class Bvh {
	public:
		Bvh() {}
		~Bvh() {}
		void setup(int prim_size,int ext_node_size,int int_node_size);
		void build(const vec3f* vertices,const vec3i* faces);
		void build(const AABB* boxs);
		void query(const AABB* boxs, const uint num,bool self);


		void reorderIntNodes();

		int& primSize() { return _primSize; }
		int& extSize() { return _extSize; }
		int& intSize() { return _intSize; }
		BOX*& bv() { return _bv; }
		ExtNodeArray& lvs() { return _extNodes; }
		IntNodeArray& tks() { return _intNodes; }
		IntNodeArray& untks() { return _unsortedTks; }
		MergeNodeArray& mgs() { return _mergeNodes; }

		int	_primSize, _extSize, _intSize;
		ExtNodeArray _extNodes;
		IntNodeArray _intNodes;
		IntNodeArray _unsortedTks;
		MergeNodeArray _mergeNodes;
		StacklessMergeNodeArray _stacklessMergeNodes;
		StacklessMergeNodeV1Array _stacklessMergeNodesV1;

		BOX* _bv;

		DeviceHostVector<uint> d_keys32;
		DeviceHostVector<int>  d_vals;
		DeviceHostVector<uint>  d_offsetTable;
		DeviceHostVector<int>  d_primMap;				///< map from primitives to leaves
		DeviceHostVector<int>  d_tkMap;

		DeviceHostVector<uint> d_count;
		DeviceHostVector<AABB> d_bv;

		DeviceHostVector<int> _cpNum;
		DeviceHostVector<int2> _cpRes;

		DeviceHostVector<int> _cpNumPerVert;
		DeviceHostVector<int> _cpResPerVert;

		int _type =  4; // 0. SOA stackless query     bottom to top tree build
						// 1. SOA stack 32 query      bottom to top tree build
						// 2. AOS stack 32 query      bottom to top tree build
						// 3. AOS stackless query     bottom to top tree build
						// 4. faster AOS stackless query     bottom to top tree build
						// 11.AOS stack 32 query      binary search tree build
						// 12.AOS stack 32 query      binary search tree build    sort query elements
	};

}

#endif // !_LBVH_H_
