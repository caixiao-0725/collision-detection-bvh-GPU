#ifndef _LBVH_H_
#define _LBVH_H_

#include "bv.h"
#include "IntNode.h"
#include "ExtNode.h"

#include "thrust/sequence.h"
#include "thrust/sort.h"
#include "thrust/device_vector.h"
#include "CudaThrustUtils.hpp"

namespace CXE {

	class Bvh {
	public:
		Bvh() {}
		~Bvh() {}
		void setup(int prim_size,int ext_node_size,int int_node_size);
		void build(const vec3f* vertices,const vec3i* faces);


		int& primSize() { return _primSize; }
		int& extSize() { return _extSize; }
		int& intSize() { return _intSize; }
		BOX*& bv() { return _bv; }
		ExtNodeArray& lvs() { return _extNodes; }
		IntNodeArray& tks() { return _intNodes; }

		int	_primSize, _extSize, _intSize;
		ExtNodeArray _extNodes;
		IntNodeArray _intNodes;
		BOX* _bv;

		thrust::device_vector<uint> d_keys32;
		thrust::device_vector<int>  d_vals;
		thrust::device_vector<int>  d_primMap;
	};

}

#endif // !_LBVH_H_
