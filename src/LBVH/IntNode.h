#ifndef _INTNODE_H_
#define _INTNODE_H_

#include "../../include/base/bv.h"
#include "DeviceHostVector.h"

namespace Lczx {
	class IntNodeArray
	{
	public:
		void setup(const int intSize);
		void clearIntNodes(int size);

		__device__ int& lc(int i) { return _lc.ptr[i]; }
		__device__ int& rc(int i) { return _rc.ptr[i]; }
		__device__ int& par(int i){ return _par.ptr[i]; }
		__device__ uint& mark(int i) { return _mark.ptr[i]; }
		__device__ int& rangex(int i) { return _rangex.ptr[i]; }
		__device__ int& rangey(int i) { return _rangey.ptr[i]; }
		__device__ uint& flag(int i) { return _flag.ptr[i]; }

		int _intSize;
		
		DeviceHostVector<int> _lc;    //left child
		DeviceHostVector<int> _rc;    //right child
		DeviceHostVector<int> _par;   //parent
		DeviceHostVector<uint> _mark;
		DeviceHostVector<uint> _flag;
		DeviceHostVector<int> _rangex;
		DeviceHostVector<int> _rangey;
		DeviceHostVector<BOX> _box;
	};
}



#endif // !_INTNODE_H_
