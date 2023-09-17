#ifndef _INTNODE_H_
#define _INTNODE_H_

#include "DeviceHostVector.h"

namespace CXE {
	class IntNodeArray
	{
	public:
		void setup(const int intSize);
		void clearIntNodes(int size);

		__device__ int& lc(int i) { return ((int*)_attribs[LC])[i]; }
		__device__ int& rc(int i) { return ((int*)_attribs[RC])[i]; }
		__device__ int& par(int i){ return ((int*)_attribs[PAR])[i]; }
		__device__ uint& mark(int i) { return ((uint*)_attribs[MARK])[i]; }
		__device__ int& rangex(int i) { return ((int*)_attribs[RANGEX])[i]; }
		__device__ int& rangey(int i) { return ((int*)_attribs[RANGEY])[i]; }
		__device__ uint& flag(int i) { return ((uint*)_attribs[FLAG])[i]; }

		int _intSize;
		enum {FLAG,LC,RC,PAR,MARK,RANGEX,RANGEY};
		void* _attribs[10];
		
	};
}



#endif // !_INTNODE_H_
