#ifndef _EXTNODE_H_
#define _EXTNODE_H_

#include "DeviceHostVector.h"
#include "bv.h"

namespace CXE {
	class ExtNodeArray
	{
	public:
		void setup(const int Prims,const int ExtSize);

		int _prims;
		int _extSize;

		DeviceHostVector<int> Parent;
		DeviceHostVector<int> mtCodes;
		DeviceHostVector<int> idx;
		DeviceHostVector<BOX> bvh;


	};
}

#endif // !_EXTNODE_H_

