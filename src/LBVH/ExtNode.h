#ifndef _EXTNODE_H_
#define _EXTNODE_H_

#include "DeviceHostVector.h"

namespace CXE {
	class ExtNodeArray
	{
	public:
		void setup(const int Prims,const int ExtSize);

		int _prims;
		int _extSize;

		DeviceHostVector<int> Parent;


	};
}

#endif // !_EXTNODE_H_

