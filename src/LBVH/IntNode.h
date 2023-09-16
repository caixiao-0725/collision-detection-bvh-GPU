#ifndef _INTNODE_H_
#define _INTNODE_H_

#include "DeviceHostVector.h"

namespace CXE {
	class IntNodeArray
	{
	public:
		void setup(const int intSize);


		int _intSize;

		DeviceHostVector<int> LeftChild;
		DeviceHostVector<int> RightChild;

	};
}



#endif // !_INTNODE_H_
