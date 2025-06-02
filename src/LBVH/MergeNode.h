#ifndef __MERGE_NODE_H__
#define __MERGE_NODE_H__

#include "DeviceHostVector.h"
#include "bv.h"

namespace CXE {
	class MergeNodeArray {
	public:
		void setup(const int Size);

		DeviceHostVector<bvhNode> _nodes;
	};

}
 

#endif // ! __MERGE_NODE_H__
