#ifndef __MERGE_NODE_H__
#define __MERGE_NODE_H__

#include "DeviceHostVector.h"
#include "bv.h"

namespace CXE {
	class MergeNodeArray {
	public:
		void setup(const int Size);
		void clearFlags();

		uint _size;
		DeviceHostVector<bvhNode> _nodes;
		DeviceHostVector<int> _flags;
	};

	class StacklessMergeNodeArray {
	public:
		void setup(const int Size);

		uint _size;
		DeviceHostVector<bvhNodeV2> _nodes;
	};

	class StacklessMergeNodeV1Array {
	public:
		void setup(const int Size);

		uint _size;
		DeviceHostVector<bvhNodeV1> _nodes;
	};
	
}
 

#endif // ! __MERGE_NODE_H__
