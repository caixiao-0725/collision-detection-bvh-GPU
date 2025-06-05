#include "MergeNode.h"

namespace CXE {

	void MergeNodeArray::setup(const int Size) {
		_size = Size;
		_nodes.Allocate(Size);
		_flags.Allocate(Size);
	}

	void MergeNodeArray::clearFlags() {
		cudaMemset(_flags, 0,sizeof(int) * _size);
	}

	void StacklessMergeNodeArray::setup(const int Size) {
		_size = Size;
		_nodes.Allocate(_size);
	}

	void StacklessMergeNodeV1Array::setup(const int Size) {
		_size = Size;
		_nodes.Allocate(_size);
	}
}