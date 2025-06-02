#include "MergeNode.h"

namespace CXE {

	void MergeNodeArray::setup(const int Size) {
		_nodes.Allocate(Size);
	}
}