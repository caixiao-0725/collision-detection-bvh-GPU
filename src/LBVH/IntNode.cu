#include "IntNode.h"

using namespace CXE;

void IntNodeArray::setup(const int intSize) {
	_intSize = intSize;
	cudaMalloc((void**)&_attribs[FLAG], sizeof(uint) * intSize);
	cudaMalloc((void**)&_attribs[LC], sizeof(int) * intSize);
	cudaMalloc((void**)&_attribs[RC], sizeof(int) * intSize);
	cudaMalloc((void**)&_attribs[PAR], sizeof(int) * intSize);
	cudaMalloc((void**)&_attribs[MARK], sizeof(uint) * intSize);
	cudaMalloc((void**)&_attribs[RANGEX], sizeof(uint) * intSize);
	cudaMalloc((void**)&_attribs[RANGEY], sizeof(uint) * intSize);
}

void IntNodeArray::clearIntNodes(int size) {
	cudaMemset(_attribs[FLAG], 0, sizeof(uint) * size);
}