#include "ExtNode.h"
#include "thrust/sequence.h"
#include "thrust/sort.h"
#include "thrust/device_vector.h"
#include "CudaThrustUtils.hpp"
#include "device_launch_parameters.h"

using namespace CXE;

__global__ void calcExtNodeSplitMetrics(int extsize, const uint* _codes, int* _metrics) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= extsize) return;
	_metrics[idx] = idx != extsize - 1 ? 32 - __clz(_codes[idx] ^ _codes[idx + 1]) : 33;
	if (idx < 10) printf("%d-ext node: split metric %d\n", idx, _metrics[idx]);
}


void ExtNodeArray::setup(const int Prims, const int ExtSize) {
	_prims = Prims;
	_extSize = ExtSize;
	cudaMalloc((void**)&_attribs[MTCODE], sizeof(uint) * _prims);
	cudaMalloc((void**)&_attribs[IDX], sizeof(int) * _prims);
	cudaMalloc((void**)&_attribs[AABB], sizeof(BOX) * _prims);
	cudaMalloc((void**)&_attribs[MARK], sizeof(uint) * _prims);
	cudaMalloc((void**)&_attribs[EXT_NO], sizeof(int) * _prims);
	cudaMalloc((void**)&_attribs[PAR], sizeof(int) * _prims);
	cudaMalloc((void**)&_attribs[LCA], sizeof(int) * _prims);
	cudaMalloc((void**)&_attribs[SPLIT_METRIC], sizeof(int) * _prims);
}

int ExtNodeArray::buildExtNodes(int primsize) {
	uint*   primMarks = getMarks();
	int*    extIds = getExtIds();
	int		extSize;
	checkThrustErrors(thrust::fill(getDevicePtr(primMarks), getDevicePtr(primMarks) + primsize, 1));
	checkThrustErrors(thrust::sequence(getDevicePtr(extIds), getDevicePtr(extIds) + primsize, 1));
	cudaMemcpy(&extSize, extIds + primsize - 1, sizeof(int), cudaMemcpyDeviceToHost);
	clearExtNodes(extSize);
	//printf("%d\n", extSize);
	return extSize;
}

void ExtNodeArray::clearExtNodes(int size) {
	cudaMemset(_attribs[PAR], 0xff, sizeof(int) * size);
	checkThrustErrors(thrust::fill(thrust::device_ptr<uint>((uint*)_attribs[MARK]), thrust::device_ptr<uint>((uint*)_attribs[MARK]) + size, 7));
	cudaMemset(_attribs[LCA], 0xff, sizeof(int) * (size + 1));
}

void ExtNodeArray::calcSplitMetrics(int extsize) {
	calcExtNodeSplitMetrics << <(extsize + 255) / 256, 256 >> > (extsize, (const uint*)getMtCodes(), getMetrics());
}

