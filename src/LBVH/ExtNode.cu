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
	//if (idx < 10) printf("%d-ext node: split metric %d\n", idx, _metrics[idx]);
}


void ExtNodeArray::setup(const int Prims, const int ExtSize) {
	_prims = Prims;
	_extSize = ExtSize;
	_mtcode.Allocate(ExtSize);
	_box.Allocate(ExtSize);
	_idx.Allocate(ExtSize);
	_mark.Allocate(ExtSize);
	_par.Allocate(ExtSize);
	_lca.Allocate(ExtSize+1);
	_metric.Allocate(ExtSize);
	_extId.Allocate(ExtSize);

}

int ExtNodeArray::buildExtNodes(int primsize) {
	uint*   primMarks = getMarks();
	int*    extIds = getExtIds();
	int		extSize;

	checkThrustErrors(thrust::fill(thrust::device,primMarks, primMarks + primsize, 1));

	checkThrustErrors(thrust::sequence(thrust::device, extIds, extIds + primsize,1));

	cudaMemcpy(&extSize, extIds + primsize - 1, sizeof(int), cudaMemcpyDeviceToHost);

	clearExtNodes(extSize);

	printf("extSize %d\n", extSize);
	return extSize;
}

void ExtNodeArray::clearExtNodes(int size) {
	cudaMemset(_par, 0xff, sizeof(int) * size);

	checkThrustErrors(thrust::fill(thrust::device, getMarks(), getMarks() + size, 7));

	cudaMemset(_lca, 0xff, sizeof(int) * (size + 1));
}

void ExtNodeArray::calcSplitMetrics(int extsize) {
	calcExtNodeSplitMetrics << <(extsize + 255) / 256, 256 >> > (extsize, _mtcode, _metric);
}

