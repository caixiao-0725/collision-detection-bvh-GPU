#include "IntNode.h"
#include "thrust/device_vector.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include "CudaThrustUtils.hpp"
#include "device_launch_parameters.h"

using namespace CXE;

void IntNodeArray::setup(const int intSize) {
	_intSize = intSize;
	_lc.Allocate(_intSize);
	_rc.Allocate(_intSize);
	_par.Allocate(_intSize);
	_mark.Allocate(_intSize);
	_flag.Allocate(_intSize);
	_rangex.Allocate(_intSize);
	_rangey.Allocate(_intSize);
	_box.Allocate(_intSize);
}

__global__ void kernelClearIntNodes(uint* _mark, uint* _flag, uint _markValue, uint _flagValue, int intSize) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= intSize) return;
	_mark[idx] = _markValue;
	_flag[idx] = _flagValue;
}

void IntNodeArray::clearIntNodes(int size) {
	//cudaMemset(_lc, 0xff, sizeof(int) * size);
	//cudaMemset(_rc, 0xff, sizeof(int) * size);
	//cudaMemset(_par, 0xff, sizeof(int) * size);
	//kernelClearIntNodes<<<(size+255)/256,256>>>(_mark, _flag, 7, 0, size);
	//cudaMemset(_rangex, 0xff, sizeof(int) * size);
	//cudaMemset(_rangey, 0xff, sizeof(int) * size);

	cudaMemset(_flag, 0u, sizeof(uint) * size);
}