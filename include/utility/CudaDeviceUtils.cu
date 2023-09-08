#include "CudaDeviceUtils.h"
#include <cuda_runtime.h>
__device__ bool atomicMinf(float* address, float val) {
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	if (*address <= val) return false;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	//return __int_as_float(old);
	return true;
}

__device__ bool atomicMaxf(float* address, float val) {
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	if (*address >= val) return false;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	//return __int_as_float(old);
	return true;
}