#ifndef __CUDA_DEVICE_UTILS_H_
#define __CUDA_DEVICE_UTILS_H_

#include <device_launch_parameters.h>


/// CUDA atomic operations customized 
__device__ bool atomicMinf(float* address, float val);
__device__ bool atomicMaxf(float* address, float val);

#endif