#ifndef __LBVH_KERNELS_H_
#define __LBVH_KERNELS_H_
#include <cuda_runtime.h>
#include "bv.h"

__global__ void calcMaxBV(int size, const int3 *_faces, const vec3f *_vertices, BOX* _bv);

#endif