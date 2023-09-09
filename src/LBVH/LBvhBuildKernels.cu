#include "LBvhKernels.h"
#include "AtomicFunctions.cuh"
#include "Utils.cuh"

__global__ void calcMaxBV(int size, const int3 *_faces, const vec3f *_vertices, BOX* _bv) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	BOX bv{};
	auto v = _vertices[_faces[idx].x];
	bv.combines(v.x, v.y, v.z);
	v = _vertices[_faces[idx].y];
	bv.combines(v.x, v.y, v.z);
	v = _vertices[_faces[idx].z];
	bv.combines(v.x, v.y, v.z);
	
	/// could use aggregate atomic min/max
	atomicMinf(&_bv->_min.x, bv._min.x);
	atomicMinf(&_bv->_min.y, bv._min.y);
	atomicMinf(&_bv->_min.z, bv._min.z);
	atomicMaxf(&_bv->_max.x, bv._max.x);
	atomicMaxf(&_bv->_max.y, bv._max.y);
	atomicMaxf(&_bv->_max.z, bv._max.z);
}

__global__ void calcMCs(int size, int3* _faces, vec3f* _vertices, BOX scene, uint* codes) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	//for (; idx < size; idx += gridDim.x * blockDim.x) {
		BOX bv{};
		auto v = _vertices[_faces[idx].x];
		bv.combines(v.x, v.y, v.z);
		v = _vertices[_faces[idx].y];
		bv.combines(v.x, v.y, v.z);
		v = _vertices[_faces[idx].z];
		bv.combines(v.x, v.y, v.z);
		const vec3f c = bv.center();
		const vec3f offset = c - scene._min;
		codes[idx] = morton3D(offset.x / scene.width(), offset.y / scene.height(), offset.z / scene.depth());
	//}
}