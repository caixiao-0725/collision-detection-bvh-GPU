#include "BvhBV.h"

__global__ void gatherBVs(int size, const int* gatherPos, BvhBvCompletePort from, BvhBvCompletePort to) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	int ori = gatherPos[idx];
	to.minx(idx) = from.minx(ori);
	to.miny(idx) = from.miny(ori);
	to.minz(idx) = from.minz(ori);
	to.maxx(idx) = from.maxx(ori);
	to.maxy(idx) = from.maxy(ori);
	to.maxz(idx) = from.maxz(ori);
}
__global__ void scatterBVs(int size, const int* scatterPos, BvhBvCompletePort from, BvhBvCompletePort to) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;
	int tar = scatterPos[idx];
	to.minx(tar) = from.minx(idx);
	to.miny(tar) = from.miny(idx);
	to.minz(tar) = from.minz(idx);
	to.maxx(tar) = from.maxx(idx);
	to.maxy(tar) = from.maxy(idx);
	to.maxz(tar) = from.maxz(idx);
}
