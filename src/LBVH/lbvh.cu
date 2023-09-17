#include "lbvh.h"
#include "BvhUtils.cuh"


using namespace CXE;

int const K_THREADS = 256;

void Bvh::setup(int prim_size, int ext_node_size, int int_node_size) {
	_primSize = prim_size;
	_extSize = ext_node_size;
	_intSize = int_node_size;
	_extNodes.setup(_primSize, _extSize);
	_intNodes.setup(_intSize);

	cudaMalloc((void**)&_bv, sizeof(BOX));
	d_keys32.resize(_primSize);
	d_vals.resize(_primSize);
	d_primMap.resize(_primSize);
	d_count.resize(_primSize);
}

void Bvh::build(const vec3f* vertices, const vec3i* faces) {

	BOX	bv{};
	cudaMemcpy(_bv, &bv, sizeof(BOX), cudaMemcpyHostToDevice);
	int blockDim = K_THREADS;
	int gridDim = (_primSize + blockDim - 1) / blockDim;
	//计算monton code的值并且排序

	BvhUtils::calcMaxBV << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, faces, vertices, _bv);
	BvhUtils::calcMCs<<<dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >>>(_primSize, faces, vertices, _bv,getRawPtr(d_keys32));
	checkThrustErrors(thrust::sequence(d_vals.begin(), d_vals.end()));
	checkThrustErrors(thrust::sort_by_key(d_keys32.begin(), d_keys32.end(), d_vals.begin()));
	BvhUtils::calcInverseMapping << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, getRawPtr(d_vals), getRawPtr(d_primMap));

	cudaMemcpy(_extNodes.getMtCodes(), getRawPtr(d_keys32), sizeof(int) * _primSize, cudaMemcpyDeviceToDevice);
	//给每个叶子结点填充数据

	BvhUtils::buildPrimitives << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize,lvs(), getRawPtr(d_primMap), faces, vertices);
	// build external nodes
    lvs().buildExtNodes(_primSize);
	lvs().calcSplitMetrics(_primSize);
	// build internal nodes
	printf("okk\n");
	tks().clearIntNodes(_primSize-1);
	printf("okk\n");
	BvhUtils::buildIntNodes << < dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, getRawPtr(d_count), lvs(), tks());
	
}