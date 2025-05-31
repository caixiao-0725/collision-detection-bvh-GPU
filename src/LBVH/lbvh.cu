#include "lbvh.h"
#include "BvhUtils.cuh"
#include "cub/cub.cuh"
 

using namespace CXE;

int const K_THREADS = 256;

void Bvh::setup(int prim_size, int ext_node_size, int int_node_size) {

	_primSize = prim_size;
	_extSize = ext_node_size;
	_intSize = int_node_size;
	_extNodes.setup(_primSize, _extSize);
	_intNodes.setup(_intSize);
	_unsortedTks.setup(_intSize);

	d_keys32.Allocate(_primSize);
	d_vals.Allocate(_primSize);
	d_primMap.Allocate(_primSize);
	d_count.Allocate(_primSize);
	d_bv.Allocate(1);
	d_tkMap.Allocate(_primSize);
	d_offsetTable.Allocate(_primSize);

	_cpNum.Allocate(1);
	_cpRes.Allocate(_primSize*32);
	
}

void Bvh::build(const vec3f* vertices, const vec3i* faces) {

	int blockDim = K_THREADS;
	int gridDim = (_primSize + blockDim - 1) / blockDim;

	BvhUtils::calcMaxBVWarpShuffle << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, faces, vertices, d_bv);
	BvhUtils::calcMCs<<<dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >>>(_primSize, faces, vertices, d_bv,d_keys32);
	checkThrustErrors(thrust::sequence(thrust::device, d_vals.GetDevice(), d_vals.GetDevice() + d_vals.GetSize()));
	checkThrustErrors(thrust::sort_by_key(thrust::device, d_keys32.GetDevice(), d_keys32.GetDevice() + d_keys32.GetSize(), d_vals.GetDevice()));
	BvhUtils::calcInverseMapping << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, d_vals, d_primMap);


	cudaMemcpy(_extNodes._mtcode, d_keys32, sizeof(int) * _primSize, cudaMemcpyDeviceToDevice);


	BvhUtils::buildPrimitives << < gridDim, blockDim >> > (_primSize,lvs()._idx, lvs()._box, d_primMap, faces, vertices);

	// build external nodes
    lvs().buildExtNodes(_primSize);
	lvs().calcSplitMetrics(extSize());
	// build internal nodes
	printf("okk\n");
	tks().clearIntNodes(_primSize-1);
	printf("okk\n");
	BvhUtils::buildIntNodes << < dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, d_count, lvs()._lca, lvs()._metric, lvs()._par, lvs()._mark, lvs()._box,
		untks()._rc, untks()._lc, untks()._rangey, untks()._rangex, untks()._mark, untks()._box, untks()._flag, untks()._par);
	
}

void Bvh::reorderIntNodes() {
	// 1. 声明临时存储大小并分配内存
	//void* d_temp_storage;
	//size_t temp_storage_bytes = 0;
	//uint* in, *out;
	//cudaMalloc(&in, sizeof(uint)* extSize());
	//cudaMalloc(&out, sizeof(uint) * extSize());
	//cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, in, out, extSize());
	//cudaMalloc(&d_temp_storage, temp_storage_bytes);
	//cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_count.GetDevice(), d_offsetTable.GetDevice(), extSize());

	checkThrustErrors(thrust::exclusive_scan(thrust::device,d_count.GetDevice(), d_count.GetDevice() + extSize(), d_offsetTable.GetDevice()));
	const int blockDim = K_THREADS;
	const int gridDim = (extSize() + blockDim - 1) / blockDim;
	BvhUtils::calcIntNodeOrders << < gridDim, blockDim >> > (extSize(), untks()._lc, lvs()._lca, d_count, d_offsetTable, d_tkMap);
	lvs()._lca.ReadToHost();
	d_tkMap.ReadToHost();
	checkThrustErrors(thrust::fill(thrust::device, lvs()._lca.GetDevice() + extSize(),
		lvs()._lca.GetDevice() + extSize() + 1, -1));
	
	
	BvhUtils::updateBvhExtNodeLinks << <gridDim, blockDim >> > (extSize(),d_tkMap, lvs()._lca, lvs()._par);

	BvhUtils::reorderIntNode << <(intSize() +255)/256, blockDim >> > (intSize(), d_tkMap, untks()._lc, untks()._rc, untks()._mark, untks()._par, untks()._rangex, untks()._rangey, untks()._box,
	tks()._lc, tks()._rc, tks()._mark, tks()._par,tks()._rangex, tks()._rangey, tks()._box);

}


void Bvh::build(const AABB* boxs) {


	int blockDim = K_THREADS;
	int gridDim = (_primSize + blockDim - 1) / blockDim;

	BvhUtils::calcMaxBVFromBox << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, boxs, d_bv);
	BvhUtils::calcMCsFromBox << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, boxs, d_bv, d_keys32);

	checkThrustErrors(thrust::sequence(thrust::device, d_vals.GetDevice(), d_vals.GetDevice() + d_vals.GetSize()));
	checkThrustErrors(thrust::sort_by_key(thrust::device, d_keys32.GetDevice(), d_keys32.GetDevice() + d_keys32.GetSize(), d_vals.GetDevice()));
	BvhUtils::calcInverseMapping << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, d_vals, d_primMap);

	cudaMemcpy(_extNodes._mtcode, d_keys32, sizeof(int) * _primSize, cudaMemcpyDeviceToDevice);

	BvhUtils::buildPrimitivesFromBox << < gridDim, blockDim >> > (_primSize, lvs()._idx, lvs()._box, d_primMap, boxs);

	// build external nodes
	lvs().buildExtNodes(_primSize);
	lvs().calcSplitMetrics(extSize());
	
	// build internal nodes
	
	untks().clearIntNodes(_primSize - 1);

	BvhUtils::buildIntNodes << < dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, d_count, lvs()._lca, lvs()._metric, lvs()._par, lvs()._mark, lvs()._box,
		untks()._rc, untks()._lc, untks()._rangey, untks()._rangex, untks()._mark, untks()._box,untks()._flag, untks()._par);
	
	//untks()._rc.ReadToHost();
	//untks()._par.ReadToHost();
	//untks()._lc.ReadToHost();
	//untks()._rangex.ReadToHost();
	//untks()._rangey.ReadToHost();
	//lvs()._lca.ReadToHost();
	//lvs()._par.ReadToHost();
	reorderIntNodes();
	//tks()._rc.ReadToHost();
	//tks()._par.ReadToHost();
	//tks()._lc.ReadToHost();
	//tks()._rangex.ReadToHost();
	//tks()._rangey.ReadToHost();
	//lvs()._lca.ReadToHost();
	//tks().clearIntNodes(_primSize - 1);
	//BvhUtils::refitIntNode << <(extSize() + 255) / 256, 256 >> > (extSize(), lvs()._par, tks()._par, tks()._flag, tks()._lc, tks()._rc, tks()._mark, lvs()._box, tks()._box);
	lvs()._box.ReadToHost();
	//tks()._box.ReadToHost();
}


void Bvh::query(const AABB* boxs, const uint num) {
	int blockDim = K_THREADS;
	int gridDim = (num + blockDim - 1) / blockDim;
	//BvhUtils::pureBvhInterCD << <gridDim, blockDim >> > (num, boxs, lvs()._par, lvs()._idx, lvs()._box, lvs()._lca,
	//	tks()._box, tks()._rangex, tks()._rangey,
	//	_cpNum,_cpRes
	//	);

	BvhUtils::pureBvhStackCD<<<gridDim, blockDim>>> (num, boxs, lvs()._par, lvs()._idx, lvs()._box, lvs()._lca,
		tks()._box, tks()._lc, tks()._rc, tks()._mark,
		_cpNum, _cpRes
		);

	_cpNum.ReadToHost();
	_cpRes.ReadToHost();

	printf("%d\n", _cpNum.GetHost()[0]);
	//std::vector<int> temp(num);
	//for (int i = 0;i < _cpNum.GetHost()[0];i++) {
	//	temp[_cpRes.GetHost()[i].x]++;
	//}
	//for (int i = 0;i < num-1;i++) {
	//	if (temp[i] == 0) {
	//		printf("%d\n", i);
	//	}
	//}
}