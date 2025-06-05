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
	_unsortedTks.setup(_intSize);

	switch (_type)
	{
	case 0:
		_intNodes.setup(_intSize);
		break;
	case 1:
		_intNodes.setup(_intSize);
		break;
	case 2:
		_mergeNodes.setup(_intSize);
		break;
	case 3:
		_intNodes.setup(_intSize);
		_stacklessMergeNodes.setup(_intSize);
		break;
	case 4:
		_intNodes.setup(_intSize);
		_stacklessMergeNodesV1.setup(2*_intSize+1);
	case 11:
		_mergeNodes.setup(_intSize);
		break;
	case 12:
		_mergeNodes.setup(_intSize);
		break;
	}

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
	BvhUtils::calcMCs<<<dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >>>(_primSize, faces, vertices, d_bv, _extNodes._mtcode);
	checkThrustErrors(thrust::sequence(thrust::device, d_vals.GetDevice(), d_vals.GetDevice() + d_vals.GetSize()));
	checkThrustErrors(thrust::sort_by_key(thrust::device, _extNodes._mtcode.GetDevice(), _extNodes._mtcode.GetDevice() + _extNodes._mtcode.GetSize(), d_vals.GetDevice()));
	BvhUtils::calcInverseMapping << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, d_vals, d_primMap);

	BvhUtils::buildPrimitives << < gridDim, blockDim >> > (_primSize,lvs()._idx, lvs()._box, d_primMap, faces, vertices);

	// build external nodes
    lvs().buildExtNodes(_primSize);
	lvs().calcSplitMetrics(extSize());

	// build internal nodes
	tks().clearIntNodes(_primSize-1);
	BvhUtils::buildIntNodes << < dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, d_count, lvs()._lca, lvs()._metric, lvs()._par, lvs()._mark, lvs()._box,
		untks()._rc, untks()._lc, untks()._rangey, untks()._rangex, untks()._mark, untks()._box, untks()._flag, untks()._par);
	
}

void Bvh::reorderIntNodes() {

	checkThrustErrors(thrust::exclusive_scan(thrust::device,d_count.GetDevice(), d_count.GetDevice() + extSize(), d_offsetTable.GetDevice()));
	const int blockDim = K_THREADS;
	const int gridDim = (extSize() + blockDim - 1) / blockDim;
	BvhUtils::calcIntNodeOrders << < gridDim, blockDim >> > (extSize(), untks()._lc, lvs()._lca, d_count, d_offsetTable, d_tkMap);

	checkThrustErrors(thrust::fill(thrust::device, lvs()._lca.GetDevice() + extSize(),
		lvs()._lca.GetDevice() + extSize() + 1, -1));
	
	
	BvhUtils::updateBvhExtNodeLinks << <gridDim, blockDim >> > (extSize(),d_tkMap, lvs()._lca, lvs()._par);
	const int intGridDim = (intSize() + blockDim - 1) / blockDim;
	switch (_type)
	{
	case 0:
		BvhUtils::reorderIntNode << <intGridDim, blockDim >> > (intSize(), d_tkMap, untks()._lc, untks()._rc, untks()._mark, untks()._par, untks()._rangex, untks()._rangey, untks()._box,
			tks()._lc, tks()._rc, tks()._mark, tks()._par, tks()._rangex, tks()._rangey, tks()._box);
		break;
	case 1:
		BvhUtils::reorderIntNode << <intGridDim, blockDim >> > (intSize(), d_tkMap, untks()._lc, untks()._rc, untks()._mark, untks()._par, untks()._rangex, untks()._rangey, untks()._box,
			tks()._lc, tks()._rc, tks()._mark, tks()._par, tks()._rangex, tks()._rangey, tks()._box);
		break;
	case 2:
		BvhUtils::reorderMergeNode << <intGridDim, blockDim >> > (intSize(), d_tkMap, _mergeNodes._nodes, lvs()._box, untks()._box, untks()._lc, untks()._rc, untks()._mark, untks()._par);
		break;
	case 3:
		BvhUtils::reorderIntNodeSoa << <intGridDim, blockDim >> > (intSize(), d_tkMap, untks()._lc, untks()._rc, untks()._mark, untks()._par, untks()._rangex, untks()._rangey, untks()._box,
			tks()._lc, tks()._rc, tks()._mark, tks()._par, _stacklessMergeNodes._nodes);
		break;
	case 4:
		BvhUtils::reorderIntNodeV1 << <intGridDim, blockDim >> > (intSize(), d_tkMap, lvs()._lca,lvs()._box, 
			untks()._lc, untks()._mark, untks()._rangey, untks()._box,
			_stacklessMergeNodesV1._nodes);
		
		break;
	default:
		break;
	}
	
}


void Bvh::build(const AABB* boxs) {


	int blockDim = K_THREADS;
	int gridDim = (_primSize + blockDim - 1) / blockDim;

	BvhUtils::calcMaxBVFromBox << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, boxs, d_bv);
	BvhUtils::calcMCsFromBox << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, boxs, d_bv, lvs()._mtcode);

	checkThrustErrors(thrust::sequence(thrust::device, d_vals.GetDevice(), d_vals.GetDevice() + d_vals.GetSize()));
	checkThrustErrors(thrust::sort_by_key(thrust::device, lvs()._mtcode.GetDevice(), lvs()._mtcode.GetDevice() + lvs()._mtcode.GetSize(), d_vals.GetDevice()));
	BvhUtils::calcInverseMapping << <dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, d_vals, d_primMap);

	BvhUtils::buildPrimitivesFromBox << < gridDim, blockDim >> > (_primSize, lvs()._idx, lvs()._box, d_primMap, boxs);
	if (_type > 10) {
		BvhUtils::lbvhBuildInternalKernel << < gridDim, blockDim >> > (mgs()._nodes, lvs()._par, lvs()._mtcode, lvs()._idx, _primSize);

		mgs().clearFlags();
		BvhUtils::mergeNodeRefit << < gridDim, blockDim >> > (mgs()._nodes, lvs()._par, lvs()._box, lvs()._idx, mgs()._flags, extSize());
	}
	else {
		// build external nodes
		lvs().buildExtNodes(_primSize);
		lvs().calcSplitMetrics(extSize());
	
		// build internal nodes
		untks().clearIntNodes(_primSize - 1);
		BvhUtils::buildIntNodes << < dim3(gridDim, 1, 1), dim3(blockDim, 1, 1) >> > (_primSize, d_count, lvs()._lca, lvs()._metric, lvs()._par, lvs()._mark, lvs()._box,
			untks()._rc, untks()._lc, untks()._rangey, untks()._rangex, untks()._mark, untks()._box,untks()._flag, untks()._par);
		reorderIntNodes();
	}
	//refit ?
	//BvhUtils::refitIntNode << <(extSize() + 255) / 256, 256 >> > (extSize(), lvs()._par, tks()._par, tks()._flag, tks()._lc, tks()._rc, tks()._mark, lvs()._box, tks()._box);
}


void Bvh::query(const AABB* boxs, const uint num,bool self) {
	int blockDim = K_THREADS;
	int gridDim = (num + blockDim - 1) / blockDim;
	_stacklessMergeNodesV1._nodes.ReadToHost();
	if (self) {
		switch (_type)
		{
		case 0:
			BvhUtils::pureBvhStacklessCD<true> << <gridDim, blockDim >> > (num, boxs, lvs()._par, lvs()._idx, lvs()._box, lvs()._lca,
				tks()._box, tks()._rangex, tks()._rangey,
				_cpNum, _cpRes
				);
			break;
		case 1:
			BvhUtils::pureBvhStackCD<true> << <gridDim, blockDim >> > (num, boxs, lvs()._par, lvs()._idx, lvs()._box, lvs()._lca,
				tks()._box, tks()._lc, tks()._rc, tks()._mark,
				_cpNum, _cpRes
				);
			break;
		case 2:
			BvhUtils::pureMergeBvhStackCD<true> << <gridDim, blockDim >> > (num, boxs,
				_mergeNodes._nodes, lvs()._idx,
				_cpNum, _cpRes);
			break;
		case 3:
			BvhUtils::AosBvhStacklessCD<true> << <gridDim, blockDim >> > (num, boxs, lvs()._par, lvs()._idx, lvs()._box, lvs()._lca,
				_stacklessMergeNodes._nodes,
				_cpNum, _cpRes
				);
			break;
		case 4:
			BvhUtils::AosBvhStacklessCDV1<true> << <gridDim, blockDim >> > (num, boxs, intSize(),lvs()._idx,
				_stacklessMergeNodesV1._nodes,
				_cpNum, _cpRes
				);
			break;
		case 11:
			BvhUtils::pureMergeBvhStackCD<true> << <gridDim, blockDim >> > (num, boxs,
				_mergeNodes._nodes, lvs()._idx,
				_cpNum, _cpRes);
			break;
		case 12:
			BvhUtils::pureMergeBvhStackSortElementCD<true> << <gridDim, blockDim >> > (num, boxs,
				_mergeNodes._nodes, lvs()._idx,
				_cpNum, _cpRes);
			break;
		}
	}
	else {
		switch (_type)
		{
		case 0:
			BvhUtils::pureBvhStacklessCD<false> << <gridDim, blockDim >> > (num, boxs, lvs()._par, lvs()._idx, lvs()._box, lvs()._lca,
				tks()._box, tks()._rangex, tks()._rangey,
				_cpNum, _cpRes
				);
			break;
		case 1:
			BvhUtils::pureBvhStackCD<false> << <gridDim, blockDim >> > (num, boxs, lvs()._par, lvs()._idx, lvs()._box, lvs()._lca,
				tks()._box, tks()._lc, tks()._rc, tks()._mark,
				_cpNum, _cpRes
				);
			break;
		case 2:
			BvhUtils::pureMergeBvhStackCD<false> << <gridDim, blockDim >> > (num, boxs,
				_mergeNodes._nodes, lvs()._idx,
				_cpNum, _cpRes);
			break;
		case 3:
			BvhUtils::AosBvhStacklessCD<false> << <gridDim, blockDim >> > (num, boxs, lvs()._par, lvs()._idx, lvs()._box, lvs()._lca,
				_stacklessMergeNodes._nodes,
				_cpNum, _cpRes
				);
			break;
		case 4:
			BvhUtils::AosBvhStacklessCDV1<false> << <gridDim, blockDim >> > (num, boxs, intSize(), lvs()._idx,
				_stacklessMergeNodesV1._nodes,
				_cpNum, _cpRes
				);
			break;
		case 11:
			BvhUtils::pureMergeBvhStackCD<false> << <gridDim, blockDim >> > (num, boxs,
				_mergeNodes._nodes, lvs()._idx,
				_cpNum, _cpRes);
			break;
		case 12:
			BvhUtils::pureMergeBvhStackSortElementCD<false> << <gridDim, blockDim >> > (num, boxs,
				_mergeNodes._nodes, lvs()._idx,
				_cpNum, _cpRes);
			break;
		}
	}
	


	_cpNum.ReadToHost();
	_cpRes.ReadToHost();

	printf("%d\n", _cpNum.GetHost()[0]);
}