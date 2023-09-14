#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Mesh.h"
#include "DeviceHostVector.h"
#include "bv.h"
#include "atomicFunctions.cuh"
#include "Utils.cuh"
using namespace CXE;

int const K_THREADS = 256;

int const K_REDUCTION_LAYER = 5;
int const K_REDUCTION_NUM = 1 << K_REDUCTION_LAYER;
int const K_REDUCTION_MODULO = K_REDUCTION_NUM - 1;




__global__ void KernelComputeAABBs(const int trianglesNum,
    const vec3f* dVertices,  const float thickness,
    const vec3i* dTriangles,
    AABB* dAABBs,AABB* MaxBox)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= trianglesNum) return;
    if (idx == 0) MaxBox[0].empty();
    __shared__ vec3f sMemAABBMin[K_THREADS];
    __shared__ vec3f sMemAABBMax[K_THREADS];

    // compute AABB
    vec3i a = dTriangles[idx];
    vec3f a0Curr = dVertices[a.x];
    vec3f a1Curr = dVertices[a.y];
    vec3f a2Curr = dVertices[a.z];

    vec3f aMin, aMax;
    FuncGetContinuousAABB(a0Curr, a1Curr, a2Curr, a0Curr, a1Curr, a2Curr,
        thickness, aMin, aMax);

    dAABBs[idx]._min = aMin;
    dAABBs[idx]._max = aMax;

    sMemAABBMin[threadIdx.x] = aMin;
    sMemAABBMax[threadIdx.x] = aMax;

    __syncthreads();

    if ((threadIdx.x & K_REDUCTION_MODULO) == 0)
    {
        vec3f lowerBound = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
        vec3f upperBound = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (int i = 0; i < K_REDUCTION_NUM; i++)
        {
            const unsigned int newIdx = idx + i;
            if (newIdx >= trianglesNum) break;
            vec3f aMin = sMemAABBMin[threadIdx.x + i];
            vec3f aMax = sMemAABBMax[threadIdx.x + i];
            lowerBound.x = fminf(lowerBound.x, aMin.x);
            lowerBound.y = fminf(lowerBound.y, aMin.y);
            lowerBound.z = fminf(lowerBound.z, aMin.z);
            upperBound.x = fmaxf(upperBound.x, aMax.x);
            upperBound.y = fmaxf(upperBound.y, aMax.y);
            upperBound.z = fmaxf(upperBound.z, aMax.z);
        }
        // minmax
        float* ptrLowerBound = reinterpret_cast<float*>(&(MaxBox[0]._min));
        float* ptrUpperBound = reinterpret_cast<float*>(&(MaxBox[0]._max));
        atomicMinf(&(MaxBox[0]._min.x), lowerBound.x);
        atomicMinf(&(MaxBox[0]._min.y), lowerBound.y);
        atomicMinf(&(MaxBox[0]._min.z), lowerBound.z);
        atomicMaxf(&(MaxBox[0]._max.x), upperBound.x);
        atomicMaxf(&(MaxBox[0]._max.y), upperBound.y);
        atomicMaxf(&(MaxBox[0]._max.z), upperBound.z);

    }
}

__global__ void calcMCs(int size, BOX* AABBs, BOX* scene,uint* codes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    BOX bv = AABBs[idx];
    const vec3f c = bv.center();
    const vec3f offset = c - scene[0]._min;
    codes[idx] = morton3D(offset.x / scene[0].width(), offset.y / scene[0].height(), offset.z / scene[0].depth());
}


int main() {
    Mesh meshA("../../../assets/models/teapot.obj");
    Mesh meshB("../../../assets/models/teapot.obj");

    for (int i = 0; i < meshB.nverts(); i++) {
		meshB.verts_[i].y += 0.1f;
    }
    //两个模型的gpu顶点数据
    DeviceHostVector<vec3f> vertsA, vertsB;
    vertsA.Allocate(meshA.nverts(),meshA.verts_.data());
    vertsB.Allocate(meshB.nverts(),meshB.verts_.data());

    //两个模型的gpu面数据
    DeviceHostVector<vec3i> faceA, faceB;
    faceA.Allocate(meshA.nfaces(), meshA.faces_.data());
    faceB.Allocate(meshB.nfaces(), meshB.faces_.data());

    //两个模型的gpu AABB数据
    DeviceHostVector<AABB> AABB_A, AABB_B;
    AABB_A.Allocate(meshA.nfaces());
    AABB_B.Allocate(meshB.nfaces());

    DeviceHostVector<AABB> MaxBox_A, MaxBox_B;
    MaxBox_A.Allocate(1);
    MaxBox_B.Allocate(1);


    int blockDim = K_THREADS;
    int gridDim = (faceA.GetSize() + blockDim - 1) / blockDim;

    KernelComputeAABBs << < dim3(gridDim, 1, 1), dim3(blockDim, 1, 1), 0 >> > (meshA.nfaces(), vertsA, 0.001f, faceA, AABB_A,MaxBox_A);
    KernelComputeAABBs << < dim3(gridDim, 1, 1), dim3(blockDim, 1, 1), 0 >> > (meshB.nfaces(), vertsB, 0.001f, faceB, AABB_B,MaxBox_B);
    
    //MaxBox_A.ReadToHost();  
    //MaxBox_B.ReadToHost();

    //morton code
    DeviceHostVector<uint> codesA, codesB;
    codesA.Allocate(meshA.nfaces());
    codesB.Allocate(meshB.nfaces());

    calcMCs <<< dim3(gridDim, 1, 1), dim3(blockDim, 1, 1), 0 >> > (meshA.nfaces(), AABB_A, MaxBox_A, codesA);
    calcMCs <<< dim3(gridDim, 1, 1), dim3(blockDim, 1, 1), 0 >> > (meshB.nfaces(), AABB_B, MaxBox_B, codesB);

    codesA.ReadToHost();

    return 0;
}