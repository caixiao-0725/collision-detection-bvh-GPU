#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Mesh.h"
#include "DeviceHostVector.h"
#include "bv.h"
#include "atomicFunctions.cuh"

#include "lbvh.h"

#include "thrust/sequence.h"
#include "thrust/sort.h"
#include "thrust/device_vector.h"
#include "CudaThrustUtils.hpp"
using namespace CXE;

int main() {
    Mesh meshA("../../../assets/models/teapot.obj");
    Mesh meshB("../../../assets/models/teapot.obj");

    float min_x = 100.0f;
    float min_y = 100.0f;
    float min_z = 100.0f;
    for (int i = 0; i < meshB.nverts(); i++) {
		meshA.verts_[i].x -= 0.001f;
        meshA.verts_[i].y -= 0.001f;
        meshA.verts_[i].z -= 0.001f;
        meshB.verts_[i].x -= 0.001f;
        meshB.verts_[i].y -= 0.001f;
        meshB.verts_[i].z -= 0.001f;
        if (meshA.verts_[i].x < min_x) min_x = meshA.verts_[i].x;
        if (meshA.verts_[i].y < min_y) min_y = meshA.verts_[i].y;
        if (meshA.verts_[i].z < min_z) min_z = meshA.verts_[i].z;
    }
    printf("%f \n", min_x);
    printf("%f \n", min_y);
    printf("%f \n", min_z);

    //两个模型的gpu顶点数据
    DeviceHostVector<vec3f> vertsA, vertsB;
    vertsA.Allocate(meshA.nverts(),meshA.verts_.data());
    vertsB.Allocate(meshB.nverts(),meshB.verts_.data());

    //两个模型的gpu面数据
    DeviceHostVector<vec3i> faceA, faceB;
    faceA.Allocate(meshA.nfaces(), meshA.faces_.data());
    faceB.Allocate(meshB.nfaces(), meshB.faces_.data());

    Bvh A;
    A.setup(faceA.GetSize(), faceA.GetSize(), faceA.GetSize()-1);
    A.build(vertsA.GetDevice(), faceA.GetDevice());
  


    return 0;
}