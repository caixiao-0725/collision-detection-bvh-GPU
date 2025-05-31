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

int main0() {
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

    //����ģ�͵�gpu��������
    DeviceHostVector<vec3f> vertsA, vertsB;
    vertsA.Allocate(meshA.nverts(),meshA.verts_.data());
    vertsB.Allocate(meshB.nverts(),meshB.verts_.data());

    //����ģ�͵�gpu������
    DeviceHostVector<vec3i> faceA, faceB;
    faceA.Allocate(meshA.nfaces(), meshA.faces_.data());
    faceB.Allocate(meshB.nfaces(), meshB.faces_.data());


    Bvh A;
    A.setup(faceA.GetSize(), faceA.GetSize(), faceA.GetSize()-1);
    A.build(vertsA.GetDevice(), faceA.GetDevice());
  


    return 0;
}

int main() {
    const int N = 100000;
    const float R = 0.001f;

    printf("Generating Data...\n");
    DeviceHostVector<AABB> aabbs;
    aabbs.Allocate(N);
    srand(1);
    for (size_t i = 0; i < N; i++) {
        vec3f points = vec3f(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
        //vec3f points = vec3f(i*R*1.1, i*R*1.1, i*R*1.1);
        aabbs.GetHost()[i] = AABB(points.x - R , points.y - R, points.z - R,
            points.x + R, points.y + R, points.z + R);
    }

    aabbs.ReadToDevice();

    Bvh A;
    A.setup(N, N, N - 1);
    A.build(aabbs.GetDevice());
    A.query(aabbs.GetDevice(), aabbs.GetSize());
    return 0;
}

