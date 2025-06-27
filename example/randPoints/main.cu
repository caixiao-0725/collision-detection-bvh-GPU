#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Mesh.h"
#include "DeviceHostVector.h"
#include "bv.h"
#include "atomicFunctions.cuh"
#include "lbvh.h"


using namespace CXE;

//#include "BvhUtils.cuh"
int main() {
    const int N =  100000;
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

    //float temp = 0.194304f;
    //uint a = *reinterpret_cast<uint*>(&temp);
    //
    //for (int bit = 31;bit >= 0;bit--) {
	//	printf("%d", (a>>bit)&1);
    //}
    //
    //printf("\n");
    //
	//a &= 0x7F800000; // Clear the sign bit
	//a |= 0x00002000; // Set the exponent to 1
    //for (int bit = 31;bit >= 0;bit--) {
    //    printf("%d", (a >> bit) & 1);
    //}
    //
    //temp = *reinterpret_cast<float*>(&a);
	//printf("\nCleared sign bit: %.10f\n", temp);

    aabbs.ReadToDevice();
    Bvh A;
    A._type = 6;
    A.setup(N, N, N - 1);
    A.build(aabbs.GetDevice());
    A.query(aabbs.GetDevice(), aabbs.GetSize(),true);
    return 0;
}

