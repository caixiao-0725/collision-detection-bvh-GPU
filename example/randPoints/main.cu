#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "Mesh.h"
#include "DeviceHostVector.h"
#include "bv.h"
#include "atomicFunctions.cuh"
#include "lbvh.h"


using namespace CXE;



int main() {
    const int N =  100000;
    const float R = 0.001f;

    printf("Generating Data...\n");
    DeviceHostVector<AABB> aabbs;
    aabbs.Allocate(N);
    srand(1);
    for (size_t i = 0; i < N; i++) {
        vec3f points = vec3f(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX) + vec3f(1,1,1);
        //vec3f points = vec3f(i*R*1.1, i*R*1.1, i*R*1.1)+ vec3f(1,1,1);
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
    //uint sign;
    //uint u;
    //uint result;
    //uint remainder;
    //sign = ((a >> 17U) & 0x8000U);
    //u = a & 0x7fffffffU;
    //
    //if (u >= 0x38000000U) {
    //    remainder = u << 19U;
    //    u -= 0x38000000U;
    //    result = (sign | (u >> 13U));
    //}
    //else  if (u < 0x33000001U) { // +0/-0
    //    remainder = u;
    //    result = sign;
    //}
    //else { // Denormal numbers
    //    const unsigned int exponent = u >> 23U;
    //    const unsigned int shift = 0x7eU - exponent;
    //    unsigned int mantissa = (u & 0x7fffffU);
    //    mantissa |= 0x800000U;
    //    remainder = mantissa << (32U - shift);
    //    result = (sign | (mantissa >> shift));
    //    result &= 0x0000FFFFU;
    //}
    //
    //
    //uint exponent;
    //uint b;
    //uint mantissa;
	//sign = ((result & 0x8000U) >> 14U) & 1U;
    //exponent = (result >> 10U) & 0x1fU;
	//mantissa = (result & 0x3ffU) << 13U;
    //if (exponent == 0U) { /* Denorm or Zero */
    //    if (mantissa != 0U) {
    //        unsigned int msb;
    //        exponent = 0x71U;
    //        do {
    //            msb = (mantissa & 0x400000U);
    //            mantissa <<= 1U; /* normalize */
    //            --exponent;
    //        } while (msb == 0U);
    //        mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
    //    }
    //}
    //else {
    //    exponent += 0x70U;
    //}
    //u= ((sign << 31U) | (exponent << 23U) | mantissa);
    //
    //temp = *reinterpret_cast<float*>(&u);
	//printf("\nCleared sign bit: %.10f\n", temp);

    aabbs.ReadToDevice();
    Bvh A;
    A._type = 5;
    A.setup(N, N, N - 1);
    A.build(aabbs.GetDevice());
    A.query(aabbs.GetDevice(), aabbs.GetSize(),true);
    return 0;
}


