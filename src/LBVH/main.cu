#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "LBvhKernels.h"
#include "Mesh.h"
//#include "bv.h"

int main() {
    Mesh mesh("../../../assets/models/teapot.obj");
    //AABB box();
    return 0;
}