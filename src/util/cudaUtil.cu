
#include <util/cudaUtil.cuh>

/*
 *  Assume that we are working with linear coordinates
 */

/* In this kernel call return my global ID */
__device__ int globalID() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ int globalThreadCount() {
    return blockDim.x * gridDim.x;
}

int getMaximumNumberOfBlocks(void) {

    int deviceID;
    CudaTest(cudaGetDevice(&deviceID));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,deviceID);

    int blocks = 32;
    if(prop.major < 3) {
        blocks = 8;
    }else if(prop.major < 5) {
        blocks = 16;
    }
    return blocks * prop.multiProcessorCount;
}


void CudaBoolean::init() {
    CudaTest(cudaMalloc(&value,sizeof(uint)));
}
void CudaBoolean::shutdown() {
    CudaTest(cudaFree(value));
}
