
#ifndef UTIL_CUH_
#define UTIL_CUH_

#include <iostream>

const int MaxThreads = 1024;

#define CUDA_TEST(Test,File,Line) do{                               \
    if((Test) != cudaSuccess) {                                     \
        std::cerr << "Error during GPU operation at line " << Line  \
        << " in file " << File                                      \
        << ": " << cudaGetErrorString(Test)                         \
        << std::endl;                                               \
        std::exit(1);                                               \
    }                                                               \
}while(0);

#define CreateSharedArray(T,S,N) __shared__ T N[S]
#define CreateSharedVariable(T,N,V) __shared__ T N; N = V;


#define CudaTest(Test) CUDA_TEST(Test,__FILE__,__LINE__)


#ifdef __CUDACC__
template<class T>
__device__ void copy(T* __restrict__ to, T* __restrict__ from, size_t length) {
    for(size_t idx = threadIdx.x ; idx < length; idx += blockDim.x) {
        T tmp = from[idx];
        to[idx] = tmp;
    }
}
#endif


#define RunKernel(NAME,BLOCKS,THREADS,...)  \
    do {                                            \
    NAME<<<BLOCKS,THREADS>>>(__VA_ARGS__);          \
        CudaTest(cudaDeviceSynchronize());          \
    } while(0);

#define RunCudaOp(OP)                           \
    do {                                        \
        CudaTest(OP);                           \
        CudaTest(cudaDeviceSynchronize());      \
    } while(0);



/*
 *  The way cache works is that you give it through the constructor an underlying 
 *  buffer of shared memory.  The cache will use this memory as a temporary storage
 *  area.
 *
 *  The Cache takes three arguments to its template: the data type stored, and how 
 *  many elements the cache holds, and the size of each data element.
 *
 *
 */
template<typename T>
struct Cache {
    
    const uint Size,Stride;
    T* data;

#ifdef __CUDACC__
    __device__ Cache(char* buffer, uint CacheSize, uint CacheStride) : data((T*)buffer), Size(CacheSize), Stride(CacheStride) {
        /* Empty */
    }

    // Given an array we copy that array to the cache starting at the beginning
    __device__ size_t set(T* array, uint arrayLength) {

        // only copy what we can
        size_t const elementsToCopy = min(Size,arrayLength);

        // Compute number of bytes to copy over
        size_t const N = elementsToCopy * Stride;

        // Copy over the bytes
        for(int i = threadIdx.x; i < N; i += blockDim.x) {
            data[i] = array[i];
        }

        // Return how many elements are actually stored
        return elementsToCopy;
    }

    // Given a position return a pointer to that element in 
    // the cache 
    __device__ T* get(size_t pos) {
        return data + (pos * Stride);
    }
#endif
};


struct CudaBoolean {
    uint *value;

    void init();
    void shutdown();

#ifdef __CUDACC__
    __device__ void toggle() {
        *value = !(*value);
    }
    __device__ void set(uint val) {
        *value = val;
    }
    __device__ uint get() {
        return *value;
    }
#endif
};


int getMaximumNumberOfBlocks(void);

#ifdef __CUDACC__
__device__ int globalThreadCount();
__device__ int globalID();
#endif

#endif