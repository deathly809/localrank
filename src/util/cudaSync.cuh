#ifndef SYNC_CUH
#define SYNC_CUH

#include <util/cudaUtil.cuh>

/*
 *  Synchronization wrapper for CUDA
 *
 *  This supports locking, synchronization across and within blocks.
 *
 */
struct Sync {

    const static int Unlocked = 0;

    uint *count;
    int *lock;

    void Init() {
        cudaMalloc(&count,sizeof(int));
        cudaMalloc(&lock,sizeof(int));

        cudaMemset(count,0,sizeof(int));
        cudaMemset(lock,0,sizeof(int));
    }

    void Cleanup() {
        cudaFree(count);
        cudaFree(lock);
    }

    __device__ void Block() {
        __syncthreads();
    }

    __device__ void Global() {

        const int curr = blockIdx.x;
        const int next = blockIdx.x + 1;

        const int Max = gridDim.x;
    
        // Wait for my turn, when my turn increment
        if(threadIdx.x == 0) while(atomicCAS(count,curr,next) != curr);

        // If we have seen each block, reset
        if(threadIdx.x == 0) atomicCAS(count,Max,0);

        // Wait for reset
        if(threadIdx.x == 0) while(atomicCAS(count,0,0) != 0);

        Block();
    }

    // This locks globally
    __device__ void Lock() {
        
        while(*lock != blockIdx.x) {
            Block();
            if(threadIdx.x == 0) {
                // If unlocked, try to grab it
                atomicCAS(lock,Unlocked,(int)blockIdx.x);
            }
        }
        Block();
    }
    // This unlocks globally
    __device__ void Unlock() {
        if(*lock == blockIdx.x) {
            *lock = Unlocked;
        }
        Block();
    }
};


Sync SyncFactory();
void SyncCleanup(Sync&);


#endif