
#include <util/cudaSync.cuh>

Sync SyncFactory() {
    Sync result;

    CudaTest(cudaMalloc(&result.count,sizeof(int)));
    CudaTest(cudaMalloc(&result.lock,sizeof(int)));

    CudaTest(cudaMemset(result.count,0,sizeof(int)));
    CudaTest(cudaMemset(result.lock,result.Unlocked,sizeof(int)));

    return result;
}

void SyncCleanup(Sync & sync) {
    cudaFree(sync.count);
    cudaFree(sync.lock);

    sync.count  = nullptr;
    sync.lock   = nullptr;
}