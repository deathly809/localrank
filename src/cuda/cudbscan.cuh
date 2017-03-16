
#ifndef DBSCAN_CUH_
#define DBSCAN_CUH_

#include <vector>

#include <util/dbscan.h>

#include <util/types.h>
#include <util/cudaUtil.cuh>

// dbscan takes in a set of data with the parameters for dbscan 
// and returns a list of clusters each element belongs to
//
//  data    - The data we want to cluster on
//  width   - the width of each data element
//  length  - how many data elements there are
//  k       - how many elements needed to become a cluster point
//  dist    - how far away do we look
//
//
//

struct DBSCANConfig {
    float* data;

    uint cacheSize,cacheStride;
    uint width,length,k;
    float epsSquared;

    void init();
    void shutdown();

#ifdef __CUDACC__
    __device__ __host__ float* getPoint(uint idx) {
        return data + (idx * width);
    }
#endif

};

struct DBSCANResult {

    uint N;
    uint* clusterIDs;
    uint* counts;

    void init();
    void shutdown();

};

class DBSCANCuda : public DBSCAN {
    private:

        // Cuda details
        DBSCANConfig d_Config;
        DBSCANResult d_Result;
        CudaBoolean d_Bool;
        uint numBlocks,numThreads;
        
        // Cpu details
        std::vector<std::vector<float>> data;
        std::vector<uint> counts;
        std::vector<uint> ids;

        uint width;

    public:

        DBSCANCuda(const std::vector<std::vector<float>> & features, uint K, float epsilon, Math::Metric & m);
        ~DBSCANCuda();

        void locateClusterPoints();
        void mergeClusters();
        void aggregateNoise();
        void run();

        std::vector<uint> getIDs() { return ids; }
        std::vector<uint> getCounts() { return counts;}

};

#endif