/*
 *
 *  Idea
 *  ==============================
 *
 *      What we want to do is perform DBScan by breaking it into multiple steps.
 *
 *
 *      Step 1: Find all core points
 *
 *          We can do this by going through each pair of points and if they are  
 *          within distance EPSILON increment the number of nodes.
 *
 *      Step 2: Group core points together
 *
 *          This is done by iterating over each point and if possible label it  
 *          according to it's closest core point.  We do this until no further 
 *          updates are possible.
 *
 *      Step 3: Group noise into single group.
 *
 *          Go over each point and if it is not a core point and is in it's own
 *          cluster we label it as noise.
 *
 */


#include <cuda/cudbscan.cuh>

#include <util/assertion.h>
#include <util/cudaUtil.cuh>
#include <util/cudaSync.cuh>


const uint KB = 1024;
const uint MB = 1024 * KB;
const uint MaxIterations = 16 * MB;

__managed__ Sync dSync;
__shared__ char SharedBuffer[47*KB];

#define isClusterPoint(ARRAY,K,C) (ARRAY[C] >= K)

/*
 *  distanceSquared -   returns the square of the euclidian distance of 
 *  two points.
 *
 *  @first  - The index of the first point
 *  @second - The index of the second point
 *  @config - configuration needed to run DBSCAN
 *
 *  @return - The square of distance
 *
 */
__device__ float withinDistance(const float* const & first, const float* const & second, const DBSCANConfig & config) {
    float sum = 0;
    for(uint i = 0 ; i < config.width; i++) {
        float tmp = first[i] - second[i];
        sum += tmp * tmp;
        //if(sum > config.epsSquared) return false; // Most points will know early on
    }
    return sum < config.epsSquared;
}

// Logging tool
__device__ void log(char* msg) {
    if(threadIdx.x == 0) printf("%d: %s\n",blockIdx.x,msg);
} 

template<typename T>
__device__ void abs(T a) {
    return (a < 0) ? -a : a;
}

__device__ void validate(float* first, float* second,uint N) {    
    for(int tid = threadIdx.x; tid < N; tid += blockDim.x) {
        if(first[tid] != second[tid]) {
            printf("Validation failed: %d - expected %f, found %f\n",tid,first[tid],second[tid]);
        }
    }
    dSync.Block();
}

#define Vector(NAME) NAME[0],NAME[1],NAME[2],NAME[3],NAME[4]


//Count 
template<bool DEBUG = false>
__device__ void findNeighbors(DBSCANConfig config, DBSCANResult result,const int & front,Cache<float> & cache) {

    uint const firstIdx = globalID() + front;
    float *first = (firstIdx < config.length)? config.getPoint(firstIdx) : nullptr;

    for(int back = config.length-1; front < back; back -= cache.Size) {

        const uint backFirst = max(front,(int)(back - cache.Size)) + 1;
        const uint NumCached = cache.set(config.getPoint(backFirst),back-backFirst+1);
        uint cacheIdx = back;
        dSync.Block();

        if(firstIdx < config.length) {
            uint localCount = 0;
            for(int k = NumCached - 1 ; (k >= 0 && cacheIdx > firstIdx); --k,--cacheIdx) {
                const float* second = cache.get(k);   
                if(withinDistance(first,second,config)) {
                    ++localCount;
                    atomicAdd(result.counts + cacheIdx,1u);  // TODO: Move to cache as well?

                }
            }
            atomicAdd(result.counts + firstIdx,localCount);
            dSync.Block(); // wait for everyone to finish using the cache
        }        
    }
}

/*
 *  LabelCorePoints -   For each point we compute the number of points within
 *      distance EPSILON.  Additionally, we put each point in it's own cluster. 
 * 
 *  @config -   The DBSCAN configration describing all the parameters to DBSCAN 
 *      along with providing the data.
 *  @result -   All results will be stored here.  We assume that all values are 
 *      initialized to 0.
 *
 *  @runtime    -   Current work is O(kN^2) were N is the number of points and k is 
 *      the dimenion of a point.
 *
 */
 template<bool DEBUG = false>
__global__ void LabelCorePoints(DBSCANConfig config, DBSCANResult result) {

    Cache<float> cache(SharedBuffer,config.cacheSize,config.cacheStride);
    uint const NumThreads = globalThreadCount();

    if(DEBUG) {
        for(int i = globalID(); i < config.length; i += NumThreads) {
            const float* first = config.getPoint(i);
            for(int j = i + 1; j < config.length; ++j) {
                const float* second  = config.getPoint(j);
                if(withinDistance(first,second,config)) {
                    atomicAdd(result.counts + i,1u);
                    atomicAdd(result.counts + j,1u);
                }
            }
        }
    } else {
        for(int front = 0;front < config.length;front += NumThreads) {
            findNeighbors(config,result,front,cache);
        }
    }
}






// Given a data point which might be a cluster point, merge it 
// with cluster points withing epsilon distance
template<bool DEBUG = false>
__device__ void merge(DBSCANConfig config, DBSCANResult result,
    const int & front,Cache<float> & cache, uint & done) {

    uint const firstIdx = globalID() + front;
    float* first = nullptr;
    bool firstIsClusterPoint;
    uint firstClusterID;

    if(firstIdx < config.length) {
        firstIsClusterPoint = result.counts[firstIdx] >= config.k;
        firstClusterID   = result.clusterIDs[firstIdx];
        first = config.getPoint(firstIdx);
    }

     // Scan from last element up to element after front
    for(int back = config.length-1; front < back; back -= cache.Size) {
        const uint backFirst = max(front,(int)(back - cache.Size)) + 1;
        const uint NumCached = cache.set(config.getPoint(backFirst),back-backFirst+1);
        uint lastIdx = back;
        dSync.Block();

        if( firstIdx < config.length) {
            for(int k = NumCached - 1 ; k >= 0 && lastIdx > firstIdx; --k, --lastIdx) {
                const bool lastIsClusterPoint = result.counts[lastIdx] >= config.k;

                if(firstIsClusterPoint || lastIsClusterPoint) {
                    uint const lastClusterID = result.clusterIDs[lastIdx];

                    if(firstClusterID != lastClusterID) {
                        const float* const second = cache.get(k);

                        if(withinDistance(first,second,config)) {
                            uint const r = min(firstClusterID,lastClusterID);
                            atomicMin(result.clusterIDs + firstIdx,r);
                            atomicMin(result.clusterIDs + lastIdx,r);
                            done = 0;
                        }
                    }
                }
            }
        }
        dSync.Block();
    }
}


/*
 *  LabelClusters   -   Given a set of points, some labeled as core points, we 
 *      place them in clusters as decribed by DBSCAN.
 *
 *  @config -   The DBSCAN configration describing all the parameters to DBSCAN 
 *      along with providing the data.
 *  @result -   All results will be stored here.  We assume that the count array
 *      has been filled already.
 *  @Done   -   A variable used by all threads to determine if computation is done.
 *
 *
 *  @runtime    -   Current work is O(kN^2) were N is the number of points and k is 
 *      the dimenion of a point.
 *
 */
 template<bool DEBUG = false>
__global__ void LabelClusters(DBSCANConfig config, DBSCANResult result, CudaBoolean Done) {

    __shared__ uint done; done = 0;

    Cache<float> cache(SharedBuffer,config.cacheSize,config.cacheStride);
    uint const NumThreads = globalThreadCount();

    Done.set(0);
    dSync.Block();

    uint iter = 0;
    while(Done.get() == 0 && iter < MaxIterations)  {
        ++iter;

        dSync.Global();
        Done.set(1);
        done = 1;

        for(int front = 0;front < config.length;front+=NumThreads) {
            merge(config,result,front,cache,done);
        }
        
        if(done == 0) Done.set(0);
        dSync.Global();
    }
}




/*
 *  ClusterNoise    -   Place all points which are classified as noise into
 *      a single cluster.  We assume that the user has called LabelCorePoints
 *      and LabelClusters, in that order, previously.
 *
 *
 *  @config -   The DBSCAN configration describing all the parameters to DBSCAN 
 *      along with providing the data.
 *  @result -   All results will be stored here.
 *
 *
 *
 *
 *
 */
 template<bool DEBUG = false>
__global__ void ClusterNoise(DBSCANConfig config, DBSCANResult result) {
    uint gID = globalID();
    uint const NumThreads = globalThreadCount();

    for(uint elementID = gID; elementID < config.length; elementID += NumThreads) {
        uint clusterID = result.clusterIDs[elementID];
        uint count = result.counts[elementID];
        if(count < config.k && clusterID == elementID) {
            result.clusterIDs[elementID] = config.length;
        }
    }
}

/*
 *  getMaxBlocks    -   Compute the maximum number of blocks which can be executed
 *      concurrently.
 *
 *  @length -   The number of points.
 *
 *
 *  @return -   The maximum number of blocks. 
 *
 *
 */
uint getMaxBlocks(uint length) {
    uint Max = getMaximumNumberOfBlocks();
    return min(Max,(length + 1023) / 1024);
}

void printCache(DBSCANConfig & config) {
    std::cout << "Cache info" << std::endl;
    std::cout << "\tNumber Elements:\t" << config.length << std::endl;
    std::cout << "\tElement Width:\t\t" << config.width << std::endl;
    std::cout << "\tEpsilon^2:\t\t" << config.epsSquared << std::endl;
    std::cout << "\tK:\t\t\t" << config.k << std::endl;
    std::cout << "\tCache Size:\t\t" << config.cacheSize << std::endl;
    std::cout << "\tCache Stride:\t\t" << config.cacheStride << std::endl;

}

void DBSCANConfig::init() {
    CudaTest(cudaMalloc(&data,sizeof(float) * width * length));
}

void DBSCANConfig::shutdown() {
    CudaTest(cudaFree(data));
}

void DBSCANResult::init() {
    CudaTest(cudaMalloc(&clusterIDs,sizeof(uint) * N));
    CudaTest(cudaMalloc(&counts,sizeof(uint) * N));
}

void DBSCANResult::shutdown() {
    CudaTest(cudaFree(clusterIDs));
    CudaTest(cudaFree(counts));
}


DBSCANCuda::DBSCANCuda(const std::vector<std::vector<float>> & features, uint K, float epsilon, Math::Metric & m) : 
    data(features.size()),
    counts(features.size(),1),
    DBSCAN(features.size(),K,epsilon) {

    CudaTest(cudaDeviceReset());
    this->width = features[0].size();

    data = features;
    
    d_Config.k = this->K;
    d_Config.width = this->width;
    d_Config.length = this->N;
    d_Config.epsSquared = this->epsilon * this->epsilon;
    d_Config.cacheStride = this->width;
    d_Config.cacheSize = min(this->N, ( (47*KB/4) / this->width));

    d_Result.N = N;

    d_Config.init();
    d_Result.init();
    d_Bool.init();
    
    // Initialize local copy
    for(uint i = 0 ; i < N; ++i) {
        ids.push_back(i);
    }
    
    for(int i = 0; i < data.size(); ++i) {
        CudaTest(cudaMemcpy(d_Config.data + i * width,&data[i][0],sizeof(float) * width,cudaMemcpyHostToDevice));
    }

    CudaTest(cudaMemcpy(d_Result.counts,&counts[0],sizeof(uint)*N,cudaMemcpyHostToDevice));
    CudaTest(cudaMemcpy(d_Result.clusterIDs,&ids[0],sizeof(uint)*N,cudaMemcpyHostToDevice));

    numThreads = min(1024,N);
    numBlocks = min(8,(N+1023)/1024);
    

    dSync.Init();

}

DBSCANCuda::~DBSCANCuda() {
    dSync.Cleanup();
    d_Bool.shutdown();
    d_Config.shutdown();
    d_Result.shutdown();
}

void DBSCANCuda::locateClusterPoints() {
    RunKernel(LabelCorePoints<true>,numBlocks,numThreads,d_Config,d_Result);
    RunCudaOp(cudaMemcpy(&counts[0],d_Result.counts,sizeof(uint)*N,cudaMemcpyDeviceToHost));
}

void DBSCANCuda::mergeClusters() {
    RunKernel(LabelClusters,numBlocks,numThreads,d_Config,d_Result,d_Bool);
    RunCudaOp(cudaMemcpy(&ids[0],d_Result.clusterIDs,sizeof(uint)*N,cudaMemcpyDeviceToHost));
}

void DBSCANCuda::aggregateNoise() {
    RunKernel(ClusterNoise,numBlocks,numThreads,d_Config,d_Result);
    RunCudaOp(cudaMemcpy(&ids[0],d_Result.clusterIDs,sizeof(uint)*N,cudaMemcpyDeviceToHost));
}

void DBSCANCuda::run() {
    locateClusterPoints();
    mergeClusters();
    aggregateNoise();    
}













