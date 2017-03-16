#include <chrono>

#include <iostream>

#include <vector>
#include <algorithm>

#include <cuda/cudbscan.cuh>
#include <util/dbscan.h>

#include <util/sort.h>
#include <util/assertion.h>

#include <tests/random.h>

#include <util/testing.h>

Math::Euclidian met;
const uint N = 5000;
const uint W = 5;
const float epsilon = 0.01;

const uint numClusters = W;
const uint clusterSize = N / numClusters;


struct Config {
    uint NumElements,ElementWidth,NumClusters,NumNeighbors;
    float Epsilon;

    Config(uint numElements, uint width, uint numNeighbors, float epsilon, uint numClusters) {
        NumElements = numElements;
        ElementWidth = width;
        NumNeighbors = numNeighbors;
        Epsilon = epsilon;
        NumClusters = numClusters;   
    }

    friend std::ostream& operator<<(std::ostream& os, const Config& c);

};

std::ostream& operator<<(std::ostream& os, const Config& c) {
    os << "Configuration {" 
        << c.NumElements 
        << "," 
        << c.ElementWidth 
        << "," 
        << c.NumNeighbors 
        << "," 
        << c.Epsilon 
        << "," 
        << c.NumClusters 
        << "}";
    return os;
}


/*
 *
 * c - Configuration information for dbscan data
 *
 *  @return -   Returns an array of length specified in Config object
 *              and is up to the user to free the memory.
 */
const std::vector<std::vector<float>> generateData(Config c) {
    std::vector<float> means(c.ElementWidth);
    std::vector<float> sigmas(c.ElementWidth);
    std::vector<std::vector<float>> result;
    
    float startingPoint = -1E4;
    float delta = 1E2;

    uint remaining = c.NumElements;
    uint numElements = c.NumElements / c.NumClusters; 

    // Init
    std::fill(sigmas.begin(),sigmas.end(),0.0001f * numElements * c.Epsilon);
    std::fill(means.begin(),means.end(),0.0);

    for(uint clusterID = 0 ; clusterID < c.NumClusters - 1; clusterID++) {
        means[0] = startingPoint; 
        for(uint elementID = 0; elementID < numElements; elementID++) {
            std::vector<float> noise = generateGaussianVectorNoise(c.ElementWidth,means,sigmas);
            noise[clusterID] = startingPoint;
            result.push_back(noise);
            Assert(noise.size(),c.ElementWidth,true);
        }
        startingPoint += delta;
        remaining -= numElements;
    }

    std::cout << "cluster=" << (c.NumClusters - 1) << std::endl;

    means[0] = startingPoint;
    for(uint elementID = 0; elementID < remaining; elementID++) {
        std::vector<float> noise = generateGaussianVectorNoise(c.ElementWidth,means,sigmas);
        noise[c.ElementWidth - 1] = startingPoint;
        result.push_back(noise);
    }
    
    return result;
}

// This should take in the results and data and output information about 
// the clusters along with information for each cluster 
void displayClusters(std::vector<uint> && results,Config c) {

    std::sort(results.begin(),results.end());

    std::vector<uint> clusterInformation;
    uint count = 1;
    
    for(uint i = 1 ; i < c.NumElements;i++) {
        if(results[i] != results[i-1]) {
            std::cout << "pushing back a new count:" << clusterInformation.size() << std::endl;
            clusterInformation.push_back(count);
            count = 0;
        }
        ++count;
    }
    clusterInformation.push_back(count);

    uint NumClusters = clusterInformation.size();
    std::cout << "Number of clusters detected: " << NumClusters << std::endl;
    std::cout << "\tNumber expected: " << c.NumClusters << std::endl;
    std::cout << std::endl;

    if(NumClusters > 10) {
        std::cout << "Too many cluster... printing first 10 clusters only" << std::endl;
        NumClusters = 10;
    }

    for(uint i = 0 ; i < NumClusters;i++) {
        std::cout << "Cluster " << (i+1) << std::endl;
        std::cout << "\tNumber of elements: " << clusterInformation[i] << std::endl;
        std::cout << std::endl;
    }
}


const std::vector<Config> Configurations = {
    Config(10000,4,5,5,5)  // 20000 elements, of width 4, requires 5 neighbors, withing 5 epsilon of each other
};

void validateInitial(std::vector<uint> && counts, std::vector<uint> && ids, uint numClusters, uint clusterSize) {
    for(uint i = 0; i < numClusters*clusterSize; ++i) {
        Assert(counts[i],1,true);
        Assert(ids[i],i,true);
    }
}

void validateCounts(std::vector<uint> && counts, uint numClusters, uint clusterSize) {
    Assert(numClusters*clusterSize,counts.size(),true);
    for(uint i = 0; i < numClusters*clusterSize; ++i) {
        Assert(counts[i],clusterSize,true);
    }
}

void validateIds(std::vector<uint> && ids, uint numClusters, uint clusterSize) {
    uint pos = 0;
    for(uint i = 0; i < numClusters; ++i) {
        for(uint j = 0;j < clusterSize; ++j) {
            Assert(ids[pos],i*clusterSize,true);
            ++pos;
        }
    }
}

void validateNoise(std::vector<uint> && ids, uint numClusters, uint clusterSize) {
    for(uint i = 0; i < numClusters*clusterSize; ++i) {
        Assert(ids[i],-1,false);
    }
}

std::vector<std::vector<float>> data(N,std::vector<float>(W,0));

void validateData(const std::vector<std::vector<float>> & data, Config & c) {
    Assert(data.size(),c.NumElements,true);
    for(int i = 0 ; i < data.size(); ++i) {
        Assert(data[i].size(),c.ElementWidth,true);
    }
}

void init() {
    uint pos = 0;
    for(auto it = data.begin(); it != data.end(); ++it) {
        (*it)[pos / clusterSize] = 1;
        ++pos;
    }

    for(int i = 0 ; i < numClusters;++i) {
        for(int j = 0; j < clusterSize; ++j) {
            for(int k = 0; k < W; ++k) {
                Assert(data[i*clusterSize + j][k],1,k == i);
            }
        }
    }
}

template<class C>
void testCount() {
    C alg(data,W,epsilon,met);
    validateInitial(alg.getCounts(),alg.getIDs(),numClusters,clusterSize);
    alg.locateClusterPoints();
    validateCounts(alg.getCounts(),numClusters,clusterSize);
}


template<class C>
void testMergeClusters() {
    C alg(data,W,epsilon,met);
    alg.locateClusterPoints();
    alg.mergeClusters();
    validateIds(alg.getIDs(),numClusters,clusterSize);
}

template<class C>
void testAggregateNoise() {
    C alg(data,W,epsilon,met);
    alg.run();
    validateNoise(alg.getIDs(),numClusters,clusterSize);
}

template<class C>
void Run(std::string label,const std::vector<std::vector<float>> & data,Config & c) {
    std::cout << label << std::endl;

    C alg(data,c.NumNeighbors,c.Epsilon,met);

    validateData(data,c);
    //validateData(alg.data,c);
    
    auto start = std::chrono::high_resolution_clock::now();
    alg.locateClusterPoints();
    alg.mergeClusters();
    
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "ms\n" << std::endl;

    std::cout << "Display Cluster" << std::endl;

    displayClusters(alg.getIDs(),c);
}

void testAll() {   
    for(Config c : Configurations) {

        std::cout << c << std::endl;
        const std::vector<std::vector<float>> data = generateData(c);
#ifdef __CUDA_ARCH__
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);

        if(deviceCount > 0){
            Run<DBSCANCuda>("GPU",data,c);
        }
#endif
        Run<DBSCANCPU>("CPU",data,c);
        
    }
}

int main() {

    init();
    RunTest("testCount<DBSCANCPU>",testCount<DBSCANCPU>,NOP);
    RunTest("testMergeClusters<DBSCANCPU>",testMergeClusters<DBSCANCPU>,NOP);
    RunTest("testAggregateNoise<DBSCANCPU>",testAggregateNoise<DBSCANCPU>,NOP);
    std::cout << std::endl;

#ifdef __CUDA_ARCH__

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if(deviceCount > 0){
        RunTest("testCount<DBSCANCuda>",testCount<DBSCANCuda>,NOP);
        RunTest("testMergeClusters<DBSCANCuda>",testMergeClusters<DBSCANCuda>,NOP);
        RunTest("testAggregateNoise<DBSCANCuda>",testAggregateNoise<DBSCANCuda>,NOP);
    }
#endif


    RunTest("testAll",testAll,NOP);
}
