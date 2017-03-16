
/*
 *  Description
 *  ######################
 *  
 *      Our algorithm implements a naive parallelized version of DBSCAN.  We break our algorithm 
 *      into three steps.
 *
 *          1.  Construct the set of neighbors for each point.
 *
 *          2.  For each node we have two cases: it is a core point, or not
 *
 *              [core-point].   For each neighbor if it is not a core-point add it to our cluster, 
 *                              and if it is a core point, place each point in the smaller cluster.
 *
 *              [otherwise].    If the neighbor is a core point, add us to it's cluster, otherwise     
 *                              do nothing.
 *
 *          3.  For each node, if it is not a core point and has not been placed in a cluster we 
 *              label it as an outlier.
 *
 */

#include <util/dbscan.h>
#include <util/assertion.h>

#include <algorithm>
#include <random>

#include <thread>
#include <mutex>
#include <functional>


const uint Threads = 16;
const uint WorkPerThread = 2048;

DBSCANCPU::DBSCANCPU(const std::vector<std::vector<float>> & features,uint K, float epsilon, Math::Metric & m) :
    data(features),
    counts(features.size(),1),
    neighbors(features.size()), 
    DBSCAN(features.size(),K,epsilon),
    tree(features,m) {

    for(uint i = 0 ; i < N;++i) {
        ids.push_back(i);
    }
}

// Used for parallel work
struct ParallelArguments {
    std::vector<std::vector<float>> & data;
    std::vector<uint> & counts;
    std::vector<uint> & ids;
    std::vector<std::vector<uint>> & neighbors;
    KD::KDTree & tree;
    float epsilon;
    uint K,N,M;
    std::mutex &mtx;
};

void parallelLocateClusterPoints(uint start, ParallelArguments w ) {
    uint work = (w.N + w.M - 1) / w.M;
    start = start * work;
    uint end = std::min(w.N,start + work);

    for( int i = start; i < end; ++i) {
        w.neighbors[i] = w.tree.findNeighborIndices(w.data[i],w.epsilon);
        w.counts[i] = w.neighbors[i].size();
    }
}

void DBSCANCPU::locateClusterPoints() {
    std::thread threads[Threads];
    uint M = std::min(Threads,(N+WorkPerThread-1)/WorkPerThread);

    std::mutex lock;
    ParallelArguments w = {data,counts,ids,neighbors,tree,epsilon,K,N,M,lock};

    
    
    for(int j = 1 ; j < M; ++j) {
        threads[j] = std::thread(parallelLocateClusterPoints,j,w);
    }

    parallelLocateClusterPoints(0,w);

    for( int j = 1 ; j < M ; ++j) {
        threads[j].join();
    }

    #if defined(VERIFY)
    std::cout << "VERIFY" << std::endl;
    Math::Euclidian met;
    for(int i = 0 ; i < N; ++i) {
        int count = 0;
        for(int j = 0 ; j < N; ++j) {
            if(met.within(data[i],data[j],epsilon)) ++count;
        }
        Assert(counts[i],neighbors[i].size(),true);
        Assert(count,neighbors[i].size(),true);
    }
    #endif
    
}

uint getID(uint idI, uint idJ, uint countsI, uint countsJ, uint K) {
    if(countsI >= K && countsJ >= K) {
        return std::min(idI,idJ);
    }
    if(countsJ >= K) return idJ;
    
    return idI;
}

void parallelMerge(uint start, ParallelArguments w, bool & done) {
    uint work = (w.N + w.M - 1) / w.M;
    start = start * work;
    uint end = std::min(w.N,start + work);

    for(int i = start; i < end; ++i) {
        
        // Not a core point, and in a cluster
        if(w.counts[i] < w.K && w.ids[i] != i) {

            // cluster id might have updated
            w.mtx.lock();
            if(w.ids[i] != w.ids[w.ids[i]]) {
                w.ids[i] = w.ids[w.ids[i]];
                done = false;
            }
            w.mtx.unlock();
            continue;
        }

        for( auto j : w.neighbors[i] ) {

            if(j == i) continue;

            w.mtx.lock();
            uint id = getID(w.ids[i],w.ids[j],w.counts[i],w.counts[j],w.K);
            if(id != w.ids[i]) {
                w.ids[i] = id;
                done = false;
            }
            w.mtx.unlock();

        }
    }
}

void DBSCANCPU::mergeClusters() {
    std::thread threads[Threads];
    std::mutex lock;

    uint M = std::min(Threads,(N+WorkPerThread-1)/WorkPerThread);

    ParallelArguments w = {data,counts,ids,neighbors,tree,epsilon,K,N,M,lock};

    while(true) {
        bool done = true;
            
        for(int j = 1 ; j < M; ++j) {
            threads[j] = std::thread(parallelMerge,j,w,std::ref(done));
        }

        parallelMerge(0,w,done);

        for( int j = 1 ; j < M ; ++j) {
            threads[j].join();
        }

        if(done) break;
    }

    #if defined(VERIFY)
    for(int i = 0 ; i < N; ++i) {

        auto idI = ids[i];
        auto iIsCore = counts[i] >= K;

        bool found = false;

        for(auto j : neighbors[i]) {

            auto idJ = ids[j];
            auto jIsCore = counts[j] >= K;

            // If both are core points we 
            // better be in the same cluster
            if(iIsCore && jIsCore) {
                Assert(idI,idJ,true);
            }else if(jIsCore) {
                if(idJ == idI) {
                    found = true;
                }
            }
        }

        // I am not a core-point, but my id has changed,
        // therefore one of my neighbors is a core point,
        // and my id is the same as one of theirs
        if( (!iIsCore) && (idI != i) ) AssertTrue(found);
    }
    #endif
}

void parallelAggregateNoise(uint start,ParallelArguments w) {
    for(uint i = start ; i < w.N; i += w.M) {
        int C = 0;
        int myID = w.ids[i];
        for(auto j : w.neighbors[i]) {
            if(w.ids[j] == myID) {
                C++;
            }
        }

        if( C < w.K) {
            w.ids[i] = w.N;
        }
    }
}

void DBSCANCPU::aggregateNoise() {
    std::thread threads[Threads];
    std::mutex lock;
    uint M = std::min(Threads,(N+WorkPerThread-1)/WorkPerThread);

    ParallelArguments w = {data,counts,ids,neighbors,tree,epsilon,K,N,M,lock};
    
    for(int j = 1 ; j < M; ++j) {
        threads[j] = std::thread(parallelAggregateNoise,j,w);
    }

    parallelAggregateNoise(0,w);

    for( int j = 1 ; j < M ; ++j) {
        threads[j].join();
    }

}

void DBSCANCPU::run() {
    locateClusterPoints();
    mergeClusters();
    aggregateNoise();
}

std::ostream& operator << (std::ostream& os, const std::nullptr_t ptr) {
    return os << std::string("nullptr");
}
