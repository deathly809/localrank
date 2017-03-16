

/*
 *  
 *  DBSCAN
 *
 *      DBSCAN is a clustering algorithm which designates points either core points, edge points, 
 *      or outliers.
 *
 *      The algorithm takes in two parameters, epsilon and K, which represents how far away we look
 *      for points and how many we need to be a core point respectively.
 *
 *      A point is a core point if there at least K points, including yourself, within an epsilon 
 *      neighborhood of that point.  All points within this neighborhood are called reachable 
 *      points, non-core points are called edge points, and all points not in any 
 *      epsilon-neighborhood of a core point is called an outlier.
 *
 *      From these definitions a cluster can be definied as a set of core points, and their edge 
 *      points, such that given a any two core points we can find a path of core points from one   
 *      to the other using epsilon-neighborhoods as defining the connections.
 *
 *  TODO(deathly809)    :   Possible reword in a better way
 */



#ifndef DBSCAN_H_
#define DBSCAN_H_

#include <iostream>
#include <vector>

#include <util/kdtree.h>
#include <util/types.h>

template<class T>
void printVector(const std::vector<T> & vec) {
    std::cout << "<";
    for(auto it = vec.begin(); it != vec.end(); ++it) {
        if((it + 1) == vec.end()) {
            std::cout << *it;
        }else {
            std::cout << *it << ", ";
        }
    }
    std::cout << ">";
}

class DBSCAN {

    protected:
        uint K,N;
        float epsilon;

        DBSCAN(uint N, uint K,float E) : N(N), K(K), epsilon(E) { /* EMPTY */}

    public:

        virtual void locateClusterPoints() = 0;
        virtual void mergeClusters() = 0;
        virtual void aggregateNoise() = 0;

        virtual void run() = 0;

        uint getK() { return K;}
        uint getN() { return N;}
        float getEpsilon() {return epsilon;}

        virtual std::vector<uint> getIDs() = 0;
        virtual std::vector<uint> getCounts() = 0;

};


class DBSCANCPU : public DBSCAN {

    private:
        KD::KDTree tree;
        std::vector<uint> counts;
        std::vector<uint> ids;
        std::vector<std::vector<float>> data;
        std::vector<std::vector<uint>> neighbors;

    public:
        DBSCANCPU(const std::vector<std::vector<float>> & features,uint K, float epsilon,Math::Metric & m);

        void locateClusterPoints();
        void mergeClusters();
        void aggregateNoise();
        void run();

        std::vector<uint> getIDs() { return ids; }
        std::vector<uint> getCounts() { return counts;}

};

std::ostream& operator << (std::ostream& os, const std::nullptr_t ptr);

#endif

