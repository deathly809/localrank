
#include <iostream>

#include <vector>
#include <tuple>

#include <cmath>


#include <util/types.h>
#include <util/kdtypes.h>
#include <util/metric.h>


#ifndef KDTREE_H_
#define KDTREE_H_

#include <iostream>

namespace KD {

    class KDTree {

        private:
            Math::Metric & metric;
            KDNode *root;
            KDData data;
            uint W;

        public:

            KDTree();
            KDTree(const std::vector<Point> & data, Math::Metric & metric);
            KDTree(KDTree && other);
            KDTree(const KDTree & other);
            ~KDTree();

            KDTree & operator=(KDTree && other);
            KDTree & operator=(const KDTree & other);

            std::vector<std::vector<float>> findNeighbors(const Point & point,float epsilon);
            std::vector<uint> findNeighborIndices(const Point & point,float epsilon);
            bool contains(const Point & point);

    };

}

#endif