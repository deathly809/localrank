

/*
 *      Construct a KDTree
 *
 *  TODO(deathly809) - Remove recursion for construction and retrieval
 *
 */

#include <thread>

#include <algorithm>

#include <util/kdtree.h>
#include <util/assertion.h>

#include <exception>

namespace KD {

    Math::Euclidian HIDDEN_euc;
    typedef std::vector<std::tuple<uint,uint>> Indices;


    #ifndef BUCKET_SIZE
    const uint BucketSize = 4;
    #else
    const uint BucketSize = BUCKET_SIZE;
    #endif

    /* Helper Functions */

    template<typename T>
    void printVector(const std::vector<T> & vec) {
        std::cout << "{";
        for( auto it = vec.begin(); it < vec.end(); ++it) {
            std::cout << "<" << *it << ">";
        }
        std::cout << "}";
    }

    void printTree(KDData & data, KDNode *node,std::string depth) {
        if(node == nullptr) return;

        printTree(data,node->left,depth + "\t");
        std::cout << depth << node->dim << ": <" <<  node->value << " " << node->leftIndex << " " << node->rightIndex << " " << node->minValue << " " << node->maxValue << "> ";
        if(node->left == node->right) {
            std::cout << "[";
            for(int i = node->leftIndex ; i < node->rightIndex; ++i) {
                std::cout << " ";
                std::cout << i << " : " << data[i].origIDX << " ";
                printVector(data[i].p);
            }
            std::cout << "]";
        }
        std::cout << std::endl;
        printTree(data,node->right,depth + "\t");
    }


    uint validateTree(KDData & data, KDNode* node, std::vector<std::tuple<bool,float>> & test ) {
        if(node == nullptr) return 0;

        uint result = 0;

        if(node->left == node->right) {
            for(int i = node->leftIndex; i < node->rightIndex; ++i) {

                const bool Right = true;
                const auto vec = data[i].p;
                const uint W = vec.size();

                for(int j = 0 ; j < test.size(); ++j) {
                    float value = vec[j%vec.size()];

                    bool splitDir   = std::get<0>(test[j]);
                    float pivot = std::get<1>(test[j]);

                    if(splitDir == Right) AssertTrue(pivot <= value);
                    else AssertTrue(value <= pivot);

                }
            }
            result += node->rightIndex - node->leftIndex;
        }else {
            test.push_back(std::make_tuple(false,node->value));
            result += validateTree(data,node->left,test);
            test.pop_back();

            test.push_back(std::make_tuple(true,node->value));
            result += validateTree(data,node->right,test);
            test.pop_back();
        }
        return result;
    }

    
    template<class BidirIt>
    void validatePartition(BidirIt first, BidirIt mid, BidirIt last, uint dim, float pivot) {

            int state = 0;
            for(auto it = first; it != mid; it++) {
                auto v = (*it).p[dim];
                AssertTrue(v <= pivot);
            }

            for(auto it = mid; it != last; it++) {
                auto v = (*it).p[dim];
                AssertTrue(v >= pivot);
            }
    }

    template<class BidirIt,typename Comp>
    BidirIt part(BidirIt first, BidirIt last, Comp &cmp) {
        BidirIt mid = first + (last - first) / 2;
        std::nth_element(first,mid,last,cmp);
        return mid;
    }

    void CheckRanges(int left, int right, int N) {
        AssertTrue(left < right);

        AssertTrue(left >= 0);
        AssertTrue(right > 0);

        AssertTrue(left < N);
        AssertTrue(right <= N);

        AssertTrue((right-left) <= N);
    }

    void CompareCheck(const Tuple & f, const Tuple & s, uint dim) {
        AssertTrue(s.p.size() > 0);
        AssertTrue(f.p.size() > 0);

        AssertTrue(dim < s.p.size());
        AssertTrue(dim < f.p.size());

        Assert(f.p.size(),s.p.size(),true);
    }

    void initRoot(KDNode* root,uint dim,float pivot,int left, int right,float min, float max){
        root->dim = dim;
        root->value = pivot;
        root->leftIndex = left;
        root->rightIndex = right;
        root->minValue = min;
        root->maxValue = max;
    }

    // [left,right)
    KDNode* createRoot(KDData & data, int left, int right, uint dim) {
        KDNode* root = new KDNode();

        CheckRanges(left,right,data.size());

        auto cmp = [dim](const Tuple & f, const Tuple & s) {
            CompareCheck(f,s,dim);
            return f.p[dim] < s.p[dim];
        };

        int size = right - left;

        auto first = data.begin() + left;
        auto last = data.begin() + right;

        if( size <= BucketSize) {
            auto min = (*std::min_element(first,last,cmp)).p[dim];
            auto max = (*std::max_element(first,last,cmp)).p[dim];
            initRoot(root,dim,right-left,left,right,min,max);
        }else {
            uint width      = data[0].p.size();
            uint nextDim    = (dim + 1) % width;

            int pivotPos = (left + right) / 2;

            auto mid = data.begin() + pivotPos;
            nth_element(first,mid,last,cmp);
            float pivot = (*mid).p[dim];

            auto min = (*std::min_element(first,mid,cmp)).p[dim];
            auto max = (*std::max_element(mid,last,cmp)).p[dim];
            initRoot(root,dim,pivot,left,right,min,max);

    #if defined(VERIFY)
            validatePartition(first,mid,last,dim,pivot);
    #endif

            root->left  = createRoot(data,left,pivotPos,nextDim);
            root->right  = createRoot(data,pivotPos,right,nextDim);
        }
        return root;
    }


    /***************************************************************************************************************/
    /****************************************** KDTree class  ******************************************************/
    /***************************************************************************************************************/

    KDTree::KDTree() : metric(HIDDEN_euc) {

    }

    void printTree(KDData &data, int left, int right, std::string prefix,uint dim, uint W) {
        if( (right - left) <= BucketSize) {
            std::cout << prefix << "[" << left << ":" << right << "] : " << data[(right+left)/2].p[dim] << ":\t";
            for(int i = left; i < right; ++i) {
                printVector(data[i].p);
                std::cout << " ";
            }
            std::cout << std::endl;
        }else {
            printTree(data,left,(right+left)/2,prefix + "\t",(dim+1)%W,W);
            std::cout << prefix << "[" << left << ":" << right << "] : " << data[(right+left)/2].p[dim] << std::endl; 
            printTree(data,(right+left)/2,right,prefix + "\t",(dim+1)%W,W);
        }
    }

    // O(n * d * log(n) )
    KDTree::KDTree(const std::vector<Point> & d, Math::Metric & m) : data(d.size()), metric(m) {
        if(d.size() > 0) {
            this->W = d[0].size();
        }else {
            this->W = 0;
        }

        if(this->W != 0) {

            for(uint i = 0 ; i < d.size(); ++i) {
                if(d[i].size() != this->W) {
                    throw std::runtime_error("all points must have same dimension");
                }
                this->data[i].origIDX = i;
                this->data[i].p = d[i];
            }

    #if defined(DEBUG) || defined(VERIFY)
            Assert(this->data.size(),d.size(),true);
            for(int i = 0 ; i < data.size(); ++i) {
                Assert(data[i].origIDX,i,true);
            }
    #endif

    #if defined(EXPERIMENTAL)

            std::vector<std::tuple<uint,uint,uint>> WorkQueue = {std::make_tuple(0,data.size(),0)};
            while(WorkQueue.size() > 0) {
                auto work = WorkQueue.back();
                WorkQueue.pop_back();

                uint left   = std::get<0>(work);
                uint right  = std::get<1>(work);
                uint dim    = std::get<2>(work);

                uint mid = (left + right) / 2;

                auto cmp = [dim](const Tuple& f, const Tuple& s) {
                    return f.p[dim] < s.p[dim];
                };

                if( (right - left) <= BucketSize) {
                    data[mid].min = (*std::min_element(data.begin() + left,data.begin() + right,cmp)).p[dim];
                    data[mid].max = (*std::max_element(data.begin() + left,data.begin() + right,cmp)).p[dim];
                }else {

                    nth_element(data.begin() + left,data.begin() + mid,data.begin()+right,cmp);

                    data[mid].min = (*std::min_element(data.begin() + left,data.begin() + mid,cmp)).p[dim];
                    data[mid].max = (*std::max_element(data.begin() + mid + 1,data.begin()+right,cmp)).p[dim];

                    WorkQueue.push_back(std::make_tuple(left,mid,(dim+1)%W));
                    WorkQueue.push_back(std::make_tuple(mid+1,right,(dim+1)%W));
                }
            }
            #if defined(VERIFY)
            printTree(data,0,data.size(),"",0,W);
            #endif
            root = nullptr;
    #else
            root = createRoot(this->data,0,data.size(),0);
    #if defined(VERIFY)
            printTree(data,root,"");
            std::vector<std::tuple<bool,float>> vec;
            Assert(validateTree(data,root,vec),data.size(),true);
    #endif

    #endif
        }else {
            root = new KDNode;
        }
    }

    KDNode* cloneRoot(KDNode* root) {
        if(root == nullptr) return nullptr;
        KDNode* result = new KDNode();

        result->value   = root->value;
        result->dim     = root->dim;
        result->leftIndex = root->leftIndex;
        result->rightIndex = root->rightIndex;
        result->left    = cloneRoot(root->left);
        result->right   = cloneRoot(root->right);

        return result;
    }

    KDTree::KDTree(const KDTree & other) : metric(other.metric) {
        if(this != &other) {
            delete root;    // TODO(deathly809) - can leave in bad state
            this->metric = other.metric;
            this->root = cloneRoot(other.root);
            this->data = other.data;
            this->W = other.W;
        }
    }

    KDTree::KDTree(KDTree && other) : metric(other.metric) {
        if(this != &other) {
            std::swap(this->root,other.root);
            std::swap(this->data,other.data);
            this->W = other.W;
        }
    }

    KDTree::~KDTree() {
        delete root;
    }

    KDTree& KDTree::operator=(const KDTree & other) {
        if(this != &other) {
            delete root;    // TODO(deathly809) - can leave in bad state
            this->metric = other.metric;
            this->root = cloneRoot(other.root);
            this->data = other.data;
            this->W = other.W;
        }
        return *this;
    }

    KDTree& KDTree::operator=(KDTree && other) {
        if(this != &other) {
            auto prev = this->root;
            std::swap(this->root,other.root);
            this->metric = other.metric;
            this->data.swap(other.data);
            this->W = other.W;
        }
        return *this;
    }

    const float BOUND = 1E-3;

    Indices find(const KDData & data, KDNode* curr, uint dim, const Math::Metric & m, const Point & point, float epsilon, uint threads) {
        Indices result;

        if(curr == nullptr) return result;
        if(curr->minValue > (point[dim] + BOUND + epsilon)) return result;
        if(curr->maxValue < (point[dim] - BOUND - epsilon)) return result;

        if(curr->left == curr->right ) {
            AssertTrue(nullptr==curr->left);
            AssertTrue(nullptr==curr->right);
            for(uint i = curr->leftIndex; i < curr->rightIndex; ++i) {
                auto vec = data[i].p;
                result.reserve(curr->rightIndex - curr->leftIndex);
                if(m.within(point,vec,epsilon)) {
                    int idx = data[i].origIDX;
                    result.push_back(std::make_tuple(i,idx));
                }
            }
        }else {
            float l_value = point[dim] - epsilon;
            float r_value = point[dim] + epsilon;

            Indices l_result,r_result;

            if(l_value <= curr->value) {
                l_result = find(data,curr->left,(dim+1)%point.size(),m,point,epsilon,threads/2);
            }

            if(r_value >= curr->value) {
                r_result = find(data,curr->right,(dim+1)%point.size(),m,point,epsilon,threads/2);
            }

            result.reserve(l_result.size()+r_result.size());
            result.insert(result.end(),l_result.begin(),l_result.end());
            result.insert(result.end(),r_result.begin(),r_result.end());

        }

        return result;
    }

    std::vector<Point> KDTree::findNeighbors(const Point & point,float epsilon) {
        if(point.size() != this->W) {
            throw std::runtime_error("dimension mismatch");
        }

        std::vector<Point> result;

        if(this->W != 0 && data.size() > 0) {
            #if defined(EXPERIMENTAL)
            std::vector<std::tuple<uint,uint,uint>> WorkQueue = {std::make_tuple(0,data.size(),0)};
            while(WorkQueue.size() > 0) {
                auto work = WorkQueue.back();
                WorkQueue.pop_back();

                uint left   = std::get<0>(work);
                uint right  = std::get<1>(work);
                uint dim    = std::get<2>(work);

                uint mid = (right + left) / 2;

                if(data[mid].min > (point[dim] + BOUND + epsilon)) continue;
                if(data[mid].max < (point[dim] - BOUND - epsilon)) continue;

                if( (right - left) <= BucketSize) {
                    for(auto i = left; i < right; ++i) {
                        if(metric.within(data[i].p,point,epsilon)) {
                            result.push_back(data[i].p);
                        }
                    }
                }else {

                    float l_value = point[dim] - epsilon;
                    float r_value = point[dim] + epsilon;

                    if(l_value > data[mid].max) continue;
                    if(r_value < data[mid].min) continue;
                    
                    float pivot = data[mid].p[dim];

                    if(metric.within(data[mid].p,point,epsilon)) {
                        result.push_back(data[mid].p);
                    }

                    if(l_value <= pivot) {
                        WorkQueue.push_back(std::make_tuple(left,mid,(dim+1)%W));
                    }

                    if(r_value >= pivot) {
                        WorkQueue.push_back(std::make_tuple(mid+1,right,(dim+1)%W));
                    }
                }
            }
            #else
            auto pairs = find(data,root,0,metric,point,epsilon,0);
            result.reserve(pairs.size());
            for(auto p : pairs) {
                auto idx = std::get<0>(p);
                auto vec = data[idx].p;
                result.push_back(vec);
            }
            #endif
        }
        return result;
    }

    std::vector<uint> KDTree::findNeighborIndices(const Point & point,float epsilon) {
        if(point.size() != this->W) {
            throw std::runtime_error("dimension mismatch");
        }

        std::vector<uint> result;

        if(this->W != 0 && data.size() > 0) {
            #if defined(EXPERIMENTAL)
            std::vector<std::tuple<uint,uint,uint>> WorkQueue = {std::make_tuple(0,data.size(),0)};
            while(WorkQueue.size() > 0) {
                auto work = WorkQueue.back();
                WorkQueue.pop_back();

                uint left   = std::get<0>(work);
                uint right  = std::get<1>(work);
                uint dim    = std::get<2>(work);

                if( (right - left) <= BucketSize) {
                    for(auto i = left; i < right; ++i) {
                        if(metric.within(data[i].p,point,epsilon)) {
                            result.push_back(data[i].origIDX);
                        }
                    }
                }else {
                    uint mid = (right + left) / 2;

                    uint l_value = point[dim] - epsilon;
                    uint r_value = point[dim] + epsilon;

                    uint pivot = data[mid].p[dim];

                    if(metric.within(data[mid].p,point,epsilon)) {
                        result.push_back(data[mid].origIDX);
                    }

                    if(l_value <= pivot) {
                        WorkQueue.push_back(std::make_tuple(left,mid,(dim+1)%W));
                    }

                    if(r_value >= pivot) {
                        WorkQueue.push_back(std::make_tuple(mid+1,right,(dim+1)%W));
                    }
                }

            }
            #else
            auto pairs = find(data,root,0,metric,point,epsilon,4);
            result.reserve(pairs.size());
            for( auto p : pairs ) {
                result.push_back(std::get<1>(p));
            }
            #endif
        }
        return result;
    }


    static bool treeContains(KDData &data, KDNode* curr, const Point & p,uint W) {
        auto dim = 0;
        while(curr != nullptr) {

            if(curr->left == curr->right) {
                for(auto i = curr->leftIndex; i < curr->rightIndex; ++i) {
                    auto vec = data[i].p;
                    if(vec == p) return true;
                }
            }

            if(p[dim] < curr->value) {
                curr = curr->left;
            }else {
                if(treeContains(data,curr->left,p,W)) return true;
                curr = curr->right;
            }

            dim = (dim + 1) % W;
        }
        return false;
    }

    bool KDTree::contains(const Point & point) {
        if(point.size() != this->W) {
            throw std::runtime_error("dimension mismatch");
        }
    #if defined(EXPERIMENTAL)
        std::vector<std::tuple<uint,uint,uint>> WorkQueue = {std::make_tuple(0,data.size(),0)};
        while(WorkQueue.size() > 0) {
            auto work = WorkQueue.back();
            WorkQueue.pop_back();

            uint left   = std::get<0>(work);
            uint right  = std::get<1>(work);
            uint dim    = std::get<2>(work);

            if( (right - left) <= BucketSize) {
                for(auto i = left; i < right; ++i) {
                    if(data[i].p == point) {
                        return true;
                    }
                }
            }else {
                uint mid = (right + left) / 2;

                uint pivot = data[mid].p[dim];

                if(pivot < pivot) {
                    WorkQueue.push_back(std::make_tuple(left,mid,(dim+1)%W));
                }else {
                    WorkQueue.push_back(std::make_tuple(left,mid,(dim+1)%W));
                    WorkQueue.push_back(std::make_tuple(mid,right,(dim+1)%W));
                }
            }
        }
        return false;
        #else
        return treeContains(data,root,point,W);
        #endif
    }

}