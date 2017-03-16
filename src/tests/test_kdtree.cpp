
#include <iostream>
#include <random>
#include <thread>

#include <util/testing.h>
#include <util/kdtree.h>

using namespace KD;

static Math::Euclidian e;

float randomFloat() {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(100, 100000);
    return dist(e2);
}

inline Point randomPoint(uint D) {
    Point p(D);
    for(int j = 0 ; j < D; ++j) {
        p[j] = randomFloat();
    }
    return p;
}

std::vector<Point> createData(uint N, uint D) {
    const int NumThreads = std::thread::hardware_concurrency();

    std::vector<Point> results(N);
    std::thread threads[NumThreads];

    uint work = N / NumThreads;

    for(int i = 0 ; i < NumThreads; ++i) {
        threads[i] = std::thread([N,D,work,&results](uint tid) {
            uint start = tid * work;
            for(int i = 0; i < work; ++i,++start) {
                results[start] = randomPoint(D);
            }
        },i);
    }
    uint elements = work * NumThreads;
    for(int i = elements; i < N; ++i) {
        results[i] = randomPoint(D);
    }
    
    for(int i = 0 ; i < NumThreads; ++i) threads[i].join();

    return results;
}

void plainTest() {
    // 3 dimensions
    std::vector<Point> data = {
        {1,1,7},
        {2,2,6},
        {3,3,5},
        {4,4,4},
        {5,5,3},
        {6,6,2},
        {7,7,1}
    };
    
    KDTree tree(data,e);

    auto point  = data[0];
    auto eps    = 10;
    auto result = tree.findNeighbors(point,eps);

    for( auto r : result) {
        TestAssert(e(point,r) <= eps);
    }
}


void allSameTest() {
    // 3 dimensions
    Point init = {1,1,1,1,1};
    std::vector<Point> data = std::vector<Point>(5000,init);

    for(auto d : data) {
        TestAssert(d == init);
    }
        
    KDTree tree(data,e);

    auto point  = data[0];
    auto eps    = 10;
    auto result = tree.findNeighbors(point,eps);

    TestSame(data.size(),result.size());
}

void allSameButOneTest() {
    Point init = {1,1,1,1,1};
    std::vector<Point> data = std::vector<Point>(500,init);
    data.push_back({1,1,10,1,1});

    KDTree tree(data,e);

    auto point  = data[data.size()-1];
    auto eps    = 8;
    auto result = tree.findNeighbors(point,eps);

    TestSame(1,result.size());

    eps    = 10;
    result = tree.findNeighbors(point,eps);
    TestSame(data.size(),result.size());
}

void noDataTest() {
    std::vector<Point> data = {};
    KDTree tree(data,e);

    auto point  = Point{};
    auto eps    = 10;
    auto result = tree.findNeighbors(point,eps);

    TestAssert(result.size() == 0);
}

void zeroLengthDataTest() {
    std::vector<Point> data = { {} };
    KDTree tree(data,e);

    auto point  = Point{};
    auto eps    = 10;
    auto result = tree.findNeighbors(point,eps);

    TestSame(result.size(),0);
}

void differentLengthDataTest() {

    std::vector<Point> data = { {1} };
    KDTree tree(data,e);

    auto point  = Point{100};
    auto eps    = 10;
    auto result = tree.findNeighbors(point,eps);

    TestAssert(result.size() == 0);

}

void largeRandomDataTestWithTenDimensions() {
    const uint N = 10000;
    const uint D = 10;

    // 3 dimensions
    std::vector<Point> data = createData(N,D);
    KDTree tree(data,e);

    auto point  = Point{0,0,0,0,0,0,0,0,0,0};
    auto eps    = 0.01;
    auto result = tree.findNeighbors(point,eps);

    TestAssert(result.size() == 0);

}

void largeRandomDataTestWith100Dimensions() {
    const uint N = 100000;
    const uint D = 100;

    // 3 dimensions
    std::vector<Point> data = createData(N,D);
    KDTree tree(data,e);

    auto point  = Point(100);
    auto eps    = 10;
    auto result = tree.findNeighbors(point,eps);

    TestAssert(result.size() == 0);
    
}

void largeInsertCheck() {
    const uint N = 10000;
    const uint D = 10;

    // 3 dimensions
    std::vector<Point> data = createData(N,D);
    KDTree tree(data,e);

    auto eps = 10;
    for(auto d : data) {
        auto result = tree.findNeighbors(d,eps);
        TestAssert(result.size() > 0);
    }
}


int main() {
    RunTest("plainTest",plainTest,NOP);
    RunTest("allSameTest",allSameTest,NOP);
    RunTest("allSameButOneTest",allSameButOneTest,NOP);
    RunTest("noDataTest",noDataTest,NOP);
    RunTest("zeroLengthDataTest",zeroLengthDataTest,NOP);
    RunTest("differentLengthDataTest",differentLengthDataTest,NOP);
    RunTest("largeRandomDataTestWithTenDimensions",largeRandomDataTestWithTenDimensions,NOP);
    RunTest("largeRandomDataTestWith100Dimensions",largeRandomDataTestWith100Dimensions,NOP);
    RunTest("largeInsertCheck",largeInsertCheck,NOP);
}

