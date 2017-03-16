
#include <iostream>
#include <random>

#include <util/testing.h>
#include <util/kdtree.h>
#include <util/assertion.h>

using namespace KD;

Math::Euclidian euc;

float randomFloat() {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(100, 100000);
    return dist(e2);
}


std::vector<Point> data;
KDTree tree;

template<const uint D>
void InitializeVector(const uint N) {
    data.reserve(N);
    for(int i = data.size() ; i < N; ++i) {
        Point p(D);
        for(int j = 0 ; j < D; ++j) {
            p.push_back(randomFloat());
        }
        data.push_back(p);
    }
}

template<uint D>
void BenchCreateWithDim(void) {
    KDTree tree(data,euc);
}


template<uint D>
void InitSearch(const uint N) {
    InitializeVector<D>(N);
    tree = KDTree(data,euc);
}

template<uint D>
void BenchRetrieveWithDim(void) {
    AssertTrue(data.size() > 0);
    float eps = 10;
    auto randIDX = ((int)randomFloat()) % data.size();
    auto randVec = data[randIDX];

    auto near = tree.findNeighbors(randVec,eps);
    AssertTrue(near.size() > 0);
    for( auto v : near ) {
        AssertTrue( euc(randVec,v) <= eps);
    }
    
}

#define BENCH_WRAP(FUNC,INIT) data.clear();RunBenchmark(Start,Step,End,#FUNC,FUNC,INIT);


int main(void) {

    {
        const uint Start = 1000;
        const uint Step = Start;
        const uint End = Start * 10;
        

        BENCH_WRAP(BenchCreateWithDim<1>,InitializeVector<1>);
        BENCH_WRAP(BenchCreateWithDim<2>,InitializeVector<2>);
        BENCH_WRAP(BenchCreateWithDim<3>,InitializeVector<3>);
        BENCH_WRAP(BenchCreateWithDim<4>,InitializeVector<4>);
        BENCH_WRAP(BenchCreateWithDim<5>,InitializeVector<5>);
        BENCH_WRAP(BenchCreateWithDim<6>,InitializeVector<6>);
        BENCH_WRAP(BenchCreateWithDim<7>,InitializeVector<7>);
        BENCH_WRAP(BenchCreateWithDim<8>,InitializeVector<8>);
        BENCH_WRAP(BenchCreateWithDim<9>,InitializeVector<9>);
        BENCH_WRAP(BenchCreateWithDim<10>,InitializeVector<10>);
        
        BENCH_WRAP(BenchCreateWithDim<200>,InitializeVector<200>);

        BENCH_WRAP(BenchRetrieveWithDim<1>,InitSearch<1>);
        BENCH_WRAP(BenchRetrieveWithDim<2>,InitSearch<2>);
        BENCH_WRAP(BenchRetrieveWithDim<3>,InitSearch<3>);
        BENCH_WRAP(BenchRetrieveWithDim<4>,InitSearch<4>);
        BENCH_WRAP(BenchRetrieveWithDim<5>,InitSearch<5>);
        BENCH_WRAP(BenchRetrieveWithDim<6>,InitSearch<6>);
        BENCH_WRAP(BenchRetrieveWithDim<7>,InitSearch<7>);
        BENCH_WRAP(BenchRetrieveWithDim<8>,InitSearch<8>);
        BENCH_WRAP(BenchRetrieveWithDim<9>,InitSearch<9>);
        BENCH_WRAP(BenchRetrieveWithDim<10>,InitSearch<10>);

        BENCH_WRAP(BenchRetrieveWithDim<200>,InitSearch<200>);
    }

#ifdef BIG_BENCH
    {
        std::cout << "BIG BENCH" << std::endl;
        const uint Start = 100000;
        const uint Step = Start;
        const uint End = Start * 10;

        BENCH_WRAP(BenchCreateWithDim<1>,InitializeVector<1>);
        BENCH_WRAP(BenchCreateWithDim<2>,InitializeVector<2>);
        BENCH_WRAP(BenchCreateWithDim<3>,InitializeVector<3>);
        BENCH_WRAP(BenchCreateWithDim<4>,InitializeVector<4>);
        BENCH_WRAP(BenchCreateWithDim<5>,InitializeVector<5>);
        BENCH_WRAP(BenchCreateWithDim<6>,InitializeVector<6>);
        BENCH_WRAP(BenchCreateWithDim<7>,InitializeVector<7>);
        BENCH_WRAP(BenchCreateWithDim<8>,InitializeVector<8>);
        BENCH_WRAP(BenchCreateWithDim<9>,InitializeVector<9>);
        BENCH_WRAP(BenchCreateWithDim<10>,InitializeVector<10>);

        BENCH_WRAP(BenchRetrieveWithDim<1>,InitSearch<1>);
        BENCH_WRAP(BenchRetrieveWithDim<2>,InitSearch<2>);
        BENCH_WRAP(BenchRetrieveWithDim<3>,InitSearch<3>);
        BENCH_WRAP(BenchRetrieveWithDim<4>,InitSearch<4>);
        BENCH_WRAP(BenchRetrieveWithDim<5>,InitSearch<5>);
        BENCH_WRAP(BenchRetrieveWithDim<6>,InitSearch<6>);
        BENCH_WRAP(BenchRetrieveWithDim<7>,InitSearch<7>);
        BENCH_WRAP(BenchRetrieveWithDim<8>,InitSearch<8>);
        BENCH_WRAP(BenchRetrieveWithDim<9>,InitSearch<9>);
        BENCH_WRAP(BenchRetrieveWithDim<10>,InitSearch<10>);
    }
#endif
    return 0;
}
