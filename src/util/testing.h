
#ifndef TESTING_H_
#define TESTING_H_

const std::string Line          = "------------------------------------------------------------";

#include <iostream>
#include <chrono>
#include <functional>

#include <sstream>

const std::string PassMessage   = "\tTest Passed";
const std::string FailMessage   = "\tTest Failed";


void NOP(uint) {}

void TestFail(std::string msg) {
    throw std::runtime_error(msg);
}

void TestAssert(bool value) {
    if(!value) TestFail("expected true, found false");
}

template<typename A, typename B>
void TestSame(A && expected, B && found) {
    if( expected != found) {
        std::stringstream ss;
        ss << "expected " << expected << " but found " << found;
        TestFail(ss.str());
    }
}


template<typename TestFunction,typename InitFunction,bool StopOnFail = true>
void RunTest(std::string name, TestFunction F,InitFunction Init) {
    std::cout << "Running Test: ";
    std::cout << name << std::endl;
    std::cout << Line << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    try {
        Init(0);
        F();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end-start;
        std::cout << PassMessage;
        std::cout << ": ";
        std::cout << diff.count();
        std::cout << "s" << std::endl;
    } catch(const std::exception &exc) {
        std::cout << exc.what() << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end-start;
        std::cout << FailMessage;
        std::cout << ": ";
        std::cout << diff.count();
        std::cout << "s" << std::endl;
        if(StopOnFail) std::exit(1);

    }
    std::cout << std::endl << Line;
    std::cout << std::endl << std::endl;
}

uint Iterations = 5;

template<typename BenchFunction,typename InitFunction>
void RunBenchmark(uint Start,uint Step, uint End, std::string name, BenchFunction F,InitFunction Init) {
    std::cout << "Running benchmark: ";
    std::cout << name << std::endl;
    std::cout << Line << std::endl;
    for(auto i = Start; i <= End; i += Step) {
        Init(i);
        double total = 0.0;
        for(int j = 0 ; j < Iterations; ++j) {
            auto start = std::chrono::high_resolution_clock::now();
            try {
                F();
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = end-start;
                total += diff.count();
            } catch(...) {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = end-start;
                total += diff.count();
            }
        }
        std::cout << "Benchmark[";
        std::cout << i << "]: ";
        std::cout << total/10;
        std::cout << "s" << std::endl;
    }
    std::cout << std::endl << Line;
    std::cout << std::endl << std::endl;
}

#endif