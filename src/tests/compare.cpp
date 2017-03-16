
#include <iostream>
#include <vector>

#include<iterator>

#include <chrono>
#include <random>

#include <kdtree++/kdtree.hpp>

#include <util/kdtree.h>
#include <util/assertion.h>

typedef std::vector<float> Point;
typedef std::vector<Point> Data;

typedef unsigned int uint;

template<class C = std::chrono::high_resolution_clock>
struct Timer {
	typedef	std::chrono::microseconds us;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
	typedef std::chrono::time_point<C> TP;

	TP startTime, endTime;	

    Timer() {
		reset();
    }

	float start() {
		startTime = C::now();
		auto d = startTime.time_since_epoch();
		auto dur = std::chrono::duration_cast<std::chrono::microseconds>(d);
		return dur.count();
	}

	float stop() {
		endTime = C::now();
		auto d = endTime.time_since_epoch();
		auto dur = std::chrono::duration_cast<std::chrono::microseconds>(d);
		return dur.count();
	}

	float duration() {
		std::chrono::duration<float> dur = (endTime - startTime);
		auto d = std::chrono::duration_cast<us>(dur);
		return d.count();
	}

	void reset() {
		startTime = endTime = C::now();
	}
};

float randomFloat() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1, 10000000.0);
	return dist(gen);
}

Data genData(uint N,uint D) {
	Data result(N,Point(D));
	for(uint i = 0; i < N; ++i) {
		for(uint j = 0; j < D; ++j) {
			result[i][j] = randomFloat();
		}
	}
	return result;
}

int main(void) {
	const uint N = 100000;
	const uint D = 5;
	const float eps = 100;
	Data data = genData(N,D);

	Math::Euclidian met;
	Timer<> t;

	t.start();
	KDTree::KDTree<D,Point> tree1(data.begin(),data.end());
	t.stop();
	std::cout << t.duration() / 1E+6 << " seconds to construct libkdtree"<< std::endl;

	t.start();
	KD::KDTree tree2(data,met);
	t.stop();


	std::cout << t.duration() / 1E+6 << " seconds to construct my kdtree"<< std::endl;

	float t1 = 0.0;
	float t2 = 0.0;

	std::cout << "validating results" << std::endl;
	for(auto d : data) {
		Data s;

		t.start();
		tree1.find_within_range(d,eps,std::back_insert_iterator<Data>(s));
		t.stop();
		t1 += t.duration();

		t.start();
		auto f = tree2.findNeighborIndices(d,eps);
		t.stop();
		t2 += t.duration();

		Assert(f.size(),s.size(),true);
		AssertTrue(f.size()>0);
		for(auto v : f) {
			AssertTrue(std::find(s.begin(),s.end(),data[v]) != s.end());
		}
	}

	std::cout << t1 / 1E6 << " seconds on average to find for libkdtree" << std::endl;
	std::cout << t2 / 1E6 << " seconds on average to find for my kdtree" << std::endl;

	return 0;
}
