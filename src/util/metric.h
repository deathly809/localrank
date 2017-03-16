
/*
 *
 *  Metrics are put here (L1 and L2 are provided)
 *
 */

#ifndef METRIC_H_
#define METRIC_H_

#include <vector>

namespace Math {

    typedef std::vector<float> Point;

    class Metric {
        protected: 
            Metric() { /* Empty */}
        public:
            virtual float operator()(Point const & a, const Point & b) const {
                throw std::runtime_error("operator(): unsupported operation");
            }

            virtual bool within(const Point & a, const Point & b, float eps) const {
                throw std::runtime_error("within: unsupported operation");
            }
    };

    class Euclidian : public Metric {
        public:
            Euclidian() { /* Empty */}

            float operator()(Point const & a, const Point & b) const {
                float result = 0;
                for( int i = 0 ; i < std::min(a.size(),b.size()); ++i) {
                    float t = a[i] - b[i];
                    result += t * t;
                }
                return std::sqrt(result);
            }

            bool within(const Point & a, const Point & b, float eps) const {
                float result = 0;
                float eps2 = eps * eps;
                for( int i = 0 ; i < std::min(a.size(),b.size()); ++i) {
                    float t = a[i] - b[i];
                    result += t * t;
                    if(result > (eps2 + 1E-2)) return false;
                }
                return true;
            }
    };

    class Manhattan : public Metric {
        public:
            Manhattan() { /* Empty */}

            float operator()(Point const & a, const Point & b) const {
                float result = 0;
                for( int i = 0 ; i < std::min(a.size(),b.size()); ++i) {
                    result += std::abs(a[i] - b[i]);
                }
                return result;
            }

            bool within(const Point & a, const Point & b, float eps) const {
                float result = 0;
                for( int i = 0 ; i < std::min(a.size(),b.size()); ++i) {
                    result += std::abs(a[i] - b[i]);
                    if(result > eps) return false;
                }
                return true;
            }
    };

    class LInf : public Metric {
        public:
            LInf() { /* Empty */}

            float operator()(Point const & a, const Point & b) const {
                float result = 0;
                for( int i = 0 ; i < std::min(a.size(),b.size()); ++i) {
                    auto r = std::abs(a[i] - b[i]);
                    if( r > result) {
                        r = result;
                    }
                }
                return result;
            }

            bool within(const Point & a, const Point & b, float eps) const {
                for( int i = 0 ; i < std::min(a.size(),b.size()); ++i) {
                    auto r = std::abs(a[i] - b[i]);
                    if( r > eps) {
                        return false;
                    }
                }
                return true;
            }
    };
    
}

#endif