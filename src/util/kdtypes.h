
/*
 *  Contains all data types used by kdtree.  Mostly put here to keep kdtree.h clean
 *
 *
 */

#ifndef KDTYPES_H_
#define KDTYPES_H_

namespace KD {

    class Tuple;
    struct KDNode;
    
    typedef std::vector<float>  Point;
    typedef std::vector<Tuple> KDData;

    struct Tuple {
        uint origIDX;
        Point p;

        float min,max;

        Tuple() : origIDX(0), min(0), max(0) {
            /* Empty */
        }

        Tuple(uint o, Point && p) : min(0), max(0) { 
            this->origIDX = o;
            this->p.swap(p);
        }

        Tuple(uint  o, const Point & p) : min(0), max(0) {
            this->origIDX = o;
            this->p = p;
            /* Empty */
        }

        Tuple(const Tuple & other) {
            this->min = other.min;
            this->max = other.max;
            this->origIDX = other.origIDX;
            this->p = other.p;
            /* Empty */
        }

        Tuple(Tuple && other) {
            this->min = other.min;
            this->max = other.max;
            this->origIDX = other.origIDX;
            this->p.swap(other.p);
        }

        Tuple & operator=(const Tuple & other) {
            this->min = other.min;
            this->max = other.max;
            this->origIDX = other.origIDX;
            this->p = other.p;
            return *this;
        }

        Tuple & operator=(Tuple && other) {
            this->min = other.min;
            this->max = other.max;
            this->origIDX = other.origIDX;
            this->p.swap(other.p);
            return *this;
        }

    };

    // TODO(deathly809) - Try to create Object oriented version of this
    struct KDNode {
        uint dim;
        float value,minValue, maxValue;
        KDNode *left,*right;
        uint leftIndex,rightIndex;

        KDNode() {
            left = right = nullptr;
            leftIndex = rightIndex = 1;
        }

        ~KDNode() {
            delete left;
            delete right;
        }
    };

}

#endif