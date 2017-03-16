
#ifndef SORT_H_
#define SORT_H_

#include <vector>
#include <util/types.h>

#include <util/assertion.h>


template<class T>
bool validateSort(const std::vector<T> & data) {
    auto prev = data.begin();
    auto curr = prev + 1;
    while(curr != data.end()) {
        if(*curr < *prev) {
            return false;
        }

        ++curr;
        ++prev;
    }
    return true;
}

// validate a partition
template<class T>
bool validatePartition(T pivot, const std::vector<T> & keys,int left,int right) {
    bool isLeft = true;
    for(int i = left; i <= right;i++) {
        if(keys[i] > pivot) {
            if(isLeft) {
                isLeft = false;
            }else {
                return false;
            }
        }
    }
    return true;
}

/*
 *  keys - what we are sorting on
 *  values - satellite data
 *  left - left most key we can examine
 *  right - right most key we can examine
 */
 template<typename K,typename V>
void insertionSort(std::vector<K> & keys, std::vector<V> & data, uint left, uint right) {
    for(uint i = left + 1; i <= right; i++) {
        int j = i;
        while(j > 0 && keys[j-1] > keys[j]) {
            std::swap(keys[j],keys[j-1]);
            std::swap(data[j],data[j-1]);
            --j;
        }

    }
}

template<typename K,typename V>
void partition(std::vector<K> & keys, std::vector<V> & values, uint left, uint right, uint & lt, uint & gt) {

    const uint mid = left + (right - left) / 2;
    const K pivot = keys[mid];

    std::swap(keys[mid],keys[left]);
    std::swap(values[mid],values[left]);

    uint i = left;
    uint j = left;
    uint n = right - 1;

    while(j <= n) {

        Assert(j<keys.size(),true,true);
        Assert(i<keys.size(),true,true);
        Assert(n<keys.size(),true,true);

        if(keys[j] < pivot) {
            std::swap(keys[i],keys[j]);
            std::swap(values[i],values[j]);
            ++i;
            ++j;
        }else if( keys[j] > pivot) {
            std::swap(keys[j],keys[n]);
            std::swap(values[j],values[n]);
            --n;
        }else {
            ++j;
        }
    }
    lt = i;
    gt = n;
}

/*
 *  keys - what we are sorting on
 *  values - satellite data
 *  left - left most key we can examine
 *  right - right most key we can examine
 */
 template<typename K,typename V>
void quickSort(std::vector<K> & keys, std::vector<V> & values, uint left, uint right) {

    if(left >= right) return;

    AssertTrue(left < keys.size());
    AssertTrue(right <= keys.size());

    if(right - left <= 100) {
        insertionSort(keys,values,left,right);
        return;
    }

    uint lt = 0, gt = 0;
    partition(keys,values,left,right,lt,gt);

    quickSort(keys,values,left,lt);
    quickSort(keys,values,gt+1,right);

}


template<typename K,typename V>
void sort(std::vector<K> & keys,std::vector<V> & data) {    
    quickSort(keys,data,0,keys.size());
}

#endif
