

#ifndef ASSERTION_H_
#define ASSERTION_H_

#include <iostream>

void stackTrace();

#define ASSERT(T,C,E,F,L) do {                              \
    auto FOUND = (T);                                       \
    auto EXPECTED = (C);                                    \
    bool ASSERT_TEST = (FOUND)==(EXPECTED);                 \
    if( ASSERT_TEST != (E) ) {                              \
        std::cerr <<                                        \
            "Assertion failed: " <<                         \
            "expected " << EXPECTED <<                      \
            ", found " << FOUND << " "                      \
            F <<                                            \
            ":" <<                                          \
            L <<                                            \
            std::endl;                                      \
        stackTrace();                                       \
        std::quick_exit(1);                                 \
    }                                                       \
} while(0)

#define Assert(test,cmp,expected) ASSERT(test,cmp,expected,__FILE__,__LINE__)
#define AssertTrue(test) Assert(test,true,true)
#define AssertFalse(test)Assert(test,false,true)

#define AssertLessThan(F,S)         \
do {                                \
    if( (F)>=(S) ) {                \
        std::cerr <<                \
        "Assertion failed: " <<     \
        "expected " << (F) <<       \
        " < " << (S) <<             \
        std::endl;                  \
        stackTrace();               \
        std::quick_exit(1);         \
    }                               \
}while(0);


#define TestAndSet(T,D,V) do {  \
    if(T) D = (V);              \
}while(0);

#endif