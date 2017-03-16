
SRC=../
LIB=$(SRC)
INCL=-I$(LIB)
TEST=testSort

OUTPUT=../../${OUTDIR}

OBJECTS= $(OUTPUT)/cudaSync.o $(OUTPUT)/cudaUtil.o $(OUTPUT)/dbscan.o $(OUTPUT)/cudbscan.o $(OUTPUT)/sort.o $(OUTPUT)/assertion.o $(OUTPUT)/kdtree.o

all: $(OUTPUT)/$(TEST) $(OUTPUT)/test_kdtree $(OUTPUT)/bench_kdtree $(OUTPUT)/compare 

debug: $(OUTPUT)/test_kdtree_debug $(OUTPUT)/bench_kdtree_debug

big: 	test_kdtree big_kdtree_bench

$(OUTPUT)/$(TEST): tests.cpp $(SRC)cuda/cudbscan.cuh $(SRC)cuda/cudbscan.cu $(SRC)util/dbscan.h $(SRC)util/dbscan.cpp
	nvcc $(INCL) $(STD) --compiler-options='-rdynamic' $(ARCH) $(FLAGS) $(OBJECTS) tests.cpp -o $(OUTPUT)/testSort


$(OUTPUT)/test_kdtree_debug:
	g++ -g -I../ -std=c++11 -pthread test_kdtree.cpp $(OUTPUT)/assertion.o $(OUTPUT)/kdtree.o -o $(OUTPUT)/test_kdtree_debug
	
$(OUTPUT)/test_kdtree:
	g++ -O3 -I../ -std=c++11 -pthread $(FLAGS) test_kdtree.cpp $(OUTPUT)/assertion.o $(OUTPUT)/kdtree.o -o $(OUTPUT)/test_kdtree

$(OUTPUT)/bench_kdtree:
	g++ -O3 -I../ -std=c++11 -pthread $(FLAGS) bench_kdtree.cpp $(OUTPUT)/assertion.o $(OUTPUT)/kdtree.o -o $(OUTPUT)/bench_kdtree
	
$(OUTPUT)/bench_kdtree_debug:
	g++ -g -I../ -std=c++11 -pthread bench_kdtree.cpp $(OUTPUT)/assertion.o $(OUTPUT)/kdtree.o -o $(OUTPUT)/bench_kdtree_debug
	
$(OUTPUT)/big_kdtree_bench:
	g++ -O3 -I../ -std=c++11 -pthread bench_kdtree.cpp $(OUTPUT)/assertion.o $(OUTPUT)/kdtree.o -DBIG_BENCH -o $(OUTPUT)/big_kdtree_bench
	
$(OUTPUT)/big_kdtree_bench_debug:
	g++ -g -I../ -std=c++11 -pthread bench_kdtree.cpp $(OUTPUT)/assertion.o $(OUTPUT)/kdtree.o -DBIG_BENCH -o $(OUTPUT)/big_kdtree_bench_debug
	
$(OUTPUT)/compare:
	g++ -g -I../ -std=c++11 -pthread compare.cpp $(OUTPUT)/assertion.o $(OUTPUT)/kdtree.o -o $(OUTPUT)/compare

clean:
	-@rm -fv $(OUTPUT)/$(TEST)
	-@rm -fv $(OUTPUT)/test_kdtree_debug
	-@rm -fv $(OUTPUT)/test_kdtree
	-@rm -fv $(OUTPUT)/bench_kdtree
	-@rm -fv $(OUTPUT)/bench_kdtree_debug
	-@rm -fv $(OUTPUT)/big_kdtree_bench
	-@rm -fv $(OUTPUT)/big_kdtree_bench_debug
	-@rm -fv $(OUTPUT)/compare


nuke: clean
