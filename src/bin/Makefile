
LIB		:=../
INCL=-I$(LIB)
OUTPUT=../../output

OBJECTS= $(OUTPUT)/cudaSync.o $(OUTPUT)/cudaUtil.o $(OUTPUT)/sort.o $(OUTPUT)/assertion.o $(OUTPUT)/cudbscan.o $(OUTPUT)/dbscan.o $(OUTPUT)/kdtree.o

.PHONY: all

all: $(OUTPUT)/gpuCluster $(OUTPUT)/dbscan

$(OUTPUT)/gpuCluster: gpuCluster.cpp $(OBJECTS)
	nvcc $(INCL) $(STD) $(ARCH) $(FLAGS) $(OBJECTS) gpuCluster.cpp -o $(OUTPUT)/gpuCluster 

$(OUTPUT)/dbscan: dbscan.cpp
	nvcc $(INCL) $(STD) $(ARCH) $(FLAGS) $(OBJECTS) dbscan.cpp -o $(OUTPUT)/dbscan 

clean:
	-@rm -fv $(OUTPUT)/dbscan
	-@rm -fv $(OUTPUT)/gpuCluster

nuke: clean