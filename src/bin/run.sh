#!/bin/bash

INPUT_FILE=../../data/converted/S1_C.txt
OUTPUT_DIR=experiments

K_START=5
K_STEP=3
K_END=50

EPS_START=0.2
EPS_STEP=0.05
EPS_END=4

mkdir -p ${OUTPUT_DIR}

echo "Running..."
echo

for K in `seq ${K_START} ${K_STEP} ${K_END}`;
do
    for EPS in `seq ${EPS_START} ${EPS_STEP} ${EPS_END}`;
    do
        ./output/gpuCluster ${INPUT_FILE} ${OUTPUT_DIR}/${K}_${EPS}.txt ${K} ${EPS} 
    done
done

