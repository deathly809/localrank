#!/bin/bash

FROM=experiments
DIRECTORY=tmp
OUTPUT=filtered.txt

K_START=5
K_STEP=3
K_END=50

EPS_START=0.2
EPS_STEP=0.05
EPS_END=4

MIN_C=5
MAX_C=20

rm -rf ${DIRECTORY}
mkdir ${DIRECTORY}


cp ${FROM}/* ${DIRECTORY}/
rm -f ${OUTPUT}
touch ${OUTPUT}

./output/filter -input ${DIRECTORY} -output ${OUTPUT} -epsStart ${EPS_START} -epsStep ${EPS_STEP} -epsEnd ${EPS_END} -kStart ${K_START} -kStep ${K_STEP} -kEnd ${K_END} -min ${MIN_C} -max ${MAX_C}


