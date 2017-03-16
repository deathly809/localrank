#!/bin/bash

START=0
END=45


if ! [[ -v FOLD ]];
then
	FOLD="Fold1"
fi

if ! [[ -v GROW ]];
then
	GROW="shrinking"
	_GROW=""
else
	GROW="growing"
	_GROW="--grow"
fi

if ! [[ -v FILENAME ]];
then
	FILENAME=${FOLD}_${GROW}.subset
fi


TRAIN=data/MQ2007/${FOLD}/train.txt
TEST=data/MQ2007/${FOLD}/test.txt

BASE="./scripts/subset_selection.py --lib external/RankLib.jar --train ${TRAIN} --test ${TEST} --result ${FILENAME} --ranker 2 ${_GROW} --score"

if ! [[ -v EVAL ]];
then
	${BASE} 
else
	for K in `seq ${START} ${END}`;
	do
		echo "feature " $(head -n $(expr $K + 1) ${FILENAME} | tail -n 1) " " $(${BASE} --eval --ignore ${K} --score ${GROW})
	done
fi
