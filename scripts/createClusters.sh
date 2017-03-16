#!/bin/bash

if ! [[ -v EXEC_LOC ]];
then
    EXEC_LOC=output
fi


if ! [[ -v K ]];
then
	K=4
fi

if ! [[ -v ITER ]];
then
	ITER=200
fi


echo
echo "creating directory ${CLUSTER_OUT}"
echo

mkdir -p ${CLUSTER_OUT}

echo
echo "running clustering with K=${K}"
echo

${EXEC_LOC}/kmeans -input ${CLUSTER_DATA} -out ${CLUSTER_INFO} -k ${K} -iter ${ITER}
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
