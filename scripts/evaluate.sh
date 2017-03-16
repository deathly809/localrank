#!/bin/bash

if ! [[ -v EXEC_LOC ]];
then
    EXEC_LOC=output
fi

if ! [[ -v METHODS ]];
then
	echo "Methods not specified..."
    METHODS="RANKBOOST RANKNET LAMBDA ADARANK"
	exit
fi


export GODEBUG=cgocheck=0

# dbscan files
CENTROID_LOCATION=${CLUSTER_LOCATION}/${CLUSTER_INFO_FILE}
# DBSCAN parameters

# Prefix for running RankLib
RANKLIB="java -jar external/RankLib.jar"

for METHOD in ${METHODS};
do

	MODEL_LOCATION=${TRAIN_DIR}/${METHOD}
	OUTPUT_DIR=${RESULTS}/${METHOD}

	echo "$METHOD"

	RANKER=0

	if [ "${METHOD}" == "RANKNET" ]
	then
		RANKER=1
	elif [ "${METHOD}" == "LAMBDA" ]
	then
		RANKER=6
		#./scripts/lambda.sh
	elif [ "${METHOD}" == "ADARANK" ]
	then
		RANKER=3
	elif [ "${METHOD}" == "RANKBOOST" ]
	then
		RANKER=2
		#./scripts/rankboost.sh
	elif [ "${METHOD}" == "BOB" ]
	then
		echo "bob!"
		exit
	fi

	echo
    echo "Creating directory ${OUTPUT_DIR}"
    echo

	mkdir -p ${OUTPUT_DIR}

	echo
	echo "Evaluating on validation data: ${VALIDATION_FILE}"
	echo

	echo "Centroid Location" ${CENTROID_LOCATION}
	echo "Model location" ${MODEL_LOCATION}

	${EXEC_LOC}/partitionData -exp -centroids ${CENTROID_LOCATION} ${MERGE_FLAG} -models ${K} -data ${VALIDATION_FILE} -dir ${MODEL_LOCATION} -ranker ${RANKER} -pairwise > ${OUTPUT_DIR}/partition_results
	rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

	perl scripts/evalScript.pl ${RAW_VALIDATION_FILE} ${OUTPUT_DIR}/partition_results ${OUTPUT_DIR}_results 0
	rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi


done
