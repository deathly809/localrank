#!/bin/bash

export GODEBUG=cgocheck=0

CLUSTER_OUT=${CLUSTER_LOCATION}/cluster_${CLUSTER_INFO_FILE}

if ! [[ -v METHODS ]];
then
    echo "Methods not defined, defining..."
    METHODS="RANKBOOST RANKNET LAMBDA ADARANK"
    exit
fi



# Prefix for running RankLib
RANKLIB="java -jar external/RankLib.jar"

for METHOD in ${METHODS};
do

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
        OPTS="-round 500 -tc 15"
		#./scripts/rankboost.sh
	elif [ "${METHOD}" == "BOB" ]
	then
		echo "bob!"
		exit
	fi

	EXEC="${RANKLIB} -ranker ${RANKER} ${OPTS} -metric2t NDCG@10"
    
    echo
    echo "training on clusters with ${METHOD} (${RANKER})"
    echo

	OUTPUT_DIR=${TRAIN_DIR}/${METHOD}

    echo
    echo "Creating directory ${OUTPUT_DIR}"
    echo

    mkdir -p ${OUTPUT_DIR}

    PARTS=`ls ${CLUSTER_LOCATION}/part_[^report]*`
    N=0
    for P in $PARTS;do
        ((N++))
    done

    I=0

    #
    #	In this section we go over each cluster and create a model which is saved in the output directory
    #	In addition we combine clusters and learn a model for those as well
    #
	JUMP=6

    # train over entire data set
    ${EXEC} -train ${CLUSTER_DATA} ${FEATURES} -save ${OUTPUT_DIR}/${METHOD} &

    while  [ $I -lt $N ]; do

        # spin up sub-processes
        END=$((I+JUMP))
        while [ $I -lt $END ] && [ $I -lt $N ] ; do
            # train on clusters
            ${EXEC} -train "${CLUSTER_LOCATION}/part_${I}" ${FEATURES} -save ${OUTPUT_DIR}/${METHOD}_${I} &
            ((I++))
        done

        # wait for subprocesses to finish
        wait
    done

done
