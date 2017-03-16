#!/bin/bash

export GODEBUG=cgocheck=0

if ! [[ -v METHODS ]];
then
    METHODS="RANKBOOST RANKNET LAMBDA ADARANK"
fi

CLUSTER_OUT=${CLUSTER_LOCATION}/cluster_${CLUSTER_INFO_FILE}

if ! [[ -v EXEC_LOC ]];
then
    EXEC_LOC=output
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
		#./scripts/rankboost.sh
	elif [ "${METHOD}" == "BOB" ]
	then
		echo "bob!"
		exit
	fi

	EXEC="${RANKLIB} -ranker ${RANKER} -metric2t NDCG@10"
    
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

    

    #
    #	In this section we go over each cluster and create a model which is saved in the output directory
    #	In addition we combine clusters and learn a model for those as well
    #
	JUMP=8
    # special case
    I=0
    while  [ $I -lt $N ]; do

        O_FILE=${OUTPUT_DIR}/${METHOD}_${I}

        ${EXEC} -train "${CLUSTER_LOCATION}/part_${I}" -save ${OUTPUT_DIR}/${METHOD}_${I} &

        # train on pairs
        J=$((I+1))
        while [ $J -lt $N ]; do

			START=$J	
			END=$((START+JUMP))

			while [ $START -lt $END ] && [ $START -lt $N ] ; do
                P3="merged_${I}_${START}"
                O_FILE=${OUTPUT_DIR}/${METHOD}_${I}_${START}
                #if ! [[ -e ${O_FILE} ]];
                #    then

                    ${EXEC_LOC}/mergeClusters -in1 "${CLUSTER_LOCATION}/part_$I" -in2 "${CLUSTER_LOCATION}/part_$START" -out ${CLUSTER_LOCATION}/${P3}
                    rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

                    ${EXEC} -train ${CLUSTER_LOCATION}/${P3} -save ${OUTPUT_DIR}/${METHOD}_${I}_${START} &

                    ((START++))
                #fi

			done
                
			wait
            J=$END

        done
        ((I++))

        wait
    done
    
    wait
done
