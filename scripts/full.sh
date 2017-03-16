#!/bin/bash

if ! [[ -v TESTING ]];
then 
	TESTING="NO"
fi

if ! [[ -v PREPROCESS ]];
then
	PREPROCESS="YES"
fi

if ! [[ -v CLUSTER ]];
then
	CLUSTER="YES"
fi

if ! [[ -v TRAINING ]];
then
	TRAINING="YES"
fi

if ! [[ -v EVALUATE ]];
then
	EVALUATE="YES"
fi

if ! [[ -v SUBSET ]];
then
	SUBSET="NO"
fi

if ! [[ -v MERGED ]];
then
	MERGED="YES"
fi

SUFFIX="all"

# Base directories
if [[ "${MERGED}" == "YES" ]];
then
	export MERGE_FLAG="-merged"
	PREFIX="merged"
else
	export MERGE_FLAG=""
	PREFIX="global"
fi

export PREPROCESS_CONFIG=./scripts/preprocess_all.conf
export FEATURES=""


if [[ "${SUBSET}" == "YES" ]];
then
	export PREPROCESS_CONFIG=./scripts/preprocess_sub.conf
	export FEATURES="-feature features"
	SUFFIX="sub"
fi

export OUTDIR=${PREFIX}_${SUFFIX}
export RAW_DATA=data/MQ2007

if [[ "$TESTING" == "YES" ]];
then
	export OUTDIR="testing_${OUTDIR}"
fi
# base directories used for each stage
export PROCESSED_DIR=${OUTDIR}/processed
export CLUSTER_DIR=${OUTDIR}/clusters
export RESULT_DIR=${OUTDIR}/results
export MODEL_DIR=${OUTDIR}/models

echo "TESTING ${TESTING}"

echo "OUTDIR ${OUTDIR}"
echo "RAW_DATA ${RAW_DATA}"
echo "PREPROCESS ${PREPROCESS}"
echo "CLUSTER ${CLUSTER}"
echo "TRAINING ${TRAINING}"
echo "EVALUATE ${EVALUATE}"

echo "SUBSET ${SUBSET}"
echo "MERGED '${MERGED}'"
echo "MERGE_FLAG '${MERGE_FLAG}'"

echo "PREPROCESS_CONFIG ${PREPROCESS_CONFIG}"
echo "FEATURES ${FEATURES}"

mkdir -p ${PROCESSED_DIR}
mkdir -p ${CLUSTER_DIR}
mkdir -p ${RESULT_DIR}
mkdir -p ${MODEL_DIR}

# Names of files
export TRAIN=train.txt
export TEST=test.txt
export VALI=vali.txt

### Cluster variables 

# Save cluster information here
export CLUSTER_INFO_FILE=cluster_info.txt
# Number of iterations to perform
export ITER=300

# starting k
if ! [[ -v MIN_K ]];
then
	MIN_K=2
fi

# ending k
if ! [[ -v MAX_K ]];
then
	MAX_K=20
fi

START_FOLD=1
END_FOLD=1

#exit

#
#	Algorithm configuration
#

# Weak Learners

if ! [[ -v METHODS ]];
then
    export METHODS="RANKBOOST RANKNET LAMBDA ADARANK"
else
	export METHODS=${METHODS}
fi

echo "Methods selected: ${METHODS}"

# Pre-process

if [[ ${PREPROCESS} == "YES" ]];
then
	echo
	echo  "Preprocess"
	echo
	(
		for i in `seq ${START_FOLD} ${END_FOLD}`;
		do
			# Extract the features needed
			OUTPUT_DIR=${PROCESSED_DIR}/Fold${i}
			mkdir -p ${OUTPUT_DIR}

			for f in train test vali;
			do
				INPUT=${RAW_DATA}/Fold${i}/${f}.txt
				OUTPUT=${OUTPUT_DIR}/${f}.txt
				echo "input: " ${INPUT}
				echo "output: " ${OUTPUT}
				./scripts/preprocess.py --config ${PREPROCESS_CONFIG} --input ${INPUT} --output ${OUTPUT}
			done
		done
	)
fi

# Cluster

if [[ "${CLUSTER}" == "YES" ]];
then
	echo
	echo "Clustering"
	echo 
	(
		for i in `seq ${START_FOLD} ${END_FOLD}`;
		do
			for k in `seq ${MIN_K} ${MAX_K}`;
			do
				export K=${k}
				export CLUSTER_OUT=${CLUSTER_DIR}/Fold${i}/${k}
				export CLUSTER_DATA=${PROCESSED_DIR}/Fold${i}/${TRAIN}
				export CLUSTER_INFO=${CLUSTER_OUT}/${CLUSTER_INFO_FILE}
				mkdir -p ${CLUSTER_OUT}
				./scripts/createClusters.sh
			done
		done
	)
fi


# train
if [[ "${TRAINING}" == "YES" ]];
then
	echo
	echo "Training"
	echo
	(
		for k in `seq ${MIN_K} ${MAX_K}`;
		do
			
			for i in `seq ${START_FOLD} ${END_FOLD}`;
			do
				export K=${k}
				export CLUSTER_LOCATION=${CLUSTER_DIR}/Fold${i}/${K}
				export CLUSTER_DATA=${PROCESSED_DIR}/Fold${i}/${TRAIN}
				export CLUSTER_INFO=${CLUSTER_LOCATION}/${CLUSTER_INFO_FILE}
				
				export TRAIN_DIR=${MODEL_DIR}/Fold${i}/${K}
				
				echo
				echo "creating partitions"
				echo

				./output/createPartitions -min 10 -input ${CLUSTER_INFO} -data ${CLUSTER_DATA} -out ${CLUSTER_LOCATION}
				rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
			
				export TRAINING_FILE=${PROCESSED_DIR}/Fold${i}/${TRAIN}	
				if [[ "${MERGED}" == "YES" ]];
				then
					./scripts/train_merged.sh
				else
					./scripts/train_global.sh
				fi

				rm -fv ${CLUSTER_LOCATION}/part_*
				rm -fv ${CLUSTER_LOCATION}/merged_*
			done
		done
	)
fi

# Run algorithm

if [[ "${EVALUATE}" == "YES" ]];
then
	echo
	echo "Evaluating"
	echo
	(
		INIT_FOLD=1
		
		INIT_J=1
		INIT_K=${MIN_K}

		START_VALI=${INIT_J}
		END_VALI=1
		for i in `seq ${START_FOLD} ${END_FOLD}`;
		do
			export TRAINING_FILE=${PROCESSED_DIR}/Fold${i}/${TRAIN}

			for j in `seq ${START_VALI} ${END_VALI}`;
			do
				export TESTING_FILE=${PROCESSED_DIR}/Fold${j}/${TEST}
				export VALIDATION_FILE=${PROCESSED_DIR}/Fold${j}/${VALI}
				export RAW_VALIDATION_FILE=${RAW_DATA}/Fold${j}/${VALI}

				for k in `seq ${INIT_K} ${MAX_K}`;
				do
					export K=${k}
					export CLUSTER_LOCATION=${CLUSTER_DIR}/Fold${i}/${K}
					export TRAIN_DIR=${MODEL_DIR}/Fold${i}/${K}
					export RESULTS=${RESULT_DIR}/Fold${i}/${K}/${j}
					./scripts/evaluate.sh
					INIT_K=${MIN_K}
				done
			done
			INIT_J=${START_VALI}
		done
	)
fi
