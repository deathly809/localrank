#/bin/bash

EXEC="java -jar external/RankLib.jar"
EVAL="perl ./scripts/evalScript.pl"

FLAG=0

TRAIN_INPUT=./data/MQ2007/Fold2/train.txt
TEST_INPUT=./data/MQ2007/Fold2/vali.txt

TRAIN_OUTPUT=train.dat
PRED_OUTPUT=pred.dat

METHODS="RANKBOOST RANKNET LAMBDA ADARANK"

for METHOD in ${METHODS};
do

    echo "METHOD"

    RANKER=0

	if [ "${METHOD}" == "CLUSTER" ] 
	then
		./scripts/logistic.sh
	elif [ "${METHOD}" == "SVMRANK" ]
	then
		./scripts/svmrank.sh
	elif [ "${METHOD}" == "RANKNET" ]
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

    EVAL_OUTPUT=./results/Fold2/${METHOD}/${METHOD}_combine

    ${EXEC} -ranker ${RANKER}       -metric2t NDCG@10       -train ${TRAIN_INPUT} -save ${TRAIN_OUTPUT}
    ${EXEC}  -load   ${TRAIN_OUTPUT} -rank   ${TEST_INPUT} -score ${PRED_OUTPUT}
    ${EVAL} ${TEST_INPUT} ${PRED_OUTPUT} ${EVAL_OUTPUT} ${FLAG}

done