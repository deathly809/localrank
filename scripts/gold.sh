
DATA=./data/MQ2007/Fold1/train.txt
VALI=./data/MQ2007/Fold1/vali.txt
FOLD=1
GOLD_DIR=./gold

# Prefix for running RankLib
RANKLIB="java -jar external/RankLib.jar"

METHODS="RANKBOOST RANKNET LAMBDA ADARANK"

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

	${EXEC} -train ${DATA} -feature features -save ${GOLD_DIR}/feat_${METHOD}.model 

	${EXEC} -load ${GOLD_DIR}/feat_${METHOD}.model -rank ${VALI} -score ${GOLD_DIR}/feat_${METHOD}_${FOLD}.vali

	perl scripts/evalScript.pl ${VALI} ${GOLD_DIR}/feat_${METHOD}_${FOLD}.vali ${GOLD_DIR}/feat_${METHOD}_${FOLD}.ndcg 0
    
done

