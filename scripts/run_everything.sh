
export PREPROCESS=YES
export CLUSTER=YES
export TRAINING=YES
export EVALUATE=YES
export MIN_K=2
export MAX_K=20

YN="YES NO"

for S in ${YN};
do
	export SUBSET=${S}
	for M in ${YN};
	do
		export MERGED=${M}
		./script/full.sh
	done
done
