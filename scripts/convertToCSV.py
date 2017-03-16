#!/usr/bin/python3

#cat train.txt | grep "qid:[0-9]*" | gawk '{ split($3,a,":"); printf(""a[2]); for(i=4;i<=47;i++) { split($i,a,":"); printf(", "a[2]); } printf("\n");}' 

import numpy as np
import sys
from sklearn.ensemble import GradientBoostingRegressor

np.seterr(all='raise')

labels = []
features = []
f = open(sys.argv[1],"r")
for line in f:
    # strip off comments
    line = line[:line.find('#') - 1]
    ls = line.split()
    labels.append(int(ls[0]))
    features.append([float(x[x.find(':') + 1:]) for x in ls[2:]])
f.close()

labels = np.asarray(labels, dtype=np.int32)
features = np.asarray(features)

# to test with gbm
np.savetxt('labels.csv', labels, delimiter=',')
np.savetxt('features.csv', features, delimiter=',')

