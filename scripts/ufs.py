#!/usr/bin/python3.5

import argparse
from scipy.stats import pearsonr
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def parseLine(line):
	line = line.split('#')[0].split()
	X = line[2:]
	Y = float(line[0])
	for i in range(0,len(X)):
		X[i] = float(X[i].split(':')[1])	
	return np.asarray(X),Y

def readData(filename):
	X = []
	Y = []
	with open(filename) as f:
		lines = f.readlines()
		for line in lines:
			x , y = parseLine(line)
			X = X + [x]
			Y = Y + [y]
	return np.asarray(X),np.asarray(Y)

def chooseBest(X,y):
	print(X.shape)
	X_new = SelectKBest(chi2).fit_transform(X, y)
	print(X_new.shape)


parser = argparse.ArgumentParser(description="Compute pearson coefficient")
parser.add_argument('filename', metavar='i')

args = parser.parse_args()



X,Y = readData(args.filename)

chooseBest(X,Y)

for i in range(0,len(X[0])):
	C = []
	for j in range(0,len(X)):
		C = C + [X[j][i]]
	Xa = np.asarray(C)
	print(pearsonr(Xa,Y))

