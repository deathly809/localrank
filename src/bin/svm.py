from sklearn import svm
from sklearn.datasets import load_svmlight_file

import argparse

parser = argparse.ArgumentParser(description='Classify cluster.')
parser.add_argument( '-i', '--input', help = 'file we wish to perform classification on', required=True)
parser.add_argument( '-o', '--output', help = 'file where results are saved', required=True)

args = parser.parse_args()


# load X, load y
X,Y = load_svmlight_file(args.input)

clf = svm.SVC()
clf.fit(X,Y)

import pickle

s = pickle.dump(clf,open(args.output,mode="w+"))