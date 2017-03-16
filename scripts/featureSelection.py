#!/usr/bin/python3
"""perform feature selection on some data"""

import numpy
import sys
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest

def load_csv(filename):
    """load a csv file"""
    return pd.read_csv(filename, sep=",", header=None)

def var_thresh(thresh, data):
    """return attributes within threshold"""
    sel = VarianceThreshold(thresh)
    return sel.fit_transform(data), sel.get_support(True)

def best_k(k_value, features, labels):
    """return the best K attributes"""
    select = SelectKBest(k=k_value)
    new_features = select.fit_transform(features, labels)
    indices = select.get_support(True)
    return new_features, indices

def main():
    """main method"""
    if len(sys.argv) != 4:
        print("usaged:", sys.argv[0], "features labels K")
        exit()
    threshold = 0.01
    features = load_csv(sys.argv[1])
    features, _ = var_thresh(threshold, features)
    max_attr = int(sys.argv[3])
    labels = load_csv(sys.argv[2])
    labels = labels.values.ravel()
    new_features, ind = best_k(max_attr, features, labels)
    orig_indices = []
    tran = numpy.array([ind])
    numpy.savetxt("indices" + str(max_attr) + ".txt", tran, fmt='%d', delimiter=",")
    numpy.savetxt("output.csv", new_features, delimiter=",")

main()
