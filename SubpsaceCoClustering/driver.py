#!/usr/bin/env python
# coding: utf-8

## Driver Code ##
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from subspaceProjectedClustering import projectedClustering
from plotOutput import plotClustering

def computeAccuracy(pred, expect):

    if len(pred) != len(expect):
        raise Exception("pred and expect must have the same length.")

    unique = np.unique(pred)

    acc = 0.0

    for cl in unique:
        points = np.where(pred == cl)[0]
        pclasses = expect[points]
        uclass = np.unique(pclasses)
        counts = [len(np.where(pclasses == u)[0]) for u in uclass]
        mcl = uclass[np.argmax(counts)]
        acc += np.sum(np.repeat(mcl, len(points)) == expect[points])

    acc /= len(pred)

    return acc


if __name__ == '__main__':
    
    ## Import Data ##
    data = pd.read_csv("../classic3.csv") 
    
    ## Label the documents type (classes) ##
    data.iloc[:,0] = data.iloc[:,0].replace('cran',int(0))
    data.iloc[:,0] = data.iloc[:,0].replace('cisi',int(1))
    data.iloc[:,0] = data.iloc[:,0].replace('med',int(2))
    
    X = data.to_numpy()
    groundTruth = X[:,0]
    X = X[:,1:]
    
    ## Dimnesionality Reduction ##
    
    #pca = PCA(n_components=10)
    #X = pca.fit_transform(X)

    #svd = TruncatedSVD(n_components=10)
    #X = svd.fit_transform(X)

    nmf =  NMF(n_components=10, init=None, random_state=0)
    X = nmf.fit_transform(X)
    
    ## Run the Projected Clustering Algorithm ##
    rseed = 327530

    M, D, expectedLabels = projectedClustering(X, k = 3, l = X.shape[1], seed = rseed, A = 5, B = 3)
    accuracy = computeAccuracy(expectedLabels, groundTruth)*100
    print("****Accuracy****:", accuracy)
    
    #Dims= [0,9]
    #plotClustering(X, M, expectedLabels, D = Dims)



