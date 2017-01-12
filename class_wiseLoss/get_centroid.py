#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:13:29 2016

@author: XFZ
"""

import mxnet as mx
import numpy as np
from sklearn.neighbors import NearestNeighbors
def get_centroid(DataIter,featureExtractor,
                 featureSize,numClasses,kNeighbors):
  centroids = np.zeros((numClasses,featureSize))
  cnt = np.zeros(numClasses)
  ii = 0
  for batch in DataIter:
    if ii%50==0:
      print ' update centroids:',ii
    ii +=1
    features = featureExtractor.predict(batch.data[0])
    features = np.squeeze(features)
    labels = batch.label[0].asnumpy()
    for i, label in enumerate(labels):
      centroids[int(label)] = (centroids[int(label)]*cnt[int(label)] \
                              +features[i]) / (cnt[int(label)]+1)
      cnt[int(label)] +=1
  nbrs = NearestNeighbors(n_neighbors=kNeighbors+1).fit(centroids)
  _,indices = nbrs.kneighbors(centroids)
  indices = [n[1:] for n in indices]
  neighbors = np.array(indices)
  return centroids,neighbors 
  