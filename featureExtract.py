#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 18:37:37 2016

@author: XFZ
"""

##test extract feature 
import mxnet as mx
import cv2
from sklearn.cluster import KMeans
import numpy as np
import cPickle
def provideBatch(data,batchSize=128):
    nRound = len(data)/batchSize
    nSample = len(data)
    for start in range(nRound+1):
        batch = np.zeros((batchSize,3,224,224))
        for i in range(batchSize) :
            ii=start*batchSize+i
            if ii<nSample:
                img = data[ii]
                img = img.reshape(32,32,3,order='F')
                img = cv2.resize(img,(224,224))

                img=img.swapaxes(2,0)
                batch[i]=img
        yield batch
def clusterLabel(classData,featureExector, batchSize,nClusters,featureSize):
    nSample = len(classData)
    remains = nSample%batchSize
    features = np.zeros((nSample,featureSize))
    i=0
    
    for batch in provideBatch(classData,batchSize):
        f = featureExector.predict(batch)
        f = np.squeeze(f)
        if (i+1)*batchSize <= nSample :
            features[i*batchSize:(i+1)*batchSize]=f
        else:
            features[i*batchSize:nSample-1]=f[0:(remains-1)]
        i+=1
    
    kmeans = KMeans(n_clusters =nClusters).fit(features)
    
    centers = kmeans.cluster_centers_
    labels  = kmeans.labels_
    count = np.zeros(nClusters)
    for i in labels:
        count[i] += 1
    
    return labels,centers,count
if __name__ == "__main__":   
    prefix = "../model/inceptionBn/Inception-BN"
    num_round = 126
    model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=128)
    internals = model.symbol.get_internals()
    # get feature layer symbol out of internals
    fea_symbol = internals["global_pool_output"]
    # Make a new model by using an internal symbol. We can reuse all parameters from model we trained before
    # In this case, we must set ```allow_extra_params``` to True
    # Because we don't need params from FullyConnected symbol
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=128,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)
    # predict feature
    f=open('data/batch_class_0','rb')
    data=cPickle.load(f)
    f.close()
    labels,centers,count = clusterLabel(data,feature_extractor,128,5)