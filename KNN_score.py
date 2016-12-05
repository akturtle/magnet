#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 22:19:30 2016

@author: XFZ
"""

import mxnet as mx
import cPickle
import numpy as np
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
import cv2
from magnetLoss import *
def unpickle(file):	
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict
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
        
def KNN_test(data,label,featureExector,splitRatio,n_neighbors,hash_len):
	#extrat feature by provide featureEx
    batchSize = 128
    nSample = len(data)
    featureSize=hash_len
    remains = nSample%batchSize
    features = np.zeros((nSample,featureSize))
    i=0
    print 'extract features'
    for batch in provideBatch(data,batchSize):
        f = featureExector.predict(batch)
        f = np.squeeze(f)
        if (i+1)*batchSize <= nSample :
            features[i*batchSize:(i+1)*batchSize]=f
        else:
            features[i*batchSize:nSample-1]=f[0:(remains-1)]
        i+=1
        print i
    data_label = zip(features,label)
    shuffle(data_label)
    splitPoint = int(len(data_label)*splitRatio)
    test = data_label[0:splitPoint]
    train = data_label[splitPoint:]
    test_data = [x for (x,y) in test]
    test_label = [y for (x, y) in test]
    train_data = [x for (x, y) in train]
    train_label = [y for (x, y) in train]
    neigh = KNeighborsClassifier(metric='euclidean',\
    		n_neighbors=n_neighbors)
    neigh.fit(train_data, train_label)
    score = neigh.score(test_data,test_label)
    return score
#def main():
#    prefix =  'model/' 
#    num_round = 5
#    hash_len = 32
#    model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=128)
#    internals = model.symbol.get_internals()
#    fea_symbol = internals["fch_output"]
#    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=128,
#                                             arg_params=model.arg_params, aux_params=model.aux_params,
#                                             allow_extra_params=True)
#
#    dic=unpickle('../test/data/cifa10/test_batch')
#    data = dic['data']
#    label = dic['labels']
#    score = KNN_test(data,label,feature_extractor,0.2,1,hash_len)
#    print score
#if __name__ == '__main__':
#	main()
prefix =  'model/' 
num_round = 10
hash_len = 128
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(), numpy_batch_size=128)
internals = model.symbol.get_internals()
fea_symbol = internals["fch_output"]
feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=128,
                                         arg_params=model.arg_params, aux_params=model.aux_params,
                                         allow_extra_params=True)

dic=unpickle('../test/data/cifa10/test_batch')
data = dic['data']
label = dic['labels']
score = KNN_test(data,label,feature_extractor,0.2,5,hash_len)
print score