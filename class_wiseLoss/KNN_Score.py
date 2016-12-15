#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:00:21 2016

@author: XFZ
"""

import mxnet as mx
import numpy as np
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from newLoss import *
from get_simple_inception import get_simplet_inception
def get_net(feature_len):
    label =  mx.sym.Variable('label')  
    flatten = get_simplet_inception()
    fc = mx.symbol.FullyConnected(data=flatten, \
                                  num_hidden=feature_len, name='fc')
    bn_fc = mx.sym.L2Normalization(data=fc,name = 'bn_fc')
    myloss=mx.symbol.Custom(data=bn_fc,label=label,\
                                    name='myLoss',op_type = 'newLoss',\
                                    nNeighbors = 5,alpha = 0.7,\
                                    nClass = 10)
    loss = mx.symbol.MakeLoss(data=myloss,name='loss',)
    return loss
def KNN_test(DataIter,featureExector,splitRatio,n_neighbors,hash_len):
	#extrat feature by provide featureEx
    i=0
    print 'extract features'
    labels=[]
    features=[]
    for batch in DataIter:
        f = featureExector.predict(batch.data[0])
        f = np.squeeze(f)
        label = batch.label[0].asnumpy()
        labels.extend(label)
        features.extend(f)
        i+=1
        #print i
    data_label = zip(features,labels)
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
def centroidScore(DataIter,featureExector,hash_len,centroids):
    numClass = len(centroids)
    CKN= KNeighborsClassifier(metric='euclidean',\
    		n_neighbors=1)
    centroid_label = [x for x in range(numClass)] 
    CKN.fit(centroids,centroid_label)                  
    batchSize = DataIter.provide_data[0][1][0]
    totalScore = 0
    cnt = 0
    print 'extract features'
    for batch in DataIter:
        f = featureExector.predict(batch.data[0])
        f = np.squeeze(f)
        label = batch.label[0].asnumpy()
        score = CKN.score(f,label)
        totalScore += batchSize *score
        cnt += batchSize
    return totalScore/cnt
  
  
  
  
  
  
  
load_prefix = './cifar_new_128'
load_epoch=79
feature_size = 128
sym,arg_params,aux_params = mx.model.load_checkpoint(load_prefix, load_epoch)
net = get_net(feature_size)
batchSize = 128
input_shapes = {'data':(batchSize, 3, 28,28 ),'label':(batchSize,)} 
executor = net.simple_bind(ctx = mx.gpu(), **input_shapes)
arg_arrays = dict(zip(net.list_arguments(), executor.arg_arrays))
data = arg_arrays['data']
label = arg_arrays['label']
init = mx.init.Uniform(scale=2)
print "load model"
for key in executor.arg_dict.keys():
    if key in arg_params:
#        print key, arg_params[key].shape, executor.arg_dict[key].shape
        arg_params[key].copyto(executor.arg_dict[key])
    else:
        if key not in ['label','data']:
            print key ,executor.arg_dict[key].shape
            init(key,executor.arg_dict[key])
for key in executor.aux_dict.keys():
    if key in aux_params:
#        print key, aux_params[key].shape, executor.arg_dict[key].shape
        aux_params[key].copyto(executor.aux_dict[key])
    else:
        print key ,executor.aux_dict[key].shape
        init(key,executor.aux_dict[key])

#loading dataIter

total_batch = 50000 / 128 + 1
# Train iterator make batch of 128 image, and random crop each image into 3x28x28 from original 3x32x3
test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/test.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,28,28),
        batch_size=batchSize,
        round_batch=False,
        preprocess_threads=1)
internals = net.get_internals()
# get feature layer symbol out of internals
#fea_symbol = internals["_minusscalar0_output"]
fea_symbol = internals["bn_fc_output"]
feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, \
                                         numpy_batch_size=128,
                                         arg_params=executor.arg_dict,\
                                         aux_params=executor.aux_dict,
                                       allow_extra_params=True)
score = KNN_test(DataIter=test_dataiter,\
                 featureExector=feature_extractor,\
                 splitRatio=0.1,
                 n_neighbors=50,
                 hash_len=feature_size) 
print 'knn score :',score
centroids = aux_params['myLoss_centroid_bias'].asnumpy()
test_dataiter.reset()
cScore = centroidScore(DataIter=test_dataiter,\
                       featureExector=feature_extractor,\
                       hash_len=32,
                       centroids = centroids)
print cScore