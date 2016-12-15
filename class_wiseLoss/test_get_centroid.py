#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:58:56 2016

@author: XFZ
"""
import mxnet as mx 
import numpy as np
import get_simple_inception
import get_centroid
from train_class_wise_cifar10 import get_net
from sklearn.neighbors import NearestNeighbors
load_prefix = 'cifar10_'
load_epoch=1
sym,arg_params,aux_params = mx.model.load_checkpoint(load_prefix, load_epoch)
net = get_net(1024)
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
fea_symbol = internals["fc_output"]
feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, \
                                         numpy_batch_size=128,
                                         arg_params=executor.arg_dict,\
                                         aux_params=executor.aux_dict,
                                       allow_extra_params=True) 
numClasses = 10
featureSize = 1024
DataIter = test_dataiter
centroids = np.zeros((numClasses,featureSize))
cnt = np.zeros(numClasses)
kNeighbors = 5
for batch in DataIter:
  features = feature_extractor.predict(batch.data[0])
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
