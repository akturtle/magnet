#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:47:46 2016

@author: XFZ
"""
import mxnet as mx
import numpy as np
from get_simple_inception import get_simplet_inception
from get_centroid import get_centroid
from class_wiseLoss import *
from math import isnan
def get_net(feature_len):
    label =  mx.sym.Variable('label')  
    flatten = get_simplet_inception()
    fc = mx.symbol.FullyConnected(data=flatten, \
                                  num_hidden=feature_len, name='fc')
    bn_fc = mx.sym.BatchNorm(data=fc,name = 'bn_fc')
    myloss=mx.symbol.Custom(data=bn_fc,label=label,\
                                    name='myLoss',op_type = 'myLoss',\
                                    nNeighbors = 5,alpha = 0.7,\
                                    nClass = 10)
    loss = mx.symbol.MakeLoss(data=myloss,name='loss',)
    return loss
    

#loading pretrianed  model 
load_prefix = './cifar_myLoss_1024'
load_epoch=3
featureSize = 1024
numClass = 10
numNeighbors = 5
sym,arg_params,aux_params = mx.model.load_checkpoint(load_prefix, load_epoch)
net = get_net(featureSize)
batchSize = 128
input_shapes = {'data':(batchSize, 3, 28,28 ),'label':(batchSize,)} 
executor = net.simple_bind(ctx = mx.gpu(), **input_shapes)
arg_arrays = dict(zip(net.list_arguments(), executor.arg_arrays))
data = arg_arrays['data']
label = arg_arrays['label']
init = mx.init.Uniform(scale=2)
all_centroids = []
all_neighbors = []
print "loading model"
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
# Train iterator make batch of 128 image, and random crop each image into 3x28x28 from original 3x32x32
train_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=True,
        rand_mirror=True,
        data_shape=(3,28,28),
        batch_size=batchSize,
        preprocess_threads=4)
center_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        data_shape=(3,28,28),
        batch_size=batchSize,
        preprocess_threads=4)
lr_scheduler =  mx.lr_scheduler.FactorScheduler(step = 200,factor = 0.998)
#setting updater
opt = mx.optimizer.SGD(
    learning_rate=0.001,
    momentum=0.9,
    wd=0.00001,
    rescale_grad=1.0/batchSize,
    lr_scheduler = lr_scheduler
    )
optt = mx.optimizer.Adam()
updater = mx.optimizer.get_updater(optt)
updateStep = total_batch # after howmany batch update centroids and neighbors 
uStep = updateStep
t = 0  
pref = './cifar_myLoss_1024'
#print 'update centroids'
#internals = net.get_internals()
#fea_symbol = internals["bn_fc_output"]
#feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, \
#                                   numpy_batch_size=128,
#                                   arg_params=executor.arg_dict,\
#                                   aux_params=executor.aux_dict,
#                                 allow_extra_params=True) 
##reset data iterator
#center_dataiter.reset()
#centroids,neighbors=get_centroid(DataIter=center_dataiter,\
#             featureExtractor=feature_extractor,\
#             featureSize=featureSize,\
#             numClasses=numClass,\
#             kNeighbors=numNeighbors)
##copy centorid and neighbors to GPU
#c = mx.nd.array(centroids)
#c.copyto(executor.aux_dict['myLoss_centroid_bias'])
#n = mx.nd.array(neighbors)
#n.copyto(executor.aux_dict['myLoss_neighbors_bias'])
for batch in train_dataiter:
  data[:] = batch.data[0]
  label[:] = batch.label[0]
  executor.forward(is_train=True)
  internals = net.get_internals()
  fea_symbol = internals["bn_fc_output"]
  feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, \
                                     numpy_batch_size=128,
                                     arg_params=executor.arg_dict,\
                                     aux_params=executor.aux_dict,
                                   allow_extra_params=True) 
  f = feature_extractor.predict(data)
  check=executor.outputs[0].asnumpy()
  break
    