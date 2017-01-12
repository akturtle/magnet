#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 15:45:01 2016

@author: XFZ
"""
import mxnet as mx 
import numpy as np
from inception_bn import get_inception_symbol
from get_centroid import get_centroid
from newLoss import *
from math import isnan
def get_net(feature_len):
    label =  mx.sym.Variable('label')  
    flatten = get_inception_symbol()
    fc = mx.symbol.FullyConnected(data=flatten, \
                                  num_hidden=feature_len, name='fc')
    bn_fc = mx.sym.L2Normalization(data=fc,name = 'bn_fc')
    myloss=mx.symbol.Custom(data=bn_fc,label=label,\
                                    name='myLoss',op_type = 'newLoss',\
                                    nNeighbors = 10,alpha = 0,\
                                    nClass = 21)
    loss = mx.symbol.MakeLoss(data=myloss)
    #quant = mx.sym.abs(data = fc,name = 'abs')
    #quant = mx.symbol.MakeLoss(data=quant)
    #loss = mx.sym.Group([myLoss,quant])
    return loss
    
featureSize = 32
numClass = 21
numNeighbors = 10

load_prefix='./nus_inBn_'+str(featureSize)
load_epoch = 15
sym,arg_params,aux_params = mx.model.load_checkpoint(load_prefix, load_epoch)
net = get_net(featureSize)
batchSize = 64
input_shapes = {'data':(batchSize, 3, 224,224 ),'label':(batchSize,)} 
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
center_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="/home/XFZ/dataSet/nus_wide/train_linux.rec",
        data_shape=(3,224,224),
        batch_size=batchSize,
        preprocess_threads=4)
internals = net.get_internals()


#fea_symbol = internals["bn_fc_output"]
#feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, \
#                                   numpy_batch_size=batchSize,
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
