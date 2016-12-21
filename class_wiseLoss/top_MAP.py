#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:55:08 2016

@author: XFZ
"""

import mxnet as mx
import numpy as np
from random import shuffle
from sklearn.neighbors import KDTree
from newLoss import *
from inception_bn import get_inception_symbol
def get_net(feature_len):
    label =  mx.sym.Variable('label')  
    flatten = get_inception_symbol()
    fc = mx.symbol.FullyConnected(data=flatten, \
                                  num_hidden=feature_len, name='fc')
    bn_fc = mx.sym.L2Normalization(data=fc,name = 'bn_fc')
    myloss=mx.symbol.Custom(data=bn_fc,label=label,\
                                    name='myLoss',op_type = 'newLoss',\
                                    nNeighbors = 5,alpha = 0,\
                                    nClass = 10)
    loss = mx.symbol.MakeLoss(data=myloss,name='loss',)
    return loss
def get_feature(DataIter,featureExector):
    i=0
    print 'extract features'
    labels=[]
    features=[]
    for batch in DataIter:
        f = featureExector.predict(batch.data[0])
        f = np.sign(np.squeeze(f))
        label = batch.label[0].asnumpy()
        labels.extend(label)
        features.extend(f)
        i+=1
        print 'batch :',i
    return features,labels

load_prefix = './cifar_inBn_32'
load_epoch=50
feature_size = 32
sym,arg_params,aux_params = mx.model.load_checkpoint(load_prefix, load_epoch)
net = get_net(feature_size)
batchSize = 32
input_shapes = {'data':(batchSize, 3, 224,224 ),'label':(batchSize,)} 
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
        path_imgrec="/home/XFZ/dataSet/cifar10/cifar_224/cifarTest_224.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,224,224),
        batch_size=batchSize,
        round_batch=False,
        preprocess_threads=4)
internals = net.get_internals()
# get feature layer symbol out of internals
#fea_symbol = internals["_minusscalar0_output"]
fea_symbol = internals["bn_fc_output"]
feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, \
                                         numpy_batch_size=batchSize,
                                         arg_params=executor.arg_dict,\
                                         aux_params=executor.aux_dict,
                                       allow_extra_params=True)
train_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="/home/XFZ/dataSet/cifar10/cifar_224/cifarTrain_224.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,224,224),
        batch_size=batchSize,
        preprocess_threads=4)

trainFeature, trainLable = get_feature(train_dataiter,feature_extractor)
testFeature, testLable = get_feature(test_dataiter,feature_extractor)
tree = KDTree(trainFeature)
i = 0
MAP = 0
for tF,tL in zip(testFeature,testLable):
    _,inds = tree.query([tF],k=5000)
    score = 0
    for ind in inds[0]:
      #print trainLable[ind],tL  
      if trainLable[ind] == tL:
            score +=1
    score = float(score) / 5000
    print score
    MAP += score
    i += 1

print MAP/i 