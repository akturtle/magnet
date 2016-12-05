#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:20:58 2016

@author: XFZ
"""

import mxnet as mx
from symbol_inception_bn import get_symbol
import cPickle
import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
from getMagnet import *
from featureExtract import clusterLabel
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
prefix = "../model/inceptionBn/Inception-BN"
num_round = 126

sym,arg_params, aux_params = mx.model.load_checkpoint(prefix,num_round)
M=8
D=4
batchSize = 128
K=5
featureSize = 32
#initialize dataIter

#initialize net
net = get_mag_net(mClass=M,dSample=D,batchSize=batchSize,\
                          featureSize=featureSize)
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
internals = net.get_internals()
# get feature layer symbol out of internals
#fea_symbol = internals["_minusscalar0_output"] #hashing
fea_symbol = internals["fch_output"] 

feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=128,\
                                         arg_params=executor.arg_dict,\
                                         aux_params=executor.aux_dict,
                                         allow_extra_params=True)
dataByClass = []
for i in range(9):
    f=open('data/batch_class_'+str(i),'rb')
    data=cPickle.load(f)
    f.close()
    dataByClass.append(data[0:500])
#from magDataIter import DataIter
#DIter=DataIter(dataByClass,\
#                     feature_extractor,K=K,\
#                    batch_size=batchSize,\
#                    mCluster=M,nSample=D,
#                    featureSize = featureSize)
#minbatch,minlabel = DIter.generate_minbatch()
clusterlabel = []
clusterCount = []
K=5
clusterCentroid = []
print 'extract feature'        
for i,classData in enumerate(dataByClass):
    labels,centers,count = clusterLabel(classData,\
                                        feature_extractor,\
                                        128,K,32)
    print 'class'+str(i)
    clusterlabel.append(labels)
    clusterCount.append(count)
    clusterCentroid.extend(centers)

