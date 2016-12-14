#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:07:44 2016

@author: XFZ
"""

import mxnet as mx
import numpy as np
import os
import sys
mxnet_root = '~/sandbox/new-mxnet/mxnet'
sys.path.append(os.path.join( mxnet_root, 'tests/python/common'))
from get_CNNF import *
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def get_symbol(num_class):
    """
    CNNF
    """
    cnnf = get_vgg()
    fc = mx.symbol.FullyConnected(data = cnnf, num_hidden = num_class,name = 'fc')
    softMax  = mx.sym.SoftmaxOutput(data = fc,name = 'softmax')    
    return softMax
def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=new_sym, context=devs)
    mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    mod.set_params(new_args, aux_params, allow_missing=True)
    mod.fit(train, val, 
        num_epoch=8,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),        
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc') 
# Use utility function in test to download the data
# or manualy prepar
prefix = "/home/XFZ/sandbox/model/vgg/vgg16"
num_classes = 10
sym,arg_params,aux_params = mx.model.load_checkpoint(prefix,0)


(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes, layer_name='drop7')
batch_size = 32
total_batch = 50000 / 128 + 1
train_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec='/home/XFZ/dataSet/cifar10/cifarTrain.bin',
        rand_crop=True,
        rand_mirror=True,
        data_shape=(3,224,224),
        batch_size=batch_size,
        preprocess_threads=1)
# test iterator make batch of 128 image, and center crop each image into 3x28x28 from original 3x32x32
# Note: We don't need round batch in test because we only test once at one time
test_dataiter = mx.io.ImageRecordIter(
        path_imgrec='/home/XFZ/dataSet/cifar10/cifarTest.bin',        
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,224,224),
        batch_size=batch_size,
        round_batch=False,
        preprocess_threads=1)
num_epoch = 200
fit(new_sym, new_args,aux_params,train_dataiter,test_dataiter,\
    batch_size,1)
