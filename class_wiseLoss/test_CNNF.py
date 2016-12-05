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
    cnnf = get_CNNF()
    fc = mx.symbol.FullyConnected(data = cnnf, num_hidden = num_class,name = 'fc')
    softMax  = mx.sym.SoftmaxOutput(data = fc,name = 'softmax')    
    return softMax

net = get_symbol(10)
# Use utility function in test to download the data
# or manualy prepar
import get_data
get_data.GetCifar10()
# After we get the data, we can declare our data iterator
# The iterator will automatically create mean image file if it doesn't exist
batch_size = 128
total_batch = 50000 / 128 + 1
# Train iterator make batch of 128 image, and random crop each image into 3x28x28 from original 3x32x32
train_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=True,
        rand_mirror=True,
        data_shape=(3,28,28),
        batch_size=batch_size,
        preprocess_threads=1)
# test iterator make batch of 128 image, and center crop each image into 3x28x28 from original 3x32x32
# Note: We don't need round batch in test because we only test once at one time
test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/test.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,28,28),
        batch_size=batch_size,
        round_batch=False,
        preprocess_threads=1)
num_epoch = 200
model = mx.model.FeedForward(ctx=mx.gpu(), symbol=net, num_epoch=num_epoch,
                             learning_rate=0.05, momentum=0.9, wd=0.00001)
model.fit(X=train_dataiter,
          eval_data=test_dataiter,
          eval_metric="accuracy",
          batch_end_callback=mx.callback.Speedometer(batch_size))
