#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:33:40 2016

@author: XFZ
"""

import mxnet as mx
"""References:
Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
def get_vgg():
    ## define alexnet
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = mx.symbol.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = mx.symbol.Pooling(
        data=relu4_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    pool5 = mx.symbol.Pooling(
        data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # output
    return drop7
def get_CNNF():
  input_data    = mx.sym.Variable(name='data')
  #group 1
  conv1 = mx.symbol.Convolution(
              data= input_data,kernel = (11,11),stride=(4,4),num_filter = 64)
  relu1 = mx.symbol.Activation(data = conv1, act_type="relu")
  lrn1 = mx.symbol.LRN(data = relu1,alpha=0.0001,beta = 0.75, knorm=1,nsize=5)
  pool1 = mx.symbol.Pooling(data=lrn1,pool_type = "max",\
                            kernel = (2,2),stride = (1,1))
  
  #group2
  conv2 = mx.symbol.Convolution(
              data= pool1,kernel = (5,5),stride=(1,1),num_filter = 256,pad =(2,2))
  relu2 = mx.symbol.Activation(data = conv2, act_type="relu")
  lrn2 = mx.symbol.LRN(data = relu2,alpha=0.0001,beta = 0.75, knorm=1,nsize=5)
  pool2 = mx.symbol.Pooling(data=lrn2,pool_type = "max",\
                            kernel = (2,2),stride = (1,1))
  
  #group3
  conv3 = mx.symbol.Convolution(
              data= pool2,kernel = (3,3),stride=(1,1),\
              num_filter = 256,pad =(1,1))
  relu3 = mx.symbol.Activation(data = conv3, act_type="relu")
  conv4 = mx.symbol.Convolution(
            data= relu3,kernel = (3,3),stride=(1,1),\
            num_filter = 256,pad =(1,1))
  relu4 = mx.symbol.Activation(data = conv4, act_type="relu")
  conv5 = mx.symbol.Convolution(
            data= relu4,kernel = (3,3),stride=(1,1),\
            num_filter = 256,pad =(1,1))
  relu5 = mx.symbol.Activation(data = conv5, act_type="relu")
  pool3 = mx.symbol.Pooling(data=relu5,pool_type = "max",\
                            kernel = (2,2),stride = (1,1))
  
  flatten = mx.symbol.Flatten(data=pool3)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
  relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
  drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
  fc2 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096)
  relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
  drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
  return drop7
  