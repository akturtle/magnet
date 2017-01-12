#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:01:07 2017

@author: XFZ
"""

import mxnet as mx
def get_iterators(batch_size, data_shape=(3, 224, 224)):
  data_prefix = '/home/XFZ/dataSet/voc2012/'  
  train = mx.io.ImageRecordIter(
        path_imgrec         = data_prefix+'ilsvrc12_30_train.rec',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
  val = mx.io.ImageRecordIter(
        path_imgrec         = data_prefix+'ilsvrc12_30_test.rec', 
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False)
  return (train, val)

def get_fine_tune_model(sym, arg_params, num_classes, layer_name='flatten'):
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
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    mod.set_params(new_args, aux_params, allow_missing=True)
    mod.fit(train, val, 
        num_epoch=8,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),        
        kvstore='device',
        optimizer='adam',
        optimizer_params={'learning_rate':0.001},
        eval_metric='acc')
    return mod.score(val)

load_prefix = '../../model/inceptionBn/Inception-BN'
load_epoch=126
sym,arg_params,aux_params = mx.model.load_checkpoint(load_prefix, load_epoch)
num_classes = 30
batch_per_gpu = 64
num_gpus = 1

(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)

batch_size = batch_per_gpu * num_gpus
(train, val) = get_iterators(batch_size)
mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
assert mod_score > 0.77, "Low training accuracy."