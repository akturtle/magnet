#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 23:41:31 2016

@author: XFZ
"""
import mxnet as mx 
from  magDataIter import DataIter
import cPickle
import numpy as np

from getMagnet import get_mag_net
from getMagnet import get_mag_hashing_net
#loading cifar10 data

    
class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('auc')
        self.sum_metric=0
        self.num_inst=0
    def update( self,labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)
    def reset(self):
        self.sum_metric = 0
        self.num_inst = 0
dataByClass = []
for i in range(10):
    f=open('data/batch_class_'+str(i),'rb')
    data=cPickle.load(f)
    f.close()
    dataByClass.append(data)
#pretrain args
prefix = "../model/inceptionBn/Inception-BN"
num_round = 126

sym,arg_params, aux_params = mx.model.load_checkpoint(prefix,num_round)
M= 4
D= 16
batchSize = 128
K=1
hash_len = 128
#initialize dataIter

##initialize net
#net = get_mag_hashing_net(mClass=M,dSample=D,batchSize=batchSize,\
#                          hashing_len=hash_len,quantScale=5,l=1)
net = get_mag_net(mClass=M,dSample=D,batchSize=batchSize,\
                          featureSize=hash_len)

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
#start training :
#create ls_scheduler
lr_scheduler =  mx.lr_scheduler.FactorScheduler(step = 300,factor = 0.5)
# create an optimizer for updating weights
opt = mx.optimizer.SGD(
    learning_rate=0.0002,
    momentum=0.9,
    wd=0.00001,
    rescale_grad=1.0/batchSize,
    lr_scheduler = lr_scheduler
    )
updater = mx.optimizer.get_updater(opt)
Qmetric= Auc()
Mmetric =Auc()
data_shape=[('data',(batchSize,3,224,224))]
label_shape = [('label',(batchSize,))]


pref = 'model/'
for epoch in range(50):
    #first initialize dataIter
    internals = net.get_internals()
    # get feature layer symbol out of internals
    #fea_symbol = internals["_minusscalar0_output"]
    fea_symbol = internals["fch_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, \
                                             numpy_batch_size=128,
                                             arg_params=executor.arg_dict,\
                                             aux_params=executor.aux_dict,
                                             allow_extra_params=True) 
    print "prepare data"
    DIter=DataIter(dataByClass,\
                     feature_extractor,K=K,\
                    batch_size=batchSize,\
                    mCluster=M,nSample=D,
                    featureSize = hash_len)
    
    
    t=0
    print 'start training loop'
    for batch in DIter:
        data[:] = batch.data[0]
        label[:] = batch.label[0]
        executor.forward(is_train=True)
        executor.backward(executor.outputs[0])
        for i, pair in enumerate(zip(executor.arg_arrays, executor.grad_arrays)):
            weight, grad = pair
            updater(i, grad, weight)
        Qmetric.update(batch.label, executor.outputs)
        Mmetric.update(batch.label,executor.outputs)
        t += 1
        if t % 10 == 0:
            print 'epoch:', epoch, 'iter:', t, 'Qloss:', Qmetric.get(), 'Mloss:', Mmetric.get()
    #save model
    Qmetric.reset()
    Mmetric.reset()
    if epoch%10 == 0:
        print  'save model'
        mx.model.save_checkpoint(pref,epoch,  net,\
                             executor.arg_dict,executor.aux_dict)
            
        
