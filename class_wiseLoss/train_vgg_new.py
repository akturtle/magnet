#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 23:35:08 2016

@author: XFZ
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:44:41 2016

@author: XFZ
"""

import mxnet as mx 
import numpy as np
from vgg_symbol import get_vgg16
from get_centroid import get_centroid
from newLoss import *
from math import isnan
#define metric
class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('Loss')
        self.sum_metric=0
        self.num_inst=0
    def update( self,labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)
    def reset(self):
        self.sum_metric = 0
        self.num_inst = 0

def get_net(feature_len):
    label =  mx.sym.Variable('label')  
    flatten = get_vgg16()
    fc = mx.symbol.FullyConnected(data=flatten, \
                                  num_hidden=feature_len, name='fc')
    bn_fc = mx.sym.L2Normalization(data=fc,name = 'bn_fc')
    myloss=mx.symbol.Custom(data=bn_fc,label=label,\
                                    name='myLoss',op_type = 'newLoss',\
                                    nNeighbors = 5,alpha = 0,\
                                    nClass = 10)
    loss = mx.symbol.MakeLoss(data=myloss,name='loss',)
    return loss
    

#loading pretrianed  model 
load_prefix ='./cifar_vgg_32'
load_epoch=10
featureSize = 16
numClass = 10
numNeighbors = 5
sym,arg_params,aux_params = mx.model.load_checkpoint(load_prefix, load_epoch)
net = get_net(featureSize)
batchSize = 32
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

#loading dataIter

total_batch = 50000 / 32 + 1
# Train iterator make batch of 128 image, and random crop each image into 3x28x28 from original 3x32x32
train_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="/home/XFZ/dataSet/cifar10/cifar_224/cifarTrain_224.bin",
        rand_crop=False,
        rand_mirror=True,
        data_shape=(3,224,224),
        batch_size=batchSize,
        preprocess_threads=4)
center_dataiter = mx.io.ImageRecordIter(
        shuffle=True,
        path_imgrec="/home/XFZ/dataSet/cifar10/cifar_224/cifarTrain_224.bin",
        data_shape=(3,224,224),
        batch_size=batchSize,
        preprocess_threads=4)
# test iterator make batch of 128 image, and center crop each image into 3x28x28 from original 3x32x32
# Note: We don't need round batch in test because we only test once at one time
test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="/home/XFZ/dataSet/cifar10/cifar_224/cifarTest_224.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,224,224),
        batch_size=batchSize,
        round_batch=False,
        preprocess_threads=1)
#start training
#create ls_scheduler
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
Mmetric =Auc()
pref = './cifar_vgg_32'
#record total iter and loss 
tIter = 0
trainLoss = []
valLoss = []
for epoch in range(1,101):
    for batch in train_dataiter:   
        if uStep%updateStep ==0:
            #get centroid and neighbor relation
            print 'update centroids'
            internals = net.get_internals()
            fea_symbol = internals["bn_fc_output"]
            feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, \
                                               numpy_batch_size=128,
                                               arg_params=executor.arg_dict,\
                                               aux_params=executor.aux_dict,
                                             allow_extra_params=True) 
            #reset data iterator
            center_dataiter.reset()
            centroids,neighbors=get_centroid(DataIter=center_dataiter,\
                         featureExtractor=feature_extractor,\
                         featureSize=featureSize,\
                         numClasses=numClass,\
                         kNeighbors=numNeighbors)
            #copy centorid and neighbors to GPU
            c = mx.nd.array(centroids)
            c.copyto(executor.aux_dict['myLoss_centroid_bias'])
            n = mx.nd.array(neighbors)
            n.copyto(executor.aux_dict['myLoss_neighbors_bias'])
            uStep = 0
            if len( all_centroids) !=0:  
              c_diff = centroids - all_centroids[-1]
              c_diff = np.sum(np.square(c_diff))/centroids.shape[0]
              print 'center moves:',c_diff
            #all_centroids.append(centroids)
            #all_neighbors.append(neighbors)
            
        data[:] = batch.data[0]
        label[:] = batch.label[0]
        executor.forward(is_train=True)
        executor.backward(executor.outputs[0])
        for i, pair in enumerate(zip(executor.arg_arrays, executor.grad_arrays)):
            weight, grad = pair
            updater(i, grad, weight)
        Mmetric.update(batch.label,executor.outputs)
        t += 1
        tIter +=1 
        if t % 50 == 0:
            print 'epoch:', epoch, 'iter:', t, 'Mloss:', Mmetric.get()
            trainLoss.append((tIter,Mmetric.get()[1]))
            Mmetric.reset()
        
        uStep += 1 
    train_dataiter.reset()
    print 'validation:'
    test_dataiter.reset()
    Mmetric.reset()
    for batch in test_dataiter:
      data[:] = batch.data[0]
      label[:] = batch.label[0]
      executor.forward(is_train=True)
      Mmetric.update(batch.label,executor.outputs)
    print 'epoch validation:', epoch, 'Mloss:', Mmetric.get()
    valLoss.append((tIter,Mmetric.get()[1]))
    Mmetric.reset()
    t=0
    if (epoch)%10 == 0:
        print  'save model:epoch:',epoch
        mx.model.save_checkpoint(pref,epoch,  net,\
                             executor.arg_dict,executor.aux_dict)
#save loss-iter
import cPickle
f1 = open('new_8_train_loss.data','w')
cPickle.dump(trainLoss,f1)
f1.close()

f2= open('new_8_val_loss.data','w')
cPickle.dump(valLoss,f2)
f2.close()
     
      
            

      