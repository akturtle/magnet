#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 22:32:12 2016

@author: XFZ
"""

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
#loading cifar10 data
class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('auc')

    def update(self, labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)
dataByClass = []
for i in range(9):
    f=open('data/batch_class_'+str(i),'rb')
    data=cPickle.load(f)
    f.close()
    dataByClass.append(data)
#pretrain args
prefix = "../model/inceptionBn/Inception-BN"
num_round = 126

sym,arg_params, aux_params = mx.model.load_checkpoint(prefix,num_round)
internals = sym.get_internals()
# get feature layer symbol out of internals
fea_symbol = internals["global_pool_output"]
feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=128,
                                         arg_params=arg_params, aux_params=aux_params,
                                         allow_extra_params=True)
M=8
D=4
batchSize = 128
K=5
#initialize dataIter

#initialize net
net = get_mag_net()
#input_shapes = {'data':(batchSize, 3, 224,224 ),'label':(batchSize,)} 
#executor = net.simple_bind(ctx = mx.gpu(), **input_shapes)
#arg_arrays = dict(zip(net.list_arguments(), executor.arg_arrays))
#data = arg_arrays['data']
#label = arg_arrays['label']
#init = mx.init.Uniform(scale=0.01)
#print "load model"
#for key in executor.arg_dict.keys():
#    if key in arg_params:
##        print key, arg_params[key].shape, executor.arg_dict[key].shape
#        arg_params[key].copyto(executor.arg_dict[key])
#for key in executor.aux_dict.keys():
#    if key in aux_params:
##        print key, aux_params[key].shape, executor.arg_dict[key].shape
#        aux_params[key].copyto(executor.aux_dict[key])
#    else:
#        print key ,executor.aux_dict[key].shape
#        init(key,executor.aux_dict[key])
#start training :
# create an optimizer for updating weights
opt = mx.optimizer.SGD(
    learning_rate=0.1,
    momentum=0.9,
    wd=0.00001,
    rescale_grad=1.0/batchSize)
updater = mx.optimizer.get_updater(opt)
metric= Auc
data_shape=[('data',(batchSize,3,224,224))]
label_shape = [('label',(batchSize,))]
mod = mx.mod.Module(symbol=net,context = mx.gpu())
mod.bind(data_shapes=data_shape,label_shapes=label_shape)
mod.set_params(arg_params,aux_params,allow_missing=True)

#for epoch in range(100):
#    #first initialize dataIter
#    internals = net.get_internals()
#    # get feature layer symbol out of internals
#    fea_symbol = internals["global_pool_output"]
#    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, \
#                                             numpy_batch_size=128,
#                                             arg_params=executor.arg_dict,\
#                                             aux_params=executor.aux_dict,
#                                             allow_extra_params=True) 
#    print "prepare data"
#    DIter=DataIter(dataByClass,\
#                     feature_extractor,K=K,\
#                    batch_size=batchSize,\
#                    mCluster=M,nSample=D)
#    
#    metric.reset()
#    t=0
#    for batch in DIter:
#        data[:] = batch.data[0]
#        label[:] = batch.label[0]
#        executor.forward(is_train=True)
#        executor.backward()
#        for i, pair in enumerate(zip(executor.arg_arrays, executor.grad_arrays)):
#            weight, grad = pair
#            updater(i, grad, weight)
#        metric.update(batch.label, executor.outputs)
#        t += 1
#        if t % 50 == 0:
#            print 'epoch:', epoch, 'iter:', t, 'loss:', metric.get()
        
