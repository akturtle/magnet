#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:03:45 2016

@author: XFZ
"""
#test loss layer
import os
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'
import mxnet as mx
from magnetLoss import *

data= mx.sym.Variable('data')
label  = mx.sym.Variable('label')


center_loss = mx.sym.MakeLoss(mx.symbol.Custom(data=data, label=label, name='loss', op_type='magnetLoss',\
           M=12, D=2, batchsize=48))
                    
data= mx.sym.Variable('data')
label  = mx.sym.Variable('label')
center_loss = mx.symbol.Custom(data=data, label=label, name='loss', op_type='magnetLoss',\
           M=12, D=2, batchsize=48)
ctx = mx.gpu()
data=mx.random.normal(0,5,shape=(48,100),ctx=ctx)
nplabel = [0,0,0,0,0,0,0,0,1,1,1,1]
label = mx.nd.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0,3,3,\
                     0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0,3,3],ctx=ctx)
f=mx.nd.zeros((48,),ctx = ctx)
d=mx.nd.zeros((48,12),ctx = ctx)
h=mx.nd.zeros((48,),ctx=ctx)
c=mx.nd.zeros((24,100),ctx=ctx)
s=mx.nd.zeros((2,),ctx=ctx)
lossGrade = mx.nd.empty(shape=(48,100),ctx=ctx)
print center_loss.list_auxiliary_states()

ex=center_loss.bind(ctx=ctx,\
                    args={'data':data,'label':label},\
               args_grad={'data':lossGrade} , \
                   aux_states={'loss_f_bias':f,'loss_h_bias':h,\
                               'loss_d_bias':d,'loss_centroid_bias':c,'loss_s_bias':s})
ex.forward(is_train=True)
y = ex.outputs[0]
ex.backward(y)