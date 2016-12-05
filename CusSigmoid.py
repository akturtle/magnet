#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:26:42 2016

@author: XFZ
"""
import mxnet as mx
class SigmoidOp(mx.operator.CustomOp):
    def __init__(self,l):
        self.l = l
    
    def forward(self, is_train, req, in_data, out_data, aux):
        l= self.l
        x = in_data[0]
        y = mx.nd.exp(-x*l) 
        y = 2*(1/(1+y)-0.5)      
        
        self.assign(out_data[0], req[0], y)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = self.l
        y = out_data[0]
        z = out_grad[0]
        grad = y*(1 - y)
        grad = grad * 2 *l
        self.assign(in_grad[0], req[0], z*grad)


@mx.operator.register("Sigmoid")
class SigmoidProp(mx.operator.CustomOpProp):
    def __init__(self, l):
        super(SigmoidProp, self).__init__(need_top_grad=True)
        self.l = float(l)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        
        return [data_shape], [data_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return SigmoidOp(self.l)