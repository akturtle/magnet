#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:07:49 2016

@author: XFZ
"""

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
import mxnet as mx


class newLoss(mx.operator.CustomOp):
    def __init__(self, ctx, shapes, dtypes, nNeighbors=5,alpha=1,nClass=10):
        self.nNeighbors = nNeighbors
        self.batch_size = shapes[0][0]
        self.alpha = alpha
        self.nClass = int(nClass)
    def forward(self, is_train, req, in_data, out_data,aux):
        nNeighbors = int(self.nNeighbors)
        alpha = float(self.alpha)
        data = in_data[0]
        batchSize = self.batch_size
        labels = in_data[1].asnumpy()
        aux_h = aux[0] #summation 
        aux_diff = aux[1]
        auxCentroid = aux[2]
        auxNeighbors = aux[3]
        batchSize = data.shape[0]
        xpu = data.context        
        loss_out = mx.nd.zeros((batchSize,),ctx = xpu)
        diff= []
        for ii in range(batchSize):
            #calculate distance to the center  
            diff.append([])
            d = mx.nd.sum(mx.nd.square(data[ii] - auxCentroid[int(labels[ii])]))
            diff[ii].append(d)
            aux_diff[ii][0:1]=d
            neighbors =   auxNeighbors[int(labels[ii])].asnumpy()
            for i,neighbor in enumerate(neighbors):   
                d = mx.nd.sum(mx.nd.square(data[ii] - \
                    auxCentroid[int(neighbor)]))
                diff[ii].append(d)
                aux_diff[ii][i+1:i+2]=d
        
        for i in range(batchSize):   
            #calculate loss for data
            loss=mx.nd.zeros(1,ctx=xpu)
            frac = mx.nd.zeros(1,ctx=xpu)
            #sum the distance to centers from diferent class
            frac[:] = 0 
            for j in range(nNeighbors):
                frac += mx.nd.exp(- diff[i][j+1]) 
            aux_h[i:i+1] = frac
            f=diff[i][0]+alpha
            loss = f + mx.nd.log(frac)
            loss_out[i:i+1] = loss
                                           
        self.assign(out_data[0], req[0], loss_out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        nNeighbors = int(self.nNeighbors)
        aux_h = aux[0]
        aux_d = aux[1]
        auxCentroid = aux[2]
        auxNeighbors= aux[3]
        data = in_data[0]
        labels = in_data[1].asnumpy()
        xpu = data.context
        batchSize = data.shape[0]
        featureSize = data.shape[1]
        grad = mx.nd.zeros((batchSize,featureSize),ctx = xpu)
        #y = out_data[0].asnumpy()
        part = mx.nd.zeros((featureSize,),ctx=xpu)
        for i in range(batchSize):  
            part[:] = 0
            neighbors = auxNeighbors[int(labels[i])].asnumpy()
            for j in range(nNeighbors):
                score = mx.nd.exp(- aux_d[i][j+1:j+2])
                part += mx.nd.broadcast_mul(auxCentroid[int(neighbors[j])],\
                                                        score)
            gh = mx.nd.broadcast_div(part,aux_h[i:i+1])
            gf = data[i]-auxCentroid[int(labels[i])]
            g  = gf - data[i]*nNeighbors +gh
           #grad[ii*wrapSize+j] = g*y[ii*wrapSize+j]
            grad[i] = g
        self.assign(in_grad[0], req[0], grad)
                    
            
@mx.operator.register("newLoss")
class newLossProp(mx.operator.CustomOpProp):
    def __init__(self, nNeighbors, alpha, nClass):
        super(newLossProp, self).__init__(need_top_grad=False)

        # convert it to numbers
        self.nNeighbors = int(nNeighbors)
        self.alpha = float(alpha)
        self.nClass = int(nClass)
    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']
    
    def list_auxiliary_states(self):
        # call them 'bias' for zero initialization
        return [ 'h_bias', 'd_bias', \
        'centroid_bias','neighbors_bias']

    def infer_shape(self, in_shape):
        batchSize = in_shape[0][0]
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        d_shape = (batchSize,self.nNeighbors+1)
        h_shape = (batchSize,)
        c_shape = (self.nClass,in_shape[0][1])
        neighbors_shape = (self.nClass,self.nNeighbors)
        output_shape = (in_shape[0][0],)
        return [data_shape, label_shape], [output_shape] ,\
                    [ h_shape, d_shape,c_shape,neighbors_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return newLoss(ctx, shapes, dtypes, self.nNeighbors,\
                             self.alpha,self.nClass)

    