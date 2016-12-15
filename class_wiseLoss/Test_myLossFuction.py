#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:52:26 2016

@author: XFZ
"""
import mxnet as mx
import numpy as np
def forward(in_data,aux):
    nNeighbors = 5
    alpha = 1
    data = in_data[0]
    batchSize = 1
    labels = in_data[1].asnumpy()
    aux_h = aux[0] #summation 
    aux_diff = aux[1]
    auxCentroid = aux[2]
    auxSigma = aux[3]
    auxNeighbors = aux[4]
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
    #calculate sigma
    s=mx.nd.zeros(1,ctx = xpu)
    for j in range(batchSize):
       s += diff[j][0]
    s = s/(batchSize - 1)
    sigma = s.asnumpy()
    sigma = float(2*sigma[0])
    auxSigma[:] = s
    for i in range(batchSize):   
        #calculate loss for data
        loss=mx.nd.zeros(1,ctx=xpu)
        frac = mx.nd.zeros(1,ctx=xpu)
        #sum the distance to centers from diferent class
        frac[:] = 0 
        for j in range(nNeighbors):
            frac += mx.nd.exp(- diff[i][j+1]/sigma) 
        aux_h[i:i+1] = frac
        f=diff[i][0]/sigma+alpha
        loss = f + mx.nd.log(frac)
        loss_out[i:i+1] = loss
    return loss_out
def backward( in_data,  aux):
        nNeighbors = 5
        aux_h = aux[0]
        aux_d = aux[1]
        auxCentroid = aux[2]
        auxSigma = aux[3]
        auxNeighbors= aux[4]
        data = in_data[0]
        labels = in_data[1].asnumpy()
        xpu = data.context
        batchSize = data.shape[0]
        featureSize = data.shape[1]
        grad = mx.nd.zeros((batchSize,featureSize),ctx = xpu)
        Sigma= auxSigma.asnumpy()
        #y = out_data[0].asnumpy()
        part = mx.nd.zeros((featureSize,),ctx=xpu)
        for i in range(batchSize):
            sigma = Sigma[0]  
            part[:] = 0
            neighbors = auxNeighbors[int(labels[i])].asnumpy()
            for j in range(nNeighbors):
                score = mx.nd.exp(- aux_d[i][j+1:j+2]/(2*sigma))
                part += mx.nd.broadcast_mul(auxCentroid[int(neighbors[j])],score)
            gh = mx.nd.broadcast_div(part,aux_h[i:i+1])
            gf = data[i]-auxCentroid[int(labels[i])]
            g  = gf - data[i]*nNeighbors +gh
            g  = g/(sigma)
           #grad[ii*wrapSize+j] = g*y[ii*wrapSize+j]
            grad[i] = g
        return grad
ctx = mx.cpu()
featureSize = 100
numclass = 10
batchSize = 10
data = mx.random.normal(-5,5,shape=(batchSize,featureSize),ctx = ctx)
labels = mx.nd.zeros((batchSize,))
in_data= [data,labels]
neighbos = np.random.random_integers(0,10,(numclass,5))
aux_neighbors  = mx.nd.array(neighbos)
aux_centroids = mx.random.normal(-2,2,shape = (numclass,featureSize),ctx = ctx)
aux_h = mx.nd.zeros(batchSize,ctx =ctx)
aux_d =mx.nd.zeros(shape=(batchSize,6),ctx = ctx)
aux_s = mx.nd.zeros((1,))
aux = [aux_h,aux_d,aux_centroids,aux_s,aux_neighbors]
loss=forward(in_data,aux)
grad = backward(in_data,aux)

#gradient check 
data = mx.nd.zeros((3,featureSize))
h=mx.random.normal(-4,5,shape=(featureSize),ctx = ctx)
h = h*float(1e-5)
x = mx.random.normal(-5,5,shape=(featureSize),ctx = ctx)
x1= x-h
x2 = x+h
data[0] = x
data[1] = x1
data[2] = x2
label = mx.nd.array([1,1,1])
loss=forward([data,label],aux)
grad = backward([data,label],aux)
l=loss.asnumpy()
g = grad.asnumpy()
x = x.asnumpy()
h = h.asnumpy()
diff = l[2]-l[1]
d = np.sum(g[0]*(2*h))
ratio = diff-d
print ratio

