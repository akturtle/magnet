#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 09:39:36 2016

@author: XFZ
"""
import mxnet as mx
from CusSigmoid import *
from math import isnan
def mLoss(in_data,M,D,aux):
        alpha = 1
        data = in_data[0]
        labels = in_data[1].asnumpy() 
        wrapSize =  M * D
        batchSize = data.shape[0]
        aux_f = aux[0]
        aux_h = aux[1]
        aux_diff = aux[2]
        auxCentroid = aux[3]
        auxSigma = aux[4]
        wrapSize =  M * D
        batchSize = data.shape[0]
        xpu = data.context
        loss_out = mx.nd.zeros((batchSize,),ctx=xpu)
        for ii in range(batchSize / wrapSize):
            wrap=data[ii*wrapSize:(ii+1)*wrapSize]
            wraplables = labels[ii*wrapSize:(ii+1)*wrapSize]
            #calculate cluster centers
            mu = []
            for j in range(wrapSize/D):
                c = mx.nd.sum(wrap[j*D:(j+1)*D],axis=0) / D
                mu.append(c)
                auxCentroid[ii*M+j][:]=c
                #calculate diff 
            diff= []
            for n in range(wrapSize):
                diff.append([])
                for m in range(M):
                    d = mx.nd.sum(mx.nd.square(wrap[n] - mu[m]))
                    #d = mx.nd.sqrt(d)
                    bug_d = d.asnumpy()
                    if (isnan(bug_d)):
                        import pdb;pdb.set_trace()
                    diff[n].append(d)
                    aux_diff[ii*wrapSize+n][m:m+1]=d
            #calculate sigma
            s=mx.nd.zeros(1,ctx = xpu)
            k=0
            m=0
            for j in range(wrapSize):
               s += diff[j][m]
               k = k+1
               if k>=D : 
                   k=0
                   m+1
            sigma = s.asnumpy()/(wrapSize - 1)
            sigma = float(2*sigma[0])
            auxSigma[ii:ii+1] = s
            
            #calculate loss for wrap
            loss=mx.nd.zeros(1,ctx=xpu)
            frac = mx.nd.zeros(1,ctx=xpu)
            for j in range(wrapSize):
                frac[:] = 0 
                for i in range(M):
                    if wraplables[j] !=wraplables[i*D]:
                        frac += mx.nd.exp(- diff[j][i]/sigma) 
                aux_h[ii*wrapSize+j:ii*wrapSize+j+1] = frac
                f=diff[j][int(j/D)]/sigma+alpha
                loss += f + mx.nd.log(frac)
            loss = loss / wrapSize
            mx.nd.broadcast_to( loss,\
                               out=loss_out[ii*wrapSize:(ii+1)*wrapSize],\
                                            shape = (wrapSize))
        return loss_out
        
def get_mGrad( out_data,in_data,M,D, aux):
    aux_f = aux[0]
    aux_h = aux[1]
    aux_d = aux[2]
    auxCentroid = aux[3]
    auxSigma = aux[4]
    data = in_data[0]
    xpu = data.context
    labels = in_data[1].asnumpy()
    M = int(M)
    D = int (D)
    wrapSize =  M * D
    batchSize = data.shape[0]
    featureSize = data.shape[1]
    grad = mx.nd.zeros((batchSize,featureSize),ctx = xpu)
    Sigma= auxSigma.asnumpy()
    part = mx.nd.zeros((featureSize,),ctx=xpu)
    for ii in range(batchSize / wrapSize):
        wrap=data[ii*wrapSize:(ii+1)*wrapSize]
        wraplables = labels[ii*wrapSize:(ii+1)*wrapSize]
        sigma = Sigma[ii]  
        for j in range(wrapSize):
            part[:] = 0
            cnt = 0 
            for i in range(M):
                if wraplables[j] !=wraplables[i*D]:
                    score = mx.nd.exp(- aux_d[ii*wrapSize+j][i:i+1]/(2*sigma))
                    part += mx.nd.broadcast_mul(auxCentroid[ii*M+i],score)
                    cnt +=1
            gh = mx.nd.broadcast_div(part,aux_h[ii*wrapSize+j:ii*wrapSize+j+1])
            gf = wrap[j]-auxCentroid[ii*M+int(j/D)]
            g  = gf - wrap[j]*cnt +gh
            g  = g/(M * D * sigma)
           #grad[ii*wrapSize+j] = g*y[ii*wrapSize+j]
            grad[ii*wrapSize+j] = g
    return grad
    
def Sbackward(out_grad, out_data,l):
    y = out_data[0]
    z = out_grad[0]
    grad = y*(1 - y)
    grad = grad * 2 *l
    return z*grad
        
        
ctx = mx.gpu()
data=mx.random.normal(0,5,shape=(48,100),ctx=ctx)
ans =mx.random.normal(0,5,shape=(48,100),ctx=ctx)
nplabel = [0,0,0,0,0,0,0,0,1,1,1,1]
label = mx.nd.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0,3,3,\
                     0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,0,3,3],ctx=ctx)
f=mx.nd.zeros((48,),ctx = ctx)
d=mx.nd.zeros((48,12),ctx = ctx)
h=mx.nd.zeros((48,),ctx=ctx)
c=mx.nd.zeros((24,100),ctx=ctx)
s=mx.nd.zeros((2,),ctx=ctx)
aux = [f,h,d,c,s]
loss = mLoss([data,label],12,2,aux)
g    = get_mGrad([loss],[data,label],12,2,aux)
sgrad = Sbackward([data],[data],3)
X = mx.symbol.Variable(name='data')
sigmoid  = mx.symbol.Custom(data=X,  name='sigmoid', op_type='Sigmoid',l=1)
Y = 3*sigmoid
Z = 4*sigmoid
F = Y + 0.5*Z
x = mx.random.normal(0,5,shape=(4,100))
y = mx.random.normal(0,5,shape=(4,100))
z = mx.random.normal(0,5,shape=(4,100))
f = mx.nd.ones((4,100))*2

gx = mx.nd.zeros((4,100))
gy = mx.nd.zeros((4,100))
gf = mx.nd.zeros((4,100))
gz = mx.nd.zeros((4,100))

ex = F.bind(ctx= ctx,args = {'data':f,'sigmoid':y},\
            args_grad = {'data':gx,'sigmoid':gy} )

ex.forward()
out = ex.outputs[0]
ex.backward(out)