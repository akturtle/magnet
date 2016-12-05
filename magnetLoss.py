# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
""


# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
import mxnet as mx
from math import isnan

class magnetLoss(mx.operator.CustomOp):
    def __init__(self, ctx, shapes, dtypes, M=12, D=4,alpha=1):
       
        
        self.M = M
        self.batch_size = shapes[0][0]
        self.D = D
        self.alpha = alpha

    def forward(self, is_train, req, in_data, out_data,aux):
        M = int(self.M)
        D = int (self.D)
        alpha = float(self.alpha)
        data = in_data[0]
        labels = in_data[1].asnumpy()
        aux_h = aux[0]
        aux_diff = aux[1]
        auxCentroid = aux[2]
        auxSigma = aux[3]
        wrapSize =  M * D
        batchSize = data.shape[0]
        xpu = data.context        
        loss_out = mx.nd.zeros((batchSize,),ctx = xpu)
      
        for ii in range(batchSize / wrapSize):
            wrap=data[ii*wrapSize:(ii+1)*wrapSize]
            wraplables = labels[ii*wrapSize:(ii+1)*wrapSize]
            #calculate cluster centers
            mu = []
            for j in range(wrapSize/D):
                c = mx.nd.sum(wrap[j*D:(j+1)*D],axis=0) / D
                mu.append(c)
                auxCentroid[ii*M+j][:]=c
                #calculate distance to the center  
            diff= []
            for n in range(wrapSize):
                diff.append([])
                for m in range(M):
                    d = mx.nd.sum(mx.nd.square(wrap[n] - mu[m]))
                    #d = mx.nd.sqrt(d)
#                    bug_d = d.asnumpy()
#                    if isnan(bug_d):
#                        import pdb;pdb.set_trace()
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
                   m+=1
            sigma = s.asnumpy()/(wrapSize - 1)
            sigma = float(2*sigma[0])
            auxSigma[ii:ii+1] = s
            
            #calculate loss for wrap
            loss=mx.nd.zeros(1,ctx=xpu)
            frac = mx.nd.zeros(1,ctx=xpu)
            #sum the distance to centers from diferent class
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
        self.assign(out_data[0], req[0], loss_out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        aux_h = aux[0]
        aux_d = aux[1]
        auxCentroid = aux[2]
        auxSigma = aux[3]
        data = in_data[0]
        labels = in_data[1].asnumpy()
        xpu = data.context
        M = int(self.M)
        D = int (self.D)
        wrapSize =  M * D
        batchSize = data.shape[0]
        featureSize = data.shape[1]
        grad = mx.nd.zeros((batchSize,featureSize),ctx = xpu)
        Sigma= auxSigma.asnumpy()
        #y = out_data[0].asnumpy()
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
        self.assign(in_grad[0], req[0], grad)
                    
            
            
            

        



@mx.operator.register("magnetLoss")
class magnetLossProp(mx.operator.CustomOpProp):
    def __init__(self, M=8,D=4, alpha=1,  batchsize=128):
        super(magnetLossProp, self).__init__(need_top_grad=False)

        # convert it to numbers
        self.M = int(M)
        self.D = float(D)
        self.alpha = float(alpha)
        self.batchsize = int(batchsize)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']
    
    def list_auxiliary_states(self):
        # call them 'bias' for zero initialization
        return [ 'h_bias', 'd_bias', 'centroid_bias', 's_bias']

    def infer_shape(self, in_shape):
        wrapNum = int(self.batchsize / (self.M *self.D))
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        d_shape = (self.batchsize,self.M)
        h_shape = (self.batchsize,)
        c_shape = (self.M * wrapNum, in_shape[0][1])
        s_shape = (wrapNum,)

        output_shape = (in_shape[0][0],)
        return [data_shape, label_shape], [output_shape] ,\
                    [ h_shape, d_shape,c_shape,s_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return magnetLoss(ctx, shapes, dtypes, self.M, self.D, self.alpha)