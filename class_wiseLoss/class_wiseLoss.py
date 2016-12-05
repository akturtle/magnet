# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
import mxnet as mx
from math import isnan

class class_wiseLoss(mx.operator.CustomOp):
    def __init__(self, ctx, shapes, dtypes, nNeighbors=5,alpha=1,nClass):
       
        
        self.nNeighbors = nNeighbors
        self.batch_size = shapes[0][0]
        self.alpha = alpha

    def forward(self, is_train, req, in_data, out_data,aux):
        nNeighbors = int(self.nNeighbors)
        alpha = float(self.alpha)
        data = in_data[0]
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
        for ii in range(batchsize):
            #calculate distance to the center  
            diff.append([])
            d = mx.nd.sum(mx.nd.square(data[ii] - auxCentroid[int(labels[ii])]))
            diff[i].append(d)    
            for n in range(nNeighbors):  
                neighbors =   auxNeighbors[int(labels[ii])].asnumpy()
                for m in range(nNeighbors):
                    d = mx.nd.sum(mx.nd.square(data[n] - \
                        auxCentroid[int(neighbors[m])]))
                    diff[i].append(d)
                    aux_diff[ii][m+1:m+2]=d
        #calculate sigma
        s=mx.nd.zeros(1,ctx = xpu)
        for j in range(batchSize):
           s += diff[j][0]
        sigma = s.asnumpy()/(batchSize - 1)
        sigma = float(2*sigma[0])
        auxSigma = s
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
                                           
        self.assign(out_data[0], req[0], loss_out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        nNeighbors = int(self.nNeighbors)
        aux_h = aux[0]
        aux_d = aux[1]
        auxCentroid = aux[2]
        auxSigma = aux[3]
        neighbors = aux[4]
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
            neighbors = auxNeighbors[int(labels[1])].asnumpy()
            for j in range(nNeighbors):
                score = mx.nd.exp(- aux_d[i][j:j+1]/(2*sigma))
                part += mx.nd.broadcast_mul(auxCentroid[int(neighbors[j])],score)
            gh = mx.nd.broadcast_div(part,aux_h[i:i+1])
            gf = data[j]-auxCentroid[int(labels[i])]
            g  = gf - data[i]*nNeighbors +gh
            g  = g/(sigma)
           #grad[ii*wrapSize+j] = g*y[ii*wrapSize+j]
            grad[i] = g
        self.assign(in_grad[0], req[0], grad)
                    
            
@mx.operator.register("class_wiseLoss")
class class_wiseProp(mx.operator.CustomOpProp):
    def __init__(self, nNeighbors, alpha,nClass):
        super(magnetLossProp, self).__init__(need_top_grad=False)

        # convert it to numbers
        self.nNeighbors = int(nNeighbors)
        self.alpha = float(alpha)
    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']
    
    def list_auxiliary_states(self):
        # call them 'bias' for zero initialization
        return [ 'h_bias', 'd_bias', 'centroid_bias', 's_bias','neighbors_bias']

    def infer_shape(self, in_shape):
        batchSize = in_shape[0][0]
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        d_shape = (batchSize,self.nNeighbors+1)
        h_shape = (batchSize,)
        c_shape = (self.nClass,)
        s_shape = (1,)
        neighbors_shape = (self.nClass,self.nNeighbors)
        output_shape = (in_shape[0][0],)
        return [data_shape, label_shape], [output_shape] ,\
                    [ h_shape, d_shape,c_shape,s_shape,neighbors_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return magnetLoss(ctx, shapes, dtypes, self.neighbors,\
                             self.alpha,self.nClass)

    