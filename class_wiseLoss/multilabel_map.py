#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 10:37:03 2016

@author: XFZ
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:55:08 2016

@author: XFZ
"""

import mxnet as mx
import numpy as np
from random import shuffle
from sklearn.neighbors import KDTree
from beaconLoss import *
from inception_bn import get_inception_symbol
def get_net(feature_len,weight=100):
    label =  mx.sym.Variable('label')  
    flatten = get_inception_symbol()
    fc = mx.symbol.FullyConnected(data=flatten, \
                                  num_hidden=feature_len, name='fc')
    myloss=mx.symbol.Custom(data=fc,label=label,\
                                    name='myLoss',op_type = 'beaconLoss',\
                                    nNeighbors = 10,alpha = 0,\
                                    nClass = 21)
    myloss = mx.symbol.MakeLoss(data=myloss,name='mloss')
    
    leftRelu = mx.sym.Activation(data = -1.5-fc,act_type='relu',name = 'leftR')
    rightRelu = mx.sym.Activation(data = fc-1.5,act_type='relu',name = 'rightR')
    s = leftRelu+rightRelu
    lc = mx.sym.sum(data =s,axis = 1,keepdims = 1,name = 'lc')
    lcLoss = lc*weight/feature_len
    lcLoss = mx.sym.MakeLoss(data= lc,name='lcLoss')
    loss=mx.sym.Group([myloss,lcLoss] ) 
    return loss
def get_feature(DataIter,featureExector):
    i=0
    print 'extract features'
    labels=[]
    features=[]
    for batch in DataIter:
        f = featureExector.predict(batch.data[0])
        f = np.sign(np.squeeze(f))
        label = batch.label[0].asnumpy()
        labels.extend(label)
        features.extend(f)
        i+=1
        if i%10==0:
          print 'batch :',i
    return features,labels
def get_mlabel_lst(fileName):
    f=open(fileName,'r')
    lines = f.readlines()
    f.close()
    
    mlabels=[]
    for line in lines:
        sp = line.split('\t')
        label = sp[1:-1]
        label = [int(x) for x in label]    
        print label
        mlabels.append(label)
    return mlabels
    
  

feature_size = 32
load_prefix = 'nus_wide/nus_inBn_all_'+str(feature_size)
load_epoch=4
sym,arg_params,aux_params = mx.model.load_checkpoint(load_prefix, load_epoch)
net = get_net(feature_size)
batchSize = 64
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

#loading dataIter

# Train iterator make batch of 128 image, and random crop each image into 3x28x28 from original 3x32x3
test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="/home/XFZ/dataSet/nus_wide/test_500_m.rec",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,224,224),
        batch_size=batchSize,
        round_batch=False,
        preprocess_threads=1)
internals = net.get_internals()
# get feature layer symbol out of internals
#fea_symbol = internals["_minusscalar0_output"]
fea_symbol = internals["fc_output"]
feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, \
                                         numpy_batch_size=batchSize,
                                         arg_params=executor.arg_dict,\
                                         aux_params=executor.aux_dict,
                                       allow_extra_params=True)
train_dataiter = mx.io.ImageRecordIter(
        path_imgrec="/home/XFZ/dataSet/nus_wide/train_500_m.rec",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,224,224),
        batch_size=batchSize,
        preprocess_threads=1)

trainFeature, _ = get_feature(train_dataiter,feature_extractor)
testFeature, _ = get_feature(test_dataiter,feature_extractor)
pref = "/home/XFZ/dataSet/nus_wide/"
trainLabel = get_mlabel_lst(pref+'train_500_m.lst')
testLabel = get_mlabel_lst(pref+'test_500_m.lst')
numTrain = len(trainLabel)
trainFeature=trainFeature[0:numTrain]
numTest = len(testLabel)
testFeature=testFeature[0:numTest]
tree = KDTree(trainFeature)
i = 0
MAP = 0
collectScore =[]
for tF,tL in zip(testFeature,testLabel):
    _,inds = tree.query([tF],k=5000)
    score = 0
    cnt = 0
    for ii,ind in enumerate(inds[0]):
      #print trainLable[ind],tL  
      testSet = set(tL)
      trainSet =set(trainLabel[ind])
      if len(testSet&trainSet):
        cnt += 1 
        ap = float(cnt)/(ii+1)
        score += ap
    if cnt != 0:
        score = float(score) / cnt
    print score
    MAP += score
    i += 1

print MAP/i 
#import cPickle
#
#f=open('trainHash.data','w')
#cPickle.dump(trainFeature,f)
#f.close()
#f=open('trainLabel.data','w')
#cPickle.dump(trainLable,f)
#f.close()
#f=open('testHash.data','w')
#cPickle.dump(testFeature,f)
#f.close()
#f=open('testLabel.data','w')
#cPickle.dump(testLable,f)
#f.close()

