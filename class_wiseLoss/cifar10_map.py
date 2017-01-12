#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:32:23 2017

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
from sklearn.neighbors import KNeighborsClassifier
from beaconLoss import *
from inception_bn import get_inception_symbol
def centroidScore(DataIter,featureExector,hash_len,centroids):
    numClass = len(centroids)
    CKN= KNeighborsClassifier(metric='euclidean',\
    		n_neighbors=1)
    centroid_label = [x for x in range(numClass)] 
    CKN.fit(np.sign(centroids),centroid_label)                  
    batchSize = DataIter.provide_data[0][1][0]
    totalScore = 0
    cnt = 0
    print 'extract features'
    for batch in DataIter:
        f = featureExector.predict(batch.data[0])
        f = np.sign(np.squeeze(f))
        #f = np.squeeze(f)
        label = batch.label[0].asnumpy()
        score = CKN.score(f,label)
        totalScore += batchSize *score
        cnt += batchSize
    return totalScore/cnt
def get_net(feature_len,weight=100):
    label =  mx.sym.Variable('label')  
    flatten = get_inception_symbol()
    fc = mx.symbol.FullyConnected(data=flatten, \
                                  num_hidden=feature_len, name='fc')
    myloss=mx.symbol.Custom(data=fc,label=label,\
                                    name='myLoss',op_type = 'beaconLoss',\
                                    nNeighbors = 5,alpha = 0,\
                                    nClass = 10)
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
        #f = np.squeeze(f)
        label = batch.label[0].asnumpy()
        labels.extend(label)
        features.extend(f)
        i+=1
        if i%20 ==0:
          print 'batch :',i
    return features,labels

feature_size = 24
#load_prefix = './cifar_inBn_'+str(feature_size)
load_prefix = './nus_wide/nus_inBn_i_32'
load_prefix = './cifar/cifar_new_'+str(feature_size)
load_epoch=80
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
data_prefix = '/home/XFZ/dataSet/voc2012/'

total_batch = 50000 / 128 + 1
# Train iterator make batch of 128 image, and random crop each image into 3x28x28 from original 3x32x3
test_dataiter = mx.io.ImageRecordIter(
        #path_imgrec="/home/XFZ/dataSet/cifar10/cifar_224/cifarTest_224.bin",
        path_imgrec = "/home/XFZ/dataSet/cifar10/cifar_224/cifarTest_224.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,224,224),
        batch_size=batchSize,
        round_batch=False,
        preprocess_threads=4)
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
        shuffle=True,
        #path_imgrec="/home/XFZ/dataSet/cifar10/cifar_224/cifarTrain_224.bin",
        path_imgrec="/home/XFZ/dataSet/cifar10/cifar_224/cifarTrain_224.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,224,224),
        batch_size=batchSize,
        preprocess_threads=4)

trainFeature, trainLable = get_feature(train_dataiter,feature_extractor)
testFeature, testLable = get_feature(test_dataiter,feature_extractor)
tree = KDTree(trainFeature)
i = 0
MAP = 0
collectScore =[]
total = len(trainFeature)
for tF,tL in zip(testFeature,testLable):
    _,inds = tree.query([tF],k=total)
    score = 0
    cnt = 0
    for ii,ind in enumerate(inds[0]):
      #print trainLable[ind],tL  
      if trainLable[ind] == tL:
        cnt += 1 
        ap = float(cnt)/(ii+1)
        score += ap
    if cnt != 0:
        score = float(score) / cnt
    MAP += score
    if i %20 ==0 :
      print i
    i += 1

print MAP/i 
centroids = aux_params['myLoss_centroid_bias'].asnumpy()
test_dataiter.reset()
train_dataiter.reset()
cScore = centroidScore(DataIter=test_dataiter,\
                       featureExector=feature_extractor,\
                       hash_len=32,
                       centroids = centroids)
print cScore
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

