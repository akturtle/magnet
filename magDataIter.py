#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:29:26 2016

@author: XFZ
"""
import  random
import mxnet as mx
import numpy as np
from operator import itemgetter
import cPickle
from sklearn.neighbors import KDTree
import cv2
from featureExtract import clusterLabel
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DataIter(mx.io.DataIter):
    def __init__(self ,dataByClass,\
                 feature_extractor,K=5,\
                 batch_size=128,\
                 mCluster=8,nSample=4,\
                 featureSize = 16):
        super(DataIter, self).__init__()
        self.data = dataByClass
        self.mCluster = mCluster
        self.nSample = nSample
        self.batch_size = batch_size
        self.provide_data = [('data', (batch_size, 3, 224, 224))]
        self.provide_label = [('label', (batch_size, ))]
        self.clusterLabel = []
        self.clusterCount = []
        self.K=K
        self.clusterCentroid = []
        print 'extract feature'        
        for i,classData in enumerate(self.data):
            labels,centers,count = clusterLabel(classData,\
                                                feature_extractor,\
                                                batch_size,K,featureSize)
            print 'class'+str(i)
            self.clusterLabel.append(labels)
            self.clusterCount.append(count)
            self.clusterCentroid.extend(centers)
        self.tree = KDTree(self.clusterCentroid)
    def generate_minbatch(self):
        n = random.sample(range(len(self.data)*self.K),1)
        n = n[0]
        dist,ind = self.tree.query([self.clusterCentroid[n]],self.mCluster)
        minBatch = []
        label = []
        for clusterInd in ind[0]:
            l1 = int(clusterInd/self.K) #class label
            l2 = clusterInd%self.K #cluster label
            
            sampleInd = random.sample(range(0,\
                                            int(self.clusterCount[l1][l2])),self.nSample)
            sampleInd.sort()
            count = 0
            i = 0
            for k,c in enumerate(self.clusterLabel[l1]):
                if c == l2:
                    if count == sampleInd[i]:
                        i=i+1
                        img= self.data[l1][k]
                        img = img.reshape(32,32,3,order='F')
                        img = cv2.resize(img,(224,224))
                        img=img.swapaxes(2,0)
                        minBatch.append(img)
                        label.append(l1)
                        if i >=self.nSample:
                            break
                    count +=1
        return minBatch,label

    def __iter__(self):
        print 'begin'
        count =500
        minBatchSize=self.mCluster*self.nSample
        nRound = int(self.batch_size/minBatchSize)
        for i in range(count):
            batch = []
            label = []
            for i in range(nRound):
                minbatch,minlabel = self.generate_minbatch()
                batch.extend(minbatch)
                label.extend(minlabel)                                   
            data_all = [mx.nd.array(batch)]
            label_all = [mx.nd.array(label)]
            data_names =['data']
            label_names = ['label']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass
