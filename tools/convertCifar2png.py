#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:06:10 2016

@author: XFZ
"""
import cPickle
import numpy as np
import cv2
from random import shuffle

def unpickle(file):
    fo = open(file, 'rb')
    dic = cPickle.load(fo)
    fo.close()
    return dic
def main():
  dataRoot = '../data/cifar10/data_batch_'
  imgSavePath = '/home/XFZ/dataSet/cifar10/'
  imgCnt = 0
  lines=[]
  for i in range(1,6):
    dic = unpickle(dataRoot+str(i))
    datas = dic['data']
    labels =dic['labels']
    print 'process batch: ',i  
    for data,label in zip(datas,labels):
      im = data.reshape(3,32,32).transpose(1,2,0)
      imgName=imgSavePath+str(imgCnt)+'.jpg'
      cv2.imwrite(imgName,im)
      lines.append(str(imgCnt)+"\t"+str(label)+"\t"+imgName+'\n')
      imgCnt += 1
  shuffle(lines)
  f=open("trainList.txt",'w')
  for line in lines:
    f.write(line)
  f.close()
  dic = unpickle('../data/cifar10/test_batch')
  datas = dic['data']
  labels =dic['labels']
  lines=[]  
  print 'process test batch'  
  for data,label in zip(datas,labels):
    im = data.reshape(3,32,32).transpose(1,2,0)
    imgName=imgSavePath+str(imgCnt)+'.jpg'
    cv2.imwrite(imgName,im)
    lines.append(str(imgCnt)+"\t"+str(label)+"\t"+imgName+'\n')
    imgCnt += 1
  shuffle(lines)
  f=open("testList.txt",'w')
  for line in lines:
    f.write(line)
  f.close()
  print 'Done'
if __name__ == "__main__":  
  main()