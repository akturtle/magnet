#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 10:17:45 2016

@author: XFZ
"""

import cPickle 
import matplotlib.pyplot as plt
f =open('IB_new_nusW32_train_loss.data','r')
loss = cPickle.load(f)
f.close()
trainX = [x for (x,y) in loss] 
trainY = [y for (x,y) in loss] 
plt.plot(trainX,trainY)
          
