#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:46:03 2016

@author: XFZ
"""

import numpy as np
from math import isnan

x=np.array([1, 2,5,float('nan')])
for y in x:
  if isnan(y):
    import pdb
    pdb.set_trace()
  