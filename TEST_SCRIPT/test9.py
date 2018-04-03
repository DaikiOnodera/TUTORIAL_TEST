#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

def activate(x):
    return 1./(1. + np.exp(-x))

x1 = np.linspace(-1000, 1000, 2001)
y1 = activate(x1)
if any(np.isnan(y1)):
    x1 = x1[~np.where(np.isnan(y1))[0]]
    y1 = y1[~np.where(np.isnan(y1))[0]]

x2 = np.linspace(-1000, 1000, 2001)
y2 = activate(-x2)
if any(np.isnan(y2)):
    x2 = x2[~np.where(np.isnan(y2))[0]]
    y2 = y2[~np.where(np.isnan(y2))[0]]

x3 = np.linspace(-1000, 1000, 2001)
y3 = activate(x3+50)
if any(np.isnan(y3)):
    x3 = x3[~np.where(np.isnan(y3))[0]]
    y3 = y3[~np.where(np.isnan(y3))[0]]

dic1 = dict(zip(x1, y1))
dic2 = dict(zip(x2, y2))
dic3 = dict(zip(x3, y3))

func = lambda x,y:x+y

for k,v in dic1.items():
    dic2[k] = func(dic1[k], dic2[k]) if k in dic2 else v
for k,v in dic2.items():
    dic3[k] = func(dic2[k], dic3[k]) if k in dic3 else v

x = list(dic3.keys())
y = list(dic3.values())

plt.scatter(x, y, s=0.1)
plt.show()

