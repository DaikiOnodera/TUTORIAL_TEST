#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

def activate(x):
    return 1./(1. + np.exp(-x))

def calc_val(x):
    return activate(x) - activate(0.5*x) + activate(0.3*x - 30) - activate(0.7*x + 50)

x = np.linspace(-100, 100, 201)
y = np.linspace(-100, 100, 201)
z = calc_val(x)

#plt.scatter(x, y, s=0.1)
plt.plot(x, y, z)
plt.show()

