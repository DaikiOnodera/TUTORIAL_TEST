#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def activate(x):
    return 1./(1. + np.exp(-x))

def calc_val(x, y):
    return activate(0.5*x) - activate(0.3*x+1.2) + activate(0.4*y-0.4) - activate(0.7*y+0.09)

x = np.linspace(-100, 100, 201)
y = np.linspace(-100, 100, 201)
X, Y = np.meshgrid(x, y)
temp = np.c_[np.ravel(X), np.ravel(Y)]
z = calc_val(temp[:,0],temp[:,1]).reshape(-1, 201)

print("X:{}".format(X.shape))
print("Y:{}".format(Y.shape))
print("z:{}".format(z.shape))

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_surface(X, Y, z, linewidth=0)
ax.set_title("Surface Plot")
plt.show()

