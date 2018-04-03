#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def activate(x):
    return np.maximum(x, 0)

def calc_val(x, y):
    return -2.2*(activate((-1.654*x-0.025) + (1.710*y-0.025))) + \
           1.1*(activate((-0.436*x+0.5894) + (1.190*y+0.5894))) + \
           (-1.2*(activate((-0.9183*x-0.0111) + (0.9194*y-0.0111)))) + \
           (-0.2*(activate((-0.5190*x+0.0) + (-0.4397*y+0.0)))) + \
           (-1.8*(activate((1.4265*x-0.004) + (-1.359*y-0.004))))

x = np.linspace(-3, 3, 7)
y = np.linspace(-3, 3, 7)
X, Y = np.meshgrid(x, y)
temp = np.c_[np.ravel(X), np.ravel(Y)]
z = calc_val(temp[:,0],temp[:,1]).reshape(-1, 7)

print("0 0:{}".format(calc_val(0, 0)))
print("1 0:{}".format(calc_val(1, 0)))
print("0 1:{}".format(calc_val(0, 1)))
print("1 1::{}".format(calc_val(1, 1)))

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot_surface(X, Y, z, linewidth=0)
ax.set_title("Surface Plot")
plt.show()
