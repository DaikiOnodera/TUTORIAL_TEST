#!/usr/bin/env python
# encoding:utf-8

import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = 2
mean = np.zeros(m)
sigma = np.eye(m)

N = 1000
x1 = np.linspace(-5, 5, N)
x2 = np.linspace(-5, 5, N)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]

Y_plot = multivariate_normal.pdf(x=X, mean=mean, cov=sigma)
Y_plot = Y_plot.reshape(X1.shape)
print("x1:{}".format(X1))
print("x2:{}".format(X2))
print("Y:{}".format(Y_plot))
1/0

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X1, X2, Y_plot, linewidth=0)
ax.set_title("Surface Plot")
fig.savefig("temp.png")

