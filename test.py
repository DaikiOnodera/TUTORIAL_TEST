#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import renom as rm
from renom.optimizer import Sgd

X = np.array([[1,1],
              [1,0],
              [0,1],
              [0,0]])

y = np.array([[1],
              [0],
              [0],
              [1]])

print("X:\n{}".format(X.shape))
print("y:\n{}".format(y.shape))

class Mnist(rm.Model):
    def __init__(self):
        self.layer1 = rm.Dense(output_size=100)
        self.layer2 = rm.Dense(1)

    def forward(self, x):
        t1 = self.layer1(x)
        t2 = rm.relu(t1)
        t3 = self.layer2(t2)
        return t3

epoch = 50
batch = 1
N = len(X)

optimizer = Sgd()

for i in range(epoch):
    perm = np.random.permutation(N)
    loss = 0
    for j in range(0, N // batch):
        train_batcn = X[perm[j * batch : (j+1) * batch]]
        res

