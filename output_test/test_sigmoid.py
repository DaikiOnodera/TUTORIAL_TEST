#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import renom as rm
from renom.optimizer import Sgd, Adam
import matplotlib.pyplot as plt

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
        self.layer1 = rm.Dense(output_size=50)
        self.layer2 = rm.Dense(1)

    def forward(self, x):
        print("")
        print("input:\n{}".format(x))
        print("input shape:{}".format(x.shape))
        print("")
        t1 = self.layer1(x)
        print("input x hidden weight:\n{}".format(self.layer1.params.w))
        print("input x hidden bias:\n{}".format(self.layer1.params.b))
        print("")
        print("hidden:\n{}".format(t1))
        print("hidden shape:{}".format(t1.shape))
        t2 = rm.sigmoid(t1)
        print("")
        print("relu:\n{}".format(t2))
        print("relu shape:{}".format(t2.shape))
        print("")
        t3 = self.layer2(t2)
        print("hidden x output weight:\n{}".format(self.layer2.params.w))
        print("hidden x output bias:\n{}".format(self.layer2.params.b))
        print("")
        print("output:\n{}".format(t3))
        print("output shape:{}".format(t3.shape))
        print("")
        return t3

epoch = 200
batch = 1
N = len(X)

optimizer = Sgd(lr=0.005)

network = Mnist()

learning_curve = []
for i in range(epoch):
    perm = np.random.permutation(N)
    loss = 0
    for j in range(0, N // batch):
        train_batch = X[perm[j*batch : (j+1)*batch]]
        response_batch = y[perm[j*batch : (j+1)*batch]]
        with network.train():
            result = network(train_batch)
            #l = rm.sigmoid_cross_entropy(result, response_batch)
            l = rm.mse(result, response_batch)
        grad = l.grad()
        grad.update(optimizer)
        loss += l
    train_loss = loss / ( N // batch)
    print("train_loss:{}".format(train_loss))
    learning_curve.append(train_loss)
print(network(X))
plt.plot(learning_curve, linewidth=3, label="train")
plt.ylabel("error")
plt.xlabel("epoch")
plt.legend()
plt.show()
