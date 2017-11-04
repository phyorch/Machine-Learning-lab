# to construct a aritificial neural network
import csv
import math
import numpy as np
import scipy as sp
import pandas as pd


class Layer(object):
    def __init__(self, layer_order, layer_size):
        self.size = layer_size
        self.order = layer_order
    def Theta_init(self, next_layer_size):
        epsilon = 0.5
        self.theta = np.random.rand(next_layer_size, self.size)
        self.theta = self.theta * 2 * epsilon - epsilon
        self.bias = np.random.rand(1, next_layer_size)
    def Node_init(self):
        self.node = np.random.rand(1, self.size)
    def Error_init(self):
        self.error = np.random.rand(1, self.size)


def activation(x):
    x = 1/(1 + np.exp(-x))
    return x

def Network_inital(X, Y, size, layer_num=3):
    M = len(Y)
    Network = []
    Error_total = []
    for i in range(layer_num):
        layer = Layer(i, size[i])
        layer.Node_init()
        layer.Error_init()
        if i<layer_num-1:
            layer.Theta_init(size[i+1])
        Network.append(layer)
    return Network

def FP_process(Network, input):  # input is array
    for l in range(len(Network)):
        layer = Network[l]
        if layer.order==0: # if we are calculating the input layer to seconde layer
            #node_matrix.append(input)
            layer.node = input
        else: #  we use previous node array to calculate the new one
            prior = Network[l-1]
            prior_node = prior.node
            prior_theta = prior.theta
            b = np.dot(prior.theta, prior_node.T)
            c = prior.bias
            layer.node = activation(b.T + c)
    return Network

#data_X, data_Y, a = data_read('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab3/train.csv')
#test_X, test_Y, labels = data_read('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab3/test.csv')
data = pd.read_csv('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab3/train.csv')
def dataread(data):
    for i in range(data.shape[0]):
        a = data[i]
    return 0
'''X = data_X[:39000,:]
Y = data_Y[:39000,:]
X = X/255
Y = Y/255
sizelist = [784, 30, 10]

Network = Network_inital(X, Y, sizelist)
x = X[0]
FP_process(Network, x)'''
a = dataread(data)