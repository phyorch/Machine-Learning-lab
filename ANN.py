# to construct a aritificial neural network
import numpy as np
import pandas as pd

#main paramenters
#node_matrix   list[array]  containing every layer node value  first element in is input last is output
#Theta         list[arrat]  containing every layer theta parameters as a 2D array
#dataset_x
#dataset_y     2D array

# Each of the image contains 28*28=784 pixels
# labels is the y value list of each output so it should be turned to a matrix or 2-D array
# to construct a aritificial neural network

# main paramenters
# node_matrix list[array]containing every layer node valuefirst element in is input last is output
# Theta list[arrat]containing every layer theta parameters as a 2D array
# dataset_x
# dataset_y 2D array

# Each of the image contains 28*28=784 pixels
# labels is the y value list of each output so it should be turned to a matrix or 2-D array

# contains the unit amount of each hidden layer
# first element is input and last is output
# Theta contains the randomly initialized theta matrix from current layer to next
class Layer(object):
    def __init__(self, layer_order, layer_size):
        self.size = layer_size
        self.order = layer_order
        self.node = np.random.randn(self.size, 1)
        self.error = np.random.randn(self.size, 1)
        self.bias = np.random.randn(self.size, 1)
    def Theta_init(self, next_layer_size):
        self.theta = np.random.randn(next_layer_size, self.size)

def activation(x):
    x = 1 / (1 + np.exp(-x))
    return x

def FP_process(Network, input):# input is array
    for l in range(len(Network)):
        layer = Network[l]
        if layer.order == 0:  # if we are calculating the input layer to seconde layer
            # node_matrix.append(input)
            layer.node = input
        else:  # we use previous node array to calculate the new one
            layer = Network[l]
            prior = Network[l - 1]
            layer.node = activation(np.dot(prior.theta, prior.node) + layer.bias)
    return Network


def BP_process(Network, y):# output is one of the dataset's y vector
    for l in range(len(Network)-1, -1, -1):
        if l == len(Network) - 1:
            layer = Network[l]
            layer.error = (layer.node - y) * layer.node * (1 - layer.node)
        else:
            layer = Network[l]
            post_layer = Network[l + 1]
            layer.error = np.dot(layer.theta.T, post_layer.error) * layer.node * (1 - layer.node)  # back propagation
    return Network

def Network_initail(size, layer_num=3):
    Network = []
    for i in range(layer_num):
        layer = Layer(i, size[i])
        if i < layer_num - 1:
            layer.Theta_init(size[i + 1])
        Network.append(layer)
    return Network


def Training(Network, X, Y, learning_rate=0.2):
    M = len(Y)
    for m in range(M):
        x = X[m]
        y = Y[m]
        x.shape = (x.shape[0],1)
        y.shape = (y.shape[0],1)
        Network = FP_process(Network, x)
        Network = BP_process(Network, y)
        for l in range(len(Network) - 1):
            ll = l + 1
            layer = Network[l]
            post_layer = Network[ll]
            # Network[l].node.shape = (1,Network[l].size)
            layer.theta -= learning_rate * np.dot(post_layer.error, layer.node.T)
            post_layer.bias -= learning_rate * post_layer.error
    return Network

def Test(Network, X, Y):
    predict = []
    right = 0
    for i in range(X.shape[0]):
        x = X[i]
        x.shape = (x.shape[0], 1)
        Network = FP_process(Network, x)
        evaluate = Network[-1].node
        evaluate.shape = (1, evaluate.shape[0])
        evaluate = evaluate[0]
        evaluate = evaluate.tolist()
        values = evaluate.index(max(evaluate))
        predict.append(values)
        if values==Y[i]:
            right += 1
    accuracy = right/len(Y)
    return right, accuracy, predict


def code(Y):
    y_network = np.zeros((len(Y), 10))
    for i in range(len(Y)):
        y_network[i, Y[i]] = 1
    return y_network

data = pd.read_csv('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab3/train.csv')
data.as_matrix
data = np.array(data)
labels = data[:, 0]
labels_network = code(labels)
data = data[:, 1:]
X = data[:39000, :]/255
Y = labels_network[:39000, :]
sizelist = [784, 30, 10]
test_X = data[39000:42000, :]/255
test_Y = labels[39000:42000]

Network = Network_initail(sizelist)
Network = Training(Network, X, Y)
right, accuracy, predict = Test(Network, test_X, test_Y)
