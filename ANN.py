# to construct a aritificial neural network
import csv
import math
import numpy as np
import scipy as sp

#main paramenters
#node_matrix   list[array]  containing every layer node value  first element in is input last is output
#Theta         list[arrat]  containing every layer theta parameters as a 2D array
#dataset_x
#dataset_y     2D array


# Each of the image contains 28*28=784 pixels
# labels is the y value list of each output so it should be turned to a matrix or 2-D array
def data_read(adress):
    data_train = csv.reader(
        open(adress))
    dataset_x = []
    labels = []

    for row in data_train:
        labels.append(row.pop(0))
        dataset_x.append(row)
    del dataset_x[0]
    del labels[0]
    dataset_x = [[int(elem) for elem in row] for row in dataset_x]
    labels = [int(elem) for elem in labels]
    dataset_x = np.array(dataset_x)
    dataset_y = np.zeros((len(dataset_x),10))
    for i in range(len(dataset_y)):
        pos = labels[i]
        dataset_y[i,pos] = 1
    return dataset_x, dataset_y, labels



# contains the unit amount of each hidden layer
# first eloement is input and last is output
# Theta contains the randomly initialized theta matrix from current layer to next
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



def BP_process(Network, y): # output is one of the dataset's y vector
    for l in range(len(Network)-1,-1,-1):
        if l==len(Network)-1:
            n = Network[l].node
            Network[l].error = Network[l].node - y
            q = Network[l].error
        else:
            layer = Network[l]
            post_layer = Network[l+1]
            e = post_layer.error
            d = np.dot(layer.theta.T, post_layer.error.T)
            layer.error = np.dot(layer.theta.T, post_layer.error.T).T * layer.node * (1-layer.node) # back propagation
    return Network


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


def Training(Network, X, Y, learning_rate=0.2):
    M = len(Y)
    for m in range(M):
        x = X[m]
        y = Y[m]
        Network = FP_process(Network, x)
        Network = BP_process(Network, y)
        for l in range(len(Network)-1):
            ll = l + 1
            Network[l].node.shape = (1,Network[l].size)
            Network[l].theta -= learning_rate * np.dot(Network[ll].error.T, Network[l].node)
            Network[l].bias -= learning_rate * Network[ll].error
    return Network

def Test(Network, X, Y):
    values = []
    for i in range(X.shape[0]):
        Network = FP_process(Network, X[i])
        result = Network[-1].node
        values.append(result)
        #for elem in range(len(result)):
        #    if result[elem]==1:
        #        values.append(elem)
        #        break
    return values


data_X, data_Y, a = data_read('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab3/train.csv')
test_X, test_Y, labels = data_read('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab3/test.csv')
X = data_X[:39000,:]
Y = data_Y[:39000,:]
X = X/255
Y = Y/255
sizelist = [784, 30, 10]
Network = Network_inital(X, Y, sizelist)
Network = Training(Network, X, Y)
test_X = data_X[39999:41000,:]
test_Y = data_Y[39999:41000,:]
values = np.array(Test(Network, test_X, test_Y))
answer = labels[39999:41000]
b = 1
