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
def data_read():
    data_train = csv.reader(
        open('C:/Users/Phyorch/Desktop/Learning/Mchine learning/project and homework/lab3/train.csv'))
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
    return dataset_x, dataset_y


def activation(x):
    for elem in x:
        elem = 1/(1 + math.exp(-elem))
    return x

class Layer(object):
    def __init__(self, layer_order, layer_size):
        self.size = layer_size
        self.order = layer_order
    def Theta_init(self, next_layer_size):
        epsilon = 0.5
        self.theta = np.random.rand(next_layer_size, self.size)
        self.theta = self.theta * 2 * epsilon - epsilon
        self.bias = np.random.rand(next_layer_size, 1)
    def Node_init(self):
        self.node = np.random.rand(self.size, 1)
    def Error_init(self):
        self.error = np.random.rand(self.size, 1)


class Error(object):
    def __init__(self, layer_order, layer_size):
        self.size = layer_size
        self.order = layer_order
    def Error_init(self):
        self.error = np.zeros((self.size, 1))

# contains the unit amount of each hidden layer
# first eloement is input and last is output
# Theta contains the randomly initialized theta matrix from current layer to next
def Theta_init(layer_size):
    epsilon = 0.04
    Theta = []
    for i in range(len(layer_size)-1):
        theta = np.random.rand(layer_size[i+1], layer_size[i]+1)
        theta = theta * 2 * epsilon - epsilon
        Theta.append(theta)
    return Theta

def deritive_zero(Theta):
    deritive = []
    for elem in Theta:
        der = np.zeros(elem.shape)
        deritive.append(der)
    return deritive

def FP_process(Network, input):  # input is array
    for l in range(len(Network)):
        layer = Network[l]
        if layer.order==0: # if we are calculating the input layer to seconde layer
            #node_matrix.append(input)
            layer.node = input
        else: #  we use previous node array to calculate the new one
            prior = Network[l-1]
            prior_node = prior.node
            a = prior.theta.shape
            a1 = prior_node.shape
            b = np.dot(prior.theta, prior_node)
            b1 = b.shape
            c = prior.bias
            d = prior.bias.shape
            e = (b+c).shape
            layer.node = activation(b + c)
    return Network


def BP_process(Network, y): # output is one of the dataset's y vector
    for l in range(len(Network)):
        ll = len(Network) - l
        if ll==len(Network)-1:
            Network[ll].error = Network[ll].node - y
        else:
            layer = Network[ll]
            post_layer = Network[ll+1]
            layer.error = np.multiply(np.multiply(np.dot(layer.theta.T, post_layer.error), layer.node),
                                      np.ones(layer.node.shape)-layer.node) # back propagation
    return Network


def cost_cal(y, Network):

    cost = y*np.log(Network[-1].node) + (1-y)*np.log(1-np.log(Network[-1].node))

    return cost



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

        error = Error(i, size[i])
        error.Error_init()
        Error_total.append(error)
    return Network, Error_total

def Training(Network,Error_total, X, Y, iteration=20, learning_rate=0.2):
    M = len(Y)
    J = 0
    for m in range(M):
        x = X[m]
        y = Y[m]
        Network = FP_process(Network, x)
        Network = BP_process(Network, y)
        Error_total.error += 1/m * Network.error
        #J += -1/m * cost_cal(y, Network)
    for i in range(iteration):
        for l in Network:
            ll = l + 1
            Network[l].theta -= learning_rate * np.dot(Network[ll].error, Network[l].node.T)
            Network[l].bias -= learning_rate * Network[ll].error
    return Network


X, Y = data_read()
X = X[:100,:]
Y = Y[:100,:]
sizelist = [784, 20, 10]
Network, Error_total = Network_inital(X, Y, sizelist)
q = Network[0].bias
w = Network[0].bias.shape
e = Network[0].node
r = Network[0].node.shape
t = Network[0].order
y = Network[1].node
u = Network[1].node.shape
Network = Training(Network, Error_total, X, Y)













