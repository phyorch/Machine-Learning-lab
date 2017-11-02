# to construct a aritificial neural network
import csv
import math
import numpy as np
import scipy as sp

#main paramenters
#node_matrix   list[array]  containing every layer node value  first element in is input last is output
#Theta         list[arrat]  containing every layer theta parameters as a 2D array
#dataset_x
#dataset_y  2D array




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
    sigmoid = 1/(1 + math.exp(-x))
    return sigmoid

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


def FP_process(Theta, input):  # input is array
    node_matrix = []
    for layer in Theta:
        if len(node_matrix)==0: # if we are calculating the input layer to seconde layer
            #node_matrix.append(input)
            input = input.append(1)
            node_layer = [activation(np.dot(row, input)) for row in layer]
            node_matrix.append(node_layer)
        else: #  we use previous node list to calculate the new one
            prior = node_matrix[-1]
            prior.append(1)
            node_layer = [activation(np.dot(row, prior)) for row in layer]
            node_matrix.append(node_layer)
    node_matrix2 = node_matrix[:]  # node_matrix2 is used for gradient claculating
    node_matrix2.insert(0, input)
    del node_matrix2[-1]
    return node_matrix, node_matrix2


def BP_process(node_matrix, y, Theta): # output is one of the dataset's y vector
    err_matrix = []
    node_matrix = node_matrix[::-1] # first element means output   forward recursion
    node_matrix = node_matrix[::-1]
    for layer in range(len(Theta)):
        if len(err_matrix)==0:
            error = node_matrix[0] - y
            err_matrix.append(error)
        else:
            post = err_matrix[0]  # error of post layer
            theta = Theta[layer]
            theta = theta[:len(theta)-1] # the last row is bias so we delete it
            node = node_matrix[layer]
            error = np.multiply(np.multiply(theta.T * post, node), np.ones(node.shape)-node) # back propagation
            err_matrix.insert(0,error)
    return err_matrix


def cost_cal(y, pridiction):
    cost = 0
    for k in range(len(y)):
        cost += y[k]*np.log(pridiction[k]) + (1-y[k])*np.log(1-np.log(pridiction[k]))
    return cost

def J_cal(X, Y, Theta):


def training(X, Y, deritive, ,lamida, layer_size=3):
    M = len(Y)
    Theta = Theta_init(layer_size)
    deritive = deritive_zero(Theta)
    for m in range(M):
        x = X(m)
        y = Y(m)
        node_matrix, node_matrix2 = FP_process(Theta, x)
        err_matrix = BP_process(node_matrix, y, Theta)
        for layer in range(len(deritive)):
            for j in range(len(node_matrix2[layer])):
                for i in range(len(err_matrix[layer])):
                    deritive[layer][i][j] += 1/m * (node_matrix2[layer][j] +err_matrix[layer][i]) + lamida/m * Theta[layer][i][j]
                    # this is the partial deritives of J











