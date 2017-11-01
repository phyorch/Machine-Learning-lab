# to construct a aritificial neural network
import csv
import numpy as np


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

x, y = data_read()
a = 1
