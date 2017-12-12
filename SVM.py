import numpy as np
import scipy.io as sio
import matplotlib as plt
import SMO_algorithm
import SVM_plot


def SVM_demo(kernel):
    data = sio.loadmat('C:/Users/Phyorch/Desktop/Learning/Python repository/hello-world/data/ex6data2.mat')
    X = data['X']
    Y = data['y']
    Y[np.argwhere(Y==0)] = -257
    SVM_plot.init_plot(X, Y)

    if kernel=='linear':
        C = 1
        K = SMO_algorithm.linearKernel(X)
    elif kernel=='polynomial':
        C = 1
        K = SMO_algorithm.polynomialKernel(X)
    elif kernel=='Gaussian':
        K = SMO_algorithm.GaussianKernel(X)
        C = 1
    w, b, alpha = SMO_algorithm.SMO(X, Y, K, C)
    SVM_plot.result_plot(w, b, alpha, X, Y, kernel, linesize=0.1)

SVM_demo('Gaussian')
a = 1





'''meanlist = [[-3,-5],[3,3]]  #,[5,-5],[8,-2]
covlist = [ [[5,1],[1,5]], [[5,3],[3,5]]]  #, [[4,-2],[-2,4]], [[2,0],[0,1]]
sizelist = [40,40]  #,500,650

def datagenerate_linear(meanlist, covlist, sizelist):
    #xlist = []
    #ylist = []
    labels1 = -np.ones((sizelist[0], 1))
    labels2 = np.ones((sizelist[1], 1))
    data1 = np.random.multivariate_normal(meanlist[0],covlist[0],sizelist[0])
    data2 = np.random.multivariate_normal(meanlist[1],covlist[1],sizelist[1])
    data1 = np.column_stack((data1, labels1))
    data2 = np.column_stack((data2, labels2))
    data = np.row_stack((data1, data2))
    return data

    data = datagenerate_linear(meanlist, covlist, sizelist)
        np.random.shuffle(data)
        X = data[:, 0:2]
        Y = data[:, -1]
        Y.shape = (Y.shape[0], 1)
    '''
