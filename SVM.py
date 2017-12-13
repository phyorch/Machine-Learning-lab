import numpy as np
import scipy.io as sio
import matplotlib as plt
import SVM_algorithm
import SVM_plot


def SVM_demo(kernel, data, order):
    X = data['X']
    Y = np.int8(data['y'])
    Y[np.argwhere(Y==0)] = -1
    SVM_plot.init_plot(X, Y)
    if order==1:
        #kernel=='linear'
        C = 1
        K = SVM_algorithm.linearKernel(X)
    elif order==2:
        #kernel == 'Gaussian'
        K = SVM_algorithm.GaussianKernel(X)
        C = 2
    elif order==3:
        #kernel=='Gaussian'
        K = SVM_algorithm.linearKernel(X)
        C = 2
    elif order==4:
        #kernel=='polynomial'
        C = 1
        K = SVM_algorithm.polynomialKernel(X)

    w, b, alpha = SVM_algorithm.SMO(X, Y, K, C)
    SVM_plot.result_plot(w, b, alpha, X, Y, kernel, order, linesize=0.1)


def data4_generate():
    meanlist = [[-1.5, -1.5], [-1.5, 1.5], [1.5, -1.5], [1.5, 1.5]]
    X = np.random.rand(200, 2)
    Y = np.zeros((200, 1))
    for i in range(len(meanlist)):
        X[(i * 50):((i + 1) * 50), 0] += meanlist[i][0]
        X[(i * 50):((i + 1) * 50), 1] += meanlist[i][1]
        if meanlist[i][0] * meanlist[i][1] < 0:
            Y[(i * 50):((i + 1) * 50)] = -1
        else:
            Y[(i * 50):((i + 1) * 50)] = 1
    return [X, Y]
def poly_demo(kernel, data, order):
    X = data[0]
    Y = data[1]
    SVM_plot.init_plot(X, Y)
    C = 1
    K = SVM_algorithm.polynomialKernel(X)
    w, b, alpha = SVM_algorithm.SMO(X, Y, K, C)
    SVM_plot.result_plot(w, b, alpha, X, Y, kernel, order, linesize=0.1)


data1 = sio.loadmat('C:/Users/Phyorch/Desktop/Learning/Python repository/hello-world/data/ex6data1.mat')
data2 = sio.loadmat('C:/Users/Phyorch/Desktop/Learning/Python repository/hello-world/data/ex6data2.mat')
data3 = sio.loadmat('C:/Users/Phyorch/Desktop/Learning/Python repository/hello-world/data/ex6data3.mat')
data4 = data4_generate()

#SVM_demo('linear', data1, 1)
#SVM_demo('Gaussian', data2, 2)
#SVM_demo('Gaussian', data3, 3)
poly_demo('polynomial', data4, 4)



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
