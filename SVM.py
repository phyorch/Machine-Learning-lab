import numpy as np
import SMO_algorithm
import matplotlib as plt
import SVM_plot

meanlist = [[-3,-5],[3,3]]  #,[5,-5],[8,-2]
covlist = [ [[5,1],[1,5]], [[5,3],[3,5]]]  #, [[4,-2],[-2,4]], [[2,0],[0,1]]
sizelist = [40,40]  #,500,650

def datagenerate(meanlist, covlist, sizelist):
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


data = datagenerate(meanlist, covlist, sizelist)
np.random.shuffle(data)
X = data[:,0:2]
Y = data[:,-1]
Y.shape = (Y.shape[0],1)
K = SMO_algorithm.GaussianKernel(X)
C = 1
alpha, b = SMO_algorithm.SMO(X, Y, K, C)
w = np.dot(Y.T * alpha.T, X)
SVM_plot.plot(w, b, X, Y)
#def classfiy(dataset):
