import numpy as np
import matplotlib.pyplot as plt

def plot(w, b, X, Y):
    pos1 = np.argwhere(Y==1)
    pos2 = np.argwhere(Y==-1)
    data1 = X[pos1[:,0]]
    data2 = X[pos2[:,0]]
    x = np.arange(-10, 10, 0.1)
    y = (-w[0,0]*x - b) / w[0,1]
    plt.plot(data1[:,0], data1[:,1], 'ro', data2[:, 0], data2[:, 1], 'bs', x, y, 'g--')
    plt.axis()
    plt.show()