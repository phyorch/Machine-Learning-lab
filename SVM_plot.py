import numpy as np
import matplotlib.pyplot as plt
import SVM_algorithm
def init_plot(X, Y):
    pos1 = np.argwhere(Y==1)
    pos2 = np.argwhere(Y==-1)
    data1 = X[pos1[:,0]]
    data2 = X[pos2[:,0]]
    plt.plot(data1[:,0], data1[:,1], 'ro', data2[:, 0], data2[:, 1], 'bs')
    plt.axis()
    plt.show()

def result_plot(w, b, alpha, X, Y, kernel, linesize=0.1):
    pos1 = np.argwhere(Y == 1)
    pos2 = np.argwhere(Y == -1)
    data1 = X[pos1[:, 0]]
    data2 = X[pos2[:, 0]]
    up0 = max(X[:,0]) *1.3
    low0 = min(X[:,0]) *1.3
    up1 = max(X[:, 1]) * 1.3
    low1 = min(X[:, 1]) * 1.3
    if kernel=='linear':
        x = np.arange(low0, up0, linesize)
        y = (-w[0, 0] * x - b) / w[0, 1]
        plt.plot(data1[:, 0], data1[:, 1], 'ro', data2[:, 0], data2[:, 1], 'bs', x, y, 'g--')
        plt.axis()
        plt.show()
    else:
        x1 = np.linspace(low0, up0, 100)
        x2 = np.linspace(low1, up1, 100)
        X_grid, Y_grid = np.meshgrid(x1, x2)
        vals = np.zeros_like(X_grid)
        for i in range(X_grid.shape[1]):
            this_X = np.vstack((X_grid[:, i], Y_grid[:, i]))
            this_X = this_X.T
            vals[:, i] = SVM_algorithm.predict(X, Y, this_X, w, b, alpha, kernel='Gaussian')
        plt.contour(X, Y, vals)
    plt.xticks(())
    plt.yticks(())
    plt.show()