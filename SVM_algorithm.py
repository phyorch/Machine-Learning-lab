import numpy as np
import scipy.spatial.distance as dist

def random_pair(i, m):
    j = i
    while j == i:
        j = np.random.uniform(0, m)
    return j


def alpha_clip(alpha, H, L):
    if alpha >= H:
        alpha = H
    if alpha <= L:
        alpha = L
    return alpha

# consider 2 class at first

def SMO(X, Y, K, C, threshould=0.001, iteration=50):
    m = X.shape[0]
    n = X.shape[1]
    # initialize the parameters
    alpha = np.zeros((m, 1))
    b = 0
    itr = 0
    while itr < iteration:
        changed_num = 0
        for i in range(m):
            k = K[:, i]
            k.shape = (k.shape[0],1)
            predi = np.dot(k.T, Y * alpha)[0] + b
            Ei = predi - Y[i]
            if ((Y[i] * Ei < -threshould) and (alpha[i] < C)) or \
                    ((Y[i] * Ei > threshould) and \
                             (alpha[i] > 0)):
                j = int(random_pair(i, m))
                k = K[:, j]
                k.shape = (k.shape[0], 1)
                predj = np.dot(k.T, Y * alpha)[0] + b
                Ej = predj - Y[j]
                if Y[i] != Y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H:
                    print('L=H')
                    continue

                eta =  2 * K[i,j] - K[i,i] - K[j,j]
                if eta >= 0:
                    print('eta >=0')
                    continue

                # update alpha
                alphaJ = alpha[j].copy()
                alphaI = alpha[i].copy()
                alpha[j] -= Y[j] * (Ei - Ej) / eta
                alpha[j] = alpha_clip(alpha[j], H, L)
                if abs(alpha[j] - alphaJ) < threshould:
                    print('j not moving enough')
                    alpha[j] = alphaJ
                    continue
                alpha[i] += Y[i] * Y[j] * (alphaJ - alpha[j])
                # update b
                b1 = b - Ei - Y[i] * (alpha[i] - alphaI) * K[i,j] \
                     - Y[j] * (alpha[j] - alphaJ) * K[i,j]
                b2 = b - Ej - Y[i] * (alpha[i] - alphaI) * K[i,j] \
                     - Y[j] * (alpha[j] - alphaJ) * K[j,j]

                if 0 < alpha[i] and alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] and alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                changed_num += 1
                print ('iter:  %d  i:  %d, pairs changed:  %d', itr, i, changed_num)
        if changed_num == 0:
            itr += 1
        else:
            itr = 0
        print('update 1')
    w = np.dot(Y.T * alpha.T, X)
    return w, b, alpha

def linearKernel(X):
    size = X.shape[0]
    K = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            K[row, col] = np.dot(X[row,:], X[col,:])
    return K

def polynomialKernel(X):
    d = 2
    K = (np.dot(X, X.T))**d
    return K

def GaussianKernel(X, sigma=2):
    '''size = X.shape[0]
    K = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            K[row, col] = -0.5 * (np.linalg.norm(X[row,:]-X[col,:]))**2 / (sigma**2)'''
    K = dist.cdist(X, X)
    K = np.exp(-K ** 2  / (2 * sigma ** 2))
    return K

def predict(X, Y, x_pred, w, b, alpha, kernel):
    if kernel=='Gaussian':
        sigma = 0.1
        K = dist.cdist(x_pred, X)
        K = np.exp(-K**2 / (2*sigma**2))
    if kernel=='linear':
        K = np.dot(x_pred, X.T)
    if kernel=='polynomial':
        d = 2
        K = (np.dot(x_pred, X.T))**d
    K = Y.T * K
    K = alpha.T * K
    pred = np.sum(K, axis=1) + b
    pos = np.argwhere(pred >= 0)[:, 0]
    pred[pos] = 1
    np.argwhere(pred < 0)[:, 0]
    pred[pos] = -1
    return pred


