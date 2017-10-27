import numpy as np
import scipy as sp
import math

def GMM_Process(Data, Pie, Miu, Sigma, iteration):
    K = len(Pie)
    pi = math.pi
    D = len(Data[0])
    Data = np.array(Data)
    for i in range(iteration):
        Gaussian = Gauss_cal(Data, Miu, Sigma, D, K, pi)
        Gama = Gama_cal(Pie, Sigma)
        Sigma = Sigma_update(Gama, Data, Miu)
        Miu = Miu_update(Gama, Data)





def Gauss_cal(Data, Miu, Sigma, D, K, pi):
    Miu = np.mat(Miu)
    Sigma = np.mat(Sigma)
    Gaussian = np.zeros((K,D,D))
    C = 1/((2*pi)**(D/2)) * (1/(np.linalg.det(Sigma) **0.5))

    for row in Data
        for i in range(K):
            Gaussian[i] =



def Gama_cal(Pie, Sigma)

def Sigma_update(Gama, Data, Miu)

def Miu_update(Gama, Data)