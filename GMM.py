import numpy as np
import math
import copy
import cluster_init
import cluster_plot

#main parameters
#Pie 1 * K          is the probability for randimly choosing a model form K Gaussian models
#Data N * D         is our dataset
#Miu K * D          is the mean value matrix of our K Gaussian model
#Sigma K * D * D    is the conviriance matrix of our K Gaussian model
#Gama N * K         is the probability matrix for the probability that a data comes from k component of K models

# clculate the probability for every point in every Gaussian cluster
def Gauss_cal(Data, Miu, Sigma, D, K, pi): #Sigma is a list of matrix
    #Miu = np.mat(Miu)
    Gaussian = np.zeros((len(Data), K))

    for i in range(len(Data)):
        for k in range(K):
            C = 1 / ((2 * pi) ** (D / 2)) * (1 / ((np.linalg.det(Sigma[k])) ** 0.5))
            Gaussian[i][k] = C * math.exp(-1/2 * ((Data[i]- Miu[k]) * Sigma[k].I * (Data[i]-Miu[k]).T))
    return Gaussian #the output Gaussian is matrix format

# update Gama for a new loop
def Gama_cal(Pie, Gaussian, N, K):
    Pie = np.mat(Pie)
    Gama = np.zeros((N, K))
    for i in range(N):  # this loop calculat the Gama matrix N*K
        for k in range(K):
            elem = copy.deepcopy(Gaussian[i])
            elem.shape = (K,1)
            Gama[i][k] = (Pie[0,k]*Gaussian[i][k]) / (Pie*elem)  # for one dimension matrix, the transpose is invlid operation by .T
    return Gama

# update Sigma for a new loop
def Sigma_update(Sigma, Gama, Data, Miu, N):
    D = Data.shape[1]
    for k in range(len(Sigma)):  #K
        Miu_row = Miu[k]
        Miu_matrix = Miu[k]
        for row in range(len(Data)-1):
            Miu_matrix = np.row_stack((Miu_matrix,Miu_row))
        comp  = np.zeros((N, D))
        Nk = 0
        for row in range(N):
            Nk += Gama[row,k]
            comp[row] = Gama[row,k] * (Data[row]-Miu_matrix[row])
        Sigma[k] = ((Data-Miu_matrix).T * comp) / Nk  #calculate the new sigma kth matrix using the formular
    return Sigma

# update Miu for a new loop
def Miu_update(Miu, Gama, Data, K, N):
    post_Miu = Miu[:]
    D = Data.shape[1]
    Nlist = sum(Gama)
    Nlist = Nlist.tolist()
    for k in range(K):
        '''comp = np.zeros((N, D))
        for row in range(N):  # N
            comp[row,:] = Gama[row, k] * Data[row,:]'''
        Gama_colum = Gama[:,k]
        Gama_matrix = Gama[:,k]
        Gama_matrix = np.column_stack((Gama_matrix,Gama_colum))
        post_Miu[k] = 1/Nlist[k] * sum(np.multiply(Gama_matrix,Data))
        #post_Miu[k] = 1/Nlist[k] * sum(comp)
        Pie = [Nlist[i]/sum(Nlist) for i in range(K)]
    return Pie, post_Miu

# GMM implementation
def GMM_Process(Data, Pie, Miu, Sigma, K):
    N = len(Data)
    pi = math.pi
    D = len(Data[0])
    Data = np.mat(Data)
    Gaussian = Gauss_cal(Data, Miu, Sigma, D, K, pi)
    Gama = Gama_cal(Pie, Gaussian, N, K)
    post_Sigma = Sigma_update(Sigma, Gama, Data, Miu, N)
    post_Pie, post_Miu = Miu_update(Miu, Gama, Data, K, N)
    return post_Pie, post_Miu, post_Sigma

# initialize all the parameters for GMM Gaussian model
def Para_init(dataset, centers, K): #init Miu Sigma Pie
    D = len(dataset[0])
    Miu = np.mat(centers) # Using initial centers
    Pie = [1/K for i in range(K)]
    Sigma = [np.zeros((D, D)) for i in range(K)]
    distance_value, idx_list = cluster_init.data_class(dataset, centers)
    class_datalist = [[] for i in centers]
    for i in range(len(idx_list)):  # this loop is to build the cluster to related centroids
        class_datalist[idx_list[i]].append(dataset[i]) # class_datalist is a K list containing each cluster
    for k in range(K):
        class_n = len(class_datalist[k])
        data = np.mat(class_datalist[k])
        Sigma[k] = np.mat(np.cov(data.T))
    return Pie, Miu, Sigma

#test
K = 4
iteration = 50
meanlist = [[0,5],[3,0],[5,-5],[12,-2]]
covlist = [ [[5,1],[1,5]], [[5,3],[3,5]], [[4,-2],[-2,4]], [[5,0],[0,1]] ]
sizelist = [400,400,500,650]

dataset = cluster_init.datagenerate(meanlist, covlist, sizelist)
cluster_plot.initial_plot(dataset)
init_centers = cluster_init.centers_init(dataset, K)
random_centers = cluster_init.centers_random(K)
Pie, Miu, Sigma = Para_init(dataset, random_centers, K)
post_Pie, post_Miu, post_Sigma = GMM_Process(dataset, Pie, Miu, Sigma, K)
for i in range(iteration):
    post_Pie, post_Miu, post_Sigma = GMM_Process(dataset, post_Pie, post_Miu, post_Sigma, K)
    cluster_plot.result_plot(dataset, post_Miu, init_centers)
a = 1




