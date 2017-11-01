import cluster_init
import cluster_plot
import numpy as np

# here we use exhausted algorithm
# here if we use PAM (Patitioning Around Medoids) will get a faster solution but it can not always find the optimal solution
# Manhattan distance is used

def data_class(dataset, medoids):  # dataset and medoids here is transfers to numpy array
    dataset = np.array(dataset)
    medoids = np.array(medoids)
    center_distance = [[sum(abs(elem - data)) for elem in medoids] for data in dataset]
    distance_value = [min(elem) for elem in center_distance]
    idx_list = [elem.index(min(elem)) for elem in center_distance]
    return distance_value, idx_list

def k_medoids(dataset, medoids):
    distance_value, idx_list = data_class(dataset, medoids)

    class_datalist = [[] for i in medoids]
    for i in range(len(idx_list)):  # this loop is to build the cluster to related centroids
        class_datalist[idx_list[i]].append(dataset[i])
    a = 1
    for i in range(len(class_datalist)):
        cluster_x = class_datalist[i]
        cluster_y = cluster_x[:]
        cost_matrix = [[sum(abs(elem1 - elem2)) for elem2 in cluster_y] for elem1 in cluster_x]
        cost_list = [sum(elem) for elem in cost_matrix]
        medoids[i] = class_datalist[i][cost_list.index(min(cost_list))] # find one point in the cluseter has a totall minimun distance
    return medoids


iteration = 20
k = 4
meanlist = [[0,5],[3,0],[5,-5],[8,-2]]
covlist = [ [[5,1],[1,5]], [[5,3],[3,5]], [[4,-2],[-2,4]], [[2,0],[0,1]] ]
sizelist = [400,400,500,650]

data = cluster_init.datagenerate(meanlist, covlist, sizelist)
cluster_plot.initial_plot(data)
centers = cluster_init.centers_init(data, k)
init_centroids = centers[:]

for i in range(iteration):
    centers = k_medoids(data, centers)
    cluster_plot.kmeans_plot(data, centers, init_centroids)