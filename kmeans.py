import numpy as np
import matplotlib as plot
import matplotlib.pyplot as plt

def datagenerate(meanlist, covlist, sizelist):
    #xlist = []
    #ylist = []
    dataset = []
    for i in range(len(meanlist)):
        mean = meanlist[i]
        cov = covlist[i]
        size = sizelist[i]
        data = np.random.multivariate_normal(mean,cov,size)
        dataset.append(data)
    return dataset


def data_class(dataset, centers):  # dataset and centers here is transfers to numpy array
    dataset = np.array(dataset)
    centers = np.array(centers)
    center_distance = [[np.linalg.norm(data - elem) for elem in centers] for data in dataset]
    distance_value = [min(elem) for elem in center_distance]
    idx_list = [elem.index(min(elem)) for elem in center_distance]
    return distance_value, idx_list


def centers_init(dataset, k):
    data = []
    centers = []
    idx = np.random.random_integers(1,500)
    centers.append(dataset[idx])  #choose first point
    while len(centers)<k:
        distance_value, idx_list = data_class(dataset, centers)
        next_idx = distance_value.index(max(distance_value))
        centers.append(dataset[next_idx])

    return centers


def kmeans(dataset, k, centroids):
    centroids = centers_init(dataset, k)
    init_centroids = centroids[:]
    distance_value, idx_list = data_class(dataset, centroids)
    class_datalist = [[] for i in centroids]
    for i in range(len(idx_list)):  #this loop is to build the cluster to related centroids
        class_datalist[idx_list[i]].append(dataset[i])
    for i in range(len(class_datalist)):  #this loop is getting every mean of the cluster and get the new centrod
        x = [element[0] for element in class_datalist[i]]
        y = [element[1] for element in class_datalist[i]]
        x_mean = sum(x)/len(x)
        y_mean = sum(y)/len(y)
        centroids[i] = [x_mean,y_mean]
    return init_centroids, centroids





