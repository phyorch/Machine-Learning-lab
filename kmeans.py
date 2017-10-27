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
    #centroids = centers_init(dataset, k)
    #init_centroids = centroids[:]
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




meanlist = [[0,5],[3,0],[5,-5],[12,-2]]
covlist = [ [[5,1],[1,5]], [[5,3],[3,5]], [[4,-2],[-2,4]], [[5,0],[0,1]] ]
sizelist = [200,250,200,150]
dataset = datagenerate(meanlist,covlist,sizelist)
datashow = []
for i in dataset:
    datashow.extend(i)
xshow = [elem[0] for elem in datashow]
yshow = [elem[1] for elem in datashow]
plt.plot(xshow, yshow, 'ro')
plt.axis()
plt.show()


iteration = 10
k = 4
centroids = centers_init(datashow, k)
init_centroids = centroids[:]
for i in range(iteration):
    init_centers, centers = kmeans(datashow, k, centroids)
    xinit_center = [elem[0] for elem in init_centers]
    yinit_center = [elem[1] for elem in init_centers]
    xcenter = [elem[0] for elem in centers]
    ycenter = [elem[1] for elem in centers]
    plt.plot(xshow, yshow, 'ro', xinit_center, ycenter, 'g^')
    plt.plot(xshow, yshow, 'ro', xcenter, ycenter, 'bs')
    plt.axis()
    plt.show()



