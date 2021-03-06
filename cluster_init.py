import numpy as np


# This function is used to generate ant amount of data
# as long as you provide the basic requirement in implementation test
def datagenerate(meanlist, covlist, sizelist):
    #xlist = []
    #ylist = []
    dataset = []
    for i in range(len(meanlist)):
        mean = meanlist[i]
        cov = covlist[i]
        size = sizelist[i]
        data = list(np.random.multivariate_normal(mean,cov,size))
        dataset.extend(data)
    return dataset

# This function is frequently used in these algorithm to classify the data to different clusters
def data_class(dataset, centers):  # dataset and centers here is transfers to numpy array
    dataset = np.array(dataset)
    centers = np.array(centers)
    center_distance = [[np.linalg.norm(data - elem) for elem in centers] for data in dataset]
    distance_value = [min(elem) for elem in center_distance]
    idx_list = [elem.index(min(elem)) for elem in center_distance] # find every data belong to which center
    return distance_value, idx_list

# Using greedy method to generate k centers
def centers_init(dataset, k):
    '''centers = [[np.random.random(),np.random.random()] for i in range(k)]
    return centers'''
    data = []
    centers = []
    idx = np.random.random_integers(1,500)
    centers.append(dataset[idx])  #choose first point
    while len(centers)<k:
        distance_value, idx_list = data_class(dataset, centers)
        next_idx = distance_value.index(max(distance_value))
        centers.append(dataset[next_idx])
    return centers
# Using random method to generate k centers
def centers_random(k):
    centers = np.random.rand(4,2)
    centers[:,0] = 5 * centers[:,0]
    centers[:, 1] = 8* centers[:,1]
    return centers

