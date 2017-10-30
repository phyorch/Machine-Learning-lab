import kmeans
import cluster_init
import matplotlib.pyplot as plt


def initial_plot(data):
    '''meanlist = [[-5, 5], [3, 3], [5, -5], [12, -2]]
    covlist = [[[5, 1], [1, 5]], [[5, 3], [3, 5]], [[4, -2], [-2, 4]], [[5, 0], [0, 1]]]
    sizelist = [800, 750, 600, 450]
    datashow = cluster_init.datagenerate(meanlist, covlist, sizelist)'''
    xshow = [elem[0] for elem in data]
    yshow = [elem[1] for elem in data]
    plt.plot(xshow, yshow, 'ro')
    plt.axis()
    plt.show()

def result_plot(data, centers, init_centers):
    xshow = [elem[0] for elem in data]
    yshow = [elem[1] for elem in data]
    xcenter = [elem[0,0] for elem in centers]
    ycenter = [elem[0,1] for elem in centers]
    xinit_center = [elem[0] for elem in init_centers]
    yinit_center = [elem[1] for elem in init_centers]
    plt.plot(xshow, yshow, 'ro', xcenter, ycenter, 'bs')#xinit_center, yinit_center, 'g^',
    plt.axis()
    plt.show()

'''iteration = 10
k = 4
centroids = cluster_init.centers_init(datashow, k)
init_centroids = centroids[:]
for i in range(iteration):
    init_centers, centers = kmeans.kmeans(datashow, k, centroids)
    xinit_center = [elem[0] for elem in init_centers]
    yinit_center = [elem[1] for elem in init_centers]
    xcenter = [elem[0] for elem in centers]
    ycenter = [elem[1] for elem in centers]
    plt.plot(xshow, yshow, 'ro', xinit_center, ycenter, 'g^')
    plt.plot(xshow, yshow, 'ro', xcenter, ycenter, 'bs')
    plt.axis()
    plt.show()'''