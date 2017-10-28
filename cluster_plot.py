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