'''
- Counting the number of points in each z
- Creating a scatter plot with z in the x axis and number of points in the y axis
- Clustering points using agglomerative clustering
- Finding the centroids of each cluster
- Picking the centroid with highest number of points(y)
- Seeing if it is within the first 10 z's or the last 10 z's 
- If it is - find the z spread in that cluster and determine where to chop off the z  
- Should I check if the cluster centers are significantly different instead?


'''

#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy 
import csv
from matplotlib import style
style.use("ggplot")
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

fig = plt.figure( )

print("Removing the cortex")

#Variables for C1
Z = list()
uniqueZ = list()

with open ('C1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        Z.append(line[2])
        
with open ('C2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        Z.append(line[2])

for z in Z:
    if z not in uniqueZ:
        uniqueZ.append(z)

Z = numpy.array(Z); Z = Z.astype(float)
uniqueZ = numpy.array(uniqueZ); uniqueZ = uniqueZ.astype(float)
uniqueZ = numpy.sort(uniqueZ)

print(uniqueZ)

zCount = list()
for uz in uniqueZ:
    counter = 0
    for z in Z:
        if (z == uz):
            counter = counter+1
    zCount.append(counter)
print(zCount)

plt.scatter(uniqueZ, zCount)
plt.show()

#How should we decide which is cortex
#Trying different clustering algorithms
clusterInput = numpy.array(list(zip(uniqueZ,zCount)))
clusterInput = numpy.array(clusterInput ); clusterInput  = clusterInput .astype(float)

'''
#Clustering - mean shift 
ms = MeanShift()
ms.fit(clusterInput)
labels = ms.labels_
cluster_centers = ms.cluster_centers_ #Predicted cluster centers

n_clusters = len(numpy.unique(labels))

print("Number of estimated clusters: ", n_clusters)

colors = 10*['r.','g.','b.','c.','k.']
print(colors)
print(labels)

for i in range (len(clusterInput)):
    plt.plot(clusterInput[i][0], clusterInput[i][1],colors[labels[i]], markersize = 10)

plt.scatter(cluster_centers[:,0], cluster_centers[:,1] , marker = 'x', s=10)
plt.show()
'''

#Clustering - Agglomerative Hierarchical 
dendrogram = sch.dendrogram(sch.linkage(clusterInput, method='ward'))
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(clusterInput)
cluster_centers = hc.cluster_centers_ 
plt.show()

plt.scatter(clusterInput[y_hc ==0,0], clusterInput[y_hc == 0,1], s=100, c='red')
plt.scatter(clusterInput[y_hc==1,0], clusterInput[y_hc == 1,1], s=100, c='black')
plt.scatter(clusterInput[y_hc ==2,0], clusterInput[y_hc == 2,1], s=100, c='blue')
plt.scatter(clusterInput[y_hc ==3,0], clusterInput[y_hc == 3,1], s=100, c='cyan')
plt.show()


