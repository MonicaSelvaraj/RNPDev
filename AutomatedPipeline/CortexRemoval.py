'''
- Counting the number of points in each z - plotting it out - see if there is a large drop
'''

#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy 
import csv
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import MeanShift #as ms

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
#Clustering
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
