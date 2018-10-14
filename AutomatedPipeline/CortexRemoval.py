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

#Variables for cortex removal
Z = list()
uniqueZ = list()

#Variables to store Channel 1 data after cortex removal
X1 = list(); Y1 = list(); Z1 = list()
#Variables to store Channel 2 data after cortex removal 
X2 = list(); Y2 =  list(); Z2 = list()

with open ('ClusteredC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        Z.append(line[2])
        
with open ('ClusteredC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        Z.append(line[2])

for z in Z:
    if z not in uniqueZ:
        uniqueZ.append(z)

Z = numpy.array(Z); Z = Z.astype(float)
uniqueZ = numpy.array(uniqueZ); uniqueZ = uniqueZ.astype(float)
uniqueZ = numpy.sort(uniqueZ) #This is a list of Z's from highest to lowest

zCount = list()
for uz in uniqueZ:
    counter = 0
    for z in Z:
        if (z == uz):
            counter = counter+1
    zCount.append(counter)

plt.scatter(uniqueZ, zCount)
plt.show()

#The input for clustering is in the form [[z, count], [z, count], .. ]
clusterInput = numpy.array(list(zip(uniqueZ,zCount)))
clusterInput = numpy.array(clusterInput ); clusterInput  = clusterInput .astype(float)

#Agglomerative Clustering

#Creating Dendrogram
dendrogram = sch.dendrogram(sch.linkage(clusterInput, method='ward'))
plt.show()

#Creating four clusters 
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(clusterInput)#Labels each point as cluster 0,1,2, or 3

#Displaying the clusters 
plt.scatter(clusterInput[y_hc ==0,0], clusterInput[y_hc == 0,1], s=100, c='red')
plt.scatter(clusterInput[y_hc==1,0], clusterInput[y_hc == 1,1], s=100, c='black')
plt.scatter(clusterInput[y_hc ==2,0], clusterInput[y_hc == 2,1], s=100, c='blue')
plt.scatter(clusterInput[y_hc ==3,0], clusterInput[y_hc == 3,1], s=100, c='cyan')
plt.show()


#Finding the cluster centers 
centroidX = list(); centroidY = list()

for c in range(4):
    zList = list(); zDensity = list() #Creating  a new list for each cluster 
    zList = clusterInput[y_hc ==c,0] #Creates a list of all the z's in that cluster
    zDensity = clusterInput[y_hc == c,1] #Creates a list of densities for the z's of that cluster 
    #Finding the centroids
    centroidX.append(sum(zList) / len(zList))
    centroidY.append(sum(zDensity) / len(zDensity))

#Displaying the clusters with centroids
plt.scatter(clusterInput[y_hc ==0,0], clusterInput[y_hc == 0,1], s=100, c='red')
plt.scatter(clusterInput[y_hc==1,0], clusterInput[y_hc == 1,1], s=100, c='black')
plt.scatter(clusterInput[y_hc == 2,0], clusterInput[y_hc == 2,1], s=100, c='blue')
plt.scatter(clusterInput[y_hc ==3,0], clusterInput[y_hc == 3,1], s=100, c='cyan')
plt.scatter(centroidX, centroidY, s = 200, c = 'green')
plt.show()

centroidX = numpy.array(centroidX, dtype = float)
print(centroidX)
centroidY = numpy.array(centroidY, dtype = float)
print(centroidY)

#Finding the cluster center with highest zDensity
highestDensityCluster = numpy.amax(centroidY)
#Finding the position of that cluster 
highestDensityPos = numpy.where(highestDensityCluster == centroidY)
highestDensityPos = highestDensityPos[0][0]

PossibleCortexZPositions = numpy.where(y_hc == highestDensityPos)
PossibleCortexZs = list()
for z in PossibleCortexZPositions:
    PossibleCortexZs.append(clusterInput[z, 0])


#Mean and standard deviation of the z density 
MeanDensity = numpy.mean(centroidY, axis = 0)
print(MeanDensity)
sdDensity= numpy.std(centroidY, axis = 0)
print(sdDensity)

cortexFound = True
#Checking if the standard deviation of the density of the clusters is greater than 10
if(sdDensity >= 5):
    #Checking if the cluster center with highest density of greater than one sd away from the mean density
    if(centroidY[highestDensityPos] >= (sdDensity + MeanDensity)):
        if(centroidX[highestDensityPos] >= uniqueZ[10]): #Within the last 10 z's
            #Finding the lowest z in that cluster and removing all the z's after it
            lowestZ = numpy.amin(PossibleCortexZs)
            with open ('ClusteredC1.csv', 'r') as csv_file:
                csv_reader = csv.reader (csv_file)
                for line in csv_reader:
                    if(float(line[2]) < lowestZ):
                        X1.append(line[0])
                        Y1.append(line[1])
                        Z1.append(line[2])
            with open ('ClusteredC2.csv', 'r') as csv_file:
                csv_reader = csv.reader (csv_file)
                for line in csv_reader:
                    if(float(line[2]) < lowestZ):
                        X2.append(line[0])
                        Y2.append(line[1])
                        Z2.append(line[2])
        elif(centroidX[highestDensityPos] <= uniqueZ[len(uniqueZ) - 10]): #Within the first 10 z's
            #Finding the highest z in that cluster and all the z's before it
            highestZ = numpy.amax(PossibleCortexZs)
            with open ('ClusteredC1.csv', 'r') as csv_file:
                csv_reader = csv.reader (csv_file)
                for line in csv_reader:
                    if(float(line[2]) > highestZ):
                        X1.append(line[0])
                        Y1.append(line[1])
                        Z1.append(line[2])
            with open ('ClusteredC2.csv', 'r') as csv_file:
                csv_reader = csv.reader (csv_file)
                for line in csv_reader:
                    if(float(line[2]) < highestZ):
                        X2.append(line[0])
                        Y2.append(line[1])
                        Z2.append(line[2])
        else:
            print("No cortex found")
            cortexFound = False
    else:
        print("No cortex found")
        cortexFound = False
else:
    print("No cortex found")
    cortexFound = False

if(!cortexFound)
    with open ('ClusteredC1.csv', 'r') as csv_file:
                csv_reader = csv.reader (csv_file)
                for line in csv_reader:
                    X1.append(line[0])
                    Y1.append(line[1])
                    Z1.append(line[2])
            with open ('ClusteredC2.csv', 'r') as csv_file:
                csv_reader = csv.reader (csv_file)
                for line in csv_reader:
                    X2.append(line[0])
                    Y2.append(line[1])
                    Z2.append(line[2])

    
X1 = numpy.array(X1, dtype=float); Y1 = numpy.array(Y1, dtype=float); Z1 = numpy.array(Z1, dtype=float)
X2 = numpy.array(X2, dtype=float); Y2 = numpy.array(Y2, dtype=float); Z2 = numpy.array(Z2, dtype=float)

#Saving each channel's data after removing the cortex in new files
numpy.savetxt("CortexRemovedC1.csv", numpy.column_stack((X1, Y1, Z1)), delimiter=",", fmt='%s')
numpy.savetxt("CortexRemovedC2.csv", numpy.column_stack((X2, Y2, Z2)), delimiter=",", fmt='%s')




