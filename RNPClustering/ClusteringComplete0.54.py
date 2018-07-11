'''
Applying the clustering algorithm to
a. 0.54, 0.45, 0.36, 0.27
Calculations 
x-y variation is 0.54, 0.45, 0.36, 0.27
z variation is 0.362, 0.543, 0.724, 0.905
'''

#!/usr/bin/python
import sys, os 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv
import math

sys.setrecursionlimit(10000)
fig = plt.figure( )

#Original data 
X1 = list(); Y1 = list(); Z1 = list() #0.27
X2 = list(); Y2 = list(); Z2 = list() #0.36
X3 = list(); Y3 = list(); Z3 = list() #0.45
X4 = list(); Y4 = list(); Z4 = list() #0.54

with open ('manualzRed.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        #each line has X,Y,Z, radius
        if (float(line[3])<0.28):
            X1.append(line[0])
            Y1.append(line[1])
            Z1.append(line[2])
        elif (float(line[3])>0.28 and float(line[3])<0.37):
            X2.append(line[0])
            Y2.append(line[1])
            Z2.append(line[2])
        elif (float(line[3])>0.37 and float(line[3])<0.46):
            X3.append(line[0])
            Y3.append(line[1])
            Z3.append(line[2])
        else:
            X4.append(line[0])
            Y4.append(line[1])
            Z4.append(line[2])

X1 = numpy.array(X1); Y1 = numpy.array(Y1); Z1 = numpy.array(Z1)
X2 = numpy.array(X2); Y2 = numpy.array(Y2); Z2 = numpy.array(Z2)
X3 = numpy.array(X3); Y3 = numpy.array(Y3); Z3 = numpy.array(Z3)
X4 = numpy.array(X4); Y4 = numpy.array(Y4); Z4 = numpy.array(Z4)
X1 = X1.astype(float); Y1= Y1.astype(float); Z1= Z1.astype(float)
X2 = X2.astype(float); Y2= Y2.astype(float); Z2 = Z2.astype(float)
X3 = X3.astype(float); Y3= Y3.astype(float); Z3= Z3.astype(float)
X4 = X4.astype(float); Y4= Y4.astype(float); Z4= Z4.astype(float)

#Generating a plot of the original points
ax = fig.add_subplot(1,2,1, projection = '3d')
ax.scatter (X1, Y1, Z1, c = 'r', marker='o', s=1)
ax.scatter (X2, Y2, Z2, c = 'r', marker='o', s=2)
ax.scatter (X3, Y3, Z3, c = 'r', marker='o', s=3)
ax.scatter (X4, Y4, Z4, c = 'r', marker='o', s=4)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#Start function here
def cluster(X,Y,Z, xylim,zlim):
    
    #For sorted data
    Xs = list(); Ys = list(); Zs = list()
    Xs = numpy.array(Xs); Ys = numpy.array(Ys); Zs = numpy.array(Zs)
    Xs = Xs.astype(float); Ys= Ys.astype(float); Zs= Zs.astype(float)

    #SORTING
    #Concatenating numpy arrays
    data = numpy.concatenate((X[:, numpy.newaxis], 
                       Y[:, numpy.newaxis], 
                       Z[:, numpy.newaxis]), 
                      axis=1)
    #Sorting wrt x, y, z consecutively like excel
    sortedData = data[numpy.lexsort(numpy.transpose(data)[::-1])]

    #Separating the sorted data into numpy arrays
    sortedArray = numpy.hsplit(sortedData, 3)
    Xs = numpy.concatenate(sortedArray[0], axis=0)
    Ys = numpy.concatenate(sortedArray[1], axis=0)
    Zs = numpy.concatenate(sortedArray[2], axis=0)

    #DEFINING A FUNCTION TO GENERATE A CLUSTER FROM SORTED DATA
    def generateCluster(xdata,ydata,zdata):
        
        #The maximum z-distance can be zlim
        xOfCluster = 0.0
        yOfCluster = 0.0
        zOfCluster = 0.0

        maxz = numpy.amax(zdata)
        minz = numpy.amin(zdata)
        zlength = len(zdata)
        zdist = maxz - minz #How spread apart the points are
        
        #In case I need to split up the arrays for recursion
        xarrays = list(); yarrays = list();zarrays = list()
        xarrays = numpy.array(xarrays); yarrays = numpy.array(yarrays); zarrays = numpy.array(zarrays)
        xarrays = xarrays.astype(float); yarrays = yarrays.astype(float); zarrays = zarrays.astype(float)
    
        if (zdist<zlim): #Is one cluster
            pointpos = math.floor((zlength/2.0))
            xOfCluster = xdata[pointpos]
            yOfCluster = ydata[pointpos]
            zOfCluster = zdata[pointpos]
        if (zdist>zlim):#More than one cluster
            xarrays = numpy.array_split(xdata, 2)
            yarrays = numpy.array_split(ydata, 2)
            zarrays = numpy.array_split(zdata, 2)
            generateCluster(xarrays[0],yarrays[0],zarrays[0])
            generateCluster(xarrays[1],yarrays[1],zarrays[1])
          
        clusterCenter = [xOfCluster, yOfCluster, zOfCluster]
        return (clusterCenter);
    #End of generateCluster method
                
    #CLUSTERING
    #Variable pos to keep track of the current positions being visited 
    pos1 = -1 #Starting from -1 because indexing starts from 0
    pos2 = -1

    #Array to keep track of what is visited, 1 indicates visited, 0 indicates unvisited
    visited = numpy.zeros(X.size)

    #New arrays for cluster points - This is what the cluster method returns 
    newx = list()
    newy = list()
    newz = list()

    #Iterating through Xs,Ys,Zs at the same time
    for x,y,z in zip (Xs, Ys, Zs):
        #Refreshing the lists before generating another cluster
        similarX = list()
        similarY = list()
        listofZ = list()
        pos1+=1
        if (visited[pos1] == 1):
            continue
        visited[pos1] = 1 
        currentX = x
        currentY = y
        similarX.append(x) #adding the first x to the list
        similarY.append(y) #adding the corresponding y to the list
        listofZ.append(z) #adding the corresponding z to the list
        #Iterating through the rest of the array to find similar values
        pos2 = -1 #Resetting the position before visiting the whole array again
        for a,b,c in zip (Xs, Ys, Zs):
            pos2+=1
            if visited[pos2] == 1:
                continue
            if ((a>currentX-xylim and  a<currentX+xylim) and (b>currentY-xylim and b<currentY+xylim)):
                similarX.append(a)
                similarY.append(b)
                listofZ.append(c)
                visited[pos2] = 1
                
        #Generating new points after clustering
        similarX = numpy.array(similarX); similarY = numpy.array(similarY); listofZ = numpy.array(listofZ)
        similarX = similarX.astype(float); similarY = similarY.astype(float); listofZ = listofZ.astype(float)
        newPoints = generateCluster(similarX, similarY, listofZ)
        newx.append(newPoints[0])
        newy.append(newPoints[1])
        newz.append(newPoints[2])

    newx= numpy.array(newx); newy = numpy.array(newy); newz = numpy.array(newz)
    newx = newx.astype(float); newy= newy.astype(float); newz= newz.astype(float)
    return (newx, newy, newz);
#End of cluster method

#List of x,y,z thresholds
#                     0.27  0.36  0.45,  0.54
xythreshold = [0.27, 0.36, 0.45, 0.54]
zthreshold  = [0.905, 0.724, 0.543, 0.362]

#Clustering 0.54 points first
cluster4 = cluster(X4,Y4,Z4, xythreshold[3],zthreshold[3])
cluster4= numpy.array(cluster4)
cluster4 = cluster4.astype(float)

#Appending the new cluster points to 0.45 array, and re-clustering
newX3 = numpy.append(X3, cluster4[0])
newY3 = numpy.append(Y3, cluster4[1])
newZ3 = numpy.append(Z3, cluster4[2])

cluster3 = cluster(newX3,newY3,newZ3, xythreshold[2],zthreshold[2])
cluster3= numpy.array(cluster3)
cluster3 = cluster3.astype(float)

#Appending the new cluster points to 0.36 array, and re-clustering
newX2 = numpy.append(X2, cluster3[0])
newY2 = numpy.append(Y2, cluster3[1])
newZ2 = numpy.append(Z2, cluster3[2])

cluster2 = cluster(newX2,newY2,newZ2, xythreshold[1],zthreshold[1])
cluster2= numpy.array(cluster2)
cluster2 = cluster2.astype(float)

#Appending the new cluster points to 0.27 array, and re-clustering
newX1 = numpy.append(X1, cluster2[0])
newY1 = numpy.append(Y1, cluster2[1])
newZ1 = numpy.append(Z1, cluster2[2])

cluster1 = cluster(newX1,newY1,newZ1, xythreshold[0],zthreshold[0])
cluster1= numpy.array(cluster1)
cluster1 = cluster1.astype(float)

#Generating a scatter plot of the points after clustering
ax = fig.add_subplot(1,2,2, projection = '3d')
ax.scatter (cluster1[0], cluster1[1], cluster1[2], c = 'r', marker='o', s=2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#print (X3.size)
#print (cluster4[0].size)
#print (newX3.size)
#print (cluster1[0].size)

plt.show()
