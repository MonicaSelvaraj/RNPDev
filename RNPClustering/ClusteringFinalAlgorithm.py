'''
Clustering 0.54 radius particles to find out how many 0.54 radius particles are present
- Each 0.54 radius particle can be at most 4 z's away  (0.362)
- The diameter of a 0.54 particle is 1.08
- The x and y can vary at most 1.08 units - 0.54 left or right 
Cluster: Taking one 0.54 particle, if there are other 0.54 particles 1.08, 1.08, 0.362 units away,
count them as one particle.
- Have an array with the length of the number of particles indicating if you have counted the particle before
to avoid double counts (0 indicates not counted, 1 indicates counted)
'''

#!/usr/bin/python
import sys, os #Used to communicate between the terminal and python
import matplotlib.pyplot as plt
from matplotlib import cm
#from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv
import math

sys.setrecursionlimit(100000)

fig = plt.figure( )

#0.54 - original data 
X1 = list()
Y1 = list()
Z1 = list()

#For sorted data
Xs = list()
Ys = list()
Zs = list()

with open ('manualzRed.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        #each line has X,Y,Z, radius
        if (float(line[3])>0.54):
            X1.append(line[0])
            Y1.append(line[1])
            Z1.append(line[2])

X1 = numpy.array(X1)
Y1 = numpy.array(Y1)
Z1 = numpy.array(Z1)

Xs = numpy.array(Xs)
Ys = numpy.array(Ys)
Zs = numpy.array(Zs)

X1 = X1.astype(float)
Y1= Y1.astype(float)
Z1= Z1.astype(float)

Xs = Xs.astype(float)
Ys= Ys.astype(float)
Zs= Zs.astype(float)

#Generating a plot of the original 0.54 points
ax = fig.add_subplot(1,2,1, projection = '3d')
ax.scatter (X1, Y1, Z1, c = 'b', marker='o', s=2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#SORTING
#Concatenating numpy arrays
data = numpy.concatenate((X1[:, numpy.newaxis], 
                       Y1[:, numpy.newaxis], 
                       Z1[:, numpy.newaxis]), 
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
    #The maximum z-distance can be 0.362
    xOfCluster = 0.0
    yOfCluster = 0.0
    zOfCluster = 0.0
    
    maxz = numpy.amax(zdata)
    minz = numpy.amin(zdata)
    zlength = len(zdata)
    zdist = maxz - minz #How spread apart the points areS

    #In case I need to split up the arrays for recursion
    xarrays = list()
    yarrays = list()
    zarrays = list()
    xarrays = numpy.array(xarrays)    
    yarrays = numpy.array(yarrays)
    zarrays = numpy.array(zarrays)
    xarrays = xarrays.astype(float)
    yarrays = yarrays.astype(float)
    zarrays = zarrays.astype(float)
    
    if (zdist<0.362): #Is one cluster
        pointpos = math.floor((zlength/2.0))
        xOfCluster = xdata[pointpos]
        yOfCluster = ydata[pointpos]
        zOfCluster = zdata[pointpos]
    if (zdist>0.363):#More than one cluster
        xarrays = numpy.array_split(xdata, 2)
        yarrays = numpy.array_split(ydata, 2)
        zarrays = numpy.array_split(zdata, 2)
        generateCluster(xarrays[0],yarrays[0],zarrays[0])
        generateCluster(xarrays[1],yarrays[1],zarrays[1])
          
    clusterCenter = [xOfCluster, yOfCluster, zOfCluster]
    return (clusterCenter);
                

#CLUSTERING
#Variable pos to keep track of the current positions being visited 
pos1 = -1 #Starting from -1 because indexing starts from 0
pos2 = -1

#Array to keep track of what is visited, 1 indicates visited, 0 indicates unvisited
visited = numpy.zeros(X1.size)

#New arrays for cluster points
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
        if ((a>currentX-0.54 and  a<currentX+0.54) and (b>currentY-0.54 and b<currentY+0.54)):
            similarX.append(a)
            similarY.append(b)
            listofZ.append(c)
            visited[pos2] = 1
    #Generating new points after clustering
    similarX = numpy.array(similarX)
    similarY = numpy.array(similarY)
    listofZ = numpy.array(listofZ)
    similarX = similarX.astype(float)
    similarY = similarY.astype(float)
    listofZ = listofZ.astype(float)
    newPoints = generateCluster(similarX, similarY, listofZ)
    newx.append(newPoints[0])
    newy.append(newPoints[1])
    newz.append(newPoints[2])

newx= numpy.array(newx)
newy = numpy.array(newy)
newz = numpy.array(newz)

newx = newx.astype(float)
newy= newy.astype(float)
newz= newz.astype(float)

#Generating a scatter plot of the points after clustering
ax = fig.add_subplot(1,2,2, projection = '3d')
ax.scatter (newx, newy, newz, c = 'b', marker='o', s=2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

print (X1.size)
print (newx.size)

plt.show()
