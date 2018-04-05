'''
This script implements:
- Finding the (mode) number of z's each size is split into
- Clustering by size
TODO: Right now you are counting the mode every time you make a cluster
you need to calculate the mode after making all the clusters
Possible appraoch :
a. Store similarX, similarY, listofZ in separate multi-dimensional arrays
b. Go through and find modeZ
c. Then start clustering all points, and write the new x,y,z onto a file
d. Do the rest of the analysis from the new data file 
'''

#!/usr/bin/python
import sys, os 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.stats
import numpy 
import csv
import math

sys.setrecursionlimit(10000)
fig = plt.figure( )

#Original data 
X1 = list(); Y1 = list(); Z1 = list(); S1 = list()#0.27 - Not being considered for Curve fitting 
X2 = list(); Y2 = list(); Z2 = list(); S2 = list()#0.36
X3 = list(); Y3 = list(); Z3 = list(); S3 = list() #0.45
X4 = list(); Y4 = list(); Z4 = list(); S4 = list()#0.54

with open ('newTestRed.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        #each line has X,Y,Z, radius
        if (float(line[3])<0.28):
            X1.append(line[0])
            Y1.append(line[1])
            Z1.append(line[2])
            S1.append(line[3])
        elif (float(line[3])>0.28 and float(line[3])<0.37):
            X2.append(line[0])
            Y2.append(line[1])
            Z2.append(line[2])
            S2.append(line[3])
        elif (float(line[3])>0.37 and float(line[3])<0.46):
            X3.append(line[0])
            Y3.append(line[1])
            Z3.append(line[2])
            S3.append(line[3])
        else:
            X4.append(line[0])
            Y4.append(line[1])
            Z4.append(line[2])
            S4.append(line[3])

X1 = numpy.array(X1); Y1 = numpy.array(Y1); Z1 = numpy.array(Z1); S1 = numpy.array(S1)
X2 = numpy.array(X2); Y2 = numpy.array(Y2); Z2 = numpy.array(Z2); S2 = numpy.array(S2)
X3 = numpy.array(X3); Y3 = numpy.array(Y3); Z3 = numpy.array(Z3); S3 = numpy.array(S3)
X4 = numpy.array(X4); Y4 = numpy.array(Y4); Z4 = numpy.array(Z4); S4 = numpy.array(S4)
X1 = X1.astype(float); Y1= Y1.astype(float); Z1= Z1.astype(float); S1= S1.astype(float)
X2 = X2.astype(float); Y2= Y2.astype(float); Z2 = Z2.astype(float); S2= S2.astype(float)
X3 = X3.astype(float); Y3= Y3.astype(float); Z3= Z3.astype(float); S3= S3.astype(float)
X4 = X4.astype(float); Y4= Y4.astype(float); Z4= Z4.astype(float); S4= S4.astype(float)

#Generating a plot of the original points
ax = fig.add_subplot(1,2,1, projection = '3d')
#ax.scatter (X1, Y1, Z1, c = 'r', marker='o', s=1)
ax.scatter (X2, Y2, Z2, c = 'b', marker='o', s=2)
ax.scatter (X3, Y3, Z3, c = 'g', marker='o', s=3)
ax.scatter (X4, Y4, Z4, c = 'y', marker='o', s=4)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#Start function here
#Note: zlim - is the distance between each z slice
#mz - mode of number of z splits
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
    def generateCluster(xdata,ydata,zdata,mZ):
        
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
        #TODO: Change z to mode z here
        print (zlim)
        print (mZ)
        if (zdist<(float)(zlim*mZ)): #Is one cluster
            pointpos = math.floor((zlength/2.0))
            xOfCluster = xdata[pointpos]
            yOfCluster = ydata[pointpos]
            zOfCluster = zdata[pointpos]
        else: #More than one cluster
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
        #When you come out of the inner loop, you have one set of similar z points, count length of z here
        print (similarX)
        print (similarY)
        print (listofZ)
        #zlengths stores the z length of each set of points to determine the mode  
        zlengths = list()
        zlengths.append (len(listofZ))
        #Converting zlengths to a numpy array
        zlengths = numpy.array(zlengths)
        zlengths = zlengths.astype(float)
        print (zlengths)
        #Finding the mode of zlengths for our z threshold
        modeZ = scipy.stats.mode(zlengths)
        #Generating new points after clustering
        similarX = numpy.array(similarX); similarY = numpy.array(similarY); listofZ = numpy.array(listofZ)
        similarX = similarX.astype(float); similarY = similarY.astype(float); listofZ = listofZ.astype(float)
        newPoints = generateCluster(similarX, similarY, listofZ,modeZ)
        newx.append(newPoints[0])
        newy.append(newPoints[1])
        newz.append(newPoints[2])

    newx= numpy.array(newx); newy = numpy.array(newy); newz = numpy.array(newz)
    newx = newx.astype(float); newy= newy.astype(float); newz= newz.astype(float)
    return (newx, newy, newz);
#End of cluster method

#List of x,y,z thresholds
#zDistance is the distance between each slice - to be multiplied with mode z to determine the z threshold
zDistance = Z2[Z2.size-1]
#                     0.27  0.36  0.45,  0.54
xythreshold = [0.27, 0.36, 0.45, 0.54]

#Clustering 0.54 points
clustersFormed4 = cluster(X4,Y4,Z4, xythreshold[3],zDistance)
clustersFormed4 = numpy.array(clustersFormed4)
clustersFormed4 = clustersFormed4.astype(float)

#Clustering 0.45 points
clustersFormed3 = cluster(X3,Y3,Z3, xythreshold[2],zDistance)
clustersFormed3 = numpy.array(clustersForme3)
clustersFormed3 = clustersFormed3.astype(float)

#Clustering 0.36 points
clustersFormed2 = cluster(X2,Y2,Z2, xythreshold[1],zDistance)
clustersFormed2 = numpy.array(clustersFormed2)
clustersFormed2 = clustersFormed2.astype(float)

#Generating a scatter plot of the points after clustering
ax = fig.add_subplot(1,2,2, projection = '3d')
ax.scatter (clustersFormed4[0] , clustersFormed4[1], clustersFormed4[2], c = 'y', marker='o', s=4)
ax.scatter (clustersFormed3[0] , clustersFormed3[1], clustersFormed3[2], c = 'g', marker='o', s=4)
ax.scatter (clustersFormed2[0] , clustersFormed2[1], clustersFormed2[2], c = 'b', marker='o', s=4)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#For calculations
#print ("Number of 0.36 particles: ", X2.size) #Number of 0.36 particles 
#print ("Number of 0.45 particles: ", X3.size) #Number of 0.45 particles
#print ("Number of 0.54 particles: ", X4.size) #Number of 0.54 particles
#print ("Total number of clusters: ", clustersFormed[0].size) #Total number of 0.54 and 0.36 points
#print ("Number of 0.36 particles left after clustering: ", ((X2.size + X3.size) - clustersFormed[0].size)) #Remaining number of 0.36 particles

plt.show()
                    
            
            

            

        
        
        
    


      

       




