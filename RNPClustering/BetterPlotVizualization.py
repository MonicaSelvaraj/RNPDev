'''
De-convolution approach
- Particles of each size are de-convoluted separately
For each size:
- Particles within a specific x,y are grouped
- The spread in the z is determined for each group
- The median spred in the z is set as the z threshold for de-convolution
- Particles within the x,y,z, threshold are represented as just the center point (getting rid of particles detected when out of focus)
'''

#!/usr/bin/python
import sys, os 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import style
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.stats
import statistics
import numpy 
import csv
import math

sys.setrecursionlimit(10000)
fig = plt.figure( )
plt.style.use('dark_background')

#Original data 
X1 = list(); Y1 = list(); Z1 = list(); S1 = list()#0.27 - Not being considered for Curve fitting 
X2 = list(); Y2 = list(); Z2 = list(); S2 = list()#0.36
X3 = list(); Y3 = list(); Z3 = list(); S3 = list() #0.45
X4 = list(); Y4 = list(); Z4 = list(); S4 = list()#0.54

#Reading in the data 
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
ax.grid(False)
#red = ax.scatter (X1, Y1, Z1, c = 'r', marker='o', s=S1[0])
#blue = ax.scatter (X2, Y2, Z2, c = 'b', marker='o', s=S2[0]*2, linewidths=2)
green = ax.scatter (X3, Y3, Z3, c = 'g', marker='o', s=S3[0]*4, linewidths=2)
yellow = ax.scatter (X4, Y4, Z4, c = 'y', marker='o', s=S4[0]*6, linewidths=2)
ax.set_title('Convoluted')
ax.legend((green, yellow),
           ( '0.45', '0.54'),
           scatterpoints=1,
           loc='best',
           ncol=1,
           fontsize=8)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#Sorts data by x, y, z like excel and returns sorted x,y,z numpy arrays - used as input for Align method 
def Sort(x,y,z):
    #For sorted data
    Xs = list(); Ys = list(); Zs = list()
    Xs = numpy.array(Xs); Ys = numpy.array(Ys); Zs = numpy.array(Zs)
    Xs = Xs.astype(float); Ys= Ys.astype(float); Zs= Zs.astype(float)
    
    #Concatenating numpy arrays
    data = numpy.concatenate((x[:, numpy.newaxis], 
                       y[:, numpy.newaxis], 
                       z[:, numpy.newaxis]), 
                      axis=1)
    
    #Sorting wrt x, y, z consecutively like excel
    sortedData = data[numpy.lexsort(numpy.transpose(data)[::-1])]
    
    #Separating the sorted data into numpy arrays
    sortedArray = numpy.hsplit(sortedData, 3)
    Xs = numpy.concatenate(sortedArray[0], axis=0)
    Ys = numpy.concatenate(sortedArray[1], axis=0)
    Zs = numpy.concatenate(sortedArray[2], axis=0)
    return (Xs, Ys, Zs);

#This method takes in the sorted data and finds the median of z splits, to determine the z threshold
def MedianZ(X,Y,Z,S):
    xylim = S
    #Variable pos to keep track of the current positions being visited 
    pos1 = -1 #Starting from -1 because indexing starts from 0
    pos2 = -1

    #Array to keep track of what has been visited to avoid double counts, 1 indicates visited, 0 indicates unvisited
    visited = numpy.zeros(X.size)

    #New multidimensional arrays for cluster points - This is what the cluster method returns 
    zSplits = list()

    #Iterating through x,y,z at the same time
    for x,y,z in zip (X, Y, Z):
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
        for a,b,c in zip (X, Y, Z):
            pos2+=1
            if visited[pos2] == 1:
                continue
            if ((a>=currentX-xylim and  a<=currentX+xylim) and (b>=currentY-xylim and b<=currentY+xylim)):
                similarX.append(a)
                similarY.append(b)
                listofZ.append(c)
                visited[pos2] = 1
        #When you come out of the inner loop, you have one set of aligned points
        similarX = numpy.array(similarX); similarY = numpy.array(similarY); listofZ = numpy.array(listofZ)
        similarX = similarX.astype(float); similarY = similarY.astype(float); listofZ = listofZ.astype(float)
        if (listofZ.size != 1): #We don't care about the points that are not being split
            zSplits.append(listofZ.size)
    zthreshold = math.ceil(statistics.median(zSplits))
    return (zthreshold);

#Clusters points
def cluster(X,Y,Z,S):
    #Getting the x,y threshold 
    xylim = S[0]
    print (xylim)
    #Sorting data 
    sortedData = numpy.array(0); alignedData = numpy.array(0)
    sortedData = Sort(X, Y, Z)

    #Getting the z threshold
    zlim = MedianZ(sortedData[0], sortedData[1], sortedData[2], xylim)*Z[Z.size-1] 
    print (zlim)

    #DEFINING A FUNCTION TO GENERATE A CLUSTER FROM SORTED DATA
    def generateCluster(xdata,ydata,zdata,zlim):
        
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
            generateCluster(xarrays[0],yarrays[0],zarrays[0],zlim)
            generateCluster(xarrays[1],yarrays[1],zarrays[1],zlim)
          
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
    for x,y,z in zip (sortedData[0], sortedData[1], sortedData[2]):
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
        for a,b,c in zip (sortedData[0], sortedData[1], sortedData[2]):
            pos2+=1
            if visited[pos2] == 1:
                continue
            if ((a>=currentX-xylim and  a<=currentX+xylim) and (b>=currentY-xylim and b<=currentY+xylim)):
                similarX.append(a)
                similarY.append(b)
                listofZ.append(c)
                visited[pos2] = 1
                
        #Generating new points after clustering
        similarX = numpy.array(similarX); similarY = numpy.array(similarY); listofZ = numpy.array(listofZ)
        similarX = similarX.astype(float); similarY = similarY.astype(float); listofZ = listofZ.astype(float)
        newPoints = generateCluster(similarX, similarY, listofZ,zlim)
        if (newPoints[0] <= 0 or newPoints[1] <= 0 or newPoints[2] <= 0 ): #Why do I have negative values? BUG
            continue
        newx.append(newPoints[0])
        newy.append(newPoints[1])
        newz.append(newPoints[2])

    newx= numpy.array(newx); newy = numpy.array(newy); newz = numpy.array(newz)
    newx = newx.astype(float); newy= newy.astype(float); newz= newz.astype(float)
    return (newx, newy, newz);
#End of cluster method

#Clustering 0.54 points
cluster4 = cluster(X4, Y4, Z4, S4)
cluster4= numpy.array(cluster4)
cluster4 = cluster4.astype(float)

#Clustering 0.45 points
cluster3 = cluster(X3, Y3, Z3, S3)
cluster3= numpy.array(cluster3)
cluster3 = cluster3.astype(float)

#Clustering 0.36 points
cluster2 = cluster(X2, Y2, Z2, S2)
cluster2= numpy.array(cluster2)
cluster2 = cluster2.astype(float)

#Clustering 0.27 points
cluster1 = cluster(X1, Y1, Z1, S1)
cluster1= numpy.array(cluster1)
cluster1 = cluster1.astype(float)

#Cluster 4 has x,y,z of 0.54
#Cluster 3 has x,y,z of 0.45
#Cluster 2 has x,y,z of 0.36
#Write out to a csv file named deconvoluted.csv - should contain list of x,y,z from cluster4,3,2

#First create 4 numpy arrays for x,y,z, size data
dX = list(); dY = list(); dZ = list(); dS = list()
dX = numpy.array(dX); dY = numpy.array(dY); dZ = numpy.array(dZ); dS = numpy.array(dS);
dX = dX.astype(float); dY= dY.astype(float); dZ= dZ.astype(float); dS= dS.astype(float)

dX = numpy.concatenate((cluster4[0], cluster3[0]), axis=0)
dY = numpy.concatenate((cluster4[1], cluster3[1]), axis=0)
dZ = numpy.concatenate((cluster4[2], cluster3[2]), axis=0)
print (dX)
print (dY)
print (dZ)

numpy.savetxt("deconvoluted.csv", numpy.column_stack((dX,dY,dZ)), delimiter=",", fmt='%s')

#Generating a scatter plot of the points after clustering
ax = fig.add_subplot(1,2,2, projection = '3d')
yellow = ax.scatter (cluster4[0] , cluster4[1], cluster4[2], c = 'y', marker='o', s=S4[0]*16, linewidths=2)
green = ax.scatter (cluster3[0] , cluster3[1], cluster3[2], c = 'g', marker='o', s=S3[0]*8, linewidths=2)
#blue = ax.scatter (cluster2[0] , cluster2[1], cluster2[2], c = 'b', marker='o', s=S2[0]*4, linewidths=2)
#red = ax.scatter (cluster1[0] , cluster1[1], cluster1[2], c = 'r', marker='o', s=S1[0])
ax.grid(False)
ax.legend((green, yellow),
           ('0.45', '0.54'),
           scatterpoints=1,
           loc='best',
           ncol=1,
           fontsize=8)
ax.set_title('Deconvoluted')
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
#ax.view_init(azim=180, elev=10)
plt.show()



    
    

