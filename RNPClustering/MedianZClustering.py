'''
- Aligning each size particles by x,y and finding the number of z splits
- Calculating the ceil of the median of z splits to figure out the z threshold 
- Using that z threshold to cluster points (individually for each size)
'''

#!/usr/bin/python
import sys, os 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.stats
import statistics
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
def MedianZ(X,Y,Z,xylim):
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


#For 0.54 points
sortedData = numpy.array(0); alignedData = numpy.array(0)
sortedData = Sort(X1, Y1, Z1)
medianZ = MedianZ(sortedData[0], sortedData[1], sortedData[2], S1[0])

def Cluster(X,Y,Z,xylim,zlim)




    
        

    
    


    
    
    


