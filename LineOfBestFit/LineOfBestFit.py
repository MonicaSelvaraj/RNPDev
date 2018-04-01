#Determining line of best fit 
#!/usr/bin/python

#Might need these packages 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import numpy 
import csv
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

'''
====================
Line of best fit approach
====================
Finding the first principal component - axis that spans the most variance, best fits the data 
'''

#Creating a 3D axes by using the keyword projection = '3D'
fig = plt.figure( )
#setting up the axes for the first subplot
ax = fig.add_subplot(1,1,1, projection = '3d')

#For subplot (260-578)
X2 = list()
Y2 = list()
Z2 = list()
S2 = list()

with open ('testRed.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        if (float(line[2])>260 and float(line[2])<578):
            X2.append(line[0])
            Y2.append(line[1])
            Z2.append(line[2])
            S2.append(line[3])

x2= numpy.array(X2)
y2 = numpy.array(Y2)
z2 = numpy.array(Z2)
s2 = numpy.array(S2)

x2 = x2.astype(float)
y2= y2.astype(float)
z2 = z2.astype(float)
s2 = s2.astype(float)

#Plotting the SMA's to see if the helix wraps around PC1
def movingaverage(values, window):
    weights = numpy.repeat(1.0, window)/window
    #valid only runs the sma's on valid points
    smas = numpy.convolve(values, weights, 'valid')
    return smas #returns a numpy array

ax.scatter (x2, y2, z2, c = 'r', marker='D', s=s2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

data = numpy.concatenate((x2[:, numpy.newaxis], 
                       y2[:, numpy.newaxis], 
                       z2[:, numpy.newaxis]), 
                      axis=1)
print(data)

datamean = data.mean(axis=0)
print (datamean) #This is going to be the center of my helix
ax.scatter (datamean[0], datamean[1], datamean[2], c = 'g', marker='o')

uu, dd, vv = numpy.linalg.svd(data - datamean)

#Tested, works
def findRange(x,y,z):
    minimum = numpy.amin(x)
    maximum = numpy.amax(x)
    miny = numpy.amin(y)
    maxy = numpy.amax(y)
    if(miny<minimum):
        minimum = miny
    if(maxy>maximum):
        maximum = maxy
    minz = numpy.amin(z)
    maxz = numpy.amax(z)
    if(minz<minimum):
        minimum = minz
    if(maxz>maximum):
        maximum = maxz
    return minimum, maximum

Range = findRange(x2,y2,z2)

#Taking the variation in the z dimension, because this is the dimension of PC1
#Linear algebra - figure out what exactly is happening in terms of dimensional collapsation
linepts = vv[0] * numpy.mgrid[-150:150:2j][:, numpy.newaxis]

#Moving the line to be in between the points 
linepts += datamean

#Why are we taking the transpose?
ax.plot3D(*linepts.T)

#SMA
#Plotting the line 
xline =movingaverage(movingaverage(x2, 10),5)
yline = movingaverage(movingaverage(y2, 10),5)
zline = movingaverage(movingaverage(z2, 10),5)
#ax.plot3D(xline,yline,zline,'red')

plt.show ( )



