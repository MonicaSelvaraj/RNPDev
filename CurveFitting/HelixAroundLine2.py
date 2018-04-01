#Plotting a helix around PC1 
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

datamean = data.mean(axis=0)
print (datamean) #Center of the helix

uu, dd, vv = numpy.linalg.svd(data - datamean)

#Re-write
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

linepts += datamean
print (linepts) #Points on the axis of the helix 

#Why are we taking the transpose?
ax.plot3D(*linepts.T)

#SMA - To check if the SMA is wrapping around the line 
#Plotting the line 
#xline =movingaverage(movingaverage(x2, 10),5)
#yline = movingaverage(movingaverage(y2, 10),5)
#zline = movingaverage(movingaverage(z2, 10),5)
#ax.plot3D(xline,yline,zline,'red')

#Plotting a helix around the line
zline = numpy.linspace(271,571,1000) 
xline = 6*numpy.sin((1/5)*zline) + 28.74435878 
yline = 6*numpy.cos((1/5)*zline) + 25.34122088

ax.plot3D(xline, yline, zline, 'green')

'''
Tilting the helix 
Finding the angle between the current axis and x,y,and z axis using a rearrangement of the
dot product:
theta = arccos(x/(sqrt(x^2+y^2+z^2))) - for the angle with the x axis
Replace numerator with y or z for the angles with the y and z axis
x,y,z are points on the best fit line
Getting the points from linepts array (contains 2 points)
For this plot
Center of helix: [28.74435878  25.34122088 421.89587115]
Two points on the axis of the helix:
[[ 26.27508786  21.7782561  571.83321801]
 [ 31.2136297   28.90418566 271.95852429]]
'''

x = linepts[0,0]
y = linepts[0,1]
z = linepts[0,2]

#Calculating the angle the points need to be rotated by
xtheta = numpy.arccos((z)/math.sqrt((y**2+z**2)))
ytheta = numpy.arccos((z)/math.sqrt((x**2+z**2)))
#ztheta = numpy.arccos((z)/math.sqrt((x**2+y**2+z**2)))

print (xtheta)
print (ytheta)
#print (ztheta)
#print (-math.sin(xtheta))

#Making rotation matrices -making lists, reshaping, converting numpy to arrays
rx = [1,0,0,0,math.cos(xtheta),-(math.sin(xtheta)),0,math.sin(xtheta),math.cos(xtheta)]
ry = [numpy.cos(ytheta),0,numpy.sin(ytheta),0,1,0,-numpy.sin(ytheta),0,numpy.cos(ytheta)]
#rz =[numpy.cos(ztheta),-numpy.sin(ztheta),0,numpy.sin(ztheta), numpy.cos(ztheta),0,0,0,1]
#Converting to numpy arrays
Rx= numpy.array(rx)
Ry = numpy.array(ry)
#Rz = numpy.array(rz)
Rx = Rx.astype(float)
Ry= Ry.astype(float)
#Rz = Rz.astype(float)

#Reshaping to 3x3 arrays
Rx = Rx.reshape((3,3))
Ry = Ry.reshape((3,3))
#Rz = Rz.reshape((3,3))

#print (Rx)
#print (Ry)
#print (Rz)

#Concatenating xline,yline,zline points
originalAxisPts= numpy.concatenate((xline[:, numpy.newaxis], 
                       yline[:, numpy.newaxis], 
                       zline[:, numpy.newaxis]), 
                      axis=1)
#print (newAxis)

newAxisPts1= numpy.matmul(originalAxisPts, Rx)
newAxisPts = numpy.matmul(newAxisPts1, Ry)

#print (newAxisPts)

newAxis = numpy.hsplit(newAxisPts, 3)
newxline = numpy.concatenate(newAxis[0], axis=0)
newyline = numpy.concatenate(newAxis[1], axis=0)
newzline = numpy.concatenate(newAxis[2], axis=0)

#print (newxline)
#print (newyline)
#print (newzline)


ax.plot3D(newxline, newyline, newzline, 'green')
plt.show( )






