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
Curve fitting approach
====================
1. Assume x,y, or z axis to be the axis of the helix
2. Rotate points onto the new assumed axis
3. Estimate radius, height of each turn and plot a helix wrt the axis
4. Use least squares to estimate error
5. Use hill climbing algorithm to find the helix with least error
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

ax.scatter (x2, y2, z2, c = 'r', marker='D', s=s2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

plt.show ( )

#Part I - Rotating points about an axis

