#Imputation of blobs

#Approach
#Computing the simple moving average of the x,y,z array's, and using those to plot a line of best fit for the 3D curve

#!/usr/bin/python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy 
import csv

#Creating a 3D axes by using the keyword projection = '3D'
 
fig = plt.figure( )
ax = fig.add_subplot(111, projection = '3d' )

X = list()
Y = list()
Z = list()
S1 = list()

with open ('testGreen.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)

    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z
        X.append(line[0])
        Y.append(line[1])
        Z.append(line[2])
        S1.append(line[3])
        
x = numpy.array(X)
y = numpy.array(Y)
z = numpy.array(Z)
s1 = numpy.array(S1)

#Copies the array and casts to float64
x = x.astype(float)
y= y.astype(float)
z = z.astype(float)
s1 = s1.astype(float)

#Simple moving average
#Starting at point 3

def movingaverage(values, window):
    weights = numpy.repeat(1.0, window)/window
    #valid only runs the sma's on valid points
    smas = numpy.convolve(values, weights, 'valid')
    return smas #returns a numpy array

#Plotting the line
xline =movingaverage(movingaverage(x, 10),5)
yline = movingaverage(movingaverage(y, 10),5)
zline = movingaverage(movingaverage(z, 10),5)
ax.plot3D(xline,yline,zline,'green')

#Trying to get a better line



ax.scatter (x, y, z, c = 'g', marker='D', s=s1)

ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('a axis')

plt.show ( )
