#Whole plot + subplots - red with sma's depicted 
'''
====================
3D subplots of C1
====================

List of subplots:
Breaks wrt z
0-260
260 - 578
578 - 658
659 - 1037(Broken up wrt x=20)
'''

#!/usr/bin/python

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv

fig = plt.figure( )

#Simple moving averages of subplots
def movingaverage(values, window):
    weights = numpy.repeat(1.0, window)/window
    #valid only runs the sma's on valid points
    smas = numpy.convolve(values, weights, 'valid')
    return smas #returns a numpy array

#For the whole plot
X = list()
Y = list()
Z = list()
S = list()

#For subplot (0-260)
X1 = list()
Y1 = list()
Z1 = list()
S1 = list()

#For subplot (260-578)
X2 = list()
Y2 = list()
Z2 = list()
S2 = list()

#For subplot (578 - 658)
X3 = list()
Y3 = list()
Z3 = list()
S3 = list()

#For subplot (659 .., x<20)
X4 = list()
Y4 = list()
Z4 = list()
S4 = list()

#For subplot (659 .., x>20)
X5 = list()
Y5 = list()
Z5 = list()
S5 = list()

with open ('testGreen.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)

    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z
        X.append(line[0])
        Y.append(line[1])
        Z.append(line[2])
        S.append(line[3])

with open ('testGreen.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)

    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z, radius
        if (float(line[2])>=0 and float(line[2])<260):
            X1.append(line[0])
            Y1.append(line[1])
            Z1.append(line[2])
            S1.append(line[3])
        elif (float(line[2])>260 and float(line[2])<578):
            X2.append(line[0])
            Y2.append(line[1])
            Z2.append(line[2])
            S2.append(line[3])
        elif (float(line[2])>578 and  float(line[2])<658):
            X3.append(line[0])
            Y3.append(line[1])
            Z3.append(line[2])
            S3.append(line[3])
        else:
            if(float(line[0])<20):
                X4.append(line[0])
                Y4.append(line[1])
                Z4.append(line[2])
                S4.append(line[3])
            else:
                X5.append(line[0])
                Y5.append(line[1])
                Z5.append(line[2])
                S5.append(line[3])

#Converting lists to numpy arrays for matplotlib input
x= numpy.array(X)
y = numpy.array(Y)
z = numpy.array(Z)
s = numpy.array(S)
                
x1= numpy.array(X1)
y1 = numpy.array(Y1)
z1 = numpy.array(Z1)
s1 = numpy.array(S1)

x2= numpy.array(X2)
y2 = numpy.array(Y2)
z2 = numpy.array(Z2)
s2 = numpy.array(S2)

x3= numpy.array(X3)
y3 = numpy.array(Y3)
z3 = numpy.array(Z3)
s3 = numpy.array(S3)

x4= numpy.array(X4)
y4 = numpy.array(Y4)
z4 = numpy.array(Z4)
s4 = numpy.array(S4)

x5= numpy.array(X5)
y5 = numpy.array(Y5)
z5 = numpy.array(Z5)
s5 = numpy.array(S5)

#Converting numpy arrays to float numpy arrays
x = x.astype(float)
y = y.astype(float)
z = z.astype(float)
s = s.astype(float)

x1 = x1.astype(float)
y1= y1.astype(float)
z1 = z1.astype(float)
s1 = s1.astype(float)

x2 = x2.astype(float)
y2= y2.astype(float)
z2 = z2.astype(float)
s2 = s2.astype(float)

x3 = x3.astype(float)
y3= y3.astype(float)
z3 = z3.astype(float)
s3 = s3.astype(float)

x4 = x4.astype(float)
y4= y4.astype(float)
z4 = z4.astype(float)
s4 = s4.astype(float)

x5= x5.astype(float)
y5= y5.astype(float)
z5 = z5.astype(float)
s5 = s5.astype(float)

#setting up the axes for the subplot
ax = fig.add_subplot(2,3,1, projection = '3d')
ax.scatter (x, y, z, c = 'g', marker='D', s=s)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
#Plotting the line
xline =movingaverage(movingaverage(x, 10),5)
yline = movingaverage(movingaverage(y, 10),5)
zline = movingaverage(movingaverage(z, 10),5)
ax.plot3D(xline,yline,zline,'green')

#setting up the axes for the sixth subplot
ax = fig.add_subplot(2,3,6, projection = '3d')
ax.scatter (x1, y1, z1, c = 'g', marker='D', s=s1)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
#Plotting the line
x1line =movingaverage(movingaverage(x1, 10),5)
y1line = movingaverage(movingaverage(y1, 10),5)
z1line = movingaverage(movingaverage(z1, 10),5)
ax.plot3D(x1line,y1line,z1line,'green')

#setting up the axes for the fifth subplot
ax = fig.add_subplot(2,3,5, projection = '3d')

ax.scatter (x2, y2, z2, c = 'g', marker='D', s=s2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
x2line =movingaverage(movingaverage(x2, 10),5)
y2line = movingaverage(movingaverage(y2, 10),5)
z2line = movingaverage(movingaverage(z2, 10),5)
ax.plot3D(x2line,y2line,z2line,'green')

#setting up the axes for the fourth subplot
ax = fig.add_subplot(2,3,4,projection = '3d')

ax.scatter (x3, y3, z3, c = 'g', marker='D', s=s3)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
x3line =movingaverage(movingaverage(x3, 10),5)
y3line = movingaverage(movingaverage(y3, 10),5)
z3line = movingaverage(movingaverage(z3, 10),5)
ax.plot3D(x3line,y3line,z3line,'green')

#setting up the axes for the second subplot
ax = fig.add_subplot(2,3,2, projection = '3d')

ax.scatter (x4, y4, z4, c = 'g', marker='D', s=s4)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
x4line =movingaverage(movingaverage(x4, 10),5)
y4line = movingaverage(movingaverage(y4, 10),5)
z4line = movingaverage(movingaverage(z4, 10),5)
ax.plot3D(x4line,y4line,z4line,'green')

#setting up the axes for the third subplot
ax = fig.add_subplot(2,3,3, projection = '3d')

ax.scatter (x5, y5, z5, c = 'g', marker='D', s=s5)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
x5line =movingaverage(movingaverage(x5, 10),5)
y5line = movingaverage(movingaverage(y5, 10),5)
z5line = movingaverage(movingaverage(z5, 10),5)
ax.plot3D(x5line,y5line,z5line,'green')

plt.show ( )
