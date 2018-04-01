'''
Subplots of different sizes
'''
#!/usr/bin/python

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv

fig = plt.figure( )

#0.27
X1 = list()
Y1 = list()
Z1 = list()

#0.36
X2 = list()
Y2 = list()
Z2 = list()

#0.45
X3 = list()
Y3 = list()
Z3 = list()

#0.54
X4= list()
Y4 = list()
Z4 = list()

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

X1 = numpy.array(X1)
Y1 = numpy.array(Y1)
Z1 = numpy.array(Z1)
X2 = numpy.array(X2)
Y2 = numpy.array(Y2)
Z2 = numpy.array(Z2)
X3 = numpy.array(X3)
Y3 = numpy.array(Y3)
Z3 = numpy.array(Z3)
X4 = numpy.array(X4)
Y4 = numpy.array(Y4)
Z4 = numpy.array(Z4)

X1 = X1.astype(float)
Y1= Y1.astype(float)
Z1= Z1.astype(float)
X2 = X2.astype(float)
Y2= Y2.astype(float)
Z2= Z2.astype(float)
X3 = X3.astype(float)
Y3= Y3.astype(float)
Z3= Z3.astype(float)
X4 = X4.astype(float)
Y4= Y4.astype(float)
Z4= Z4.astype(float)

#setting up the axes for the 0.27 subplot
ax = fig.add_subplot(2,3,2, projection = '3d')
ax.scatter (X1, Y1, Z1, c = 'r', marker='o', s=2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#setting up the axes for the 0.36 subplot
ax = fig.add_subplot(2,3,3, projection = '3d')
ax.scatter (X2, Y2, Z2, c = 'b', marker='o', s=4)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#setting up the axes for the 0.45 subplot
ax = fig.add_subplot(2,3,4, projection = '3d')
ax.scatter (X3, Y3, Z3, c = 'g', marker='o', s = 8)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#setting up the axes for the 0.27 subplot
ax = fig.add_subplot(2,3,5, projection = '3d')
ax.scatter (X4, Y4, Z4, c = 'y', marker='o', s=16)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#setting up the axes for the all sizes subplot
ax = fig.add_subplot(2,3,1, projection = '3d')
ax.scatter (X1, Y1, Z1, c = 'r', marker='o', s=1)
ax.scatter (X2, Y2, Z2, c = 'b', marker='o', s=2)
ax.scatter (X3, Y3, Z3, c = 'g', marker='o', s=3)
ax.scatter (X4, Y4, Z4, c = 'y', marker='o',s=4)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

plt.show( )


