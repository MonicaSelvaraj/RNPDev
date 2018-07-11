'''
Creating a scatter plot of both channels 
'''

#!/usr/bin/python

#using a library called matplotlib to make a 3D plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy 
import csv

#Creating a 3D axes by using the keyword projection = '3D'
fig = plt.figure( )
plt.style.use('dark_background')

#Variables for C1
X1 = list(); Y1 = list(); Z1 = list(); S1 = list()

#Variables for C2
X2 = list(); Y2 =  list(); Z2 = list(); S2 = list()

with open ('C1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z,S
        X1.append(line[0])
        Y1.append(line[1])
        Z1.append(line[2])
        S1.append(line[3])

with open ('C2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z,S
        X2.append(line[0])
        Y2.append(line[1])
        Z2.append(line[2])
        S2.append(line[3])

X1 = numpy.array(X1); Y1 = numpy.array(Y1); Z1 = numpy.array(Z1); S1 = numpy.array(S1)
X2 = numpy.array(X2); Y2 = numpy.array(Y2); Z2 = numpy.array(Z2); S2 = numpy.array(S2)
X1 = X1.astype(float); Y1 = Y1.astype(float); Z1 = Z1.astype(float); S1 = S1.astype(float)
X2 = X2.astype(float); Y2 = Y2.astype(float); Z2 = Z2.astype(float); S2 = S2.astype(float)

ax = fig.add_subplot(111, projection = '3d' )                            
ax.grid(False)
ax.set_xlabel ('x, axis'); ax.set_ylabel ('y axis'); ax.set_zlabel ('z axis')
ax.scatter (X1, Y1, Z1, c = 'r', marker='o', s=S1*5)
ax.scatter (X2, Y2, Z2, c = 'g', marker='o', s=S2*5)
plt.show ( )
