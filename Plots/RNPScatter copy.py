#Activating conda - where matlibplot is installed
#source activate myenv

#!/usr/bin/python

#using a library called matplotlib to make a 3D plot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy 
import csv

#Creating a 3D axes by using the keyword projection = '3D'
 
fig = plt.figure( )
ax = fig.add_subplot(111, projection = '3d' )
plt.style.use('dark_background')

 #Data for three-dimensional scattered points
#zdata = 15 * np.random.random(100)
#xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
#ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
#ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

#Red
X = list()
Y = list()
Z = list()
S1 = list()

#Green
A = list()
B =  list()
C = list()
S2 = list()

#Reading in the csv file

#opening the csv file
# 'r' specifies that we want to read this file
#csv_reader is the name of the reader object that we have created 
with open ('testRed.csv', 'r') as csv_file:
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

x = x.astype(float)
y= y.astype(float)
z = z.astype(float)
s1 = s1.astype(float)
                            
#for x in range(len(X)):
#    print (X[x])
#for y in range(len(Y)):
#    print (Y[y])
#for y in range(len(Z)):
#    print (Z[y])
        
                             


#X = [1,2,3,4,5,6,7,8,9,10]
#Y = [2,4,6,7,4,8,4,1,12,15]
#Z = [0,0,0,0,0,0,0,0,0,0,0,0]

ax.scatter (x, y, z, c = 'r', marker='D', s=s1)
ax.grid(False)

ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('a axis')

with open ('testGreen.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)

    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z
        A.append(line[0])
        B.append(line[1])
        C.append(line[2])
        S2.append(line[3])
        
a = numpy.array(X)
b = numpy.array(Y)
c = numpy.array(Z)
s2 = numpy.array(S2)

a = a.astype(float)
b= b.astype(float)
c = c.astype(float)
s2 = s2.astype(float)

ax.scatter (a, b, c, c = 'g', marker='o', s=s2)
ax.grid(False)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('a axis')

# Data for a three-dimensional line
#zline = np.linspace(0, 15, 1000)
#xline = np.sin(zline)
#yline = np.cos(zline)
#ax.plot3D(xline, yline, zline, 'green')

#Determine polynomial of best fit

plt.show ( )
