#Subplotting - red
#!/usr/bin/python

#Initial breakes wrt z - (0-260), (260-578), (578 - 658)
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy 
import csv


#Creating a 3D axes by using the keyword projection = '3D'

 #Setting up a figure
fig = plt.figure( )
#setting up the axes for the first subplot
ax = fig.add_subplot(111, projection = '3d')
#ax1, ax2, ax3, ax4 = fig.add_subplots(111, projection = '3d', sharex = True, sharey = True, sharez = True )
#fig, (ax1, ax2, ax3, ax = plt.subplots(4, sharex=True)

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

#For subplot (659 ..)
X4 = list()
Y4 = list()
Z4 = list()
S4 = list()

with open ('testRed.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)

    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z
        if (float(line[2])>0 and float(line[2])<260):
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
            X4.append(line[0])
            Y4.append(line[1])
            Z4.append(line[2])
            S4.append(line[3])

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

#for i in range (1,4,1):
#    xi= numpy.array(Xi)
#    yi = numpy.array(Yi)
#    zi = numpy.array(Zi)
#    si = numpy.array(Si)


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
  
#Simple moving average
#Starting at point 3

#def movingaverage(values, window):
#    weights = numpy.repeat(1.0, window)/window
#    #valid only runs the sma's on valid points
#    smas = numpy.convolve(values, weights, 'valid')
#    return smas #returns a numpy array

#Plotting the line
#xline =movingaverage(x, 5)
#yline = movingaverage(y, 5)
#zline = movingaverage(z, 5)
#ax.plot3D(xline,yline,zline,'red')

#Trying to get a better line



#f0r i in range (1,4,1):
ax.scatter (x1, y1, z1, c = 'r', marker='D', s=s1)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

ax = fig.add_subplot(112, projection = '3d')

ax.scatter (x2, y2, z2, c = 'r', marker='D', s=s2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

ax = fig.add_subplot(113,projection = '3d')

ax.scatter (x3, y3, z3, c = 'r', marker='D', s=s3)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

ax = fig.add_subplot(114, projection = '3d')

ax.scatter (x4, y4, z4, c = 'r', marker='D', s=s4)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

plt.show ( )
