'''
- Drawing a third degree polynomial through the whole aggregate - oriented in the z now instead of x 
- Straightening the aggregate 
'''

#!/usr/bin/python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv
from scipy.spatial import distance

print("Straightening")
f_read = open("FileNames.txt", "r")
last_line = f_read.readlines()[-1]
last_line = last_line[:-1] #Ignoring newline character
f_read.close()
plt.style.use('dark_background')

'''
args: --
returns: numpy arrays with x,y,z coorindates

Reads in deconvoluted points and stores x,y,z coordinates in numpy arrays
'''
def readAndStoreInput( ):
    x = list(); y = list(); z = list()
    x1 = list(); y1 = list(); z1 = list(); x2 = list(); y2 = list(); z2 = list()
    with open ('OrientedC1.csv', 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
            x.append(line[0]); x1.append(line[0])
            y.append(line[1]); y1.append(line[1])
            z.append(line[2]); z1.append(line[2])
    with open ('OrientedC2.csv', 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
            x.append(line[0]); x2.append(line[0])
            y.append(line[1]); y2.append(line[1])
            z.append(line[2]); z2.append(line[2])

    x = numpy.array(x, dtype = float); y = numpy.array(y, dtype = float); z = numpy.array(z, dtype = float)
    x1 = numpy.array(x1, dtype = float); y1 = numpy.array(y1, dtype = float); z1 = numpy.array(z1, dtype = float)
    x2 = numpy.array(x2, dtype = float); y2 = numpy.array(y2, dtype = float); z2 = numpy.array(z2, dtype = float)
    return (x, y, z, x1, y1, z1, x2, y2, z2);

def polyReg(X,Y,Z): 
    #Fitting a polynomial to new coordinates
    xP = numpy.polyfit(Z, X, 3)
    yP = numpy.polyfit(Z, Y, 3)
    
    Z.sort()
    fitX = list(); fitY = list()
    #Generating y and z fit points
    for z in Z:
        fitX.append((xP[0]*(z**3)) + (xP[1]*(z**2)) +(xP[2] *z) + xP[3])
        fitY.append((yP[0]*(z**3)) + (yP[1]*(z**2)) +(yP[2] *z) + yP[3])

    #Plotting the polynomial 
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.scatter (X, Y, Z, c = 'y', marker='o', s=1, linewidths=3)
    ax.set_xlabel ('x, axis')
    ax.set_ylabel ('y axis')
    ax.set_zlabel ('z axis')
    ax.plot3D(fitX, fitY, Z,'blue')
    plt.show()
    ax.grid(False)
    fig.savefig('Output/%s/Polynomial.png' % last_line)
    return(fitX, fitY, Z)

'''
args - Points on the curve, x,y,z coordinates
returns - straightened points

Finds the distance of every point on the line from the 0th point
Finds the point on the line that every coordinate is closest to
Finds a vector between the two points, and adds the extra z distance 
'''
def Straighten(LinePts,x,y,z):
    xPoints = list(); yPoints = list(); zPoints = list()
    xPoints = LinePts[0]; yPoints = LinePts[1]; zPoints = LinePts[2]

    #Finding the distance of every point from the 0th point
    linePtsDistances = list()
    for i in range (0, len(xPoints)-1,1):
        a = (xPoints[0], yPoints[0], zPoints[0])
        b = (xPoints[i], yPoints[i], zPoints[i])
        linePtsDistances.append(distance.euclidean(a,b))

    #Creating a list of the index of the point on the line that each x,y,z is closets to
    closestPointPos = 0; i=0; j=0; indexOfClosestPoints = list()
    for i in range (0, len(x)-1): #looping through all the x,y,z coordinates
        mindst = 1000 #Setting an upper limit on the minimum distance 
        for j in range(0, len(xPoints)-1): #For every coordinate, looping through each point on the line
            a = (x[i], y[i], z[i])
            b = (xPoints[j], yPoints[j], zPoints[j])
            if (distance.euclidean(a,b)) < mindst:
                mindst = (distance.euclidean(a,b))
                closestPointPos = j
        indexOfClosestPoints.append(closestPointPos)

    #Finding the vector from the coordinate to the nearest point on the line
    #Adding the additional x distance
    dx = list(); dy = list(); dz = list()
    for i in range(0, len(x)-1, 1):
        posOnLine = indexOfClosestPoints[i]
        dx.append(x[i] - xPoints[posOnLine])
        dy.append(y[i] - yPoints[posOnLine])
        dz.append((z[i] - zPoints[posOnLine])+linePtsDistances[posOnLine])
    return(dx, dy, dz)

#Reading and storing the input
In = readAndStoreInput()
X = In[0]; Y = In[1]; Z = In[2];X1 = In[3]; Y1 = In[4]; Z1 = In[5];X2 = In[6]; Y2 = In[7]; Z2 = In[8]
#Drawing a polynomial through the points
poly = polyReg(X,Y,Z)
#Straightening points
StraightenedPts1 = Straighten(poly, X1, Y1, Z1)
StraightenedPts2 = Straighten(poly, X2, Y2, Z2)
#Plotting the straightened points
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection = '3d')
ax.scatter (StraightenedPts1[0],StraightenedPts1[1],StraightenedPts1[2], c = 'r', marker='o', s=1, linewidths=2)
ax.scatter (StraightenedPts2[0],StraightenedPts2[1],StraightenedPts2[2], c = 'g', marker='o', s=1, linewidths=2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
plt.show()
ax.grid(False)
fig.savefig('Output/%s/Straightened.png' % last_line)
#Writing straightened points to a file
StraightenedPts1= numpy.array(StraightenedPts1, dtype = float)
StraightenedPts2= numpy.array(StraightenedPts2, dtype = float)
numpy.savetxt("StraightenedC1.csv", numpy.column_stack((StraightenedPts1[0], StraightenedPts1[1], StraightenedPts1[2])), delimiter=",", fmt='%s')
numpy.savetxt("StraightenedC2.csv", numpy.column_stack((StraightenedPts2[0], StraightenedPts2[1], StraightenedPts2[2])), delimiter=",", fmt='%s')



