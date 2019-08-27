'''
- Reading in the points and ordering of points from the Principal Curve script
- Straightening the points of each channel 
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

def getPoints(filename):
    x = list(); y = list(); z = list()
    with open (filename, 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
            x.append(line[0]); y.append(line[1]); z.append(line[2])
    x = numpy.array(x, dtype = float); y = numpy.array(y, dtype = float); z = numpy.array(z, dtype = float)
    return (x, y, z);

def reorderPC(x, y, z, r):
    #For reordered points
    xs = list(); ys = list(); zs = list()
    xs = numpy.array(xs, dtype=float); ys = numpy.array(ys, dtype=float); zs = numpy.array(zs, dtype=float)
    
    #Concatenating numpy arrays
    data = numpy.concatenate((r[:, numpy.newaxis],
                       x[:, numpy.newaxis], 
                       y[:, numpy.newaxis], 
                       z[:, numpy.newaxis]), 
                      axis=1)
    
    #Sorting wrt x, y, z consecutively like excel
    sortedData = data[numpy.lexsort(numpy.transpose(data)[::-1])]
    
    #Separating the sorted data into numpy arrays
    sortedArray = numpy.hsplit(sortedData, 4)
    xs = numpy.concatenate(sortedArray[1], axis=0)
    ys = numpy.concatenate(sortedArray[2], axis=0)
    zs = numpy.concatenate(sortedArray[3], axis=0)
    numpy.savetxt("reorderCheck.csv", numpy.column_stack((xs, ys, zs)), delimiter=",", fmt='%s')
    return (xs, ys, zs);
'''
args - Points on the curve, x,y,z coordinates
returns - straightened points

Finds the distance between every pair of points on the curve
Finds the point on the curve that every coordinate is closest to
Finds a vector between the point on the curve to the data point
'''
def Straighten(LinePts,x,y,z):
    xPoints = list(); yPoints = list(); zPoints = list()
    xPoints = LinePts[0]; yPoints = LinePts[1]; zPoints = LinePts[2]

    #Finding the distance between every pair of points and storing the cumulative distances to get to that point 
    linePtsDistances = list()
    cumulativeDistance = 0 #Keeps track of cumulative distance till that point
    linePtsDistances.append(0) #Don't have to add anything for the first point 
    for i in range (0, len(xPoints)-2,1):
        a = (xPoints[i], yPoints[i], zPoints[i])
        b = (xPoints[i+1], yPoints[i+1], zPoints[i+1])
        dst = distance.euclidean(a,b)
        cumulativeDistance = cumulativeDistance + dst
        linePtsDistances.append(cumulativeDistance)

    #Creating a list of the index of the point on the line that each x,y,z is closets to
    closestPointPos = 0; i=0; j=0; indexOfClosestPoints = list()
    for i in range (0, len(x)-1): #looping through all the x,y,z coordinates
        mindst = 10000 #Setting an upper limit on the minimum distance 
        for j in range(0, len(xPoints)-1): #For every coordinate, looping through each point on the line
            a = (x[i], y[i], z[i])
            b = (xPoints[j], yPoints[j], zPoints[j])
            if (distance.euclidean(a,b)) < mindst:
                mindst = (distance.euclidean(a,b))
                closestPointPos = j
        indexOfClosestPoints.append(closestPointPos)
    print(indexOfClosestPoints)

    #Finding the vector from the coordinate to the nearest point on the line
    #Adding the additional x distance
    dx = list(); dy = list(); dz = list()
    for i in range(0, len(x)-1, 1):
        posOnLine = indexOfClosestPoints[i]
        dx.append((x[i] - xPoints[posOnLine])+linePtsDistances[posOnLine])
        dy.append(y[i] - yPoints[posOnLine])
        dz.append(z[i] - zPoints[posOnLine])
    return(dx, dy, dz)


#Reading in C1 and C2 points
pointsC1 = getPoints('CortexRemovedC1.csv')
pointsC2 = getPoints('CortexRemovedC2.csv')

#Reading in principal curve points
pointsPC = getPoints('fitpoints.csv')
#Reading in order of principal curve points
ranks = list()
with open ('fitorder.csv') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        ranks.append(line[0])
ranks = numpy.array(ranks, dtype = float)

#Reordering points of the principal curve
orderedpointsPC = reorderPC(pointsPC[0], pointsPC[1], pointsPC[2], ranks)

#Plotting the original points and the principal curve
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection = '3d')
ax.scatter (pointsC1[0], pointsC1[1], pointsC1[2], c = 'r', marker='o', s=1, linewidths=2)
ax.scatter (pointsC2[0], pointsC2[1], pointsC2[2], c = 'g', marker='o', s=1, linewidths=2)
ax.scatter(pointsPC[0], pointsPC[1], pointsPC[2], c = 'b', marker = '*', s=1, linewidths=2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
plt.show()
ax.grid(False)
fig.savefig('Output/%s/PrincipalCurve.png' % last_line)

#Straightening each of the channels separately and saving the straightened points
StraightenedPts1 = Straighten(orderedpointsPC, pointsC1[0], pointsC1[1], pointsC1[2])
StraightenedPts2 = Straighten(orderedpointsPC, pointsC2[0], pointsC2[1], pointsC2[2])
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
