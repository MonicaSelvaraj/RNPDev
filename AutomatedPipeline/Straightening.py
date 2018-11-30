'''
- Drawing a Minimum Spanning Tree through the points
- Performing a breadth first search(bfs) through the minimum spanning tree
- Re-ordering x,y,z coordinates wrt the bfs ranks
- Determining the simple moving average through the ranked coordinates
- Taking 10-100 points on the moving average, and drawing a bezier curve through the points
'''

#!/usr/bin/python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv
import scipy
from scipy.spatial import distance
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import itertools
from scipy.misc import comb

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
    with open ('CortexRemovedC1.csv', 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
            x.append(line[0]); x1.append(line[0])
            y.append(line[1]); y1.append(line[1])
            z.append(line[2]); z1.append(line[2])
    with open ('CortexRemovedC2.csv', 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
            x.append(line[0]); x2.append(line[0])
            y.append(line[1]); y2.append(line[1])
            z.append(line[2]); z2.append(line[2])

    x = numpy.array(x, dtype = float); y = numpy.array(y, dtype = float); z = numpy.array(z, dtype = float)
    x1 = numpy.array(x1, dtype = float); y1 = numpy.array(y1, dtype = float); z1 = numpy.array(z1, dtype = float)
    x2 = numpy.array(x2, dtype = float); y2 = numpy.array(y2, dtype = float); z2 = numpy.array(z2, dtype = float)
    return (x, y, z, x1, y1, z1, x2, y2, z2);


'''
args: numpy arrays with x,y,z coorindates
returns: ---

Generates a 3D scatter plot
'''
def ScatterPlot(x, y, z):
    fig = plt.figure( )
    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.grid(False)
    ax.scatter (x, y, z, c = 'r', marker='o', s=1, linewidths=2)
    ax.set_xlabel ('x, axis')
    ax.set_ylabel ('y axis')
    ax.set_zlabel ('z axis')
 #   plt.show()
    return();

'''
args: numpy arrays with x,y,z coorindates
returns: minimum spanning tree 

Iterating through the X,Y,Z coordinates and creating a row, column and distance matrix
Using those matrices to create a csr matrix
Using the csr matrix to make a minimum spanning tree
'''
def minSpanningTree(X,Y,Z):
    row = list(); col = list(); dist = list()
    i = -1 #i, j keeps track of position 
    for x1,y1,z1 in zip(X,Y,Z):
        i = i+1; j = -1
        for x2,y2,z2 in zip (X,Y,Z):
            j = j+1
            if (j>i):
                col.append(i)
                a = (x1,y1,z1)
                b = (x2,y2,z2)
                row.append(j)
                dist.append(distance.euclidean(a,b))
    row= numpy.array(row); col = numpy.array(col); dist = numpy.array(dist)
    sparseMatrix = csr_matrix((dist, (row, col)), shape=(len(row), len(col))).toarray()
    MST = minimum_spanning_tree(sparseMatrix)
    return (MST);


'''
args: minimum spanning tree, x,y,z coordinates
returns: --

Converting the output csr matrix to a coo matrix
Iterating through the coo matrix to draw the minimum spanning tree
'''
def drawMinimumSpanningTree(MST, X, Y, Z):
    A = list(); B = list() #These lists store which points need to be connected A-row, B-col
    cx = scipy.sparse.coo_matrix(MST)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        A.append(i)
        B.append(j)
    #A, B have the indices of the points that need to be connected
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.scatter (X1, Y1, Z1, c = 'r', marker='o', s=1, linewidths=2)
    ax.scatter (X2, Y2, Z2, c = 'g', marker='o', s=1, linewidths=2)
    ax.set_xlabel ('x, axis')
    ax.set_ylabel ('y axis')
    ax.set_zlabel ('z axis')
    for a, b in zip(A, B):
        ax.plot3D([X[a], X[b]], [Y[a], Y[b]], [Z[a], Z[b]], c='b')
 #   plt.show()
    fig.savefig('Output/%s/MinimumSpanningTree.png' % last_line)
    return();

def polyReg(X,Y,Z):

    #Making the principal componenets the axes of the coordinates
    points = numpy.concatenate((X[:, numpy.newaxis], 
                       Y[:, numpy.newaxis], Z[:, numpy.newaxis]), axis = 1)
    center = points.mean(axis = 0)
    centered = points-center
    centeredT = numpy.transpose(centered)
    constant = 1/(len(X)-1)
    covarianceM = constant*numpy.matmul(centeredT, centered)
    w, v = numpy.linalg.eig(covarianceM)
    Pc1 = v[:,0]
    Pc2 = v[:,1]
    Pc3 = v[:, 2]
    #New points
    C1s = numpy.dot(centered , Pc1) 
    C2s = numpy.dot(centered , Pc2)
    C3s = numpy.dot(centered, Pc3)

    #Fitting a polynomial to new coordinates 
    yP = numpy.polyfit(C1s, C2s, 2)
    zP = numpy.polyfit(C1s, C3s, 2)

    C1s.sort()
    fitY = list(); fitZ = list()
    #Generating y and z fit points
    for c in C1s:
        fitY.append((yP[0]*(c**2)) +(yP[1] *c) + yP[2])
        fitZ.append((zP[0]*(c**2)) +(zP[1] *c) + zP[2])
    return(C1s, fitY, fitZ)
    
'''
args: csr graph of minimum spanning tree, original coordinates
returns: ranked coordinates

Converts csr graph to networkx graph
Performs breadth first search of minimum spanning tree
Stores the path from origin to end in a list and creates a new list of reordered points
'''
def RankPoints(MST, X, Y, Z):
    G = nx.from_scipy_sparse_matrix(MST)
    bfsDict = list(nx.bfs_successors(G,1))
    #Traversing through the dictionary and storing the values form each key in a list
    rank = list(); mergedRanks = list(); rX = list(); rY = list(); rZ = list()
    for i, j in bfsDict:
        rank.append(j)
    for k in rank: #Making an array or arrays a single list
        mergedRanks = mergedRanks + k
    #mergedRanks contains the indices of the ranks
    for r in mergedRanks:
        rX.append(X[r]); rY.append(Y[r]); rZ.append(Z[r])
    rX= numpy.array(rX, dtype = float); rY = numpy.array(rY, dtype = float);rZ= numpy.array(rZ, dtype = float)
    numpy.savetxt("RankedPoints.csv", numpy.column_stack((rX, rY, rZ)), delimiter=",", fmt='%s')
    return (rX, rY, rZ)

def movingaverage(values, window):
    weights = numpy.repeat(1.0, window)/window
    #valid only runs the sma's on valid points
    smas = numpy.convolve(values, weights, 'valid')
    return smas #returns a numpy array

'''
args: ranked x,y,z coordinates
returns: sma in x,y,z direction
'''
def drawMovingAverage(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.scatter (X1, Y1, Z1, c = 'r', marker='o', s=1, linewidths=2)
    ax.scatter (X2, Y2, Z2, c = 'g', marker='o', s=1, linewidths=2)
    ax.set_xlabel ('x, axis')
    ax.set_ylabel ('y axis')
    ax.set_zlabel ('z axis')
    xline =movingaverage(x, 40)
    yline =movingaverage(y, 40)
    zline =movingaverage(z, 40)
    ax.plot3D(xline,yline,zline,'blue')
 #  plt.show()
    fig.savefig('Output/%s/MovingAverage.png' % last_line)
    return(xline, yline, zline);

'''
args: points from the simple moving average
returns: sample points to use to draw the bezier curve

Picks the first, last and every 10th point in between,  as sample points to
draw the bezier curve
'''
def BezierInput(smaPoints):
    last = len(smaPoints[0]) #Makes sure we include the last point in the curve
    xB = list(); yB = list(); zB = list()
    interval = 0 #Keeps track of which points to add
    for i,j,k in zip(smaPoints[0], smaPoints[1], smaPoints[2]):
        interval = interval +1
        if (interval == 1):
            xB.append(i); yB.append(j);  zB.append(k)
        if (interval%1 == 0 or interval == last -1): #picking every 10 points on the sma
            xB.append(i); yB.append(j);  zB.append(k)
    xB = numpy.array(xB); yB = numpy.array(yB); zB = numpy.array(zB)
    xB = xB.astype(float); yB= yB.astype(float); zB= zB.astype(float)
    bezierPoints = numpy.concatenate((xB[:, numpy.newaxis], 
                       yB[:, numpy.newaxis], 
                       zB[:, numpy.newaxis]), 
                      axis=1)
    return (bezierPoints);

'''
The Bernstein polynomial of n, i as a function of t
'''
def bernstein_poly(i, n, t):
    return (comb(n, i) * ( t**(n-i) ) * (1 - t)**i);

'''
Given a set of control points, return the
bezier curve defined by the control points.

points should be a list of lists, or list of tuples
such as [ [1,1],
               [2,3], 
               [4,5], ..[Xn, Yn] ]
nTimes is the number of time steps, defaults to 1000

See http://processingjs.nihongoresources.com/bezierinfo/
'''
def bezier_curve(points, nTimes =1000):
    nPoints = len(points)
    xPoints = numpy.array([p[0] for p in points])
    yPoints = numpy.array([p[1] for p in points])
    zPoints = numpy.array([p[2] for p in points])

    t = numpy.linspace(0.0, 1.0, nTimes)

    polynomial_array = numpy.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
    
    xvals = numpy.dot(xPoints, polynomial_array)
    yvals = numpy.dot(yPoints, polynomial_array)
    zvals = numpy.dot(zPoints, polynomial_array)

    return (xvals, yvals,zvals);
'''
Finds the vector between two points a,b
a = [ax,ay,az]
b = [bx,by,bz]
'''
def Vector(ax,ay,az,bx,by,bz):
    dx = (bx - ax)
    dy = (by - ay)
    dz = (bz - az)
    return ()

'''
args - Points on the bezier curve, ranked x,y,z coordinates
returns - straightened points

Finds the distance of every point on the line from the 0th point
Finds the point on the line that every coordinate is closest to
Finds a vector between the two points, and adds the extra x distance 
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
    #print (len(linePtsDistances))

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
    #print(len(indexOfClosestPoints))

    #Finding the vector from the coordinate to the nearest point on the line
    #Adding the additional x distance
    dx = list(); dy = list(); dz = list()
    for i in range(0, len(x)-1, 1):
        posOnLine = indexOfClosestPoints[i]
        dx.append((x[i] - xPoints[posOnLine])+linePtsDistances[posOnLine])
        dy.append(y[i] - yPoints[posOnLine])
        dz.append(z[i] - zPoints[posOnLine])
    #print(dx); print(dy); print(dz)
    return(dx, dy, dz)

#Reading and storing the input
In = readAndStoreInput()
X = In[0]; Y = In[1]; Z = In[2];X1 = In[3]; Y1 = In[4]; Z1 = In[5];X2 = In[6]; Y2 = In[7]; Z2 = In[8]
#Producing a scatter plot of the original data
#ScatterPlot(X, Y, Z)
#Computing the minimum spanning tree for the data
#minimumSpanningTree= minSpanningTree(X, Y, Z)
#Drawing the minimum spanning tree through the data
#drawMinimumSpanningTree(minimumSpanningTree, X, Y, Z)
#Re-ordering the x,y,z coordinates to give the data a direction
#newPoints = RankPoints(minimumSpanningTree, X, Y, Z); bezierPointsLength = len(newPoints[0])
#Drawing the simple moving average through the ranked points
#smaPoints = drawMovingAverage(newPoints[0], newPoints[1], newPoints[2])
#Picking sample points as input to draw the bezier curve
#bezierPoints = BezierInput(smaPoints)
#Drawing the bezier curve through the data
#bezierLine = bezier_curve(bezierPoints)
poly = polyReg(X,Y,Z)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection = '3d')
ax.scatter (X1, Y1, Z1, c = 'r', marker='o', s=1, linewidths=2)
ax.scatter (X2, Y2, Z2, c = 'g', marker='o', s=1, linewidths=2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
ax.plot3D(poly[0], poly[1], poly[2],'blue')
#plt.show()
fig.savefig('Output/%s/Polynomial.png' % last_line)

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
#plt.show()
fig.savefig('Output/%s/Straightened.png' % last_line)

#Writing straightened points to a file
StraightenedPts1= numpy.array(StraightenedPts1, dtype = float)
StraightenedPts2= numpy.array(StraightenedPts2, dtype = float)
numpy.savetxt("StraightenedC1.csv", numpy.column_stack((StraightenedPts1[0], StraightenedPts1[1], StraightenedPts1[2])), delimiter=",", fmt='%s')
numpy.savetxt("StraightenedC2.csv", numpy.column_stack((StraightenedPts2[0], StraightenedPts2[1], StraightenedPts2[2])), delimiter=",", fmt='%s')



