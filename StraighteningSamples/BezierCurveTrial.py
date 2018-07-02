'''
- Drawing a Minimum Spanning Tree through the points
- Removing outliers
- Use moving least squares to draw a path through the points
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

plt.style.use('dark_background')


'''
args: --
returns: numpy arrays with x,y,z coorindates 
'''
def readAndStoreInput( ):
    x = list(); y = list(); z = list()
    with open ('Input.csv', 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
            x.append(line[0])
            y.append(line[1])
            z.append(line[2])

    x = numpy.array(x); y = numpy.array(y); z = numpy.array(z)
    x = x.astype(float); y = y.astype(float); z = z.astype(float)
    return (x, y, z);


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
    plt.show()
    return();

'''
args: numpy arrays with x,y,z coorindates
returns: minimum spanning tree 

Iterating through the X,Y,Z coordinates and creating a row column and distance matrix
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
    print (MST)
    return (MST);


'''
args: row and column array of points that need to be joined
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
    ax.scatter (X, Y, Z, c = 'r', marker='o', s=1, linewidths=2)
    ax.set_xlabel ('x, axis')
    ax.set_ylabel ('y axis')
    ax.set_zlabel ('z axis')
    for a, b in zip(A, B):
        ax.plot3D([X[a], X[b]], [Y[a], Y[b]], [Z[a], Z[b]], c='b')
    plt.show()
    return();

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
    for k in rank:
        mergedRanks = mergedRanks + k
    #mergedRanks contains the indices of the ranks
    for r in mergedRanks:
        rX.append(X[r]); rY.append(Y[r]); rZ.append(Z[r])
    return (rX, rY, rZ)


def movingaverage(values, window):
    weights = numpy.repeat(1.0, window)/window
    #valid only runs the sma's on valid points
    smas = numpy.convolve(values, weights, 'valid')
    return smas #returns a numpy array

def drawMovingAverage(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection = '3d')
    ax.scatter (X, Y, Z, c = 'r', marker='o', s=1, linewidths=2)
    ax.set_xlabel ('x, axis')
    ax.set_ylabel ('y axis')
    ax.set_zlabel ('z axis')
    xline =movingaverage(x, 3)
    yline =movingaverage(y, 3)
    zline =movingaverage(z, 3)
    ax.plot3D(xline,yline,zline,'green')
    plt.show()
    return(xline, yline, zline);
    

In = readAndStoreInput(); X = In[0]; Y = In[1]; Z = In[2] #X,Y,Z has the original data
ScatterPlot(X, Y, Z)
minimumSpanningTree= minSpanningTree(X, Y, Z)
drawMinimumSpanningTree(minimumSpanningTree, X, Y, Z)
newPoints = RankPoints(minimumSpanningTree, X, Y, Z)
smaPoints = drawMovingAverage(newPoints[0], newPoints[1], newPoints[2])

#Bezier Curve implementation
last = len(smaPoints[0])
xB = list(); yB = list(); zB = list()
interval = 1
for i,j,k in zip(smaPoints[0], smaPoints[1], smaPoints[2]):
    interval = interval +1
    if (interval == 1):
        xB.append(i); yB.append(j);  zB.append(k)
    if (interval%10 == 0 or interval == last -1):
        xB.append(i); yB.append(j);  zB.append(k)

print(xB)
print(yB)
print(zB)

xB = numpy.array(xB); yB = numpy.array(yB); zB = numpy.array(zB)
xB = xB.astype(float); yB= yB.astype(float); zB= zB.astype(float)

bezierPoints = numpy.concatenate((xB[:, numpy.newaxis], 
                       yB[:, numpy.newaxis], 
                       zB[:, numpy.newaxis]), 
                      axis=1)
print(bezierPoints)
        

from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = numpy.array([p[0] for p in points])
    yPoints = numpy.array([p[1] for p in points])
    zPoints = numpy.array([p[2] for p in points])

    t = numpy.linspace(0.0, 1.0, nTimes)

    polynomial_array = numpy.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = numpy.dot(xPoints, polynomial_array)
    yvals = numpy.dot(yPoints, polynomial_array)
    zvals = numpy.dot(zPoints, polynomial_array)

    return xvals, yvals,zvals

bezierLine = bezier_curve(bezierPoints)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection = '3d')
ax.scatter (X, Y, Z, c = 'r', marker='o', s=1, linewidths=2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
ax.plot3D(bezierLine[0], bezierLine[1], bezierLine[2],'green')
plt.show()


