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
returns: minimum spanning tree in the form of 3 arrays - row, column, distance between pair of points

Iterating through the X,Y,Z coordinates and creating a row column and distance matrix
Using those matrices to create a csr matrix
Using the csr matrix to make a minimum spanning tree
Converting the output csr matrix to a coo matrix
Iterating through the coo matrix to return row, column, and distance arrays
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
    
    A = list(); B = list(); C = list() #A-row, B-col, C-distance
    cx = scipy.sparse.coo_matrix(MST)
    for i, j, k in zip(cx.row, cx.col, cx.data):
        A.append(i)
        B.append(j)
        C.append(k)
    return (A, B, C);

'''
args: row and column array of points that need to be joined
returns: --

Draws the minimum spanning tree
'''
def drawMinimumSpanningTree(A, B):
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

def removeOutliers():





In = readAndStoreInput(); X = In[0]; Y = In[1]; Z = In[2]
ScatterPlot(X, Y, Z)
minimumSpanningTree= minSpanningTree(X, Y, Z)
drawMinimumSpanningTree(minimumSpanningTree[0], minimumSpanningTree[1])
    




