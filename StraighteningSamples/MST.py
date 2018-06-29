'''
Straightening samples
- Draw a minimum spanning tree through points
- Use moving least squares to draw a path through the points 
'''

#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv
import scipy
from scipy.spatial import distance
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.csgraph import minimum_spanning_tree

fig = plt.figure( )
plt.style.use('dark_background')

#Plot of input
X = list(); Y = list(); Z = list()

with open ('Input.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        X.append(line[0])
        Y.append(line[1])
        Z.append(line[2])

X = numpy.array(X); Y = numpy.array(Y); Z = numpy.array(Z)
X = X.astype(float); Y = Y.astype(float); Z = Z.astype(float)

#Generating a plot of the original points
ax = fig.add_subplot(1,1,1, projection = '3d')
ax.grid(False)
ax.scatter (X, Y, Z, c = 'r', marker='o', s=1, linewidths=2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
plt.show()


'''
This function goes through the X,Y,Z coordinates and creates a dictionary of the following form:
  point: [(next point, distance between the two points), ... for every point]
{ 0: [(1, 32.67118780822281), (2, 29.296469795872532) ... ]
  1: [(2, 6.658432275467342) ... ]
  .
  .
  .
  208: []
'''
def createDictionary(X, Y, Z):
    D = dict()
    i = -1 #i.j keeps track of position - which is the key in the dictionairy
    for x1,y1,z1 in zip(X,Y,Z):
        i = i+1; j = -1; pos = list(); distances = list()
        for x2,y2,z2 in zip (X,Y,Z):
            j = j+1
            if (j>i):
                a = (x1,y1,z1)
                b = (x2,y2,z2)
                pos.append(j)
                distances.append(distance.euclidean(a,b))
        weightedVertices = list(zip(pos,distances))
        D.update({i:weightedVertices})
    return (D);

'''
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
    return (MST);
    


def drawMinimumSpanningTree(MST):
    cx = scipy.sparse.coo_matrix(MST)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        print ("(%d, %d), %s" % (i , j, v))
    return();

minimumSpanningTree= minSpanningTree(X, Y, Z)
drawMinimumSpanningTree(minimumSpanningTree)
    
    




