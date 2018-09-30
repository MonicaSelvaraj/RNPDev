'''
Naive approach -
- We know that the cortex is always going to be aligned with one of the axes
- Draw a plane in the 8 reasonable positions for the cortex
- Check if there are lots of points near the plane
- If there is, remove the points 
'''

'''
#!/usr/bin/python

import numpy 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

fig = plt.figure( )
plt.style.use('dark_background')

xx,yy = numpy.meshgrid(range(60), range(60))
zz = xx*0

ax = fig.add_subplot(1,1,1, projection = '3d')
plt.hold(True)
ax.plot_surface(xx,yy,zz)


#Generating a scatter plot of the points after clustering
#Variables for deconvolutedC1
dX1 = list(); dY1 = list(); dZ1 = list()
#Variables for deconvolutedC2
dX2 = list(); dY2 =  list(); dZ2 = list()

with open ('ClusteredC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z
        dX1.append(line[0])
        dY1.append(line[1])
        dZ1.append(line[2])

with open ('ClusteredC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z
        dX2.append(line[0])
        dY2.append(line[1])
        dZ2.append(line[2])

dX1 = numpy.array(dX1); dY1 = numpy.array(dY1); dZ1 = numpy.array(dZ1)
dX2 = numpy.array(dX2); dY2 = numpy.array(dY2); dZ2 = numpy.array(dZ2)
dX1 = dX1.astype(float); dY1 = dY1.astype(float); dZ1 = dZ1.astype(float) 
dX2 = dX2.astype(float); dY2 = dY2.astype(float); dZ2 = dZ2.astype(float)

    
ax.grid(False)
ax.scatter (dX1, dY1, dZ1, c = 'r', marker='o', s=1, linewidths=2)
#ax.scatter (dX2, dY2, dZ2, c = 'g', marker='o', s=1, linewidths=2)
plt.show()
'''

'''
Another approach
1. Check the first five x's, y's and z's - project the points into 2D plane 
2. Find the convex hull
3. Find the surface area of the plane covered by the conved hull
4. If the points are very spread out 

'''

 


    
    


