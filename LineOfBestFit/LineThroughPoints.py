'''
Reading in De-convoluted points
Drawing the first principal component through the points 
'''

#!/usr/bin/python
import sys, os 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.stats
import statistics
import numpy 
import csv
import math
from matplotlib import style

sys.setrecursionlimit(10000)
fig = plt.figure( )
plt.style.use('dark_background')

#Original data 
X1 = list(); Y1 = list(); Z1 = list(); S1 = list()#C1 - red
X2 = list(); Y2 = list(); Z2 = list(); S2 = list()#C2 - green

#Reading in the data 
with open ('deconvolutedC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        #each line has X,Y,Z, radius
            X1.append(line[0])
            Y1.append(line[1])
            Z1.append(line[2])
            
with open ('deconvolutedC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        #each line has X,Y,Z, radius
            X2.append(line[0])
            Y2.append(line[1])
            Z2.append(line[2])
       

X1 = numpy.array(X1); Y1 = numpy.array(Y1); Z1 = numpy.array(Z1); S1 = numpy.array(S1)
X2 = numpy.array(X2); Y2 = numpy.array(Y2); Z2 = numpy.array(Z2); S2 = numpy.array(S2)
X1 = X1.astype(float); Y1= Y1.astype(float); Z1= Z1.astype(float); S1= S1.astype(float)
X2 = X2.astype(float); Y2= Y2.astype(float); Z2 = Z2.astype(float); S2= S2.astype(float)

#Generating a plot of the original points
ax = fig.add_subplot(1,1,1, projection = '3d')
ax.grid(False)
red = ax.scatter (X1, Y1, Z1, c = 'r', marker='o',linewidths=2)
green = ax.scatter (X2, Y2, Z2, c = 'g', marker='o',linewidths=2)
#ax.set_title('unclustered')
ax.legend((red, green),
           ('vasa', 'dazl'),
           scatterpoints=1,
           loc='best',
           ncol=1,
           fontsize=8)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

plt.show()




    
    
    


