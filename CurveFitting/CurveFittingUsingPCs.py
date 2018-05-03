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
from scipy.optimize import curve_fit

sys.setrecursionlimit(10000)
fig = plt.figure( )
#plt.style.use('dark_background')

#Original data 
X1 = list(); Y1 = list(); Z1 = list(); S1 = list()#C1 - red
X2 = list(); Y2 = list(); Z2 = list(); S2 = list()#C2 - green

#Reading in the data 
with open ('deconvolutedAC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        #each line has X,Y,Z, radius
            if (float(line[1]) > 10):
                    X1.append(line[0])
                    Y1.append(line[1])
                    Z1.append(line[2])
            
with open ('deconvolutedAC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        #each line has X,Y,Z, radius
            if (float(line[1]) > 15): 
                    X2.append(line[0])
                    Y2.append(line[1])
                    Z2.append(line[2])
       

X1 = numpy.array(X1); Y1 = numpy.array(Y1); Z1 = numpy.array(Z1); S1 = numpy.array(S1)
X2 = numpy.array(X2); Y2 = numpy.array(Y2); Z2 = numpy.array(Z2); S2 = numpy.array(S2)
X1 = X1.astype(float); Y1= Y1.astype(float); Z1= Z1.astype(float); S1= S1.astype(float)
X2 = X2.astype(float); Y2= Y2.astype(float); Z2 = Z2.astype(float); S2= S2.astype(float)

'''
#Generating a plot of the original points - both vasa and dazl
ax = fig.add_subplot(1,3,1, projection = '3d')
ax.grid(False)
red = ax.scatter (X1, Y1, Z1, c = 'r', marker='o',linewidths=2)
green = ax.scatter (X2, Y2, Z2, c = 'g', marker='o',linewidths=2)
ax.set_title('GP-RNP')
ax.legend((red, green),
           ('vasa', 'dazl'),
           scatterpoints=1,
           loc='best',
           ncol=1,
           fontsize=8)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
'''

def PC1(X,Y,Z):
        data = numpy.concatenate((X[:, numpy.newaxis], 
                       Y[:, numpy.newaxis], 
                       Z[:, numpy.newaxis]), 
                      axis=1)
       # print(data)
        datamean = data.mean(axis=0)
        print (datamean) #This is going to be the center of my helix
        ax.scatter (datamean[0], datamean[1], datamean[2], c = 'b', marker='o')
        uu, dd, vv = numpy.linalg.svd(data - datamean)
        #Taking the variation in the z dimension, because this is the dimension of PC1
        #Linear algebra - figure out what exactly is happening in terms of dimensional collapsation
        linepts = vv[0] * numpy.mgrid[-20:20:2j][:, numpy.newaxis]
        print (vv[0])
        print (vv[1])
        print (vv[2])
        #Moving the line to be in between the points
        linepts += datamean
        return linepts;
'''
[27.23232954 24.56108827  9.5954606 ]
[ 0.41980803  0.74475744 -0.51874616]
[-0.90745483  0.35508816 -0.22458437]
[0.01693974 0.56502102 0.82490259]
[26.27468425 24.7928729  10.52890996]
[ 0.60816783  0.61608871 -0.50056628]
[ 0.7237724  -0.68934573  0.03091892]
[-0.32601442 -0.38109995 -0.8651459 ]
'''
Points = list(zip(X1,Y1,Z1))
center = numpy.array([27.23232954, 24.56108827,  9.5954606 ])#Vasa - C1
Pc1 = numpy.array([ 0.41980803,  0.74475744, -0.51874616])
Pc2 = numpy.array([-0.90745483,  0.35508816, -0.22458437])
Pc3 = numpy.array([0.01693974, 0.56502102, 0.82490259])
#If we don't sort - it might looks like a bunch of scribbles - sorted draws a nice line
Points.sort(key=lambda i: numpy.dot(Pc1, i)) # Sorts by first PC so it draws lines nicely
C1s = numpy.dot(Points - center, Pc1) # Components in first PC direction
C2s = numpy.dot(Points - center, Pc2) 
C3s = numpy.dot(Points - center, Pc3)
plt.grid('true')
#plt.scatter(C2s, C3s) # Shows plot without first PC
plt.scatter(C2s, C1s)
#plt.scatter(C1s, C3s)
plt.show()

def helixFit(pc1, r, frequency, phase):
    return r*numpy.cos(pc1*frequency + phase) #Doesn't matter if it's sin or cos

# Need to fit each of the components separately
popt, pcov = curve_fit(helixFit, C1s, C2s, p0=[5, 0.2, -numpy.pi/2]) # Predicts C2 given C1
print("Fit parameters (radius, frequency, phase) for C1 -> C2:", popt)
C2Ps = [helixFit(c1, *popt) for c1 in C1s]

popt, pcov = curve_fit(helixFit, C1s, C3s, p0=[5, 0.2, 0]) # Predicts C3 given C1
print("Fit parameters (radius, frequency, phase) for C1 -> C3:", popt)
C3Ps = [helixFit(c1, *popt) for c1 in C1s]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X1, Y1, Z1, c = 'r', marker='o')
# Plot lines connecting nearby points in Helix
#Drawing a line throught the middle of C1, C2Ps, C3Ps
for i in range(len(C1s)-1): #Go from the center into the PC's direction by this much for each point
        start = center + C1s[i]*Pc1 + C2Ps[i]*Pc2 + C3Ps[i]*Pc3
        end = center + C1s[i+1]*Pc1 + C2Ps[i+1]*Pc2 + C3Ps[i+1]*Pc3
        ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c = 'blue')
plt.show()

plt.scatter(C1s, C2s) # True
plt.plot(C1s, C2Ps) # Fit

#Plot3D, takes a start x, end x, start y, end y, start 

#Rotate helix? - rotate to Pc1?


plt.show()


'''
#Generating a subplot of vasa-C1 and drawing PC1

ax = fig.add_subplot(1,3,2, projection = '3d')
ax.grid(False)
red = ax.scatter (X1, Y1, Z1, c = 'r', marker='o',linewidths=2)
ax.set_title('vasa PC1')
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
linePointsC1 = PC1(X1,Y1,Z1)
ax.plot3D(*linePointsC1.T)

#Generating a subplot of vasa-C1 and drawing PC1
ax = fig.add_subplot(1,3,3, projection = '3d')
ax.grid(False)
green = ax.scatter (X2, Y2, Z2, c = 'g', marker='o',linewidths=2)
ax.set_title('dazl PC1')
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')
linePointsC2 = PC1(X2,Y2,Z2)
ax.plot3D(*linePointsC2.T)

plt.show()
'''



