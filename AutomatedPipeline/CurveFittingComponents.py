'''
Fitting curves in 2D to each channel separately
Combining the 2D fits to get a 3D fit
Plots of 2D and 3D fits
'''

#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.axes import Axes
import scipy.stats
import statistics
import numpy 
import csv
import math
from matplotlib import style
from scipy.spatial import distance
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

plt.style.use('dark_background')

#2D curve fitting 
#Original data 
C1r = list(); C2r = list(); C3r = list()#C1 - red
C1g = list(); C2g = list(); C3g = list()#C2 - green

#Reading in the data 
with open ('CleanedComponentsC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
            C1r.append(line[0])
            C2r.append(line[1])
            C3r.append(line[2])
            
with open ('CleanedComponentsC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
            C1g.append(line[0])
            C2g.append(line[1])
            C3g.append(line[2])
        
       
C1r = numpy.array(C1r); C2r = numpy.array(C2r); C3r = numpy.array(C3r)
C1g = numpy.array(C1g); C2g = numpy.array(C2g); C3g = numpy.array(C3g)
C1r = C1r.astype(float); C2r= C2r.astype(float); C3r = C3r.astype(float)
C1g = C1g.astype(float); C2g= C2g.astype(float); C3g = C3g.astype(float)

#This is to input to scipy.optimize.curvefit
def helixFit(pc1, r, frequency, phase):
    return r*numpy.cos(pc1*frequency + phase) #Doesn't matter if it's sin or cos

#Need to fit each of the components separately - C1
popt, pcov = curve_fit(helixFit, C1r, C2r, p0=[5, 0.4, -numpy.pi/2]) # Predicts C2 given C1
print("Fit parameters (radius, frequency, phase) for C1 -> C2:", popt)
C2Pr = [helixFit(c1, *popt) for c1 in C1r]

popt, pcov = curve_fit(helixFit, C1r, C3r, p0=[5, 0.4, 0]) # Predicts C3 given C1
print("Fit parameters (radius, frequency, phase) for C1 -> C3:", popt)
C3Pr = [helixFit(c1, *popt) for c1 in C1r]

#Need to fit each of the components separately - C2
popt, pcov = curve_fit(helixFit, C1g, C2g, p0=[5, 0.6, -numpy.pi/2]) # Predicts C2 given C1
print("Fit parameters (radius, frequency, phase) for C1 -> C2:", popt)
C2Pg = [helixFit(c1, *popt) for c1 in C1g]

popt, pcov = curve_fit(helixFit, C1g, C3g, p0=[5, 0.6, 0]) # Predicts C3 given C1
print("Fit parameters (radius, frequency, phase) for C1 -> C3:", popt)
C3Pg = [helixFit(c1, *popt) for c1 in C1g]

#2D plots after fit
plt.scatter(C2g, C3g, c='g') # True
plt.plot(C2Pg, C3Pg, c='y') # Fit
plt.scatter(C2r, C3r, c='r') # True
plt.plot(C2Pr, C3Pr, c='b') # Fit
plt.title(' C2 vs C3 fit')
plt.ylim(-20, 20); plt.xlim(-20,20)
plt.show()

plt.scatter(C1g, C2g, c='g') # True
plt.plot(C1g, C2Pg, c= 'y') # Fit
plt.scatter(C1r, C2r, c='r') # True
plt.plot(C1r, C2Pr, c= 'b') # Fit
plt.title(' C1 vs C2 fit')
plt.ylim(-20, 20); plt.xlim(-20,20)
plt.show()

plt.scatter(C1g, C3g, c='g') # True
plt.plot(C1g, C3Pg, c='y') # Fit
plt.scatter(C1r, C3r, c='r') # True
plt.plot(C1r, C3Pr, c='b') # Fit
plt.title(' C1 vs C3 fit')
plt.ylim(-20, 20); plt.xlim(-20,20)
plt.show()

#3D fit

#Original data
#Variables for C1
X1 = list(); Y1 = list(); Z1 = list()

#Variables for C2
X2 = list(); Y2 =  list(); Z2 = list()

with open ('deconvolutedC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z,S
        if (float(line[1]) > 10):
                X1.append(line[0])
                Y1.append(line[1])
                Z1.append(line[2])
        
with open ('deconvolutedC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z,S
             if (float(line[1]) > 10):
                     X2.append(line[0])
                     Y2.append(line[1])
                     Z2.append(line[2])
        

X1 = numpy.array(X1); Y1 = numpy.array(Y1); Z1 = numpy.array(Z1)
X2 = numpy.array(X2); Y2 = numpy.array(Y2); Z2 = numpy.array(Z2)
X1 = X1.astype(float); Y1 = Y1.astype(float); Z1 = Z1.astype(float)
X2 = X2.astype(float); Y2 = Y2.astype(float); Z2 = Z2.astype(float)

def PCs(X,Y,Z):
        data = numpy.concatenate((X[:, numpy.newaxis], 
                       Y[:, numpy.newaxis], 
                       Z[:, numpy.newaxis]), 
                      axis=1)
        #print(data)
        datamean = data.mean(axis=0)
        print (datamean) #This is going to be the center of my helix
        uu, dd, vv = numpy.linalg.svd(data - datamean)
        #Taking the variation in the z dimension, because this is the dimension of PC1
        #Linear algebra - figure out what exactly is happening in terms of dimensional collapsation
        return vv[0], vv[1], vv[2], datamean;

C1PCs = PCs(X1, Y1, Z1); C2PCs = PCs(X2, Y2, Z2)
centerC1 = C1PCs[3]; centerC2 = C2PCs[3]
C1Pc1 = C1PCs[0]; C2Pc1 = C2PCs[0]
C1Pc2 = C1PCs[1]; C2Pc2 = C2PCs[1]
C1Pc3 = C1PCs[2]; C2Pc3 = C2PCs[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X1, Y1, Z1, c = 'r', marker='o')
ax.scatter3D(X2, Y2, Z2, c = 'g', marker='o')
# Plot lines connecting nearby points in Helix
#Drawing a line throught the middle of C1, C2P, C3P
for i in range(len(C1r)-1): #Go from the center into the PC's direction by this much for each point
        #Plot3D, takes a start x, end x, start y, end y, start z, end z
        start = centerC1 + C1r[i]*C1Pc1 + C2Pr[i]*C1Pc2 + C3Pr[i]*C1Pc3
        end = centerC1 + C1r[i+1]*C1Pc1 + C2Pr[i+1]*C1Pc2 + C3Pr[i+1]*C1Pc3
        ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c = 'blue', linewidth = 3)
for i in range(len(C1g)-1): #Go from the center into the PC's direction by this much for each point
        #Plot3D, takes a start x, end x, start y, end y, start z, end z
        start = centerC2 + C1g[i]*C2Pc1 + C2Pg[i]*C2Pc2 + C3Pg[i]*C2Pc3
        end = centerC2 + C1g[i+1]*C2Pc1 + C2Pg[i+1]*C2Pc2 + C3Pg[i+1]*C2Pc3
        ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c = 'yellow', linewidth = 3, alpha=0.2)
        ax.set_title('overall fit')
plt.show()

#Parameters
def Pitch(x, y):
        #x is the array of x coordinates and y is the array of y coordinates

        #Converting x, y to numpy arrays
        x = numpy.array(x); y = numpy.array(y)
        x = x.astype(float); y = y.astype(float)

        # sort the data in x and rearrange y accordingly
        sortId = numpy.argsort(x)
        x = x[sortId]
        y = y[sortId]

        # this way the x-axis corresponds to the index of x
        maxm = argrelextrema(y, numpy.greater)  
        minm = argrelextrema(y, numpy.less)
        
        #maxm and minm contains the indices of minima and maxima respectively
        maxima = maxm[0]
        
        #Now finding the distance between the first two maxima 
        p1 = (x[maxima[0]], y[maxima[0]])
        p2 = (x[maxima[1]], y[maxima[1]])
        pitch = distance.euclidean(p1,p2)
        return (pitch);

print (Pitch(C1r, C2Pr))
print (Pitch(C1r, C3Pr))
print (Pitch(C1g, C2Pg))
print (Pitch(C1g, C3Pg))

        





