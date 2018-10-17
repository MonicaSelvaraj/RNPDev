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

f_read = open("FileNames.txt", "r")
last_line = f_read.readlines()[-1]
last_line = last_line[:-1] #Ignoring newline character
f_read.close()

plt.style.use('dark_background')

print("Curve Fitting")

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
        
       
C1r = numpy.array(C1r, dtype = float); C2r = numpy.array(C2r, dtype = float); C3r = numpy.array(C3r, dtype = float)
C1g = numpy.array(C1g, dtype = float); C2g = numpy.array(C2g, dtype = float); C3g = numpy.array(C3g, dtype = float)

#This is to input to scipy.optimize.curvefit
def helixFit(pc1, r, frequency, phase):
    return r*numpy.cos(pc1*frequency + phase) #Doesn't matter if it's sin or cos

#Given x, predict the best y 
def BestFit(x,y):
#        minError = 1000; radius = 0; frequency = 0; phase = 0; yOpt = list()
#        frequencies = numpy.arange(0.0 , 0.5 , 0.1)
         #Outer loop for generating radii, Inner loop for generating frequencies
#        for r in range(0, 15, 2):
#                for f in frequencies:
#                        popt, pcov = curve_fit(helixFit, x, y, p0=[r, f, 0])
#                        StandardErr = numpy.sqrt(numpy.diag(pcov))
#                        currentFitErr = (StandardErr[0] + StandardErr[1])/2
#                        if(currentFitErr < minError):
#                                minError = currentFitErr
#                                radius = popt[0]; frequency = popt[1]; phase = popt[2]
#                                yOpt = [helixFit(c1, *popt) for c1 in x]
#        print("Fit parameters - Radius: ", radius, "Frequency: ", frequency, "Phase: ", phase, "Mean Standard Error: ", minError)

        #Alternative - constraining optimization 
        popt, pcov = curve_fit(helixFit, x, y, bounds=(0, [15, 1, 2*math.pi]))
        radius = popt[0]; frequency = popt[1]; phase = popt[2]
        yOpt = [helixFit(c1, *popt) for c1 in x]
        StandardErr = numpy.sqrt(numpy.diag(pcov))
        #print("Fit parameters - Radius: ", radius, "Frequency: ", frequency, "Phase: ", phase,  "Standard Error in each parameter: ", StandardErr)        
        with open("FitRadius.txt", "a") as text_file:
                text_file.write( str(radius) + "\n" )
        with open("FitFrequency.txt", "a") as text_file:
                text_file.write( str(frequency) + "\n" )
        with open("FitPhase.txt", "a") as text_file:
                text_file.write( str(phase) + "\n" )
        with open("FitRadiusSE.txt", "a") as text_file:
                text_file.write( str(StandardErr[0]) + "\n" )
        with open("FitFrequencySE.txt", "a") as text_file:
                text_file.write( str(StandardErr[1]) + "\n" )
        with open("FitPhaseSE.txt", "a") as text_file:
                text_file.write( str(StandardErr[2]) + "\n" )
        return(yOpt)


print("Channel1")
C2Pr = BestFit(C1r, C2r) # Predicts C2 given C1
C3Pr = BestFit(C1r, C3r)  # Predicts C3 given C1

print("Channel2")
C2Pg = BestFit(C1g, C2g) # Predicts C2 given C1
C3Pg = BestFit(C1g, C3g)  # Predicts C3 given C1

#2D plots after fit
plt.scatter(C2g, C3g, c='g') # True
plt.plot(C2Pg, C3Pg, c='y') # Fit
plt.scatter(C2r, C3r, c='r') # True
plt.plot(C2Pr, C3Pr, c='b') # Fit
plt.title(' C2 vs C3 fit')
plt.ylim(-20, 20); plt.xlim(-20,20)
plt.savefig('Output/%s/C2C3Fit.png' % last_line)
#plt.show()
plt.close()

plt.scatter(C1g, C2g, c='g') # True
plt.plot(C1g, C2Pg, c= 'y') # Fit
plt.scatter(C1r, C2r, c='r') # True
plt.plot(C1r, C2Pr, c= 'b') # Fit
plt.title(' C1 vs C2 fit')
plt.ylim(-20, 20); plt.xlim(-20,20)
plt.savefig('Output/%s/C1C2Fit.png' % last_line)
#plt.show()
plt.close()

plt.scatter(C1g, C3g, c='g') # True
plt.plot(C1g, C3Pg, c='y') # Fit
plt.scatter(C1r, C3r, c='r') # True
plt.plot(C1r, C3Pr, c='b') # Fit
plt.title(' C1 vs C3 fit')
plt.ylim(-20, 20); plt.xlim(-20,20)
plt.savefig('Output/%s/C1C3Fit.png' % last_line)
#plt.show()
plt.close()

#3D fit

#Original data
#Variables for C1
X1 = list(); Y1 = list(); Z1 = list()

#Variables for C2
X2 = list(); Y2 =  list(); Z2 = list()

with open ('StraightenedC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
            X1.append(line[0])
            Y1.append(line[1])
            Z1.append(line[2])
        
with open ('StraightenedC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
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
        #print (datamean) #This is going to be the center of my helix
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
        ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c = 'yellow', linewidth = 3) #alpha=0.5) 
        ax.set_title('overall fit')
#plt.show()
fig.savefig('Output/%s/3DFit.png' % last_line)

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

        if(len(maxm[0]) >= 2):
                maxima = maxm[0]
                #Now finding the distance between the first two maxima
                p1 = (x[maxima[0]], y[maxima[0]])
                p2 = (x[maxima[1]], y[maxima[1]])
                pitch = distance.euclidean(p1,p2)
        elif(len(minm[0]) >= 2):
                minima = minm[0]
                #Now finding the distance between the first two maxima
                p1 = (x[minima[0]], y[minima[0]])
                p2 = (x[minima[1]], y[minima[1]])
                pitch = distance.euclidean(p1,p2)
        elif(len(minm[0]) == 1 and len(maxm[0]) == 1):
                minima = minm[0]
                maxima = maxm[0]
                #Now finding the distance between the first two maxima
                p1 = (x[minima[0]], y[minima[0]])
                p2 = (x[maxima[0]], y[maxima[0]])
                pitch = distance.euclidean(p1,p2)
        else:
                pitch = 0
        
        return (pitch);

#print ("Channel1 Pitch: C1 C2 Pitch -  ", Pitch(C1r, C2Pr) ," C1 C3 Pitch - " ,Pitch(C1r, C3Pr))
#print ("Channel2 Pitch: C1 C2 Pitch -  ", Pitch(C1g, C2Pg) ," C1 C3 Pitch - " ,Pitch(C1g, C3Pg))

with open("Pitch.txt", "a") as text_file:
                text_file.write( str(Pitch(C1r, C2Pr)) + "\n" )
                text_file.write( str(Pitch(C1r, C3Pr)) + "\n" )
                text_file.write( str(Pitch(C1g, C2Pg)) + "\n" )
                text_file.write( str(Pitch(C1g, C3Pg)) + "\n" )
                
