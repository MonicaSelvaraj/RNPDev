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
from scipy.optimize import curve_fit

plt.style.use('dark_background')

#Original data 
C1r = list(); C2r = list(); C3r = list()#C1 - red
C1g = list(); C2g = list(); C3g = list()#C2 - green

#Reading in the data 
with open ('ComponentsC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
            C1r.append(line[0])
            C2r.append(line[1])
            C3r.append(line[2])
            
with open ('ComponentsC2.csv', 'r') as csv_file:
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
popt, pcov = curve_fit(helixFit, C1g, C2g, p0=[5, 0.4, -numpy.pi/2]) # Predicts C2 given C1
print("Fit parameters (radius, frequency, phase) for C1 -> C2:", popt)
C2Pg = [helixFit(c1, *popt) for c1 in C1g]

popt, pcov = curve_fit(helixFit, C1g, C3g, p0=[5, 0.4, 0]) # Predicts C3 given C1
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







