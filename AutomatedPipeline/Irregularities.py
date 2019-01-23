#Sorting consecutively by z
#Splitting the helix into four parts by z
#Calculating average of residuals in each quarter

import csv
import numpy 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X1 = list(); Y1 = list(); Z1 = list()
X2 = list(); Y2 = list(); Z2 = list()
X1Fit = list(); Y1Fit = list(); Z1Fit = list()
X2Fit = list(); Y2Fit = list(); Z2Fit = list()

#Reading in straightened points
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
            
with open ('FitC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
            X1Fit.append(line[0])
            Y1Fit.append(line[1])
            Z1Fit.append(line[2])
        
with open ('FitC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
            X2Fit.append(line[0])
            Y2Fit.append(line[1])
            Z2Fit.append(line[2])

X1 = numpy.array(X1, dtype = float); Y1 = numpy.array(Y1, dtype = float); Z1 = numpy.array(Z1, dtype = float)
X2 = numpy.array(X2, dtype = float); Y2 = numpy.array(Y2, dtype = float); Z2 = numpy.array(Z2, dtype = float)
X1Fit = numpy.array(X1Fit, dtype = float); Y1Fit = numpy.array(Y1Fit, dtype = float); Z1Fit = numpy.array(Z1Fit, dtype = float)
X2Fit = numpy.array(X2Fit, dtype = float); Y2Fit = numpy.array(Y2Fit, dtype = float); Z2Fit = numpy.array(Z2Fit, dtype = float)

fig = plt.figure( )
plt.style.use('dark_background')
ax = fig.add_subplot(111, projection = '3d' )                            
ax.grid(False)
ax.set_xlabel ('x, axis'); ax.set_ylabel ('y axis'); ax.set_zlabel ('z axis')
ax.scatter (X1, Y1, Z1, c = 'r', marker='o', s = 10)
ax.scatter (X2, Y2, Z2, c = 'g', marker='o', s = 10)
ax.scatter (X1Fit, Y1Fit, Z1Fit, c = 'b', marker='o', s = 10)
ax.scatter (X2Fit, Y2Fit, Z2Fit, c = 'y', marker='o', s = 10)
plt.show()







