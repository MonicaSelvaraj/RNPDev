#Curve fitting one of the subplots
#!/usr/bin/python

#breakes wrt z - (0-260), (260-578), (578 - 658)
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import numpy 
import csv
import math

#Creating a 3D axes by using the keyword projection = '3D'
fig = plt.figure( )
#setting up the axes for the first subplot
ax = fig.add_subplot(1,1,1, projection = '3d')

def movingaverage(values, window):
    weights = numpy.repeat(1.0, window)/window
    #valid only runs the sma's on valid points
    smas = numpy.convolve(values, weights, 'valid')
    return smas #returns a numpy array

#For subplot (260-578)
X2 = list()
Y2 = list()
Z2 = list()
S2 = list()

with open ('testRed.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        if (float(line[2])>260 and float(line[2])<578):
            X2.append(line[0])
            Y2.append(line[1])
            Z2.append(line[2])
            S2.append(line[3])
            
x2= numpy.array(X2)
y2 = numpy.array(Y2)
z2 = numpy.array(Z2)
s2 = numpy.array(S2)

x2 = x2.astype(float)
y2= y2.astype(float)
z2 = z2.astype(float)
s2 = s2.astype(float)

ax.scatter (x2, y2, z2, c = 'r', marker='D', s=s2)
ax.set_xlabel ('x, axis')
ax.set_ylabel ('y axis')
ax.set_zlabel ('z axis')

#Can't use numpy.polyfit because it takes in only x and y data

#Might be able to use scipy.optimize.curvefit
# Define Helix
#def helixGen(a,b,t):
#r = 9.00
#c = 3.183
#t = numpy.linspace(0, 2*math.pi, 100)
#z = c*t
#x = r*numpy.cos(t)
#y = r*numpy.sin(t)

#params,pcov = optimize.curve_fit(helixGen, x2, y2,z2)
#popt, pcov = curve_fit(helixGen(9, 3.183, t), 20,20)
#plt.plot(20, helixGen(20, *popt), 'r-')

#Might be able to use sklearn
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
#poly = PolynomialFeatures(degree=3)
#X_t = poly.fit_transform(x2)

#Trying to fit a line through the data
xline =movingaverage(movingaverage(movingaverage(movingaverage(x2, 10),10),75),75)
yline = movingaverage(movingaverage(movingaverage(movingaverage(y2, 10),10),75),75)
zline = movingaverage(movingaverage(movingaverage(movingaverage(z2, 10),10),75),75)
ax.plot3D(xline,yline,zline,'red')

plt.show ( )

