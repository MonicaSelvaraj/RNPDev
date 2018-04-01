#Proof that each RNP is a sphere
'''
==============================
Determining if each RNP is a sphere
==============================
- Each z is approximately 0.0905 units apart
- The radii of the RNP's are 0.27, 0.36, 0.45, 0.54 units (Note that all of the RNP's are going to be)
- Calculate the diameter
- Diameter/0.0905 will give us an estimate of how many z's they are split into
- Go up and down the z, in the same x,y to determine how many they are actually split into
- There is a possibility that the x,y varies a little: For now: Set the threshold to be less than the diameter of the RNP
- Check if there is a correlation/statistical significance between the estimate and the actual

Initial method of checking to get some idea of what is going on
- Read in the file
- Wrt size, create separate x,y,z, size for each category
- Set a limit to how many z's each RNP can possible be broken into 
- Iterate through the radii separately and find out how many have the same x,y (or with <diameter variation in x,y), going down the z
- Compare it to how many RNP's there are
'''

#!/usr/bin/python

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv

zdistance = 0.0905
radii = [0.27, 0.36, 0.45, 0.54]
radii = numpy.array(radii)
diameter = radii*2
diameter = numpy.array(diameter)
print (diameter)
possibleSplits= diameter/zdistance
possibleSplits = numpy.array(possibleSplits)
print (possibleSplits)

#Results:
#               1      2       3     4   
#radii = [0.27, 0.36, 0.45, 0.54]
#diameter = [0.54 0.72 0.9  1.08]
#possibleSplits = [ 5.96685083  7.9558011   9.94475138 11.93370166]

X1 = list()
Y1 = list()
Z1 = list()

X2 = list()
Y2 = list()
Z2 = list()

X3 = list()
Y3 = list()
Z3 = list()

X4= list()
Y4 = list()
Z4 = list()

with open ('manualzRed.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)

    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z, radius
        if (float(line[3])<0.28):
            X1.append(line[0])
            Y1.append(line[1])
            Z1.append(line[2])
        elif (float(line[3])>0.28 and float(line[3])<0.37):
            X2.append(line[0])
            Y2.append(line[1])
            Z2.append(line[2])
        elif (float(line[3])>0.37 and float(line[3])<0.46):
            X3.append(line[0])
            Y3.append(line[1])
            Z3.append(line[2])
        else:
            X4.append(line[0])
            Y4.append(line[1])
            Z4.append(line[2])

X1 = numpy.array(X1)
Y1 = numpy.array(Y1)
Z1 = numpy.array(Z1)
X2 = numpy.array(X2)
Y2 = numpy.array(Y2)
Z2 = numpy.array(Z2)
X3 = numpy.array(X3)
Y3 = numpy.array(Y3)
Z3 = numpy.array(Z3)
X4 = numpy.array(X4)
Y4 = numpy.array(Y4)
Z4 = numpy.array(Z4)

X1 = X1.astype(float)
Y1= Y1.astype(float)
Z1= Z1.astype(float)
X2 = X2.astype(float)
Y2= Y2.astype(float)
Z2= Z2.astype(float)
X3 = X3.astype(float)
Y3= Y3.astype(float)
Z3= Z3.astype(float)
X4 = X4.astype(float)
Y4= Y4.astype(float)
Z4= Z4.astype(float)

'''
=================================
Method for counting the number of repeats
=================================
- Iterating through the x
- Make a list of all the x's and y's within the threshold

You can either start with an xy
Go to the next z
Find if it could have splits in that z
do that for the next (maxPossible splits z)
have a separate array to keep track of what you have already counted as a split
- 
'''
#args: x(array of x coordinates), y(array of y coordinates), z(array of z coordinates)
#maxz (maximum number of z's to check), diameter (to check what x and y is acceptable to count as a split)
def countSplits(x,y,z, maxz,diameter):
    #A numpy array to keep track of what dot is accounted for so we don't double count dots
    #0 indicates not accounted for, 1 indicates accounted for 
    accountedFor = np.zeros(x.size()) 
    #Iterating through each x
    for a in numpy.nditer(x):
        xpos+=1 #Keeps track of position
        # I need to keep track of x position myself, because if I use a function, the position returned can be that of a repeated x value
        xvalue = a
        yvalue = y[xpos]
        zvalue = z[xpos]
        newzvalue = z[xpos+1]
        #Going through the data until we reach the next z
        if (zvalue == newzvalue):
            continue
        else:
            #Going through the x's in the next z
            if (x[xpos+1])
        
    
    







            
            
            



