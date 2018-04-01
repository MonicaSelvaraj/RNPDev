#Checking if the data is being sorted the way I want
#!/usr/bin/python

import matplotlib.pyplot as plt
from matplotlib import cm
#from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv

fig = plt.figure( )

#0.54 - original data 
X1 = list()
Y1 = list()
Z1 = list()

#For sorted data
Xs = list()
Ys = list()
Zs = list()

with open ('manualzRed.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        #each line has X,Y,Z, radius
        if (float(line[3])>0.54):
            X1.append(line[0])
            Y1.append(line[1])
            Z1.append(line[2])

X1 = numpy.array(X1)
Y1 = numpy.array(Y1)
Z1 = numpy.array(Z1)

Xs = numpy.array(Xs)
Ys = numpy.array(Ys)
Zs = numpy.array(Zs)

X1 = X1.astype(float)
Y1= Y1.astype(float)
Z1= Z1.astype(float)

Xs = Xs.astype(float)
Ys= Ys.astype(float)
Zs= Zs.astype(float)

#Concatenating numpy arrays
data = numpy.concatenate((X1[:, numpy.newaxis], 
                       Y1[:, numpy.newaxis], 
                       Z1[:, numpy.newaxis]), 
                      axis=1)
#print (data)

#Sorting wrt x, y, z consecutively like excel
sortedData = data[numpy.lexsort(numpy.transpose(data)[::-1])]
print (sortedData)

#Separating the sorted data into numpy arrays 
sortedArray = numpy.hsplit(sortedData, 3)
print (sortedArray)
Xs = numpy.concatenate(sortedArray[0], axis=0)
Ys = numpy.concatenate(sortedArray[1], axis=0)
Zs = numpy.concatenate(sortedArray[2], axis=0)

print (Xs)
print (Ys)
print (Zs)
