#!/usr/bin/python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv

x = list(); y = list(); z = list()

with open ('VariancePoints.csv', 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
            x.append(line[0])
            y.append(line[1])
            z.append(line[2])

#Sorting points
Xs = list(); Ys = list(); Zs = list()
Xs = numpy.array(Xs, dtype=float); Ys = numpy.array(Ys, dtype=float); Zs = numpy.array(Zs, dtype=float)
    
#Concatenating numpy arrays
data = numpy.concatenate((x[:, numpy.newaxis],
                          y[:, numpy.newaxis],
                          z[:, numpy.newaxis]),
                         axis=1)
    
#Sorting wrt x, y, z consecutively like excel
sortedData = data[numpy.lexsort(numpy.transpose(data)[::-1])]
    
#Separating the sorted data into numpy arrays
sortedArray = numpy.hsplit(sortedData, 3)
Xs = numpy.concatenate(sortedArray[0], axis=0)
Ys = numpy.concatenate(sortedArray[1], axis=0)
Zs = numpy.concatenate(sortedArray[2], axis=0)

xInWindow = list(); yInWindow = list(); zInWindow = list(); variance = list(); 
window = x.size()/20
for i in range(1, x.size(), window):
    for j in range(i, i+(window-1)):
        xInWindow.append(Xs[j])
        yInWindow.append(Ys[j])
        zInWindow.append(Zs[j])
    varY = numpy.var(Ys)
    varZ = numpy.var(Zs)
    meanVar = (varY + varZ)/2
    variance.append(meanVar)

plt.scatter(meanVar)
    

        


            
