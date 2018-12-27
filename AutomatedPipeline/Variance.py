#!/usr/bin/python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy 
import csv
import math

x = list(); y = list(); z = list()

with open ('VariancePoints.csv', 'r') as csv_file:
        csv_reader = csv.reader (csv_file)
        for line in csv_reader:
            x.append(line[0])
            y.append(line[1])
            z.append(line[2])
x = numpy.array(x, dtype=float); y = numpy.array(y, dtype=float); z = numpy.array(z, dtype=float)

fig = plt.figure( )
plt.style.use('dark_background')
ax = fig.add_subplot(111, projection = '3d' )                            
ax.grid(False)
ax.set_xlabel ('x, axis'); ax.set_ylabel ('y axis'); ax.set_zlabel ('z axis')
ax.scatter (x, y, z, c = 'r', marker='o')
plt.show() 

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
window = math.floor(len(x)/10)
for i in range(1, len(x)-1, window):
    for j in range(i, i+(window-1)):
        if(j == len(x) - 1): break
        xInWindow.append(Xs[j])
        yInWindow.append(Ys[j])
        zInWindow.append(Zs[j])
    varY = numpy.var(yInWindow)
    varZ = numpy.var(zInWindow)
    meanVar = (varY + varZ)/2
    variance.append(meanVar)

plt.plot(variance)
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.show()
    
      


            
