#This is just some general code for plotting points and curves in 3D using matplotlib
#!/usr/bin/python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv

#Creating a 3D axes by using the keyword projection = '3D'
 
fig = plt.figure( )
ax = fig.add_subplot(111, projection = '3d' )

 #Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');


# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'green')

plt.show ( )
