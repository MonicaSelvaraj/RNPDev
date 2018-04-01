'''
If, on the other hand, you just want to get the best fitting line to the data,
i.e. the line which, if you projected the data onto it, would minimize the squared
distance between the real point and its projection, then what you want is the first
principal component.
One way to define it is the line whose direction vector is the eigenvector of the
covariance matrix corresponding to the largest eigenvalue, that passes through the
mean of your data. That said, eig(cov(data)) is a really bad way to calculate it, since
it does a lot of needless computation and copying and is potentially less accurate
than using svd. See below:
'''
#!/usr/bin/python
import numpy as np

# Generate some data that lies along a line

#The purpose of meshgrid is to create a rectangular grid out of an array of x values and an array of y values.
#[-2:5:120j] - producing values 120 values from -2, 5, similar to linspace (-2,5, 120)
x = np.mgrid[-2:5:120j]
print (x)
y = np.mgrid[1:9:120j]
print (y)
z = np.mgrid[-5:3:120j]
print (z)

#data1 = [1,2,3,4]
#arr1 = np.array(data1)
#print (arr1)
#data2 = [5,6,7,8]
#arr2 = np.array(data2)
#data3= [9,10,11,12]
#arr3 = np.array(data3)
#print (arr2)
#print(arr3)
#conc = np.concatenate((arr1,arr2,arr3), axis = 1)
#print (conc)


#Join a sequence of arrays along an existing axis
data = np.concatenate((x[:, np.newaxis], 
                       y[:, np.newaxis], 
                       z[:, np.newaxis]), 
                      axis=1)
#axis - is the axis along which the arrays will be joined 
print (data)

# Perturb with some Gaussian noise
data += np.random.normal(size=data.shape) * 0.4

# Calculate the mean of the points, i.e. the 'center' of the cloud
datamean = data.mean(axis=0)
print (datamean)

# Do an SVD on the mean-centered data.
uu, dd, vv = np.linalg.svd(data - datamean)


# Now vv[0] contains the first principal component, i.e. the direction
# vector of the 'best fit' line in the least squares sense.

# Now generate some points along this best fit line, for plotting.

# I use -7, 7 since the spread of the data is roughly 14
# and we want it to have mean 0 (like the points we did
# the svd on). Also, it's a straight line, so we only need 2 points.
linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]

# shift by the mean to get the line in the right place
linepts += datamean

# Verify that everything looks right.

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d

ax = m3d.Axes3D(plt.figure())
ax.scatter3D(*data.T)
ax.plot3D(*linepts.T)
plt.show()
