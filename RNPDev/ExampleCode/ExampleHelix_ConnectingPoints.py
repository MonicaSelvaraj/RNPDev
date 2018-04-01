import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# Create 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define Helix
def helixGen(t, r, h, theta, std):
	x = r*np.cos(t*theta) + np.random.normal(scale=std)
	y = r*np.sin(t*theta) + np.random.normal(scale=std)
	z = t*h + np.random.normal(scale=std)
	return x, y, z

def DrawHelix(r, h, theta, std, n, color='green'):	
	# Get points for Helix
	Xs = []
	Ys = []
	Zs = []
	for i in range(n):
		t = i/n
		x, y, z = helixGen(t, r, h, theta, std)
		Xs.append(x)
		Ys.append(y)
		Zs.append(z)

	# Plot Helix
	ax.scatter3D(Xs, Ys, Zs)

	# Plot lines connecting nearby points in Helix
	for x1, y1, z1 in zip(Xs, Ys, Zs):
		for x2, y2, z2 in zip(Xs, Ys, Zs):
			if x1 > x2: # This prevents repeats
				continue
			if (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 < 25:
				ax.plot3D([x1, x2], [y1, y2], [z1, z2], color)

DrawHelix(10, 30, 10, 1, 100, color='green')
DrawHelix(10, 30, -10, 1, 100, color='red')

plt.show()
