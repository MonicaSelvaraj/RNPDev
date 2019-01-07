import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randint

#Generating a helix aligned in the z axis
n = 100
r1 = 3
r2 = 3
p1 = 14
p2 = 14
theta = np.linspace(0, 8*np.pi, n) # 3 turns

x1 = r1*np.cos(theta) + 100
y1 = r1*np.sin(theta) + 100
z1 = (p1/(2*np.pi))*theta

x2 = r2*np.cos(theta) + 100
y2 = r2*np.sin(theta)  + 100
z2 = (p2/(2*np.pi))*theta

#Generating cortex from z = 60 - 75, x, y = 50 - 150
#Actual x, y is in 104 - 96
x1Cortex = list(); y1Cortex = list(); z1Cortex = list()
for i in range(0, 500):
    z1Cortex.append(randint(60, 75))
    x1Cortex.append(randint(75, 125))
    y1Cortex.append(randint(75, 125))

x2Cortex = list(); y2Cortex = list(); z2Cortex = list()
for i in range(0, 500):
    z2Cortex.append(randint(60, 75))
    x2Cortex.append(randint(75, 125))
    y2Cortex.append(randint(75, 125))
    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, y1, z1, c = 'r', lw=2)
ax.scatter(x2, y2, z2, c = 'g', lw=2)
ax.scatter(x1Cortex, y1Cortex, z1Cortex, c = 'r')
ax.scatter(x2Cortex, y2Cortex, z2Cortex, c = 'g')
#plt.show()

X = list(); Y = list(); Z = list(); C = list(); S = list();

for i in range(0, 100):
    X.append(x1[i]);Y.append(y1[i]);Z.append(z1[i]);C.append('C1'); S.append(0.5)
for i in range(0, 500):
    X.append(x1Cortex[i]);Y.append(y1Cortex[i]);Z.append(z1Cortex[i]); C.append('C1');S.append(0.5)
for i in range(0, 100):
    X.append(x2[i]);Y.append(y2[i]);Z.append(z2[i]); C.append('C2');S.append(0.5)
for i in range(0, 500):
    X.append(x2Cortex[i]);Y.append(y2Cortex[i]);Z.append(z2Cortex[i]); C.append('C2');S.append(0.5)


#Writing points to a file
np.savetxt("TestHelix.csv", np.column_stack((C, X, Y, Z, S)), delimiter=",", fmt='%s')


