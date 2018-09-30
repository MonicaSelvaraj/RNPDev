'''
Saving x-y image needed for cortex removal
'''

#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy 
import csv


fig = plt.figure()
fig.patch.set_visible(False)
X = list(); Y = list(); Z = list()

with open ('ClusteredC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z,S
        X.append(line[0])
        Y.append(line[1])
        Z.append(line[2])
        
with open ('ClusteredC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z,S
        X.append(line[0])
        Y.append(line[1])
        Z.append(line[2])
        

X = numpy.array(X); Y = numpy.array(Y); Z = numpy.array(Z)
X = X.astype(float); Y = Y.astype(float); Z = Z.astype(float)

print("X")
print(max(X))
print(min(X))
print("Y")
print(max(Y))
print(min(Y))


plt.scatter(X,Y, c='b')
plt.axis('off')
fig.savefig('xy.png', transparent=True)
plt.show()

