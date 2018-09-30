'''
- Counting the number of points in each z - plotting it out - see if there is a large drop
'''

#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy 
import csv

fig = plt.figure( )

print("Removing the cortex")

#Variables for C1
Z = list()
uniqueZ = list()

with open ('C1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        Z.append(line[2])
        
with open ('C2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        Z.append(line[2])

for z in Z:
    if z not in uniqueZ:
        uniqueZ.append(z)

Z = numpy.array(Z); Z = Z.astype(float)
uniqueZ = numpy.array(uniqueZ); uniqueZ = uniqueZ.astype(float)
uniqueZ = numpy.sort(uniqueZ)

print(uniqueZ)

zCount = list()
for uz in uniqueZ:
    counter = 0
    for z in Z:
        if (z == uz):
            counter = counter+1
    zCount.append(counter)
print(zCount)

plt.scatter(uniqueZ, zCount)
plt.show()

#How should we decide which is cortex

    


