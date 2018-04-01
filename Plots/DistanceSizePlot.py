#Distance wrt z - size plot 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy 
import csv


Z = list() #List for z values 
S1 = list() #List for sizes
S2 = list()
S3 = list()
S4 = list()
one = 0
two = 0
three = 0
four = 0
#Reading in the csv file

#opening the csv file
# 'r' specifies that we want to read this file
#csv_reader is the name of the reader object that we have created 
with open ('testRed.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)

    #Iterating through contents in the file
    for line in csv_reader:
        #each line has X,Y,Z
        Z.append(line[2])
        if (float(line[3]) <= 0.28):
            S1.append(line[3])
            one = one+1
        elif(float(line[3]) > 0.28 and float(line[3])<=0.37):
            S2.append(line[3])
            two = two+1
        elif(float(line[3]) > 0.37 and float(line[3])<=0.46):
            S3.append(line[3])
            three = three+1
        else:
            S4.append(line[3])
            four = four+1

sizes = [one, two, three, four]

z = numpy.array(Z)
s1 = numpy.array(S1)
s2 = numpy.array(S2)
s3 = numpy.array(S3)
s4 = numpy.array(S4)
sizes = numpy.array(sizes)

z = z.astype(float)
s1= s1.astype(float)
s2= s2.astype(float)
s3= s3.astype(float)
s4= s4.astype(float)
sizes = sizes.astype(float)
#x-coordinates of the bar
objects = ('0.27', '0.36', '0.45', '0.54')
y_pos = numpy.arange(len(objects))
plt.bar(y_pos, sizes, align='center', alpha=0.5)
print (sizes)
plt.xticks(y_pos, objects)
plt.show( )

