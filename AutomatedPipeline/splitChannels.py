'''
This script reads in the csv file with coorindates of all channels, and saves the data
of each of the channels as individual files
'''
#!/usr/bin/python

import numpy 
import csv

#Variables to store Channel 1 data
X1 = list(); Y1 = list(); Z1 = list(); S1 = list()

#Variables to store Channel 2 data 
X2 = list(); Y2 =  list(); Z2 = list(); S2 = list()

#opening the csv file
# 'r' specifies that we want to read this file
#csv_reader is the name of the reader object that we have created 
with open ('vasaC1dazlC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        if float(line[4]) > 0.28:
            if line[0] == 'C1':
                X1.append(line[1])
                Y1.append(line[2])
                Z1.append(line[3])
                S1.append(line[4])
            else:
                X2.append(line[1])
                Y2.append(line[2])
                Z2.append(line[3])
                S2.append(line[4])
        
X1 = numpy.array(X1); Y1 = numpy.array(Y1); Z1 = numpy.array(Z1); S1 = numpy.array(S1)
X2 = numpy.array(X2); Y2 = numpy.array(Y2); Z2 = numpy.array(Z2); S2 = numpy.array(S2)
X1 = X1.astype(float); Y1 = Y1.astype(float); Z1 = Z1.astype(float); S1 = S1.astype(float)
X2 = X2.astype(float); Y2 = Y2.astype(float); Z2 = Z2.astype(float); S2 = S2.astype(float)

#Saving each channel's data in new files
numpy.savetxt("C1.csv", numpy.column_stack((X1, Y1, Z1, S1)), delimiter=",", fmt='%s')
numpy.savetxt("C2.csv", numpy.column_stack((X2, Y2, Z2, S2)), delimiter=",", fmt='%s')
