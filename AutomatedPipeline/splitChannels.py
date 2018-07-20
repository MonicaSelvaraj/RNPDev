'''
This script reads in the csv file with coorindates of all channels, and saves the data
of each of the channels as individual files, after removing particles with radius less than 0.28
'''
#!/usr/bin/python
import sys, os
import numpy 
import csv

#Variables to store Channel 1 data
X1 = list(); Y1 = list(); Z1 = list(); S1 = list()

#Variables to store Channel 2 data 
X2 = list(); Y2 =  list(); Z2 = list(); S2 = list()

#opening the csv file
# 'r' specifies that we want to read this file
#csv_reader is the name of the reader object that we have created 
with open (sys.argv[1], 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    #Iterating through contents in the file
    for line in csv_reader:
        #Removing points with radius less than 0.27
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
        
X1 = numpy.array(X1, dtype=float); Y1 = numpy.array(Y1, dtype=float); Z1 = numpy.array(Z1, dtype=float); S1 = numpy.array(S1, dtype=float)
X2 = numpy.array(X2, dtype=float); Y2 = numpy.array(Y2, dtype=float); Z2 = numpy.array(Z2, dtype=float); S2 = numpy.array(S2, dtype=float)

#Saving each channel's data in new files
numpy.savetxt("C1.csv", numpy.column_stack((X1, Y1, Z1, S1)), delimiter=",", fmt='%s')
numpy.savetxt("C2.csv", numpy.column_stack((X2, Y2, Z2, S2)), delimiter=",", fmt='%s')



