'''
- Reading in the clustered points
- Centering the points
- Reorienting the aggregate so PC1 is aligned with z, PC2 is aligned with x, PC3 is aligned with y 
'''
#!/usr/bin/python
import numpy 
import csv
import sys

print("Orienting aggregate")

x = list(); y = list(); z = list(); n = 0; #n keeps track of where channel 1 stops and c2 starts

with open ('ClusteredC1.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        x.append(line[0]); y.append(line[1]); z.append(line[2])
        n = n +1
with open ('ClusteredC2.csv', 'r') as csv_file:
    csv_reader = csv.reader (csv_file)
    for line in csv_reader:
        x.append(line[0]); y.append(line[1]); z.append(line[2])
        
#Converting lists to numpy arrays
x = numpy.array(x, dtype = float); y = numpy.array(y, dtype = float); z = numpy.array(z, dtype = float)

def PCA(X, Y, Z, n):
    #Making the principal componenets the axes of the coordinates
    points = numpy.concatenate((X[:, numpy.newaxis], 
                       Y[:, numpy.newaxis], Z[:, numpy.newaxis]), axis = 1)
    center = points.mean(axis = 0)
    centered = points-center
    centeredT = numpy.transpose(centered)
    constant = 1/(len(X)-1)
    covarianceM = constant*numpy.matmul(centeredT, centered)
    w, v = numpy.linalg.eig(covarianceM)
    Pc1 = v[:,0]
    Pc2 = v[:,1]
    Pc3 = v[:, 2]
    #New points 
    Znew = numpy.dot(centered , Pc1) 
    Xnew = numpy.dot(centered , Pc2)
    Ynew = numpy.dot(centered, Pc3)

    numpy.savetxt("OrientedC1.csv", numpy.column_stack((Xnew[:n], Ynew[:n], Znew[:n])), delimiter=",", fmt='%s')
    numpy.savetxt("OrientedC2.csv", numpy.column_stack((Xnew[n+1:len(Xnew)], Ynew[n+1:len(Xnew)], Znew[n+1:len(Xnew)])), delimiter=",", fmt='%s')

    return(Xnew, Ynew, Znew)

newPoints = PCA(x,y,z, n)
