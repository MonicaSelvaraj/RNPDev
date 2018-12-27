import numpy

print('Formatting Data')

fn = list(); mcd = list();sdcd = list()
r = list();f = list(); p=list(); rse = list(); fse = list();pse = list();pitch = list()
#Channel 1 
C1C1C2r = list(); C1C1C2f = list(); C1C1C2p = list(); C1C1C2rse = list(); C1C1C2fse = list(); C1C1C2pse = list()
C1C1C3r = list(); C1C1C3f = list(); C1C1C3p = list(); C1C1C3rse = list(); C1C1C3fse = list(); C1C1C3pse = list()
C1C1C2Pitch = list(); C1C1C3Pitch = list()
#Channel 2
C2C1C2r = list(); C2C1C2f = list(); C2C1C2p = list(); C2C1C2rse = list(); C2C1C2fse = list(); C2C1C2pse = list()
C2C1C3r = list(); C2C1C3f = list(); C2C1C3p = list(); C2C1C3rse = list(); C2C1C3fse = list(); C2C1C3pse = list()
C2C1C2Pitch = list(); C2C1C3Pitch = list()

a = open('FileNames.txt','r')
for line in a:
    fn.append(line.strip())

b = open('MeanCortexDensity.txt','r')
for line in b:
    mcd.append(line.strip())
    
c = open('sdCortexDensity.txt','r')
for line in c:
    sdcd.append(line.strip())

d = open('FitRadius.txt','r')
for line in d:
    r.append(line.strip())

e = open('FitFrequency.txt','r')
for line in e:
    f.append(line.strip())

g = open('FitPhase.txt','r')
for line in g:
    p.append(line.strip())

h = open('FitRadiusSE.txt','r')
for line in h:
    rse.append(line.strip())

i = open('FitFrequencySE.txt','r')
for line in i:
    fse.append(line.strip())

j = open('FitPhaseSE.txt','r')
for line in j:
    pse.append(line.strip())

k = open('Pitch.txt','r')
for line in k:
    pitch.append(line.strip())

for i in range(0,len(r),4):
    C1C1C2r.append(r[i])
for i in range(1,len(r),4):
    C1C1C3r.append(r[i])
for i in range(2,len(r),4):
    C2C1C2r.append(r[i])
for i in range(3,len(r),4):
    C2C1C3r.append(r[i])

for i in range(0,len(f),4):
    C1C1C2f.append(f[i])
for i in range(1,len(f),4):
    C1C1C3f.append(f[i])
for i in range(2,len(f),4):
    C2C1C2f.append(f[i])
for i in range(3,len(f),4):
    C2C1C3f.append(f[i])

for i in range(0,len(p),4):
    C1C1C2p.append(p[i])
for i in range(1,len(p),4):
    C1C1C3p.append(p[i])
for i in range(2,len(p),4):
    C2C1C2p.append(p[i])
for i in range(3,len(p),4):
    C2C1C3p.append(p[i])

for i in range(0,len(rse),4):
    C1C1C2rse.append(rse[i])
for i in range(1,len(rse),4):
    C1C1C3rse.append(rse[i])
for i in range(2,len(rse),4):
    C2C1C2rse.append(rse[i])
for i in range(3,len(rse),4):
    C2C1C3rse.append(rse[i])

for i in range(0,len(fse),4):
    C1C1C2fse.append(fse[i])
for i in range(1,len(fse),4):
    C1C1C3fse.append(fse[i])
for i in range(2,len(fse),4):
    C2C1C2fse.append(fse[i])
for i in range(3,len(fse),4):
    C2C1C3fse.append(fse[i])

for i in range(0,len(pse),4):
    C1C1C2pse.append(pse[i])
for i in range(1,len(pse),4):
    C1C1C3pse.append(pse[i])
for i in range(2,len(pse),4):
    C2C1C2pse.append(pse[i])
for i in range(3,len(pse),4):
    C2C1C3pse.append(pse[i])

for i in range(0,len(pitch),4):
    C1C1C2Pitch.append(pitch[i])
for i in range(1,len(pitch),4):
    C1C1C3Pitch.append(pitch[i])
for i in range(2,len(pitch),4):
    C2C1C2Pitch.append(pitch[i])
for i in range(3,len(pitch),4):
    C2C1C3Pitch.append(pitch[i])



fn = numpy.array(fn); mcd = numpy.array(mcd); sdcd = numpy.array(sdcd)
C1C1C2r = numpy.array(C1C1C2r)
C1C1C2f = numpy.array(C1C1C2f)
C1C1C2p = numpy.array(C1C1C2p)
C1C1C2rse = numpy.array(C1C1C2rse)
C1C1C2fse = numpy.array(C1C1C2fse)
C1C1C2pse = numpy.array( C1C1C2pse)
C1C1C3r = numpy.array(C1C1C3r)
C1C1C3f = numpy.array(C1C1C3f)
C1C1C3p = numpy.array(C1C1C3p)
C1C1C3rse = numpy.array(C1C1C3rse)
C1C1C3fse = numpy.array(C1C1C3fse)
C1C1C3pse = numpy.array(C1C1C3pse)
C1C1C2Pitch = numpy.array(C1C1C2Pitch)
C1C1C3Pitch  = numpy.array(C1C1C3Pitch)
    
C2C1C2r = numpy.array(C2C1C2r)
C2C1C2f = numpy.array(C2C1C2f)
C2C1C2p = numpy.array(C2C1C2p)
C2C1C2rse = numpy.array(C2C1C2rse)
C2C1C2fse = numpy.array(C2C1C2fse)
C2C1C2pse = numpy.array( C2C1C2pse)
C2C1C3r = numpy.array(C2C1C3r)
C2C1C3f = numpy.array(C2C1C3f)
C2C1C3p = numpy.array(C2C1C3p)
C2C1C3rse = numpy.array(C2C1C3rse)
C2C1C3fse = numpy.array(C2C1C3fse)
C2C1C3pse = numpy.array(C2C1C3pse)
C2C1C2Pitch = numpy.array(C2C1C2Pitch)
C2C1C3Pitch  = numpy.array(C2C1C3Pitch)


numpy.savetxt('Results.csv', numpy.column_stack((fn, mcd, sdcd,C1C1C2r,C1C1C2f,C1C1C2p, C1C1C2Pitch,
                                                 C1C1C2rse , C1C1C2fse, C1C1C2pse, C1C1C3r, C1C1C3f, C1C1C3p,C1C1C3Pitch,
                                                 C1C1C3rse, C1C1C3fse,C1C1C3pse,C2C1C2r,C2C1C2f,C2C1C2p, C2C1C2Pitch,
                                                 C2C1C2rse, C2C1C2fse, C2C1C2pse, C2C1C3r, C2C1C3f, C2C1C3p,C2C1C3Pitch,
                                                 C2C1C3rse,C2C1C3fse,C2C1C3pse)), delimiter=",", fmt='%s')

