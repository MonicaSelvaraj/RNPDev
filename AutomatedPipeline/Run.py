'''
Runs all other files
'''
#!/usr/bin/python
import sys, os

InputPath = 'Input'
os.mkdir("Output")
for filename in os.listdir(InputPath):
    if (filename == '.DS_Store'):
        continue
    os.system('echo && echo && echo'); os.system('echo Input/%s' % filename); os.system('echo')
    os.chdir('Output')
    outputFile = filename[:-4]
    os.mkdir('%s' % outputFile)
    os.chdir('..')
    #Splitting channels and visualizing the aggregate 
    os.system('python splitChannels.py Input/%s' % filename)
    os.system('python ScatterPlot.py')
    
    #Clustering
    os.system('python Clustering.py C1.csv ClusteredC1.csv')
    os.system('python Clustering.py C2.csv ClusteredC2.csv')
    os.system('python ClusteredPlots.py')

    #CortexRemoval
    os.system('python CortexRemoval.py')
    os.system('python ScatterPlotCortexRemoved.py')
    
#Straightening
    os.system('python Straightening.py')

#PCA and 2D plots
    os.system('python 2Dprojections.py StraightenedC1.csv ComponentsC1.csv r Output/%s/PrincipalComponentsC1.png' % outputFile)
    os.system('python 2Dprojections.py StraightenedC2.csv ComponentsC2.csv g Output/%s/PrincipalComponentsC2.png' % outputFile)

#Removing outliers
    os.system('python Outliers.py')

#Curve Fitting
    os.system('python CurveFittingComponents.py')

#Formatting the data collected
os.system('python DataFormatting.py')









    
