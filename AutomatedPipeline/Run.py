'''
Runs all other files
'''
#!/usr/bin/python
import sys, os
#Splitting channels and visualizing
os.system('python splitChannels.py Input/aa.aura_dnd_nanos.UW-39_63x_e1_1.csv')
os.system('python ScatterPlot.py')

#Clustering
os.system('python Clustering.py C1.csv ClusteredC1.csv')
os.system('python Clustering.py C2.csv ClusteredC2.csv')
os.system('python ClusteredPlots.py')

#Straightening
#os.system('python Straightening.py')

#PCA and 2D plots
#os.system('python 2DProjections.py StraightenedC1.csv ComponentsC1.csv r Output/PrincipalComponentsC1.png')
#os.system('python 2DProjections.py StraightenedC2.csv ComponentsC2.csv g Output/PrincipalComponentsC2.png')

#Removing outliers
#os.system('python Outliers.py')

#Curve Fitting
#os.system('python CurveFittingComponents.py')
