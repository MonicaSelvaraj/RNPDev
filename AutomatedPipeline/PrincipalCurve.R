#Importing data from excel to R
sampledata <- read.csv("PCTrial.csv")
#Converting the dataframe to a matrix 
sampledata <- data.matrix(sampledata)

#Fitting the first principal curve 
library(princurve)
fit <-(principal_curve(sampledata))

#Obtaining fitted points on the curve and ordering of points
fits <- fit$s
fitord <- fit$ord

#Exporting to csv files 
write.table(fits, "fitpoints.csv", sep = ",", row.names = F, col.names = F)
write.table(fitord, "fitorder.csv", sep = ",", row.names = F, col.names = F)


