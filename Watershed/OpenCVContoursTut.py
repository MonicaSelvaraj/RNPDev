'''
PART1: Edge detection and segmentation - https://www.youtube.com/watch?v=STnoJ3YCWus&t=57s
PART2: Using regionprops to find the largest segments and its coordinates - gives the required x-y
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
from skimage import feature 
from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage import morphology, segmentation 
from skimage import color
from sklearn.cluster import KMeans

#Reading and plotting original image 
original_image = cv2.imread('xy.png')

#Converting image to grayscale
original_image_gray = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
#ngcm= greycomatrix(im, [1], [0], 256, symmetric=False, normed=True)
#ret, thresh = cv2.threshold(imgray, 122,255,0)
plt.imshow(original_image_gray, cmap="gray")
plt.show()

#Using canny edge detector to find the edges of the image
edges = skimage.feature.canny(original_image_gray, sigma = 4)
plt.imshow(edges)
plt.show()

#Converting edge image into a landscape using distance transform
#Distance transform is the distance to the closest background pixel, applying to inverse of edge mask - background is foreground and foreground is background  
dt = distance_transform_edt(~edges)
plt.imshow(dt)
plt.show()

#Finding the locations of the fountains - local peaks 
local_max = feature.peak_local_max(dt, indices=False, min_distance=5)
plt.imshow(local_max, cmap='gray')
plt.show()

#Getting the coordinates of the peaks and vizualizing them better
peak_idx = feature.peak_local_max(dt, indices=True,min_distance=5)
plt.plot(peak_idx[:, 1], peak_idx[:,0], 'r.')
plt.imshow(dt)
plt.show()

#Labeling segments 
markers = measure.label(local_max)

#Watershed
labels = morphology.watershed(-dt, markers)
plt.imshow(segmentation.mark_boundaries(original_image_gray, labels))
plt.show()

#Displaying the segments by averaging underlying pixel values 
plt.imshow(color.label2rgb (labels, image = original_image_gray, kind= 'avg'), cmap='gray')
plt.show()

#Merging appropriate regions together
#First distinguishing foreground from background by using K-Means clustering
regions = measure.regionprops(labels, intensity_image = original_image_gray)
region_means = [r.mean_intensity for r in regions]
model = KMeans(n_clusters=2)
region_means = np.array(region_means).reshape(-1,1)
model.fit(region_means)
#print(model.cluster_centers_) - threshold for backgound and foreground 

#predicts label for each region 
bg_fg_labels = model.predict(region_means)

#Labeling image appropriately
classified_labels = labels.copy()
for bg_fg, region in zip(bg_fg_labels, regions):
    classified_labels[tuple(region.coords.T)] = bg_fg
    segmented_image = color.label2rgb(classified_labels, image = original_image_gray)
plt.imshow(segmented_image)
plt.savefig('segmented_image.png')
plt.show()

