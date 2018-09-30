#!/bin/env python

import sys, os
from math import sqrt
from scipy.spatial.distance import pdist
import numpy as np
## Set for remote plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.feature import blob_log
from skimage.filters import sobel
from scipy import ndimage as ndi
from skimage import morphology
from skimage.feature import canny
from skimage.color import label2rgb

sys.setrecursionlimit(10000) ## Default ceiling is 1000

## Set free params
germplasm_sds_thresh = 1
axis_fraction_thresh = 0.25
log_small_step = 0.0001
log_big_step = 0.005
log_delta = 40
log_start = 0.0001
log_blobs_max = 500
log_blobs_keep = 150

## Read args
tif1, tif2 = sys.argv[1], sys.argv[2]

## Initialize matricies
original_matrix = {}
germplasm_matrix = {}
rnp_matrix = {}
germplasm_indicies = {}
rnp_indicies = {}

def get_original_image(filename):
    ## Read in tif
    return(plt.imread(filename))

def clean_image(image, sdt, aft, diag_plot_flag):
    origimg = image.copy()
    ## Flatten image array and get cutoff
    allval = np.hstack(image)
    nonzero = []
    for v in allval:
        if(v>0):
            nonzero.append(v)
    nz = np.array(nonzero)
    cut = nz.mean() + nz.std()*sdt
    ## Sobel elevation map
    elevation_map = sobel(image)
    ## Define markers
    markers = np.zeros_like(image)
    markers[image < allval.mean()] = 1
    markers[image > cut] = 2
    ## Watershed segmentation
    seg = morphology.watershed(elevation_map, markers)
    ## Fill in holes
    fill_h = ndi.binary_fill_holes(seg - 1)
    ## Get rid of small stuff (below axis fraction threshold)
    size_thresh = float(image.shape[1] + image.shape[0])/2 * aft
    seg = morphology.remove_small_objects(fill_h, size_thresh)
    ## Define label
    labeled, _ = ndi.label(seg)
    ## Label based on segmentation
    image_label_overlay = label2rgb(labeled, image=image)    
    ## Create new image based on labeled segmentation
    cleanimg = image
    cleanimg[labeled == 0] = 0
    if(diag_plot_flag!='none'):
        ## Plot it out
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 12), sharex=True, sharey=True)
        axes[0].imshow(origimg, interpolation='nearest')
        axes[0].set_title('Original Image')
        axes[1].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        axes[1].contour(seg, [0.5], linewidths=1.2, colors='y')
        axes[1].set_title('Watershed Segmentation')
        axes[2].imshow(image_label_overlay, interpolation='nearest')
        axes[2].set_title('Labeled Segments')
        axes[3].imshow(cleanimg, interpolation='nearest')
        axes[3].set_title('Cleaned Image')
        for a in axes:
            a.axis('off')
            a.set_adjustable('box-forced')
        plt.tight_layout()
        plt.savefig(diag_plot_flag+'_cleaned.jpg')
    return(cleanimg)

def get_nz(img, nz):
    for x in range(0,img.shape[1]):
        for y in range(0,img.shape[0]):
            if(img[y][x] > 0):
                if(y in nz):
                    nz[y][x] = 1
                else:
                    nz[y] = {}
                    nz[y][x] = 1
    return(nz)

def log(img, diag_plot_flag, thresh):
    ## Laplacian of gaussian blob detection
    blobs_log = blob_log(img, min_sigma=2, max_sigma=4, num_sigma=10, threshold=thresh)
    ## Get radius
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    blob_num = 0
    for y, x, r in blobs_log:
        blob_num += 1
    if(blob_num > log_blobs_max+log_delta):
        thresh += log_big_step
        #sys.stderr.write("Blob num too high ("+str(blob_num)+"), trying thresh= "+str(thresh)+"\n")
        return(log(img, diag_plot_flag, thresh))
    elif(blob_num > log_blobs_max):
        thresh += log_small_step
        #sys.stderr.write("Blob num too high ("+str(blob_num)+"), trying thresh= "+str(thresh)+"\n")
        return(log(img, diag_plot_flag, thresh))
    else:
        sys.stderr.write("Blobs within range: "+str(blob_num)+"\n")
        ## Reorder the blob array
        blob_trans = {}
        for y, x, r in blobs_log:
            r_int = int(round(r))
            x_lb, x_ub = int(x-r_int), int(x+r_int)
            y_lb, y_ub = int(y-r_int), int(y+r_int)
            t, n = 0, 0
            for x_pos in range(x_lb, x_ub+1):
                for y_pos in range(y_lb, y_ub+1):
                    if(y_pos < img.shape[0]):
                        if(x_pos < img.shape[1]):
                            t += img[y_pos][x_pos]
                            n += 1
            center = str(x)+','+str(y)
            blob_trans[center] = {}
            blob_trans[center]['avg_int'] = float(t)/n
            blob_trans[center]['x'] = x
            blob_trans[center]['y'] = y
            blob_trans[center]['r'] = r_int
            blob_trans[center]['t'] = t
            blob_trans[center]['n'] = n
        blob_trans_sort = sorted(
            blob_trans.items(),
            key=lambda x: (-x[1]['avg_int'])
        )
        blob_keeps = np.zeros_like(img)
        i = 1
        ymax, xmax = img.shape[0], img.shape[1]
        for c, l in blob_trans_sort:
            if(i <= log_blobs_keep):
                #print "\t".join([str(l['x']), str(l['y']), str(l['r']), str(l['avg_int']), str(l['t']), str(l['n'])])
                x_lb, x_ub = int(l['x']-l['r']), int(l['x']+l['r'])
                y_lb, y_ub = int(l['y']-l['r']), int(l['y']+l['r'])
                for x_pos in range(x_lb, x_ub+1):
                    for y_pos in range(y_lb, y_ub+1):
                        if (y_pos < ymax) and (x_pos < xmax):
                            blob_keeps[y_pos][x_pos] = img[y_pos][x_pos]
                i += 1
        #sys.stderr.write("\t".join(['thresh', str(thresh)])+"\n")
        #sys.stderr.write("\t".join(['blobs', str(blob_num)])+"\n")
        if(diag_plot_flag != 'none'):
            ## Plot
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 12), sharex=True, sharey=True)
            axes[0].imshow(img, interpolation='nearest')
            axes[0].set_title('Cleaned Image')
            axes[1].imshow(img, interpolation='nearest')
            axes[1].set_title('Laplacian of Gaussian')
            for y, x, r in blobs_log:
                col = 'w'
                if(blob_keeps[int(y)][int(x)] > 0):
                    col = 'r'
                c = plt.Circle((x, y), r, color=col, linewidth=2, fill=False)
                axes[1].add_patch(c)
            axes[2].imshow(blob_keeps, interpolation='nearest')
            axes[2].set_title('Highest Intensity Blobs (n='+str(log_blobs_keep)+')')
            for a in axes:
                a.axis('off')
                a.set_adjustable('box-forced')
            plt.tight_layout()
            plt.savefig(diag_plot_flag+'_log.jpg')
        return(blob_keeps)
    
## Main loop
for filename in [tif1, tif2]:
    pref = os.path.splitext(filename)[0]
    if(pref=='skip'):
        continue
    original_matrix[pref] = get_original_image(filename)
    working_copy = original_matrix[pref].copy()
    working_copy = clean_image(working_copy, germplasm_sds_thresh, axis_fraction_thresh, pref+'_germplasm')
    germplasm_matrix[pref] = working_copy.copy()
    #germplasm_indicies = get_nz(germplasm_matrix[pref], germplasm_indicies)
    working_copy = log(working_copy, pref+'_rnp', log_start)
    rnp_matrix[pref] = working_copy.copy()
## Print it all out
#print "\t".join(['source','x','y', 'original', 'germplasm', 'rnp'])
#for f in original_matrix.keys():
#    for x in range(0,original_matrix[f].shape[1]):
#        for y in range(0,original_matrix[f].shape[0]):
#            print "\t".join([f, str(x), str(y), str(original_matrix[f][y][x]), str(germplasm_matrix[f][y][x]), str(rnp_matrix[f][y][x])])
## Germplasm and RNP comparisons
pref1, pref2 = os.path.splitext(tif1)[0], os.path.splitext(tif2)[0]
print "\t".join(['base','comparison', 'x','y', pref1, pref2])
for x in range(0,germplasm_matrix[pref1].shape[1]):
    for y in range(0,germplasm_matrix[pref1].shape[0]):
        print "\t".join(['full_image','full_image', str(x), str(y), str(original_matrix[pref1][y][x]), str(original_matrix[pref2][y][x])])
        if(germplasm_matrix[pref1][y][x] > 0):
            print "\t".join([pref1,'germplasm', str(x), str(y), str(germplasm_matrix[pref1][y][x]), str(germplasm_matrix[pref2][y][x])])
        if(germplasm_matrix[pref2][y][x] > 0):
            print "\t".join([pref2,'germplasm', str(x), str(y), str(germplasm_matrix[pref1][y][x]), str(germplasm_matrix[pref2][y][x])])
        if(rnp_matrix[pref1][y][x] > 0):
            print "\t".join([pref1,'rnp', str(x), str(y), str(rnp_matrix[pref1][y][x]), str(rnp_matrix[pref2][y][x])])
        if(rnp_matrix[pref2][y][x] > 0):
            print "\t".join([pref2,'rnp', str(x), str(y), str(rnp_matrix[pref1][y][x]), str(rnp_matrix[pref2][y][x])])
