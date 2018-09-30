'''
Goal: Detect the z's to remove 
Read in x-y image

'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## Set free params
germplasm_sds_thresh = 1
axis_fraction_thresh = 0.25

def get_original_image(filename):
    return(plt.imread(filename))

def Watershed(image, sdt, aft):
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


original_matrix = get_original_image('x-y.png')
print(original_matrix)
working_copy = original_matrix.copy()

