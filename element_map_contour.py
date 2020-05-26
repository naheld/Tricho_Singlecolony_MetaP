#!/usr/bin/env python3
# Note: This example code is presented for reproducibility/example purposes only. 

import matplotlib.pyplot as plt

from numpy import zeros_like, dstack
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion, binary_dilation, binary_opening, binary_closing
from skimage import io as skio
from skimage.morphology import disk
from skimage.measure import find_contours
from skimage.color import gray2rgb
from skimage.filters import threshold_triangle, threshold_li, threshold_isodata
"""scikit-image version 0.15.0"""

from skimage.filters import try_all_threshold

def gen_img(img, dpi=96):

    nrows, ncols = img.shape[0], img.shape[1]
    fig = plt.figure(figsize=(ncols/dpi, nrows/dpi), dpi=dpi)
    axes = plt.Axes(fig,[0,0,1,1])
    fig.add_axes(axes)
    axes.set_axis_off()
    axes.imshow(img, cmap=plt.cm.gray)


    return fig, axes

def make_trace(contours, image, show=True):
    trace = zeros_like(image)
    fig, ax = gen_img(image)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        for coords in contour:
            coords = tuple(int(x) for x in coords)
            trace[coords] = 1
    if show==True:
        plt.show()

    return binary_dilation(trace,structure=disk(2),iterations=1)

image = skio.imread('noncontour.png')
image_red = image[:,:,0]
image_green = image[:,:,1]
# image_blue = image[:,:,2]

image_rg = image_green + image_red

image_left = image_rg[:,:525]
image_right = image_rg[:,525:]

# fig,ax = try_all_threshold(image_right)
# plt.show()

thresh_left = threshold_triangle(image_left)
binary_left = image_left > thresh_left
binary_left = binary_erosion(binary_left, structure=disk(2), iterations=1)
binary_left = binary_dilation(binary_left, structure=disk(2),iterations=3)
# gen_img(binary_left);plt.show()
contours_left = find_contours(binary_left, 0.8)
contours_left = sorted(contours_left, key=len)

thresh_right = threshold_isodata(image_right)
binary_right = image_right > thresh_right
binary_right = binary_dilation(binary_right, structure=disk(2), iterations=1)
binary_right = binary_fill_holes(binary_right)
# gen_img(binary_right);plt.show()
contours_right = find_contours(binary_right, 0.8)
contours_right = sorted(contours_right, key=len)
contours_right_corr = []
for contour in contours_right:
    contour[:,1] = contour[:,1]+525
    contours_right_corr.append(contour)


trace_left = make_trace(contours_left[-5:], image_rg,show=True)*200
trace_right = make_trace(contours_right[-3:], image_rg,show=True)*255
trace_all = trace_left+trace_right

final = dstack((trace_all, trace_all, trace_all, trace_all))
#
# final = gray2rgb(trace_rgb, alpha=True)
# final[:,:,-1] = trace_rgb

skio.imsave("trace.png", final)
