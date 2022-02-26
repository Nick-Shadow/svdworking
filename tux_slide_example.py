size_cimg = 0
size_gimg = 0
size_usvm = 0
size_dimg = 0

from matplotlib import image, pyplot

tux_color = image.imread('assetdata/tux.png')

pyplot.imshow(tux_color)
pyplot.show()

size_cimg = tux_color.__sizeof__()

import numpy as numpy

def rgb_to_gray(c):
    return numpy.dot(c[...,:3], [0.3, 0.59, 0.11])

tux_gray = rgb_to_gray(tux_color)

size_gimg = tux_gray.__sizeof__()

pyplot.imshow(tux_gray, cmap='gray')

from numpy.linalg import svd

k = 100
u, s, v = svd(tux_gray, full_matrices=False)
size_usvm = (u[:,:k]).__sizeof__() + (s[:k]).__sizeof__() + (v[:k,:]).__sizeof__()
size_dimg = (numpy.dot(u[:,:k], numpy.dot(numpy.diag(s[:k]), v[:k,:]))).__sizeof__()


def compress_grayscale_svd(mat, k):
    u, s, v = svd(mat, full_matrices=False)
    size_usvm = (u[:,:k]).__sizeof__() + (s[:k]).__sizeof__() + (v[:k,:]).__sizeof__()
    reconstructed_matrix = numpy.dot(u[:,:k], numpy.dot(numpy.diag(s[:k]), v[:k,:]))
    return reconstructed_matrix, s

def get_compression_ratio(k, shape1,shape2):
    return 100.0 * ( k * (shape1 + shape2) + k) / (shape1 * shape2)

def show_compressed_grayscale_image(img, k):
    reconstructed_img, sigma = compress_grayscale_svd(img, k)
    figure, xy_axises = pyplot.subplots(1, 2, figsize=(8, 5))
    xy_axises[0].plot(sigma)
    compression_ratio = get_compression_ratio(k, img.shape[0], img.shape[1])
    xy_axises[1].set_title("Compression Ratio is {:.2f}%".format(compression_ratio))
    xy_axises[1].imshow(reconstructed_img, cmap='gray')
    xy_axises[1].axis('off')
    figure.tight_layout()
  
from ipywidgets import interact, fixed
interact(show_compressed_grayscale_image, img=fixed(tux_gray), k=(1, 100))

print("Color Image size: {c:}\nGray Image size: {g:}\nCompressed Size: {com:}\nDecompressed Image Size: {d:}".format(c = size_cimg, g = size_gimg, com = size_usvm, d = size_dimg))