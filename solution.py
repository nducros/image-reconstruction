# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:41:54 2024

@author: ducros
"""
# -*- coding: utf-8 -*-

#from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon, iradon

#%% Exercise 1
# load image and rescale to (0,1) todo | load ct_image_xx.png for xx = 0..2
image_file = './data/ct_image_0.png'
imag = plt.imread(image_file)

# rescale
imag = imag[:,:,:-1].sum(-1)
imag = (imag - imag.min())/(imag.max() - imag.min())

# plot
plt.imshow(imag, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.colorbar()

# Compute radon transform using skimage
theta = np.linspace(0.0, 180.0, max(imag.shape))
sinog = radon(imag, circle=False, theta=theta)

# plot | todo: specify axis and units
plt.imshow(sinog, cmap='gray')
plt.xlabel(r'$\theta$ (in degrees)')
plt.ylabel(r'$\rho$ (in pixels)')
plt.colorbar()

#%% Exercise 2 
# load sinogram
# todo | load ct_image_xx_sinog.png for xx = 4..5
image_file = './data/ct_image_4_sinog.png'
sinog = plt.imread(image_file)
sinog = sinog[:,:,:3].sum(2)/3*255

# plot
plt.imshow(sinog, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.colorbar()

# Compute inverse radon transform using skimage
theta = np.linspace(0.0, 180.0, max(imag.shape))
recon = iradon(sinog, circle=False, theta=theta)

# plot | todo: specify axis and units
plt.imshow(recon, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.colorbar()

#%% Exercise 3 
# Reconstruct with 2% additive Gaussian noise using different filters (e.g.,
# ramp, cosine, hann, None)

prct = 0.02 # noise percentage
filter_name = 'ramp' # E.g., 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None

# Add noise and reconstruct
sinog_noise = sinog + prct*sinog.max()*np.random.standard_normal(size=sinog.shape)
recon = iradon(sinog_noise, circle=False, theta=theta, filter_name=filter_name)

# plot | todo: specify axis and units
plt.imshow(recon, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.colorbar()

#%% Exercise 4
# Construct the discrete forward operator corresponding to the Radon transform
# of a 32x32 image computed under 90 view angles over [0,\pi) using a linear
# detector of 45 pixels.

from skimage.transform import rescale

img_size = 32  # we assume image is square
n_angle = 90
n_detec = 45

# Init matrix A
A = np.zeros((n_detec*n_angle, img_size*img_size)) #COMPLETE

# Build the forward operator, one column at a time. 
theta = np.linspace(0.0, 180.0, n_angle+1)[:-1]

for i in range(img_size):
    for j in range(img_size):
        # Activating a single pixel of the object image
        image = np.zeros((img_size,img_size)) ##COMPLETE
        image[i,j] = 1  ##COMPLETE
        
        # Radon transform
        sinogram = radon(image, theta, circle=False) ##COMPLETE
        sinogram = rescale(sinogram, scale=(n_detec/sinogram.shape[0],1), mode='reflect')
        
        # Concatenating results in matrix A 
        A[:,img_size*i+j] = np.reshape(sinogram, (n_detec*n_angle, ))

# Plot forward matrix | todo: specify axis
fig, ax = plt.subplots()
ax.imshow(A)
ax.set_title("A")
ax.set_ylabel(r"Projection ray $(r, \theta)$")  ##COMPLETE
ax.set_xlabel(r"Image pixel $x$")               ##COMPLETE
plt.show()
