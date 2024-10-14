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

#%% load image and rescale to (0,1)
# todo | load ct_image_xx.png for xx = 0..2
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

#%% Compute radon transform using skimage
theta = np.linspace(0.0, 180.0, max(imag.shape))
sinog = radon(imag, circle=False, theta=theta)

# plot | todo: specify axis and units
plt.imshow(sinog, cmap='gray')
plt.xlabel(r'$\theta$ (in degrees)')
plt.ylabel(r'$\rho$ (in pixels)')
plt.colorbar()

#%% load sinogram
# todo | load ct_image_xx_sinog.png for xx = 4..5
image_file = './data/ct_image_4_sinog.png'
sinog = plt.imread(image_file)
sinog = sinog[:,:,:3].sum(2)/3*255

# plot
plt.imshow(sinog, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.colorbar()

#%% Compute inverse radon transform using skimage
theta = np.linspace(0.0, 180.0, max(imag.shape))
recon = iradon(sinog, circle=False, theta=theta)

# plot | todo: specify axis and units
plt.imshow(recon, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.colorbar()