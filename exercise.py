# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:45:29 2024

@author: ducros
"""
# -*- coding: utf-8 -*-

#from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon, iradon

#%% load image and rescale to (0,1)
# todo | load ct_image_xx.png for xx = 0..2
image_file = 
imag = plt.imread(image_file)

# rescale

# plot | todo: specify axis and units
plt.imshow(imag, cmap='gray')
plt.colorbar()

#%% Compute radon transform using skimage
theta = np.linspace(0.0, 180.0, max(imag.shape))
sinog = radon(imag, circle=False, theta=theta)

# plot | todo: specify axis and units
plt.imshow(sinog, cmap='gray')
plt.colorbar()

#%% load sinogram 
# todo | load ct_image_xx.png for xx = 0..2
image_file = 
imag = plt.imread(image_file)

# rescale

# plot | todo: specify axis and units
plt.imshow(imag, cmap='gray')
plt.colorbar()


#%% Compute inverse radon transform using skimage
recon = 

# plot | todo: specify axis and units
plt.imshow(recon, cmap='gray')
plt.colorbar()