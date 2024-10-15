# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:40:52 2024

@author: ducros
"""
#from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon, iradon
from pathlib import Path

#%% load image and rescale to (0,1)
image_file = Path('./data-lesson/ct_image_0.png')
save_folder = Path('lesson')

imag = plt.imread(image_file)

# rescale
imag = imag[:,:,:-1].sum(-1)
imag = (imag - imag.min())/(imag.max() - imag.min())

# plot
plt.imshow(imag, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.colorbar() #

# save for lecture
plt.savefig(save_folder / (image_file.stem + '.pdf'),bbox_inches='tight')

#%% Compute radon transform using skimage
theta = np.linspace(0.0, 180.0, max(imag.shape))
sinog = radon(imag, circle=False, theta=theta)

# plot | todo: specify units
plt.imshow(sinog, cmap='gray')
plt.xlabel(r'$\theta$ (in degrees)')
plt.ylabel(r'$\rho$ (in pixels)')
plt.colorbar()

plt.savefig(save_folder / (image_file.stem + '_sinog.pdf'), bbox_inches='tight')

# save for lecture
plt.imsave(save_folder / (image_file.stem + '_sinog.png'), sinog, cmap='gray')

#%% Compute inverse radon transform using skimage
recon = iradon(sinog, circle=False, theta=theta)

# plot | todo: specify units
plt.imshow(recon, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.colorbar()

#%% load image and rescale to (0,1)
image_file = Path('./data-lesson/ct_image_5.png')
imag = plt.imread(image_file)

# rescale
imag = imag[:,:,:-1].sum(-1)
imag = (imag - imag.min())/(imag.max() - imag.min())

# plot
plt.imshow(imag, cmap='gray')
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
plt.colorbar()

# save for lecture
plt.savefig(save_folder / (image_file.stem + '.pdf'),bbox_inches='tight')

#%% backprojection

for n in [1,2,4,16,32,64,128]:
    
    #
    theta = np.linspace(0.0, 180.0, n+1)[:-1]
    sinog = radon(imag, circle=False, theta=theta)
    retro = iradon(sinog, circle=False, theta=theta, filter_name=None)
    
    # plot
    plt.imshow(retro, cmap='gray')
    plt.xlabel(r'$x_1$ (in pixels)')
    plt.ylabel(r'$x_2$ (in pixels)')
    plt.colorbar()
    
    # save for lecture
    plt.savefig(save_folder / f'{image_file.stem}_bp_{n:03}.pdf',
                bbox_inches='tight')
    plt.close()

#%% filtered backprojection

for n in [1,2,4,16,32,64,128]:   
    #
    theta = np.linspace(0.0, 180.0, n+1)[:-1]
    sinog = radon(imag, circle=False, theta=theta)
    retro = iradon(sinog, circle=False, theta=theta)
    
    # plot
    plt.imshow(retro, cmap='gray')
    plt.xlabel(r'$x_1$ (in pixels)')
    plt.ylabel(r'$x_2$ (in pixels)')
    plt.colorbar()
    
    # save for lecture
    plt.savefig(save_folder / f'{image_file.stem}_fbp_{n:03}.pdf',
                bbox_inches='tight')
    plt.close()

#%% load image and rescale to (0,1)
image_file = Path('./data-lesson/ct_image_5.png')
imag = plt.imread(image_file)

# rescale
imag = imag[:,:,:-1].sum(-1)
imag = (imag - imag.min())/(imag.max() - imag.min())

# Noisy synogram
theta = np.linspace(0.0, 180.0, 256+1)[:-1]
prct = 0.1      # noise percentage
vmin, vmax = None, None
np.random.seed(1)  # for reproducibility
sinog = radon(imag, circle=False, theta=theta)

filter_list = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']

for f in filter_list:
    sinog += prct*sinog.max()*np.random.standard_normal(size=sinog.shape)
    retro = iradon(sinog, circle=False, theta=theta, filter_name=f)

    # plot
    plt.imshow(retro, cmap='gray', vmin=vmin, vmax=vmax)
    plt.xlabel(r'$x_1$ (in pixels)')
    plt.ylabel(r'$x_2$ (in pixels)')
    plt.colorbar()

    # save for lecture
    plt.savefig(save_folder / f'{image_file.stem}_{f}.pdf',
                bbox_inches='tight')
    plt.close()