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
# 1. Construct the discrete forward operator corresponding to the Radon transform
# of a 32x32 image computed under 90 view angles over [0,\pi) using a linear
# detector of 45 pixels.

from skimage.transform import rescale

img_size = 32  # image assumed to be square
n_angle = 40
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
ax.set_ylabel(r"Projection ray $(r, \theta)$")  
ax.set_xlabel(r"Image pixel $x$")               
plt.show()


# 2. Check that the forward matrix works
from skimage.data import shepp_logan_phantom

phantom = shepp_logan_phantom()
phantom = rescale(phantom, scale=(img_size/phantom.shape[0]), mode='reflect')

# Radon transform with skimage function
sinog = radon(phantom, theta, circle=False) 
sinog = rescale(sinog, scale=(n_detec/sinog.shape[0], 1), mode='reflect') ##COMPLETE

# Radon transform as a matrix-vector product
f = np.reshape(phantom, (-1, 1)) ## COMPLETE
m = A @ f
sinog2 = np.reshape(m, (n_detec, n_angle)) 

# Plot both sinograms
fig, (ax1, ax2) = plt.subplots(1, 2, ) 
ax1.set_title(r"Sinogram from" + "\nRadon function")
ax1.set_xlabel(r"Projection angle $\theta$ (in deg)")
ax1.set_ylabel(r"Projection position $r$ (in pixels)")
ax1.imshow(sinog, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinog.shape[0]), aspect='auto')

ax2.set_title(r"Sinogram from" + "\n" + r"forward matrix $A$")
ax2.set_xlabel(r"Projection angle $\theta$ (in deg)")
ax2.imshow(sinog2, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinog2.shape[0]), aspect='auto')

#%% Exercise 5
# Implement the ART reconstruction algorithm and reconstruct the sinogram
# of Exercise 4.

def art(A, m, n_ite=15, f0=np.zeros((A.shape[1],1)), gamma=1):
    """
    Algebraic reconstruction technique.
    
    It solve the linear system :math:`m = Af` in an iterative manner. It is 
    known as Kaczmarz method in numerical linear algebra.
       

    Args:
        A (ndarray): System matrix :math:`A`.
        
        m (ndarray): Measurement vector :math:`m`.
        
        n_ite (int, optional): Number of iterations. Defaults to 15.
        
        f0 (ndarray, optional): Initialisation. Defaults to 0.
        
        gamma (float, optional): Relaxation parameter in )0,1). Defaults to 1.

    Returns:
        f (ndarray): Unknown vector :math:`f`.
    """
     
    # remove unrelevant (rho, theta) rays
    a = np.sum(A, 1, keepdims=True)
    ind = np.nonzero(a)[0]
    m = m[ind]
    A = A[ind,:]
    
    f = f0
    # External iteration loop
    for kk in range(n_ite):     
        # Loop over the rows of A 
        for ii in range(A.shape[0]): 
            a_m = A[ii,:]
            a_m = a_m[:,None] 
            f = f - gamma*(a_m.T @ f - m[ii])*a_m/(a_m.T @ a_m)
    
    return f

# Reconstruct using 20 iterations of ART
n_ite = 20
f_rec = art(A, m, n_ite=n_ite)
f_rec  = f_rec.reshape((img_size,img_size))

# Display results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5), layout='compressed')
ax1.set_title(r"Ground Truth")
im = ax1.imshow(phantom, cmap=plt.cm.Greys_r)
fig.colorbar(im)

ax2.set_title(f"ART ({n_ite} iterations)")
im = ax2.imshow(f_rec, cmap=plt.cm.Greys_r)
fig.colorbar(im)

ax3.set_title(f"Diff ({n_ite} iterations)")
im = ax3.imshow(phantom-f_rec, cmap=plt.cm.Greys_r)
fig.colorbar(im)