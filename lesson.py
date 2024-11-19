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
    
    
#%% Create forward operator 
from skimage.transform import rescale

img_size = 32 # we assume image is square
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
        image[i,j] = 1 ##COMLPETE
        
        # Radon transform
        sinogram = radon(image, theta, circle=False) ##COMPLETE
        sinogram = rescale(sinogram, scale=(n_detec/sinogram.shape[0],1), mode='reflect')
        
        # Concatenating results in matrix A 
        A[:,img_size*i+j] = np.reshape(sinogram, (n_detec*n_angle, ))

# A matrix visualisation
fig, ax = plt.subplots()
ax.imshow(A)
ax.imshow(A, cmap=plt.cm.Greys_r)
ax.set_title("A")
ax.set_ylabel(r"Projection ray $(r, \theta)$")
ax.set_xlabel(r"Image pixel $x$")

# save for lecture
fig.savefig(save_folder / f'forward_{n_detec*n_angle}x{img_size**2}.pdf', 
            bbox_inches='tight')


#%% Check forward operator
from skimage.data import shepp_logan_phantom

phantom = shepp_logan_phantom()
phantom = rescale(phantom, scale=(img_size/phantom.shape[0]), mode='reflect')

# Radon transform with skimage function
sinog = radon(phantom, theta, circle=False) ## COMPLETE
sinog = rescale(sinog, scale=(n_detec/sinog.shape[0], 1), mode='reflect') ##COMPLETE

# Radon transform as a matrix-vector product ## COMPLETE
f = np.reshape(phantom, (-1, 1))
m = A @ f
sinog2 = np.reshape(m, (n_detec, n_angle)) 

# Plot both sinograms
fig, (ax1, ax2) = plt.subplots(1, 2, ) #figsize=(8, 4.5)
ax1.set_title(r"Sinogram from" + "\nRadon function")
ax1.set_xlabel(r"Projection angle $\theta$ (in deg)")
ax1.set_ylabel(r"Projection position $r$ (in pixels)")
ax1.imshow(sinog, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinog.shape[0]), aspect='auto')

ax2.set_title(r"Sinogram from" + "\n" + r"forward matrix $A$")
ax2.set_xlabel(r"Projection angle $\theta$ (in deg)")
ax2.set_ylabel(r"Projection position $r$ (in pixels)")
ax2.imshow(sinog2, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinog2.shape[0]), aspect='auto')

fig.tight_layout()

# Plot sinogram
fig, ax = plt.subplots()
ax.set_title(r"Sinogram")
ax.set_xlabel(r"Projection angle $\theta$ (in deg)")
ax.set_ylabel(r"Projection position $r$ (in pixels)")
ax.imshow(sinog, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinog.shape[0]), aspect=5)

# save for lecture
fig.savefig(save_folder / f'sinog_{n_detec*n_angle}.pdf', 
            bbox_inches='tight')

# Plot image
fig, ax = plt.subplots()
ax.set_title("Image")
plt.xlabel(r'$x_1$ (in pixels)')
plt.ylabel(r'$x_2$ (in pixels)')
ax.imshow(phantom, cmap=plt.cm.Greys_r, aspect='equal')

# save for lecture
fig.savefig(save_folder / f'image_{img_size*img_size}.pdf', 
            bbox_inches='tight')

#%% Algebraic reconstruction technique

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

#%% Reconstruction using ART
step_ite = 2

f_ = art(A, m, n_ite=step_ite)                 # 1 iteration

for ii in range(9):
    
    f  = f_.reshape((img_size,img_size))
    
    # Display results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5), layout='compressed')
    ax1.set_title(r"Ground Truth")
    im = ax1.imshow(phantom, cmap=plt.cm.Greys_r)
    fig.colorbar(im)
    
    ax2.set_title(f"ART ({step_ite*(ii+1)} iterations)")
    im = ax2.imshow(f, cmap=plt.cm.Greys_r)
    fig.colorbar(im)
    
    ax3.set_title(f"Diff ({step_ite*(ii+1)} iterations)")
    im = ax3.imshow(phantom-f, cmap=plt.cm.Greys_r)
    fig.colorbar(im)
    
    fig.savefig(save_folder / f'art_nonoise_{ii+1:02}.pdf', 
                bbox_inches='tight')
    
    # ART recon
    f2_, _ = art(A, m, n_ite=step_ite, f0 = f_)   # 1 more iteration 
    f_ = np.copy(f2_)
 
#%% Solve
import scipy.linalg as lin
import time

# Use the computed sinogram
sinogram = np.reshape(sinog, (-1, 1))

# Compute the pseudoinverse
t0 = time.perf_counter()
pinv = lin.pinv(A) # COMPLETE
t0 = time.perf_counter() - t0

# Reconstruct with pseudoinverse 
t1 = time.perf_counter()
rec_pi = np.reshape(np.dot(pinv, sinogram), (img_size,img_size)) # COMPLETE
t1 = time.perf_counter() - t1
print(f'Recon with pseudoinverse: {t0:.3f} + {t1:.4f} s')

# Reconstruct with a linear solver
t2 = time.perf_counter()
rec_solv = np.reshape(lin.lstsq(A, sinogram)[0], (img_size,img_size)) # COMPLETE
t2 = time.perf_counter() - t2
print(f'Recon with solver: {t2:.3f} s')

# Display results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))
ax1.set_title(r"Ground Truth")
ax1.imshow(phantom, cmap=plt.cm.Greys_r)

ax2.set_title(r"Recon with pseudoinverse")
ax2.imshow(rec_pi, cmap=plt.cm.Greys_r)

ax3.set_title(r'Recon with solver')
ax3.imshow(rec_solv, cmap=plt.cm.Greys_r)
