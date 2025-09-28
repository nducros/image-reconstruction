# Image Reconstruction ---A short tour from analytical to data-driven methods

* The lecture handouts can be downloaded [here](https://www.creatis.insa-lyon.fr/~ducros/AR/AR_lecture_handout_2024.pdf) (2024).
* The `lesson.py` script generates the images used in the handouts.
* The `exercise.py` script is a code template for solving the exercises.
* I have absolutely no idea what `solution.py` is, but I strongly recommend that you do not look at it right now!

## Code and data
### How to get the code?
* Windows PowerShell
    ```powershell
    git clone https://github.com/nducros/image-reconstruction.git
    cd image-reconstruction
    ```

### How to download the images and sinograms?
* Windows PowerShell
    ```powershell
    wget https://www.creatis.insa-lyon.fr/~ducros/AR/data.zip -OutFile data.zip
    tar xvf data.zip
    ```

* Linux shell
    ```shell
    cd exercise
    wget https://www.creatis.insa-lyon.fr/~ducros/AR/data.zip
    unzip -xvf data.zip
    ```

## Exercises

> To complete the exercises, please refer to the code template `exercise.py`, and follow the instructions therein.

* **Exercise 1**: Compute the Radon transform of `ct_image_xx.png` for `xx` = 0, 1, and 2.

* **Exercise 2**: Reconstruct the image corresponding to `ct_image_xx_sinog.png` for `xx` = 4 and 5.

* **Exercise 3**: Reconstruct `ct_image_4_sinog.png` with 2% additive Gaussian noise using different filters (e.g., ramp, cosine, Hann, no filter).

* **Exercise 4**: 
    * Construct the forward matrix corresponding to the Radon transform of a $I_1\times I_2 = 32 \times 32$ image computed under $J = 40$ view angles over $[0,\pi)$ using a linear detector of $K = 45$ pixels.
    * Check your forward matrix by comparing the Radon transform of the Shepp-Logan phantom that is computed using the `radon` function and using your forward matrix.

* **Exercise 5**: Implement the ART reconstruction algorithm and reconstruct the sinogram of Exercise 4.

* **Exercise 6**: Reconstruct the sinogram of Exercise 4 by computing the pseudo inverse of the forward matrix and by using a solver for linear systems. How do the reconstruction times compare?

* **Hands on**: This hands-on was proposed in the [2025 Deep Learning for Medical Imaging Summer School](https://github.com/openspyrit/spyrit-examples/tree/master/2025_DLMIS). The code is organised in Jupyter notebooks and Python scripts. The notebooks contain the theoretical background, the code for the exercises and the solutions. The code is written in Python and uses the SPyRiT library for image reconstruction. The data consists of 2D and 3D images and sinograms. 