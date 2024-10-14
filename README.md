# Image Reconstruction ---A short tour from analytical to data-driven methods

* The lecture handouts can be downloaded [here](https://www.creatis.insa-lyon.fr/~ducros/AR/AR_lecture_handout_2024.pdf) (2024).
* The `lesson.py` script generates the images used in the handouts.
* The `exercise.py` script is a code template for solving the exercises.
* I have absolutely no idea what `solution.py` is, but I strongly recommend that you do not look at it right now!

## Exercise
* Compute the Radon transform of `ct_image_xx.png` for load image for `xx` = 0, 1, and 2.
* Reconstruct the image corresponding to `ct_image_xx_sinog.png` for load image for `xx` = 4 and 5.
* To do so, complete the code template `exercise.py`

### How to get the code?
* Windows PowerShell
    ```powershell
    git clone https://github.com/nducros/image-reconstruction.git
    cd image-reconstruction
    ```

### How to download the images and sinograms?
* Windows PowerShell
    ```powershell
    cd exercise
    wget https://www.creatis.insa-lyon.fr/~ducros/AR/data.zip -OutFile data.zip
    tar xvf data.zip
    ```

* Linux shell
    ```shell
    cd exercise
    wget https://www.creatis.insa-lyon.fr/~ducros/AR/data.zip
    unzip xvf data.zip
    ```