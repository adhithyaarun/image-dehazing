# image-dehazing

An implementation of the IEEE paper titled **_Effective Single Image Dehazing by Fusion_** by _Codruta Orniana Ancuti_, _Cosmin Ancuti_ and _Philippe Bekaert_.

## Contributors

For any queries, you may contact:

- Adhithya Arun (adhithya.arun@students.iiit.ac.in)
- Namratha Gopalabhatla (namratha.g@students.iiit.ac.in)

## Pre-requisites

Below are the pre-requisites to be satisfied before the package can be used.

- Make sure `python3` is installed. (Preferably `python3.7`)
- Package requirements:
  - `matplotlib`
  - `numpy`
  - `opencv-python`
  - `scikit-image`
- If the above packages are not installed,

  - You may manually install the packages individually
  - You can also install the packages using the [_requirements.txt_](requirements.txt):
    ```bash
    sudo python3.7 -m pip install -r requirements.txt
    ```

## Usage

- Once the pre-requisites are satisfied, the [demo](demo.py) can be run by running the following command:

  ```bash
  python3.7 demo.py
  ```

- Place the [_image_dehazing_](image_dehazing/) directory in the same directory as your code files that are using them.
- The **_ImageDehazing_** class from the package can be used as shown in the below example:

  ```python
  from skimage.io import imread
  import matplotlib.pyplot as plt

  # Import ImageDehazing class
  from image_dehazing.single_image import ImageDehazing

  # Read image
  path_to_image = './dataset/image_1.jpg'  # Path to hazy image
  hazy_image = imread(path_to_image)

  # Create object of ImageDehazing class
  dehazer = ImageDehazing(verbose=True)

  # Dehaze the the image using th dehaze method of the object
  dehaze_data = dehazer.dehaze(hazy_image)

  # Display dehazed image
  plt.figure()
  plt.subplot(1, 2, 1)
  plt.imshow(dehaze_data['hazed'])
  plt.title('Hazy Image')
  plt.subplot(1, 2, 2)
  plt.imshow(dehaze_data['dehazed'])
  plt.title('Dehazed Image')
  plt.axis('off')
  plt.show()
  ```
