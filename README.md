# image-dehazing

An implementation of the IEEE paper titled **_Effective Single Image Dehazing by Fusion_** by _Codruta Orniana Ancuti_, _Cosmin Ancuti_ and _Philippe Bekaert_.

## Contributors

For any queries, you may contact:

- Adhithya Arun (adhithya.arun@students.iiit.ac.in)
- Namratha Gopalabhatla (namratha.g@students.iiit.ac.in)

## Directory Structure

- _image-dehazing_
- [_image_dehazing_](image_dehazing) : Image dehazing package
  - [_\_\_init\_\_.py_](image_dehazing/__init__.py)
  - [_single_image.py_](image_dehazing/single_image.py) : ImageDehazing class
- [_demo.py_](demo.py) : Demo script
- [_height_variance.py_](height_variance.py) : Sample script to display variance in quality of dehazed results with height of image pyramid
- [_LICENSE_](LICENSE) : License
- [_README.md_](README.md) : Documentation
- [_requirements.txt_](requirements.txt) : Package requirements

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

## Class Structure

- **_ImageDehazing_**_(_[**_verbose_** = _False_]_)_
- Methods:
  - **_white_balance_**_(**image**)_
    <br/>**Description:** _Function to perform white balancing operation on an image_<br/>
    | Parameter | Description | Default |
    | --------- | -------------------------- | ------- |
    | _image_ | Image to be white balanced | _None_ |
  - **_enhance_contrast_**_(**image**)_
    <br/>**Description:** _Function to enhance contrast in an image_<br/>
    | Parameter | Description | Default |
    | --------- | ----------------------------- | ------- |
    | _image_ | Image to be contrast enhanced | _None_ |
  - **_luminance_map_**_(**image**)_
    <br/>**Description:** _Function to generate the Luminance Weight Map of an image_<br/>
    | Parameter | Description | Default |
    | --------- | -------------------------------------------- | ------- |
    | _image_ | Image whose luminance map is to be generated | _None_ |
  - **_chromatic_map_**_(**image**)_
    <br/>**Description:** _Function to generate the Chromatic Weight Map of an image_<br/>
    | Parameter | Description | Default |
    | --------- | -------------------------------------------- | ------- |
    | _image_ | Image whose chromatic map is to be generated | _None_ |
  - **_saliency_map_**_(**image**)_
    <br/>**Description:** _Function to generate the Saliency Weight Map of an image_<br/>
    | Parameter | Description | Default |
    | --------- | ------------------------------------------- | ------- |
    | _image_ | Image whose saliency map is to be generated | _None_ |
  - **_image_pyramid_**_(**image**, **pyramid_type**, **levels**)_
    <br/>**Description:** _Function to generate the Gaussian/Laplacian pyramid of an image_<br/>
    | Parameter | Description | Default |
    | -------------- | -------------------------------------------------- | -------------- |
    | _image_ | Image whose pyramid is to be generated | _None_ |
    | _pyramid_type_ | Type of pyramid: _\'gaussian\'_ or _\'laplacian\'_ | _\'gaussian\'_ |
    | _levels_ | Height of pyramid | _1_ |
  - **_fusion_**_(**inputs**, **weights**, **gaussians**)_
    <br/>**Description:** _Function to fuse the pyramids together_<br/>
    | Parameter | Description | Default |
    | ----------- | ---------------------------------------------- | ------- |
    | _inputs_ | Images to be fused | _None_ |
    | _weights_ | Image to be white balanced | _None_ |
    | _gaussians_ | Gaussian pyramids for the corresponding inputs | _None_ |
  - **_dehaze_**_(_**_image_**, [**_verbose_** = _None_, **_pyramid_height_** = _12_]_)_
    <br/>**Description:** _Driver function to dehaze the image_<br/>
    | Parameter | Description | Default |
    | ---------------- | --------------------------------------------------- | ------- |
    | _image_ | Image to be dehazed | _None_ |
    | _verbose_ | Flag denoting whether each step should be displayed | _None_ |
    | _pyramid_height_ | Height of image pyramids | _12_ |

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
  dehaze_data = dehazer.dehaze(hazy_image, pyramid_height=12)

  # Display dehazed image
  plt.figure()
  plt.subplot(1, 2, 1)
  plt.imshow(dehaze_data['hazed'])
  plt.title('Hazy Image')
  plt.axis('off')
  plt.subplot(1, 2, 2)
  plt.imshow(dehaze_data['dehazed'])
  plt.title('Dehazed Image')
  plt.axis('off')
  plt.show()
  ```
