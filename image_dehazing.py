import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# skimage imports
from skimage.io import imread
from skimage.util import img_as_ubyte, img_as_float64
from skimage.color import rgb2gray
from skimage.color import rgb2hsv

#scipy
from scipy.signal import convolve2d
from scipy import sparse

class ImageDehazing:
    def __init__(self, verbose=False):
        self.image = None
        self.verbose = verbose

    def __clip(self, image):
        image[image < 0] = 0
        image[image > 1] = 1
        return image

    def __show(self, images, titles, size, gray=False):
        if self.verbose is True:
            plt.figure(figsize=size)

            plt.subplot(1, 2, 1)
            if gray is True:
                plt.imshow(images[0], cmap='gray')
            else:
                plt.imshow(images[0])
            plt.title(titles[0])
            plt.axis('off')

            plt.subplot(1, 2, 2)
            if gray is True:
                plt.imshow(images[1], cmap='gray')
            else:
                plt.imshow(images[1])
            plt.title(titles[1])
            plt.axis('off')

            plt.show()

    def white_balance(self, image):
        image = img_as_float64(image)

        R = image[:, :, 2]
        G = image[:, :, 1]
        B = image[:, :, 0]

        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)

        mean_RGB = np.array([mean_R, mean_G, mean_B])

        grayscale = np.mean(mean_RGB)
        scale = grayscale / mean_RGB

        white_balanced = np.zeros(image.shape)

        white_balanced[:, :, 2] = scale[0] * R
        white_balanced[:, :, 1] = scale[1] * G
        white_balanced[:, :, 0] = scale[2] * B

        white_balanced = self.__clip(white_balanced)

        if self.verbose is True:
            self.__show(
                images=[self.image, white_balanced],
                titles=['Original Image', 'White Balanced Image'],
                size=(15, 15)
            )
        return white_balanced

    def enhance_contrast(self, image):
        image = img_as_float64(image)

        R = image[:, :, 2]
        G = image[:, :, 1]
        B = image[:, :, 0]

        luminance = 0.299 * R + 0.587 * G + 0.114 * B
        mean_luminance = np.mean(luminance)

        gamma = 2 * (0.5 + mean_luminance)

        enhanced = np.zeros(image.shape)
        enhanced[:, :, 2] = gamma * (R - mean_luminance)
        enhanced[:, :, 1] = gamma * (G - mean_luminance)
        enhanced[:, :, 0] = gamma * (B - mean_luminance)

        enhanced = self.__clip(enhanced)

        if self.verbose is True:
            self.__show(
                images=[self.image, enhanced],
                titles=['Original Image', 'Contrast Enhanced Image'],
                size=(15, 15)
            )

        return enhanced

    def luminance_map(self, image):
        image = img_as_float64(image)
        luminance = np.mean(image, axis=2)
        luminancemap = np.sqrt((1 / 3) * (np.square(image[:, :, 0] - luminance) + np.square(image[:, :, 1] - luminance) + np.square(image[:, :, 2] - luminance)))

        if self.verbose is True:
            self.__show(
                images=[self.image, luminancemap],
                titles=['Original Image', 'Luminanace Weight Map'],
                size=(15, 15),
                gray=True
            )
        
        return luminancemap
    
    def chromatic_map(self, image):
        image = img_as_float64(image)
        hsv = rgb2hsv(image)
        saturation = hsv[:, :, 1]
        max_saturation = 1.0
        sigma = 0.3
        chromaticmap = np.exp(-1 * (((saturation - max_saturation) ** 2) / (2 * (sigma ** 2))))

        if self.verbose is True:
            self.__show(
             images=[self.image, chromaticmap],
             titles=['Original Image', 'Chromatic Weight Map'],
             size=(15, 15),
             gray=True
        )
    
        return chromaticmap

    def saliency_map(self, image):
        image = img_as_float64(image)
        
        if(image.shape[2] > 0):
            image = rgb2gray(image)
        else:
            image = image
        
        gaussian = cv.GaussianBlur(image,(5,5),0) 
        image_mean = np.mean(image)
        
        saliencymap = np.absolute(gaussian - image_mean)
           
        if self.verbose is True:
            self.__show(
                images=[self.image, saliencymap],
                titles=['Original Image', 'Saliency Weight Map'],
                size=(15, 15),
                gray=True
            )
        
        return saliencymap

    def dehaze(self, image, verbose=None):
        self.image = image

        if verbose is None:
            pass
        elif verbose is True:
            self.verbose = True
        else:
            self.verbose = False

        wb = self.white_balance(hazed)
        en = self.enhance_contrast(hazed)
        lm = self.luminance_map(hazed)
        cm = self.chromatic_map(hazed)
        sm = self.saliency_map(hazed)

        self.image = None


hazed = imread('./dataset/haze.jpg')
obj = ImageDehazing(verbose=True)
obj.dehaze(hazed)
