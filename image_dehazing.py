import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# skimage imports
from skimage.io import imread, imshow
from skimage.util import img_as_ubyte, img_as_float64


class ImageDehazing:
    def __init__(self, verbose=False):
        self.image = None
        self.verbose = verbose

    def __show(self, images, titles, size):
        if self.verbose is True:
            plt.figure(figsize=size)

            plt.subplot(1, 2, 1)
            plt.imshow(images[0])
            plt.title(titles[0])
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(images[1])
            plt.title(titles[1])
            plt.axis('off')

            plt.show()

    def white_balance(self, image):
        pass

    def dehaze(self, image):
        pass
