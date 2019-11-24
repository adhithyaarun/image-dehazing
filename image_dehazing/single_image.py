import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# skimage imports
from skimage.util import img_as_ubyte, img_as_float64
from skimage.color import rgb2gray
from skimage.color import rgb2hsv

class ImageDehazing:
    def __init__(self, verbose=False):
        '''Function to initialize class variables'''
        self.image = None
        self.verbose = verbose

    def __clip(self, image=None):
        '''Function to clip images to range of [0.0, 1.0]'''
        # Validate parameters
        if image is None:
            return None

        image[image < 0] = 0
        image[image > 1] = 1
        return image

    def __show(self, images=None, titles=None, size=None, gray=False):
        '''Function to display images'''
        # Validate parameters
        if images is None or titles is None or size is None:
            return

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

    def white_balance(self, image=None):
        '''Function to perform white balancing operation on an image'''
        # Validate parameters
        if image is None:
            return None
        
        image = img_as_float64(image)

        # Extract colour channels
        R = image[:, :, 2]
        G = image[:, :, 1]
        B = image[:, :, 0]

        # Obtain average intensity for each colour channel
        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)

        mean_RGB = np.array([mean_R, mean_G, mean_B])

        # Obtain scaling factor
        grayscale = np.mean(mean_RGB)
        scale = grayscale / mean_RGB

        white_balanced = np.zeros(image.shape)

        # Rescale original intensities
        white_balanced[:, :, 2] = scale[0] * R
        white_balanced[:, :, 1] = scale[1] * G
        white_balanced[:, :, 0] = scale[2] * B

        # Clip to [0.0, 1.0]
        white_balanced = self.__clip(white_balanced)

        # Display result (if verbose)
        if self.verbose is True:
            self.__show(
                images=[self.image, white_balanced],
                titles=['Original Image', 'White Balanced Image'],
                size=(15, 15)
            )
        return white_balanced

    def enhance_contrast(self, image=None):
        '''Function to enhance contrast in an image'''
        # Validate parameters
        if image is None:
            return None

        image = img_as_float64(image)

        # Extract colour channels
        R = image[:, :, 2]
        G = image[:, :, 1]
        B = image[:, :, 0]

        # Obtain luminance using predefined scale factors
        luminance = 0.299 * R + 0.587 * G + 0.114 * B
        mean_luminance = np.mean(luminance)

        # Compute scale factor
        gamma = 2 * (0.5 + mean_luminance)

        # Scale mean-luminance subtracted colour chanels 
        enhanced = np.zeros(image.shape)
        enhanced[:, :, 2] = gamma * (R - mean_luminance)
        enhanced[:, :, 1] = gamma * (G - mean_luminance)
        enhanced[:, :, 0] = gamma * (B - mean_luminance)

        # Clip to [0.0, 1.0]
        enhanced = self.__clip(enhanced)

        # Display result (if verbose)
        if self.verbose is True:
            self.__show(
                images=[self.image, enhanced],
                titles=['Original Image', 'Contrast Enhanced Image'],
                size=(15, 15)
            )

        return enhanced

    def luminance_map(self, image=None):
        '''Function to generate the Luminance Weight Map of an image'''
        # Validate parameters
        if image is None:
            return None
        
        image = img_as_float64(image)

        # Generate Luminance Map
        luminance = np.mean(image, axis=2)
        luminancemap = np.sqrt((1 / 3) * (np.square(image[:, :, 0] - luminance + np.square(image[:, :, 1] - luminance) + np.square(image[:, :, 2] - luminance))))

        # Display result (if verbose)
        if self.verbose is True:
            self.__show(
                images=[self.image, luminancemap],
                titles=['Original Image', 'Luminanace Weight Map'],
                size=(15, 15),
                gray=True
            )
        return luminancemap
    
    def chromatic_map(self, image=None):
        '''Function to generate the Chromatic Weight Map of an image'''
        # Validate parameters
        if image is None:
            return None
        
        image = img_as_float64(image)
        
        # Convert to HSV colour space
        hsv = rgb2hsv(image)

        # Extract Saturation
        saturation = hsv[:, :, 1]
        max_saturation = 1.0
        sigma = 0.3
        
        # Generate Chromatic Map
        chromaticmap = np.exp(-1 * (((saturation - max_saturation) ** 2) / (2 * (sigma ** 2))))

        # Display result (if verbose)
        if self.verbose is True:
            self.__show(
             images=[self.image, chromaticmap],
             titles=['Original Image', 'Chromatic Weight Map'],
             size=(15, 15),
             gray=True
        )
    
        return chromaticmap

    def saliency_map(self, image=None):
        '''Function to generate the Saliency Weight Map of an image'''
        # Validate parameters
        if image is None:
            return None
        
        image = img_as_float64(image)
        
        # Convert image to grayscale
        if(len(image.shape) > 2):
            image = rgb2gray(image)
        else:
            image = image
        
        # Apply Gaussian Smoothing
        gaussian = cv.GaussianBlur(image,(5, 5),0) 
        
        # Apply Mean Smoothing
        image_mean = np.mean(image)
        
        # Generate Saliency Map
        saliencymap = np.absolute(gaussian - image_mean)

        # Display result (if verbose)           
        if self.verbose is True:
            self.__show(
                images=[self.image, saliencymap],
                titles=['Original Image', 'Saliency Weight Map'],
                size=(15, 15),
                gray=True
            )
        
        return saliencymap
    
    def image_pyramid(self, image=None, pyramid_type='gaussian', levels=1):
        '''Function to generate the Gaussian/Laplacian pyramid of an image'''
        # Validate parameters
        if image is None:
            return None
        
        image = img_as_float64(image)
        
        # Generate Gaussian Pyramid
        current_layer = image
        gaussian = [current_layer]
        for i in range(levels):
            current_layer = cv.pyrDown(current_layer)
            gaussian.append(current_layer)
            
        if pyramid_type == 'gaussian':
            return gaussian
        # Generate Laplacian Pyramid
        elif pyramid_type == 'laplacian':
            current_layer = gaussian[levels-1]
            laplacian = [current_layer]
            for i in range(levels - 1, 0, -1):
                shape = (gaussian[i-1].shape[1], gaussian[i-1].shape[0])
                expand_gaussian = cv.pyrUp(gaussian[i], dstsize=shape)
                current_layer = cv.subtract(gaussian[i-1], expand_gaussian)
                laplacian.append(current_layer)
            laplacian.reverse()
            return laplacian
            
    def fusion(self, inputs=None, weights=None, gaussians=None):
        '''Function to fuse the pyramids together'''
        # Validate parameters
        if inputs is None or weights is None or gaussians is None:
            return None
        
        fused_levels = []

        # Perform Fusion by combining the Laplacian and Gaussian pyramids
        for i in range(len(gaussians[0])):
            if len(inputs[0].shape) > 2:
                for j in range(inputs[0].shape[2]):
                    # Generate Laplacian Pyramids
                    laplacians = [
                        self.image_pyramid(image=inputs[0][:, :, j], pyramid_type='laplacian', levels=len(gaussians[0])),
                        self.image_pyramid(image=inputs[1][:, :, j], pyramid_type='laplacian', levels=len(gaussians[0]))
                    ]
                    
                    # Adjust rows to match
                    row_size = np.min(np.array([
                        laplacians[0][i].shape[0],
                        laplacians[1][i].shape[0],
                        gaussians[0][i].shape[0],
                        gaussians[1][i].shape[0]
                    ]))

                    # Adjust columns to match
                    col_size = np.min(np.array([
                        laplacians[0][i].shape[1],
                        laplacians[1][i].shape[1],
                        gaussians[0][i].shape[1],
                        gaussians[1][i].shape[1]
                    ]))
                    
                    if j == 0:
                        intermediate = np.ones(inputs[0][:row_size, :col_size].shape)
                    # Fusion Step
                    intermediate[:, :, j] = (laplacians[0][i][:row_size, :col_size] * gaussians[0][i][:row_size, :col_size]) + (laplacians[1][i][:row_size, :col_size] * gaussians[1][i][:row_size, :col_size])
            fused_levels.append(intermediate)
        
        # Reconstruct Image Pyramids
        for i in range(len(fused_levels)-2, -1, -1):
            level_1 = cv.pyrUp(fused_levels[i+1])
            level_2 = fused_levels[i]
            r = min(level_1.shape[0], level_2.shape[0])
            c = min(level_1.shape[1], level_2.shape[1])
            fused_levels[i] = level_1[:r, :c] + level_2[:r, :c]

        # Clip fused image to [0.0, 1.0]
        fused = self.__clip(fused_levels[0])
        if self.verbose is True:
            self.__show(
                    images=[self.image, fused],
                    titles=['Original Image', 'Fusion'],
                    size=(15, 15),
                    gray=False
                )
        return fused

    def dehaze(self, image=None, verbose=None, pyramid_height=12):
        '''Driver function to dehaze the image'''
        # Validate parameters
        if image is None:
            return None

        self.image = image
        
        if len(image.shape) > 2 and image.shape[2] == 4:
            self.image = image[:, :, :3]

        # Set verbose flag (to decide whether each step is displayed)
        if verbose is None:
            pass
        elif verbose is True:
            self.verbose = True
        else:
            self.verbose = False

        # Generating Input Images 
        white_balanced = self.white_balance(image=img_as_float64(self.image))       # First Input Image
        contrast_enhanced = self.enhance_contrast(image=img_as_float64(self.image)) # Second Input Image
        
        input_images = [
            img_as_float64(white_balanced),
            img_as_float64(contrast_enhanced)
        ]
        
        # Generating Weight Maps
        weight_maps = [
            # Weight maps for first image
            {
                'luminance': self.luminance_map(image=input_images[0]),
                'chromatic': self.chromatic_map(image=input_images[0]),
                'saliency': self.saliency_map(image=input_images[0])
            },
            
            # Weight maps for second image
            {
                'luminance': self.luminance_map(image=input_images[1]),
                'chromatic': self.chromatic_map(image=input_images[1]),
                'saliency': self.saliency_map(image=input_images[1])
            }
        ]
        
        # Weight map normalization
        # Combined weight maps
        weight_maps[0]['combined'] = (weight_maps[0]['luminance'] * weight_maps[0]['chromatic'] * weight_maps[0]['saliency'])
        weight_maps[1]['combined'] = (weight_maps[1]['luminance'] * weight_maps[1]['chromatic'] * weight_maps[1]['saliency'])
        
        # Normalized weight maps
        weight_maps[0]['normalized'] = weight_maps[0]['combined'] / (weight_maps[0]['combined'] + weight_maps[1]['combined'])
        weight_maps[1]['normalized'] = weight_maps[1]['combined'] / (weight_maps[0]['combined'] + weight_maps[1]['combined'])
        
        # Generating Gaussian Image Pyramids
        gaussians = [
            self.image_pyramid(image=weight_maps[0]['normalized'], pyramid_type='gaussian', levels=pyramid_height),
            self.image_pyramid(image=weight_maps[1]['normalized'], pyramid_type='gaussian', levels=pyramid_height)
        ]

        # Fusion Step
        fused = self.fusion(input_images, weight_maps, gaussians)
 
        # Dehazing data
        dehazing = {
            'hazed': self.image,
            'inputs': input_images,
            'maps': weight_maps,
            'dehazed': fused
        }
        
        self.image = None   # Reset image

        return dehazing