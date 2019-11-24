from image_dehazing.single_image import ImageDehazing
import matplotlib.pyplot as plt
from skimage.io import imread

# Read images
images = [
    imread('./dataset/image_1.jpg'),
    imread('./dataset/image_2.jpeg'),
    imread('./dataset/image_3.png'),
    imread('./dataset/image_4.jpg'),
    imread('./dataset/image_5.jpg'),
    imread('./dataset/image_6.jpg'),
    imread('./dataset/image_7.jpg'),
    imread('./dataset/image_8.jpg'),
    imread('./dataset/image_9.jpg'),
    imread('./dataset/image_10.jpg'),
]

# Different Heights for Pyramids
heights = [3, 5, 8, 11, 15]
results = {}

# Generate outputs for each height
for height in heights:
    results[str(height)] = []
    for image in images:
        obj = ImageDehazing(verbose=False)
        dehazed = obj.dehaze(image, pyramid_height=height)
        results[str(height)].append(dehazed['dehazed'])

# Display images
for i in range(len(images)):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(images[i])
    plt.title('Original Image')
    plt.axis('off')
    for j in range(len(heights)):
        plt.subplot(2, 3, j+2)
        plt.imshow(results[str(heights[j])][i])
        plt.title('Pyramid Height: {}'.format(heights[j]))
        plt.axis('off')
    plt.show()