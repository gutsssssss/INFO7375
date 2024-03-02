import numpy as np
import os
import keras
from skimage.transform import resize

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Resize the images to 20x20 pixels
x_train = np.array([resize(image, (20, 20)) for image in x_train])
x_test = np.array([resize(image, (20, 20)) for image in x_test])

# Rescale the pixel values to [0, 1] range
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create an image data generator with some augmentation parameters
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,  # rotate the image up to 10 degrees
    width_shift_range=0.1,  # shift the image horizontally by 10% of the width
    height_shift_range=0.1,  # shift the image vertically by 10% of the height
    zoom_range=0.1,  # zoom the image by 10% of the size
    fill_mode='constant',  # fill the empty pixels with 0
    cval=0  # constant value for fill_mode
)

# Create a folder to save the generated images

if not os.path.exists('generated_images'):
    os.makedirs('generated_images')

if not os.path.exists('generated_images2'):
    os.makedirs('generated_images2')

# Generate and save 10 images for each digit
for digit in range(10):
    # Get the first image of the digit from the training set
    image = x_train[y_train == digit][0]
    # Reshape the image to (1, 20, 20, 1) for the data generator
    image = image.reshape((1, 20, 20, 1))
    # Create a subfolder for the digit
    digit_folder = os.path.join('generated_images', str(digit))
    if not os.path.exists(digit_folder):
        os.makedirs(digit_folder)
    # Generate 10 images and save them as png files
    i = 0
    for batch in datagen.flow(image, batch_size=1, save_to_dir=digit_folder, save_prefix='digit', save_format='png'):
        i += 1
        if i == 10:
            break

# Generate and save 3 images for each digit
for digit in range(10):
    # Get the first image of the digit from the training set
    image = x_test[y_test == digit][0]
    # Reshape the image to (1, 20, 20, 1) for the data generator
    image = image.reshape((1, 20, 20, 1))
    # Create a subfolder for the digit
    digit_folder = os.path.join('generated_images2', str(digit))
    if not os.path.exists(digit_folder):
        os.makedirs(digit_folder)
    # Generate 10 images and save them as png files
    i = 0
    for batch in datagen.flow(image, batch_size=1, save_to_dir=digit_folder, save_prefix='digit', save_format='png'):
        i += 1
        if i == 10:
            break
