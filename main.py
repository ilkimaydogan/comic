import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from scipy.ndimage import gaussian_filter, median_filter
from sklearn.cluster import KMeans

# CONFIG
IMAGE_PATH = "hamburger.jpg"
SIGMA = 1.1
K_VALUE = 1.2
EPSILON = 12
NUM_OF_BINS = 4
MODE = "median"
KERNEL_SIZE = 7


def image_smoothing(image, mode="gaussian", sigma=None, kernel_size=None):
    if mode == "gaussian":
        gaussian_image = gaussian_filter(image, sigma=(sigma, sigma, 0))
        return gaussian_image
    elif mode == "median":
        median_image = median_filter(image, size=(kernel_size, kernel_size, 1))
        return median_image
    else:
        return


def edge_detection(image, sigma, k, epsilon):
    # transforming the image grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_array = np.array(image_gray)

    # defining two different Gaussian filtered images
    gaussian_1 = gaussian_filter(image_array, sigma=sigma)
    gaussian_2 = gaussian_filter(image_array, sigma=k * sigma)

    # computing the difference of gaussians
    dog_image = gaussian_1 - gaussian_2

    # thresholding the filtered image
    thresholded_image = np.where(dog_image >= epsilon, 1, 0)

    return thresholded_image


def image_quantization_hsv(image, NUM_OF_BINS):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # extract the value channel
    value_channel = hsv_image[:, :, 2].reshape((-1, 1))

    # quantize the value channel using KMeans
    kmeans = KMeans(n_clusters=NUM_OF_BINS, n_init=10)
    kmeans.fit(value_channel)
    quantized_values = kmeans.cluster_centers_[kmeans.predict(value_channel)].reshape(hsv_image[:, :, 2].shape)

    # replace the original value channel with the quantized values
    hsv_image[:, :, 2] = quantized_values

    quantized_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    return quantized_image


def image_quantization_lab(image, NUM_OF_BINS):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # extract the lab channel
    luminance_channel, a_channel, b_channel = cv2.split(lab_image)

    # quantize the value channel using KMeans
    kmeans = KMeans(n_clusters=NUM_OF_BINS, n_init=10)
    kmeans.fit(luminance_channel)
    quantized_values = kmeans.cluster_centers_[kmeans.predict(luminance_channel)].reshape(lab_image[:, :, 0].shape)

    # replace the original value channel with the quantized values
    lab_image[:, :, 0] = quantized_values

    quantized_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    return quantized_image


# reading the image
image_original = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)  # transforming to rgb to display with plt

smoothed_image = image_smoothing(image_rgb, MODE, SIGMA, KERNEL_SIZE)

start_time = time.time()
sketch_image = edge_detection(image_rgb, SIGMA, K_VALUE, EPSILON)
inverted_edge_image = 1 - sketch_image
quantized_image = image_quantization_hsv(smoothed_image, NUM_OF_BINS)
# quantized_image = image_quantization_lab(smoothed_image, NUM_OF_BINS)

cartoon_image = quantized_image * inverted_edge_image[:, :, np.newaxis]

end_time = time.time()
elapsed_time = end_time - start_time


def display_image(image, title, color=None):
    if color:
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()
    else:
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# displaying images
# display_image(image_rgb,"original image")
# display_image(smoothed_image,f"smoothed image w:{MODE} and value:{SIGMA}")
# display_image(sketch_image,f"pencil sketch image with k:{K_VALUE} and epsilon:{EPSILON}",color="cmap")
# display_image(quantized_image,f"cartoon image with num of bins:{NUM_OF_BINS}")

display_image(cartoon_image, "cartoon image")
# can print elapsed time
# print(elapsed_time)
