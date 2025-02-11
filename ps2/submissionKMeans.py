
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# Edit ps2_updated.ipynb instead.
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb
from typing import Tuple
import matplotlib.pyplot as plt

def quantize_rgb(img: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the k-means clusters for the input image in RGB space, and return
    an image where each pixel is replaced by the nearest cluster's average RGB
    value.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        k: The number of clusters to use

    Output:
        An RGB image with shape H x W x 3 and dtype "uint8"
    """
#     quantized_img = np.zeros_like(img)

    ##########################################################################
    # TODO: Perform k-means clustering and return an image where each pixel  #
    # is assigned the value of the nearest clusters RGB values.              #
    ##########################################################################
    image = np.copy(img)
    [w,h,d] = np.shape(image)
    stretch = np.reshape(img,(w*h,d))
    kmeans = KMeans(n_clusters=k, random_state = 101)

    labels = kmeans.fit_predict(stretch)

    means = kmeans.cluster_centers_
    image_out = means[labels].reshape(img.shape).astype(np.uint8)


    ##########################################################################
    ##########################################################################

    return image_out

def quantize_hsv(img: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the k-means clusters for the input image in the hue dimension of the
    HSV space. Replace the hue values with the nearest cluster's hue value. Finally,
    convert the image back to RGB.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        k: The number of clusters to use

    Output:
        An RGB image with shape H x W x 3 and dtype "uint8"
    """


    ##########################################################################
    # TODO: Convert the image to HSV. Perform k-means clustering in hue      #
    # space. Replace the hue values in the image with the cluster centers.   #
    # Convert the image back to RGB.                                         #
    ##########################################################################
    img_rgb = np.copy(img)
    img_hsv = rgb2hsv(img_rgb)
    hue_array = img_hsv[:,:,0]
    samples = hue_array.reshape((-1, 1))
    kmeans = KMeans(n_clusters=k, random_state=101)
    labels = kmeans.fit_predict(samples)
    centers = kmeans.cluster_centers_
    img_hsv[:,:,0] = centers[labels].reshape(hue_array.shape)
    quantized_img = (hsv2rgb(img_hsv) * 255).astype(np.uint8)
    return quantized_img
#     img_rgb = np.copy(img)
#     img_hsv = rgb2hsv(img_rgb)

#     w, h, d = np.shape(img_hsv)
#     image_array = np.reshape(img_hsv, (w * h, d))

#     hue_array = image_array[:, 0]
#     kmeans = KMeans(n_clusters=k)

#     # Get labels for all points
#     labels = kmeans.fit_predict(hue_array.reshape(-1, 1))

#     means = kmeans.cluster_centers_
#     image_out = np.zeros((w, h, d))
#     label_idx = 0
#     for i in range(w):
#         for j in range(h):
#             hue = means[labels[label_idx]]
#             image_out[i][j] = [hue, img_hsv[i, j, 1], img_hsv[i, j, 2]]
#             label_idx += 1
#     image_out = hsv2rgb(image_out) * 255
#     image_out = np.array(image_out, dtype='uint8')
#     return image_out


    ##########################################################################
    ##########################################################################


def compute_quantization_error(img: np.ndarray, quantized_img: np.ndarray) -> int:
    """
    Compute the sum of squared error between the two input images.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        quantized_img: Quantized RGB image with shape H x W x 3 and dtype "uint8"

    Output:

    """
    error = 0

    ##########################################################################
    # TODO: Compute the sum of squared error.                                #
    ##########################################################################
    error = np.square(img.astype('float32') - quantized_img.astype('float32'))
    return np.sum(error)


    ##########################################################################
    ##########################################################################


def get_hue_histograms(img: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the histogram values two ways: equally spaced and clustered.

    Inputs:
        img: Input RGB image with shape H x W x 3 and dtype "uint8"
        k: The number of clusters to use

    Output:
        hist_equal: The values for an equally spaced histogram
        hist_clustered: The values for a histogram of the cluster assignments
    """
    hues = rgb2hsv(img)[:, :, 0]
    hues = np.ravel(hues)
    hist_equal = plt.hist(hues, bins=k)


    kmeans = KMeans(n_clusters=k,random_state=101)
    labels = kmeans.fit_predict(hues.reshape(-1,1))
    hist_clustered = plt.hist(labels, bins=k)


    ##########################################################################
    # TODO: Convert the image to HSV. Calculate a k-bin histogram for the    #
    # hue dimension. Calculate the k-means clustering of the hue space.      #
    # Calculate the histogram values for the cluster assignments.            #
    ##########################################################################



    ##########################################################################
    ##########################################################################

    return hist_equal[0], hist_clustered[0]

    ##########################################################################
    # TODO: Convert the image to HSV. Calculate a k-bin histogram for the    #
    # hue dimension. Calculate the k-means clustering of the hue space.      #
    # Calculate the histogram values for the cluster assignments.            #
    ##########################################################################
