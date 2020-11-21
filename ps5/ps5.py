import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

from imageio import imread, imsave, mimsave
#from IPython.display import Image
def variance(data):
    n = len(data)
    mean = np.sum(data, axis=0) / n
    deviations = [(x - mean) ** 2 for x in data]
    var = np.sum(deviations, axis=0) / n
    return var


def background_subtraction(image, threshold):
    """
    Returns an image with the background removed (zeroed out) wherever the pixel value
    is greater than the threshold. 

    Inputs:
    - image: HxW matrix - the image
    - threshold: integer - the background threshold
    
    Outputs:
    - foreground_image: the output image containing only the foreground pixels
    """
    foreground_image = image.copy()

    #############################################################################
    # TODO: Add your code here (note: this should only take a few line).        #
    #############################################################################
    x, y = np.where(foreground_image > threshold)
    foreground_image[x, y] = 0
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return foreground_image


def compute_two_frame_difference(image1, image2, threshold):
    """
    Returns a binary image that is `True` where the absolute difference between
    the two images is above a threshold.

    Inputs:
    - image1: HxW matrix - the first image
    - image2: HxW matrix - the second image
    - threshold: integer - the difference threshold
    
    Outputs:
    - diff: HxW matrix the output binary image
    """
    diff = np.zeros_like(image1, dtype="bool")

    #############################################################################
    # TODO: Add your code here (note: this should only take a few line).        #
    #############################################################################
    x, y = np.where(np.abs(image1 - image2) > threshold)
    diff[x, y] = True
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return diff


def compute_motion_history_image(images, background_subtraction_threshold, difference_threshold):
    """ Returns a Motion History Image (MHI) for an action sequence.
    
    Steps:
    - Use your `background_subtraction` function to remove the background pixels.
    - Use your `compute_two_frame_difference` function to calculate $D(x,y,t)$
      for the image sequence.
    - Calculate the MHI using the equation in step 2 of the approach overview.
    - Remember to normalize the Motion History Image (step 3)
    
    Inputs:
    - images: List of HxW matrices - the image sequence
    - background_subtraction_threshold: integer - the threshold for background subtraction
    - difference_threshold: Integer - threshold for frame differencing
    
    Outputs:
    - MHI: HxW matrix - motion history image
    """
    assert len(images) > 0
    MHI = np.zeros_like(images[0])
    
    #############################################################################
    # TODO: Add your code here.                                                 #
    #############################################################################
    n = len(images)
    diff_list = []
    for i in range(n - 1):
        background_removed_image_one = background_subtraction(images[i], background_subtraction_threshold)
        background_removed_image_two = background_subtraction(images[i + 1], background_subtraction_threshold)
        diff = compute_two_frame_difference(background_removed_image_one, background_removed_image_two,
                                            difference_threshold)
        diff_list.append(diff)
    tau = n - 1
    for t in range(tau):
        diff = diff_list[t]
        x_true, y_true = np.where(diff == True)
        MHI[x_true, y_true] = tau
        x_false, y_false = np.where(diff == False)
        MHI[x_false, y_false] = MHI[x_false, y_false] - 1
        negative_x, negative_y = np.where(MHI < 0)
        MHI[negative_x, negative_y] = 0
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    max_h = MHI.max()
    if max_h != 0:
        MHI = MHI / max_h
    return MHI


def compute_hu_moments(MHI):
    """
    Calculate the 7 Hu Moments of a Motion History Image

    Inputs:
    - MHI: HxW matrix - the Motion History Image

    Returns:
    - hm: A 1d array of length 7 containing the Hu Moments
    """
    hm = np.zeros(7)

    x, y = MHI.shape
    sum_y_axis = np.sum(MHI, axis=1)
    sum_x_axis = np.sum(MHI, axis=0)
    M00 = np.sum(MHI)
    M10 = np.sum(sum_y_axis * list(range(x)))
    M01 = np.sum(sum_x_axis * list(range(y)))
    x_bar = M10 / M00
    y_bar = M01 / M00

    x_temp = [X - x_bar for X in list(range(x))]
    x_temp = np.array(x_temp)

    y_temp = [Y - y_bar for Y in list(range(y))]
    y_temp = np.array(y_temp)

    mu20 = np.sum(sum_y_axis * pow(x_temp, 2))  # good
    mu02 = np.sum(sum_x_axis * pow(y_temp, 2))  # good

    mu11 = np.sum(x_temp * np.sum(y_temp * MHI, axis=1))  # good

    mu30 = np.sum(sum_y_axis * pow(x_temp, 3))

    mu12 = np.sum(x_temp * np.sum(pow(y_temp, 2) * MHI, axis=1))

    mu21 = np.sum(pow(x_temp, 2) * np.sum(y_temp * MHI, axis=1))

    mu03 = np.sum(sum_x_axis * pow(y_temp, 3))

    mu20 = mu20/pow(M00,2)
    mu02 = mu02/pow(M00,2)

    mu11 = mu11/pow(M00,2)

    mu30 = mu30/pow(M00,2.5)
    mu03 = mu03/pow(M00,2.5)

    mu12 = mu12/pow(M00,2.5)
    mu21 = mu21/pow(M00,2.5)

    hm[0] = mu02 + mu20
    hm[1] = pow((mu02 - mu20), 2) + 4 * pow(mu11, 2)

    hm[2] = pow((mu30 - 3 * mu12), 2) + pow((3 * mu21 - mu03), 2)

    hm[3] = pow((mu30 + mu12), 2) + pow((mu03 + mu21), 2)

    hm[4] = (mu30 - 3 * mu12) * (mu30 + mu12) * (pow((mu30 + mu12), 2) - 3 * pow((mu03 + mu21), 2)) + (
                3 * mu21 - mu03) * (mu21 + mu03) * (3 * pow((mu30 + mu12), 2) - pow((mu03 + mu21), 2))
    hm[5] = (mu20 - mu02) * (pow((mu30 + mu12), 2) - pow((mu03 + mu21), 2)) + 4 * mu11 * (mu30 + mu12) * (mu21 + mu03)
    hm[6] = (3 * mu21 - mu03) * (mu30 + mu12) * (pow((mu30 + mu12), 2) - 3 * pow((mu03 + mu21), 2)) - (
                mu30 - 3 * mu12) * (mu21 + mu03) * (3 * pow((mu30 + mu12), 2) - pow((mu03 + mu21), 2))
    return hm


def normalized_euclidean_distance(x, c, var):
    """
    Returns the Euclidean distance normalized with the variance of the data sample.

    Inputs:
    - x: 7 dimensional vector containing the testMoment
    - c: A matrix of shape (N, 7) containing the trainMoments
    - var: 7 dimensional vector containing the variance of each Hu moment across the data sample.

    Outputs:
    - dists: A matrix of shape (N, 1) containing the normalized euclidean distance of
      each trainMoment from the testMoment
    """
    N = c.shape[0]
    dists = np.zeros((N, 1), dtype="float64")
    #############################################################################
    # TODO: Add your code here                                                  #
    #############################################################################
    for i in range(N):
        dists[i] = pow(np.sum((c[i] - x)**2 / var), 0.5)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dists


def predict_action(testMoment, trainMoments, trainLabels):
    """
    Predict the action label for testMoment by using
    nearest neighbour classification on trainMoments and trainLabels.

    Steps:
    - Calculate the variance of each of the 7 Hu Moments across the entire data sample (Train + Test data).
    - Use your `normalized_euclidean_distance` function to calculate the distance
      between the testMoment from all the trainMoments.
    - Return the label of the training data point in trainMoments that is nearest to the testMoment.

    Note: This function can be used to perform leave-one-out cross validation.

    Inputs:
    - testMoment: 7 dimensional Hu moment decriptor representing the Test sequence
    - trainMoments: A matrix of shape (N, 7) containing Hu moment descriptors for N training instances
    - trainLabels: A vector of shape (N, 1) containing the action category labels for the N training instances

    Returns:
    - predictedLabel: An integer from 0 to 4 denoting the predicted action label
    """
    #############################################################################
    # TODO: Using nearest neighbours predict the action label for testMoment   #
    #############################################################################

    n = len(testMoment)
    data = np.concatenate((np.reshape(testMoment, (1, n)), trainMoments))
    var = variance(data)
    dists = normalized_euclidean_distance(testMoment, trainMoments, var)
    return np.argmin(dists)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
