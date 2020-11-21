import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

from imageio import imread, imsave, mimsave


#############################################################################
# TODO: Add additional imports
#############################################################################
import math
import pdb
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
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
    background_index = foreground_image > threshold
    foreground_image[background_index] = 0
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
    diffabs = abs(image1 - image2)
    diff[diffabs > threshold] = True
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
    firstsub = background_subtraction(images[0], background_subtraction_threshold)
    for i in images[1:]:
        sub = background_subtraction(i, background_subtraction_threshold)
        diff = compute_two_frame_difference(firstsub, sub, difference_threshold)
        MHI = MHI - 1
        MHI[diff == True] = len(images) - 1
        MHI[MHI < 0] = 0
        firstsub = sub
    MHI = MHI / MHI.max()
        
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return MHI
#############################################################################
# Implement any additional helper functions (if needed) here.               #
#############################################################################


#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
def cal(p, q, MHI, X, Y, x_b, y_b):
    return np.sum(np.power((X-x_b),p) * np.power((Y-y_b),q) * MHI, dtype=np.float64)

def compute_hu_moments(MHI):
    """
    Calculate the 7 Hu Moments of a Motion History Image

    Inputs:
    - MHI: HxW matrix - the Motion History Image

    Returns:
    - hm: A 1d array of length 7 containing the Hu Moments 
    """
    hm = np.zeros(7)
    #############################################################################
    # TODO: Compute Hu Moments for the given Motion History Image (H)           #  
    #############################################################################
    x, y = np.arange(0, MHI.shape[1], 1), np.arange(0,MHI.shape[0],1)
    x1, y1 = np.meshgrid(x, y, sparse=False, indexing='xy')
    x1 += 1
    y1 += 1
    m00 = np.sum(MHI, dtype=np.float64)
    x_b = np.sum(x1*MHI, dtype=np.float64) / m00
    y_b = np.sum(y1*MHI, dtype=np.float64) / m00
    u02 = cal(0,2,MHI,x1,y1,x_b,y_b)
    u03 = cal(0,3,MHI,x1,y1,x_b,y_b)
    u11 = cal(1,1,MHI,x1,y1,x_b,y_b)
    u12 = cal(1,2,MHI,x1,y1,x_b,y_b)
    u20 = cal(2,0,MHI,x1,y1,x_b,y_b)
    u21 = cal(2,1,MHI,x1,y1,x_b,y_b)
    u30 = cal(3,0,MHI,x1,y1,x_b,y_b)
    hm = [u20 + u02, np.power((u20 - u02),2) + 4 * np.power(u11,2),
          np.power((u30 - 3 * u12),2) + np.power((3 * u21 - u03), 2), np.power((u30 + u12),2) + np.power((u21 + u03), 2),
          (u30 - 3 * u12) * (u30 + u12) * (np.power((u30 + u12),2) - 3 * np.power((u21 + u03), 2)) + (3 * u21 - u03) * (u21 + u03) * (3 * np.power((u30 + u12), 2) - np.power((u21 + u03), 2)),
          (u20 - u02) * (np.power((u30 + u12),2) - np.power((u21 + u03),2)) + 4 * u11 * (u30 + u12) * (u21 + u03), 
          (3 * u21 - u03) * (u30 + u12) * (np.power((u30 + u12),2) - 3 * np.power((u21 + u03),2)) - (u30 - 3 * u12) * (u21 + u03) * (3 * np.power((u30 + u12), 2) - np.power((u21 + u03), 2))]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return hm
#############################################################################
# Implement any additional helper functions (if needed) here                #
#############################################################################

#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

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
    for i in range(c.shape[0]):
        dists[i] = np.sqrt(np.sum(np.divide(np.power((c[i] - x), 2), var)))
        
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
    predictedLabel = 0
    #############################################################################
    # TODO: Using nearest neighbours predict the action label for testMoment   #  
    #############################################################################
    distance = np.zeros((1, trainMoments.shape[0]))
    for i in range(0, trainMoments.shape[0]):
        distance[:,i] = np.power(np.sum(((np.reshape(trainMoments[i,:], (-1, 1)) - np.reshape(testMoment,(-1, 1)))**2) / np.reshape(np.nanvar(trainMoments, axis = 0), (-1, 1))),0.5)
    sortdistance = np.argsort(distance) 
    predictedLabel = trainLabels[sortdistance][0][0]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return predictedLabel

