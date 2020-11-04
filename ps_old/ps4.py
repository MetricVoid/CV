import os
import glob
import imageio
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
 
#############################################################################
# TODO: Add additional imports
#############################################################################
from dist2 import dist2
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################


def match_descriptors(desc1, desc2):
    """ Finds the `descriptors2` that best match `descriptors1`
    
    Inputs:
    - desc1: NxD matrix of feature descriptors
    - desc2: MxD matrix of feature descriptors

    Returns:
    - indices: the index of N descriptors from `desc2` that 
               best match each descriptor in `desc1`
    """
    N = desc1.shape[0]
    indices = np.zeros((N,), dtype="int64")
    
    ############################
    # TODO: Add your code here #
    ############################
#     def pairwise_dist(x, y): # [5 pts]
#         """
#         Args:
#             x: N x D numpy array
#             y: M x D numpy array
#         Return:
#                 dist: N x M array, where dist2[i, j] is the euclidean distance between 
#                 x[i, :] and y[j, :]
#                 """
#         diff_mat = x.reshape(x.shape[0], 1, -1) - y.reshape(1, y.shape[0], -1)
#         return np.sqrt(np.sum(diff_mat**2, axis=2))
    
    dists = dist2(desc1, desc2)
    indices = np.argmin(dists, axis=1)
    ############################
    #     END OF YOUR CODE     #
    ############################
    
    return indices


def calculate_bag_of_words_histogram(vocabulary, descriptors):
    """ Calculate the bag-of-words histogram for the given frame descriptors.
    
    Inputs:
    - vocabulary: kxd array representing a visual vocabulary
    - descriptors: nxd array of frame descriptors
    
    Outputs:
    - histogram: k-dimensional bag-of-words histogram
    """
    k = vocabulary.shape[0]
    histogram = np.zeros((k,), dtype="int64")
    
    ############################
    # TODO: Add your code here #
    ############################
    matches = np.argmin(dist2(descriptors, vocabulary), axis=1)
    for match in matches:
        histogram[match] += 1
    ############################
    #     END OF YOUR CODE     #
    ############################

    return histogram


def caculate_normalized_scalar_product(hist1, hist2):
    """ Caculate the normalized scalar product between two histograms.
    
    Inputs:
    - hist1: k-dimensional array
    - hist2: k-dimensional array
    
    Outputs:
    - score: the normalized scalar product described above
    """
    score = 0
    
    ############################
    # TODO: Add your code here #
    ############################
    norm1, norm2 = (np.linalg.norm(hist1), np.linalg.norm(hist2))
    if abs(norm1) < 1e-6 or abs(norm2) < 1e-6:
        return 0
    score = hist1.dot(hist2) / (norm1 * norm2)
    ############################
    #     END OF YOUR CODE     #
    ############################
    
    return score