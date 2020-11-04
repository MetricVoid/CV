from dist2 import dist2
import numpy as np


def match_descriptors(desc1, desc2, threshold=None):
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
    dists = dist2(desc1, desc2)
    for descriptor_idx, descriptor_dists in enumerate(dists):
        min_dist, min2_dist = np.sort(descriptor_dists)[:2]
        ratio = min_dist / min2_dist
        if threshold is None or ratio < threshold:
            indices[descriptor_idx] = np.argmin(descriptor_dists)
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
    hist = np.zeros((k,), dtype="int64")
    dists = dist2(descriptors, vocabulary)
    for desc_dists in dists:
        hist[np.argmin(desc_dists)] += 1
    return hist


def caculate_normalized_scalar_product(hist1, hist2):
    """
    Calculate the normalized scalar product between two histograms.

    Inputs:
    - hist1: k-dimensional array
    - hist2: k-dimensional array

    Outputs:
    - score: the normalized scalar product described above
    """
    n1 = np.linalg.norm(hist1, ord=2)
    n2 = np.linalg.norm(hist2, ord=2)
    temp = n1 * n2 + 1E-15
    result = hist1.dot(hist2) / temp
    return result
