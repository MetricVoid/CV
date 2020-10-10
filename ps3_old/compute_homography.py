import os
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
def compute_homography(t1, t2):
    """
    Computes the Homography matrix for corresponding image points t1 and t2

    The function should take a list of N ≥ 4 pairs of corresponding points 
    from the two views, where each point is specified with its 2d image 
    coordinates.

    Inputs:
    - t1: Nx2 matrix of image points from view 1
    - t2: Nx2 matrix of image points from view 2

    Returns a tuple of:
    - H: 3x3 Homography matrix
    """
    H = np.eye(3)
    #############################################################################
    # TODO: Compute the Homography matrix H for corresponding image points t1, t2
    #############################################################################
    L0 = []
    for row in range(t1.shape[0]):
        t10i = t1[row][0]
        t11i = t1[row][1]
        t20i = t2[row][0]
        t21i = t2[row][1]
        arr = np.array([
            [t10i, t11i, 1, 0, 0, 0, -t10i * t20i, -t11i * t20i, -t20i],
            [0, 0, 0, t10i, t11i, 1, -t10i * t21i, -t11i * t21i, -t21i]
        ])
        L0.append(arr)
    L = np.vstack(L0)
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(L.T, L))
    return np.reshape(eigenvectors[:, np.argmin(eigenvalues)],(3, 3))