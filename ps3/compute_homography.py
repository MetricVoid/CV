import numpy as np

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
    max_val = max(t1.max(), t2.max())
    mats = []
    for i in range(t1.shape[0]):
        x = t1[i][0]
        y = t1[i][1]
        xp = t2[i][0]
        yp = t2[i][1]
        mat = np.array([
            [x, y, 1, 0, 0, 0, -x * xp, -y * xp, -xp],
            [0, 0, 0, x, y, 1, -x * yp, -y * yp, -yp]
        ])
        mats.append(mat)
    L = np.vstack(mats)
    eig_vals, eig_vecs = np.linalg.eig(L.T.dot(L))
    H = eig_vecs[:, np.argmin(eig_vals)].reshape((3, 3))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return H