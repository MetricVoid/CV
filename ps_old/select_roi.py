import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from selectRegion import roipoly


if __name__ == "__main__":
    mat = scipy.io.loadmat("twoFrameData.mat")
    im1 = mat.get("im1")

    plt.show()
    plt.imshow(im1)
    plt.title("Select Region of Interest from Image 1")

    roi = roipoly(color="r")
    region = np.array((roi.all_x_points, roi.all_y_points)).T
    indices = roi.get_indices(im1, mat["positions1"])
    np.save("region.npy", region)
    np.save("points.npy", indices)