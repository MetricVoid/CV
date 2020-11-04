import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from selectRegion import roipoly
import imageio


if __name__ == "__main__":
    frame_num = int(input("Frame Num: "))
    im = np.array(imageio.imread("PS4Frames/frames/friends_{}.jpeg".format(str(frame_num).zfill(10)))).astype(np.uint8)
    mat = scipy.io.loadmat("PS4SIFT/sift/friends_{}.jpeg.mat".format(str(frame_num).zfill(10)))

    plt.show()
    plt.imshow(im)
    plt.title("Select Region of Interest")

    roi = roipoly(color="r")
    region = np.array((roi.all_x_points, roi.all_y_points)).T
    indices = roi.get_indices(im, mat["positions"])
    np.save("region-{}.npy".format(frame_num), region)
    np.save("points-{}.npy".format(frame_num), indices)