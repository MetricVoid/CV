import os
import imageio
import argparse
import numpy as np

#############################################################################
# TODO: Add additional imports
#############################################################################
import matplotlib.pyplot as plt
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################


def get_parser():
    parser = argparse.ArgumentParser(description="Points Selection")
    parser.add_argument("image1", type=str, help="path to image 1")
    parser.add_argument("image2", type=str, help="path to image 2")
    return parser


def pick_points(img1, img2):
    """
    Functionality to get manually identified corresponding points from two views.

    Inputs:
    - img1: The first image to select points from
    - img2: The second image to select points from

    Output:
    - coords1: An ndarray of shape (N, 2) with points from image 1
    - coords2: An ndarray of shape (N, 2) with points from image 2
    """
    ############################################################################
    # TODO: Implement pick_points
    ############################################################################
    num_points = int(input('How many points: '))
    coords1 = np.zeros((num_points, 2))
    coords2 = coords1.copy()

    def onclick(event):
        curr_coords = coords1 if i % 2 == 0 else coords2
        curr_coords[i // 2, 0] = event.xdata
        curr_coords[i // 2, 1] = event.ydata
        if np.any(np.isnan(curr_coords[i // 2])):
            return
        plt.close()
    
    for i in range(num_points * 2):
        fig, ax = plt.subplots()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        if i % 2 == 0:
            plt.imshow(img1)
            plt.title('Pick point %d from image 1.' % (i // 2 + 1))
        else:
            plt.imshow(img2)
            plt.title('Pick point %d from image 2.' % (i // 2 + 1))
        plt.show()
        fig.canvas.mpl_disconnect(cid)
    
    return coords1, coords2
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################


if __name__ == "__main__":
    args = get_parser().parse_args()

    img1 = np.asarray(imageio.imread(args.image1))
    img2 = np.asarray(imageio.imread(args.image2))

    coords1, coords2 = pick_points(img1, img2)

    assert len(coords1) == len(coords2), "The number of coordinates does not match"

    filename1 = os.path.splitext(args.image1)[0] + ".npy"
    filename2 = os.path.splitext(args.image2)[0] + ".npy"

    assert not os.path.exists(filename1), f"Output file {filename1} already exists"
    assert not os.path.exists(filename2), f"Output file {filename2} already exists"

    np.save(filename1, coords1)
    np.save(filename2, coords2)
