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


# def pick_points(img1, img2):
def pick_points(i, r):
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
#     fig, axes = plt.subplots(1,2)
#     axes[0].imshow(img1)
#     axes[1].imshow(img2)
#     p1 = plt.ginput(10)
#     p2 = plt.ginput(10)
#     return p1,p2
    inp = (plt.figure(1)).add_subplot(1, 1, 1)
    inp.imshow(i)
    inp = Cursor(inp, useblit=True, color='red', linewidth=1)
    ref = (plt.figure(2)).add_subplot(1, 1, 1)
    ref.imshow(r)
    ref = Cursor(ref, useblit=True, color='blue', linewidth=1)
    i = 0
    input_points = np.empty((2,0))
    ref_points = np.empty((2,0))
    try:
        while True:
            plt.figure(1)
            input_point = plt.ginput(n = 1, timeout = 0)
            input0 = input_point[0][0]
            input1 = input_point[0][1]
            plt.annotate(str(i),(input0, input1))
            plt.plot(input0, input1, 'bo')
            input_points = np.hstack((input_points,np.array([[input0], [input1]])))
            plt.figure(2)
            ref_point = plt.ginput(n = 1, timeout = 0)
            ref0 = ref_point[0][0]
            ref1 = ref_point[0][1]
            plt.annotate(str(i),(ref0, ref1))
            plt.plot(ref0, ref_point[0][1], 'go')
            ref_points = np.hstack((ref_points,np.array([[ref0],[ref1]])))
            i += 1
    except KeyboardInterrupt:
        return input_points, ref_points
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
