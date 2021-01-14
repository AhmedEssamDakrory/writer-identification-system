# import the necessary packages
from skimage import feature
import numpy as np
import cv2
from matplotlib import pyplot as plt


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        # plt.imshow(lbp, cmap="Greys")
        # plt.show()
        # plt.imshow(image,cmap="Greys")
        # plt.show()
        # cv2.imshow('image', lbp)
        (hist, _) = np.histogram(lbp.ravel(), 256, [0, 256])
        #hist = cv2.calcHist(lbp.ravel(), [0], None, [256], [0, 256])
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist
