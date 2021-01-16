from skimage import feature
import numpy as np
from matplotlib import pyplot as plt 
from skimage.filters import threshold_otsu
import cv2

class FeatureExtractor:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def local_binary_pattern(self, image):
        threshold = threshold_otsu(image)
        #image = 255 - image
        #binary_img = (image > threshold)
        # threshold the image
        #binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]

        lbp = feature.local_binary_pattern(image, self.num_points, self.radius, method="uniform")
        #lbp = lbp[binary_img == True]

        return lbp

    def histogram(self, arr):
        #(hist, _) = np.histogram(arr, bins=np.arange(0, self.num_points + 3), range=(0, self.num_points + 2))
        (hist, _) = np.histogram(arr, bins= 256)
        hist = hist.astype("float")
        hist /= (hist.mean())
        #print(hist)
        return hist