from skimage import feature
import numpy as np


class FeatureExtractor:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def local_binary_pattern(self, image):
        lbp = feature.local_binary_pattern(image, self.num_points, self.radius, method="uniform")
        # (hist, _) = np.histogram(lbp.ravel(), 256, [0, 256])
        # (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.num_points + 3), range=(0, self.num_points + 2))
        # hist = hist.astype("float")
        # hist /= (hist.mean())
        return lbp

    def histogram(self, arr):
        #(hist, _) = np.histogram(arr, bins=np.arange(0, self.num_points + 3), range=(0, self.num_points + 2))
        (hist, _) = np.histogram(arr, bins= 256)
        hist = hist.astype("float")
        hist /= (hist.mean())
        return hist