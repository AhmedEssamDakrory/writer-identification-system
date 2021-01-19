import cv2
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
# from preprocessing import *
from skimage.filters import threshold_otsu
import numpy as np
import math
import os


class Preprocessor:
    @staticmethod
    def paragraph_extraction(gray):
        edges = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=5)
        # Remove noise with opening operator
        kernel = np.ones((5, 5), np.uint8)
        kernel[0, :] = 0
        kernel[4, :] = 0
        opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

        # Get horizontal lines
        lines = cv2.HoughLinesP(opening, 1, math.pi / 2, 100, None, 600, 1)

        # Get horizontal lines rows
        rows = []
        for line in lines:
            rows.append(line[0][1])
            rows.sort()

        # Get upper, lower bounds
        lower, upper = 2850, 0
        for row in rows:
            if 400 < row < 1100:
                upper = max(upper, row)
            elif 2500 < row < 2850:
                lower = min(lower, row)
        return gray[upper: lower - 20, 200:]  # Margin to remove edge pixels

    @staticmethod
    def binarization(gray_img):
        reversed_img = 255 - gray_img
        # average filter
        average_kernel = np.ones((3, 3), np.float32) / 9
        filtered_img = cv2.filter2D(reversed_img, -1, average_kernel)
        # opening and closing
        open_close_kernel = np.ones((1, 20), np.float32)
        opened_img = cv2.morphologyEx(filtered_img, cv2.MORPH_OPEN, open_close_kernel)
        closed_img = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, open_close_kernel)
        diff = closed_img - opened_img
        closed_kernel2 = np.ones((5, 5), np.float32)
        closed_img2 = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, closed_kernel2)
        closed_img2 = cv2.GaussianBlur(closed_img2, (9, 9), 0)
        # thresholding
        threshold = threshold_otsu(closed_img2)
        binary_img = closed_img2 > threshold
        return binary_img

    @staticmethod
    def get_peaks(hist):
        sum = 0
        non_zeros = 0
        for e in hist:
            sum += e
            if (e > 0):
                non_zeros += 1

        avg_peak = sum / non_zeros

        peaks_pos = []
        i = 0
        while i < len(hist):
            peak_idx = -1
            mx = 0
            while i < len(hist) and hist[i] >= avg_peak:
                if hist[i] > mx:
                    mx = hist[i]
                    peak_idx = i
                i += 1
            i += 1
            if peak_idx != -1:
                peaks_pos.append(peak_idx)
        return peaks_pos

    @staticmethod
    def get_valleys(hist, peaks_pos):
        valleys = []
        s = 0
        while hist[s] < 5:
            s += 1
        for i in range(0, len(peaks_pos), 1):
            if i != 0:
                l = peaks_pos[i - 1]
            else:
                l = s
            r = peaks_pos[i]
            mn = min(hist[l:r])
            mn_pos = []
            for j in range(l, r + 1, 1):
                if hist[j] == mn:
                    mn_pos.append(j)
            valleys.append(mn_pos[len(mn_pos) // 2])
        valleys.append(len(hist) - 1)
        return valleys

    @staticmethod
    def add_missed_valleys_and_peaks(hist, peaks_pos, valleys_pos):
        # calculate average distance between two peaks
        avg_peak_dist = 0
        for i in range(1, len(peaks_pos), 1):
            avg_peak_dist += peaks_pos[i] - peaks_pos[i - 1] + 1
        avg_peak_dist //= (len(peaks_pos) - 1)

        # if the distance between two valleys is about twice the diastance
        # between two peeks then we are missing a peak in between
        for i in range(1, len(valleys_pos), 1):
            dist = valleys_pos[i] - valleys_pos[i - 1] + 1
            if dist >= 1.5 * avg_peak_dist:
                l = valleys_pos[i - 1] + avg_peak_dist // 3
                r = valleys_pos[i] - avg_peak_dist // 3
                mn = min(hist[l:r])
                mn_pos = []
                for j in range(l, r + 1, 1):
                    if hist[j] == mn:
                        mn_pos.append(j)
                missed_valley_pos = mn_pos[len(mn_pos) // 2]
                mx_peak_in_between = max(hist[missed_valley_pos: valleys_pos[i]])
                if mx_peak_in_between >= 30:
                    valleys_pos.insert(i, missed_valley_pos)

    @staticmethod
    def remove_false_lines(hist, valleys_pos):
        i = 1
        while i < len(valleys_pos):
            l = valleys_pos[i - 1]
            r = valleys_pos[i]
            mx = max(hist[l:r])
            if mx < 30:
                valleys_pos.pop(i)
            i += 1

    @staticmethod
    def smooth_hist(hist, kernel_size):
        smoothed_hist = []
        for i in range(len(hist)):
            l = max(0, i - kernel_size)
            r = min(len(hist) - 1, i + kernel_size)
            sum = np.sum(hist[l:r])
            avg = sum // (r - l + 1)
            smoothed_hist.append(avg)
        return smoothed_hist

    @staticmethod
    def crop_hist(hist):
        i = len(hist) - 1
        while i >= 0 and hist[i] < 5:
            hist.pop()
            i -= 1

    @staticmethod
    def get_boundaries(valleys_pos, binary_img, hist):
        boundaries = []
        width = binary_img.shape[1] - 1
        for i in range(1, len(valleys_pos), 1):
            up = valleys_pos[i - 1]
            down = valleys_pos[i]
            vertical_sum = np.sum(binary_img[up:down + 1, :], axis=0)
            x1 = 0
            x2 = width
            while x1 < width and vertical_sum[x1] < 5:
                x1 += 1
            while x2 >= 0 and vertical_sum[x2] < 5:
                x2 -= 1
            while up <= down and hist[up] == 0:
                up += 1
            while down >= up and hist[down] == 0:
                down -= 1
            x1 = max(x1 - 10, 0)
            x2 = min(x2 + 10, width)
            boundaries.append((up, x1, down, x2))
        return boundaries

    @staticmethod
    def draw_segmented_lines(img, boundaries):
        for b in boundaries:
            x1 = b[0]
            y1 = b[1]
            x2 = b[2]
            y2 = b[3]
            cv2.rectangle(img, (y1, x1), (y2, x2), (0, 255, 0), 2)

    @staticmethod
    def display_image(img):
        dpi = matplotlib.rcParams['figure.dpi']
        height, width = img.shape
        # Determine the figures size in inches
        figsize = width / float(dpi), height / float(dpi)
        plt.figure(figsize=figsize)
        plt.imshow(img, cmap='gray')
        plt.show()

    @staticmethod
    def line_segmentation(cropped_img):
        binary_img = Preprocessor.binarization(cropped_img)
        hist = np.sum(binary_img, axis=1).tolist()
        hist = Preprocessor.smooth_hist(hist, 30)
        Preprocessor.crop_hist(hist)
        peaks_pos = Preprocessor.get_peaks(hist)
        valleys_pos = Preprocessor.get_valleys(hist, peaks_pos)
        Preprocessor.add_missed_valleys_and_peaks(hist, peaks_pos, valleys_pos)
        Preprocessor.remove_false_lines(hist, valleys_pos)
        boundaries = Preprocessor.get_boundaries(valleys_pos, binary_img, hist)
        #         plt.plot(hist)
        #         plt.show()
        return boundaries
