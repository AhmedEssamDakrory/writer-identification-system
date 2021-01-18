from local_binary_pattern import FeatureExtractor
from preprocessing import *
from sklearn.svm import SVC
import time
import cv2
import os
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from itertools import groupby


def check_if_tie_voting(lst):
    lst.sort()
    counts = [len(list(group)) for key, group in groupby(lst)]
    mx_cnt = max(counts)
    cnt = 0
    for i in range(len(counts)):
        if counts[i] == mx_cnt:
            cnt += 1
    return cnt > 1


def get_prediction(img, feature_extractor, model, line_voting):
    cropped_img = Preprocessor.paragraph_extraction(img)
    line_boundries = Preprocessor.line_segmentation(cropped_img)
    i = 0
    lst = []
    hist = None
    hist_acc = None
    print("predicting...........")
    for line in line_boundries:
        lbp = list(feature_extractor.local_binary_pattern(cropped_img[line[0]:line[2], line[1]:line[3]]).ravel())
        if line_voting:
            hist = feature_extractor.histogram(lbp)
            lst.append(model.predict(hist.reshape(1, -1)))
            print("line " + str(i) + " presidctions is: ")
            print(lst[i])
        hist_acc = feature_extractor.histogram_acc(lbp, hist_acc)
        i += 1
    if not line_voting:
        hist_acc /= (hist_acc.mean())
        return model.predict(hist_acc.reshape(1, -1))
    else:
        tie = check_if_tie_voting(lst)
        if tie:
            print("tie! predict on accumulated histogram")
            return model.predict(hist_acc.reshape(1, -1))
        else:
            return max(lst, key=lst.count)