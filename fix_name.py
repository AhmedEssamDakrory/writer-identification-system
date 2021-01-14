from local_binary_pattern import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
desc = LocalBinaryPatterns(24, 8)
image = cv2.imread('/home/mazen/IAM/training/d06-072.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = desc.describe(gray)
print(hist.shape)
plt.hist(hist)
plt.title("histogram")
plt.show()


dictionary = {}
file = pd.read_csv('writer-id.csv', encoding="ISO-8859-1")
for i in range(0, len(list(file['image']))):
    dictionary[file['image'][i]] = file['label'][i]

