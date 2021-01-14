from local_binary_pattern import FeatureExtractor
from matplotlib import pyplot as plt
import pandas as pd
import cv2

desc = FeatureExtractor(24, 8)
image = cv2.imread('d06-072.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = desc.local_binary_pattern(gray)
print(hist.shape)
plt.hist(hist)
plt.title("histogram")
plt.show()


dictionary = {}
file = pd.read_csv('writer-id.csv', encoding="ISO-8859-1")
for i in range(0, len(list(file['image']))):
    dictionary[file['image'][i]] = file['label'][i]

