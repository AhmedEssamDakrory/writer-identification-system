# import the necessary packages
from local_binary_pattern import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import cv2
import os

root_directory = 'test/'
for file in os.listdir(root_directory):
    desc = LocalBinaryPatterns(8, 1)
    data = []
    labels = []
    print(file)
    for samples in os.listdir(root_directory + '/' + file):
        print(samples)
        for imagePath in paths.list_images(root_directory + '/' + file + '/' + samples):
            print(imagePath)
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            labels.append(str(samples))
            data.append(hist)
    model = LinearSVC(C=100.0, random_state=42)
    model.fit(data, labels)

    image = cv2.imread(root_directory + '/' + file + '/test.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))
    print(prediction)
