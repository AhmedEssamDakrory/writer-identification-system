from local_binary_pattern import FeatureExtractor
from preprocessing import *
from sklearn.svm import LinearSVC
from imutils import paths
import time
import cv2
import os
import csv

indcies = {}
model = None

def train():
    global indcies, model

    trainset = open("trainset.txt", "r")
    
    feature_extractor = FeatureExtractor(8, 2)
    labels = []
    features = []

    for image in trainset:
        
        image = image[:-1]
        print("./dataset/"+image+".png")
        gray = cv2.imread("./dataset/"+image+".png", 0)
        #results_file.write("Image: ", image)
        #start_time = time.time()

        # Preprocess the data
        cropped = Preprocessor.paragraph_extraction(gray)
        line_boundries = Preprocessor.line_segmentation(cropped)

        # Get feature for each line
        lbp = []
        for line in line_boundries:
            if line[2]>line[0] and line[3]>line[1]: # Valid boundary
                lbp += get_features(gray[line[0]:line[2],line[1]:line[3]], feature_extractor)
        hist = feature_extractor.histogram(lbp)
        labels.append(indcies[image])
        features.append(hist)

    trainset.close()

    #print("features is ", len(features[0]))
    model = SVC()
    model.fit(features, labels)
    return

def validate():
    global indcies, model

    results_file = open("results.txt", "w")
    validationset1 = open("validationset1.txt", "r")

    feature_extractor = FeatureExtractor(8, 2)

    for image in validationset1:

        image = image[:-1]
        print(image)
        gray = cv2.imread("./dataset/"+image+".png", 0)
        prediction = get_prediction(gray, feature_extractor, model)[0]
        #time_taken = round(time.time() - start_time, 2)
        #print("Prediction: " + prediction)
        #print("Time taken: " + str(time_taken))
        results_file.write("Writer: " + indcies[image] + " and Prediction: " +prediction + '\n')
        #time_file.write(str(time_taken) + '\n')
    
    results_file.close()
    return

def get_features(img, feature_extractor):
    lbp = list(feature_extractor.local_binary_pattern(img).ravel())
    return lbp


def get_prediction(image, feature_extractor, model):
    line_boundries = Preprocessor.line_segmentation(Preprocessor.paragraph_extraction(image))
    lbp = []
    for line in line_boundries:
        if line[2]>line[0] and line[3]>line[1]:
            lbp += list(feature_extractor.local_binary_pattern(image[line[0]:line[2], line[1]:line[3]]).ravel())
    hist = feature_extractor.histogram(lbp)
    return model.predict(hist.reshape(1, -1))


def main():
    global indcies

    #time_file = open("time.txt", "w")
    
    # Get all images indcies
    with open('writer-id.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            indcies[row[0]] = row[1]
    
    train()

    validate()
    
    #time_file.close()


if __name__ == '__main__':
    main()
