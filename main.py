from local_binary_pattern import FeatureExtractor
from preprocessing import *
from sklearn.svm import SVC
from imutils import paths
import time
import cv2
import os
from matplotlib import pyplot as plt
from natsort import natsorted
from sklearn.neighbors import KNeighborsClassifier


def get_features(root_dir, file, feature_extractor):
    labels = []
    features = []
    for writer in os.listdir(os.path.join(root_dir, file)):
        print(writer)
        for sample in paths.list_images(os.path.join(root_dir, file, writer)):
            print(sample)
            gray = cv2.imread(sample, 0)
            cropped_img = Preprocessor.paragraph_extraction(gray)
            line_boundries = Preprocessor.line_segmentation(cropped_img)
            for line in line_boundries:
                lbp = list(
                    feature_extractor.local_binary_pattern(cropped_img[line[0]:line[2], line[1]:line[3]]).ravel())
                hist = feature_extractor.histogram(lbp)
                labels.append(str(writer))
                features.append(hist)
    return features, labels


def get_prediction(root_dir, file, feature_extractor, model):
    gray = cv2.imread(os.path.join(root_dir, file, 'test.png'), 0)
    cropped_img = Preprocessor.paragraph_extraction(gray)
    line_boundries = Preprocessor.line_segmentation(cropped_img)
    i = 0
    lst = []
    for line in line_boundries:
        if line[2] > line[0] and line[3] > line[1]:
            lbp = list(feature_extractor.local_binary_pattern(cropped_img[line[0]:line[2], line[1]:line[3]]).ravel())
            hist = feature_extractor.histogram(lbp)
            lst.append(model.predict(hist.reshape(1, -1)))
            #print("line " + str(i) + " presidctions is: ")
            #print(lst[i])
            i += 1
    return max(lst, key=lst.count)


def main():
    root_dir = 'test'
    time_file = open("time.txt", "w")
    results_file = open("results.txt", "w")
    actual_result=open("actual_result.txt", "r")
    actual_result_lines=actual_result.readlines()
    feature_extractor = FeatureExtractor(24, 8)
    correct_classification = 0
    tests = natsorted(os.listdir(root_dir))
    num_of_tests = len(tests)
    for test in tests:
        print("Test: ", test)
        start_time = time.time()
        features, labels = get_features(root_dir, test, feature_extractor)

        model = KNeighborsClassifier(n_neighbors=3)
#         model = SVC(C=0.5, gamma='auto', probability=True)
        model.fit(features, labels)

        prediction = get_prediction(root_dir, test, feature_extractor, model)[0]
        time_taken = round(time.time() - start_time, 2)
        if int(prediction)==int(actual_result_lines[int(test)-1]):
            correct_classification += 1
        else:
            print("Wrong classified at TestCase ",test)
        print("Prediction: " + prediction)
        print("Time taken: " + str(time_taken))
        results_file.write(prediction + '\n')
        time_file.write(str(time_taken) + '\n')
    results_file.close()
    time_file.close()

    acc = ( correct_classification / num_of_tests ) * 100.0
    print("Accuracy is : "+str(acc)+' % ')


if __name__ == '__main__':
    main()