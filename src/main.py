import time
import cv2
import os
import pandas as pd

from train import *
from predict import *
from constants import *
from preprocessing import *

from local_binary_pattern import FeatureExtractor
from sklearn.svm import SVC
from imutils import paths
from matplotlib import pyplot as plt
from natsort import natsorted
from sklearn.neighbors import KNeighborsClassifier
from itertools import groupby

train_line_voting = True
predict_line_voting = True


def main_1():
    root_dir = TEST_CASES_DIR
    time_file = open(OUTPUTS_DIR + "/time.txt", "w")
    results_file = open(OUTPUTS_DIR + "/results.txt", "w")
    actual_result=open(INPUTS_DIR + "/actual_results.txt", "r")
    actual_result_lines=actual_result.readlines()
    feature_extractor = FeatureExtractor(8, 6)
    correct_classification = 0
    tests = natsorted(os.listdir(root_dir))
    num_of_tests = len(tests)
    for test in tests:
        print("Test: ", test)
        start_time = time.time()
        model = train_1(root_dir, test, feature_extractor, train_line_voting)

        test_image_dir = os.path.join(root_dir, test, 'test.png')
        test_img = cv2.imread(test_image_dir, 0)
        prediction = get_prediction(test_img, feature_extractor, model, predict_line_voting)[0]
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

    acc = (correct_classification / num_of_tests) * 100.0
    print("Accuracy is : "+str(acc)+' % ')


def main_2():
    training_dir = TRAINING_DATA_DIR
    validation_dir = VALIDATION_DATA_DIR

    # get writers of images --------------------------------
    writers_dic = {}
    file = pd.read_csv(INPUTS_DIR + '/writer-id.csv', encoding="ISO-8859-1")
    for i in range(0, len(list(file['image']))):
        writers_dic[file['image'][i]] = file['label'][i]

    feature_extractor = FeatureExtractor(8, 6)
    model = train_2(training_dir, feature_extractor, writers_dic, train_line_voting)

    # testing
    tests = os.listdir(validation_dir)
    correct_classification = 0
    num_of_tests = len(tests)
    for sample in tests:
        print("Test image: ", sample)
        test_image_dir = os.path.join(validation_dir, sample)
        test_img = cv2.imread(test_image_dir, 0)
        prediction = get_prediction(test_img, feature_extractor, model, predict_line_voting)[0]
        actual_writer = writers_dic[sample[:-4]]
        print("Actual writer: ", actual_writer)
        print("Predicted writer: ", prediction)
        if int(prediction) == int(actual_writer):
            correct_classification += 1
        else:
            print("Wrong classification!!")
    acc = (correct_classification / num_of_tests) * 100.0
    print("Accuracy is : "+str(acc)+' % ')


if __name__ == '__main__':
    main_2()