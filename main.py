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
from itertools import groupby

train_line_voting = True
predict_line_voting = True

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
            hist = None
            for line in line_boundries:
                lbp = list(
                    feature_extractor.local_binary_pattern(cropped_img[line[0]:line[2], line[1]:line[3]]).ravel())
                if train_line_voting:
                    hist = feature_extractor.histogram(lbp)
                    labels.append(str(writer))
                    features.append(hist)
#                     plt.plot(hist)
#                     plt.show()
                else:
                    hist = feature_extractor.histogram_acc(lbp, hist)
            if not train_line_voting:
                hist /= (hist.mean())
                labels.append(str(writer))
                features.append(hist)
            
#                 plt.plot(hist)
#                 plt.show()
            
#             Preprocessor.draw_segmented_lines(cropped_img, line_boundries)
#             Preprocessor.display_image(cropped_img)
    return features, labels

def check_if_tie_voting(lst):
    lst.sort()
    counts = [len(list(group)) for key, group in groupby(lst)]
    tie = False
    for i in range(1, len(counts), 1):
        if counts[i] == counts[i-1]:
            tie = True
            break
    return tie

def get_prediction(root_dir, file, feature_extractor, model):
    gray = cv2.imread(os.path.join(root_dir, file, 'test.png'), 0)
    cropped_img = Preprocessor.paragraph_extraction(gray)
    line_boundries = Preprocessor.line_segmentation(cropped_img)
    i = 0
    lst = []
    hist = None
    hist_acc = None
    for line in line_boundries:
        if line[2] > line[0] and line[3] > line[1]:
            lbp = list(feature_extractor.local_binary_pattern(cropped_img[line[0]:line[2], line[1]:line[3]]).ravel())
            if predict_line_voting:
                hist = feature_extractor.histogram(lbp)
                lst.append(model.predict(hist.reshape(1, -1)))
                print("line " + str(i) + " presidctions is: ")
                print(lst[i])
            hist_acc = feature_extractor.histogram_acc(lbp, hist_acc)
            i += 1
    if not predict_line_voting:
        hist_acc /= (hist_acc.mean())
        return model.predict(hist_acc.reshape(1, -1))
    else:
        tie = check_if_tie_voting(lst)
        if tie:
            print("tie! predict on accumulated histogram")
            return model.predict(hist_acc.reshape(1, -1))
        else:
            return max(lst, key=lst.count)


def main():
    root_dir = 'TestCases'
    time_file = open("time.txt", "w")
    results_file = open("results.txt", "w")
    actual_result=open("actual_results.txt", "r")
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