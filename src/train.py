from local_binary_pattern import FeatureExtractor
from preprocessing import *
from sklearn.svm import SVC
import cv2
import os
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from imutils import paths


def get_features(img, feature_extractor, line_voting):
    cropped_img = Preprocessor.paragraph_extraction(img)
    line_boundries = Preprocessor.line_segmentation(cropped_img)
    hist = None
    features = []
    for line in line_boundries:
        lbp = list(feature_extractor.local_binary_pattern(cropped_img[line[0]:line[2], line[1]:line[3]]).ravel())
        if line_voting:
            hist = feature_extractor.histogram(lbp)
            features.append(hist)
        else:
            hist = feature_extractor.histogram_acc(lbp, hist)

    if not line_voting:
        hist /= (hist.mean())
        features.append(hist)

    return features


def training_model():
	model = SVC(C=5.0, gamma='auto', probability=True)
	return model
  
def train_1(root_dir, file, feature_extractor, line_voting):
    labels = []
    all_features = []
    for writer in os.listdir(os.path.join(root_dir, file)):
        if writer == 'test.png':
            continue
        print('trainging.........writer:', writer)
        for sample in paths.list_images(os.path.join(root_dir, file, writer)):
            gray_img = cv2.imread(sample, 0)
            features = get_features(gray_img, feature_extractor, line_voting)
            for f in features:
                all_features.append(f)
                labels.append(str(writer))
    model = training_model()
    model.fit(all_features, labels)
    return model


def train_2(root_dir, feature_extractor, writers_dic, line_voting):
    print("training.........")
    labels = []
    all_features = []
    unique_writers = set()
    for sample in os.listdir(root_dir):
        writer = writers_dic[sample[:-4]]
        unique_writers.add(writer)
        print("writer.......", writer)
        gray_img = cv2.imread(os.path.join(root_dir, sample), 0)
        features = get_features(gray_img, feature_extractor, line_voting)
        for f in features:
            all_features.append(f)
            labels.append(str(writer))
    print("trained on ", len(unique_writers), "unique writers")
    model = training_model()
    model.fit(all_features, labels)
    return model
