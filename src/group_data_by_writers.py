from random import random
from src.constants import *
import pandas as pd
import os
import secrets
import shutil
from imutils import paths

dictionary = {}
file = pd.read_csv(INPUTS_DIR+'/writer-id.csv', encoding="ISO-8859-1")
for i in range(0, len(list(file['image']))):
    dictionary[file['image'][i]] = file['label'][i]
writer_dict = {}
for key, value in dictionary.items():
    writer_images = [k for k, v in dictionary.items() if v == value]
    writer_dict[value] = writer_images
desired_writers = {}
for key, value in writer_dict.items():
    if len(value) >= 1:
        desired_writers[key] = value

# Create Work Directory
directory = "all_writers"
parent_dir = DATA_DIR
path = os.path.join(parent_dir, directory)
os.mkdir(path)
for key, value in desired_writers.items():
    current_directory = path + '/' + str(key)
    os.mkdir(current_directory)
    for image in value:
        shutil.copyfile(DATASET_DIR + "/" + image + ".png", current_directory + "/" + image + ".png")