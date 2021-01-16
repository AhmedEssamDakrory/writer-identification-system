from random import random

import pandas as pd
import os
import secrets
import shutil
from imutils import paths

dictionary = {}
file = pd.read_csv('writer-id.csv', encoding="ISO-8859-1")
for i in range(0, len(list(file['image']))):
    dictionary[file['image'][i]] = file['label'][i]
writer_dict = {}
for key, value in dictionary.items():
    writer_images = [k for k, v in dictionary.items() if v == value]
    writer_dict[value] = writer_images
desired_writers = {}
for key, value in writer_dict.items():
    if len(value) >= 3:
        desired_writers[key] = value

# Create Work Directory
directory = "tests"
cwd = os.getcwd()
parent_dir = cwd
path = os.path.join(parent_dir, directory)
os.mkdir(path)
for key, value in desired_writers.items():
    current_directory = path + '/' + str(key)
    os.mkdir(current_directory)
    for image in value:
        shutil.copyfile("/home/mazen/IAM/training/" + image, current_directory + "/" + image)

directory = "tests"
cwd = os.getcwd()
parent_dir = cwd
path = os.path.join(parent_dir, directory)
# Create TestCases
list_sub_folders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
list_ids = [list_sub_folders_with_paths[i][len(path) + 1:] for i in range(0, len(list_sub_folders_with_paths))]
directory = "TestCases"
path = os.path.join(parent_dir, directory)
main_path = path
os.mkdir(path)
results = open(cwd + '/results.txt', "a")
for i in range(1, 101):
    path = main_path
    path = path + '/' + str(i)
    os.mkdir(path)
    first = secrets.choice(list_ids)
    second = secrets.choice(list_ids)
    while second == first:
        second = secrets.choice(list_ids)
    third = secrets.choice(list_ids)
    while third == first or third == second:
        third = secrets.choice(list_ids)
    os.mkdir(path + '/' + str(first))
    os.mkdir(path + '/' + str(second))
    os.mkdir(path + '/' + str(third))

    cwd = os.getcwd() + '/tests/' + str(first)
    counter = 0
    for samples in paths.list_images(cwd):
        if counter == 2:
            break
        shutil.copyfile(samples, path + '/' + str(first) + '/' + str(samples[len(cwd):]))
        counter += 1

    cwd = os.getcwd() + '/tests/' + str(second)
    counter = 0
    for samples in paths.list_images(cwd):
        if counter == 2:
            break
        shutil.copyfile(samples, path + '/' + str(second) + '/' + str(samples[len(cwd):]))
        counter += 1

    cwd = os.getcwd() + '/tests/' + str(third)
    counter = 0
    for samples in paths.list_images(cwd):
        if counter == 2:
            break
        shutil.copyfile(samples, path + '/' + str(third) + '/' + str(samples[len(cwd):]))
        counter += 1

    test_image_id = secrets.choice([first, second, third])
    cwd = os.getcwd() + '/tests/' + str(test_image_id)
    counter = 0
    for samples in paths.list_images(cwd):
        counter += 1
        if counter == 3:
            # shutil.copyfile(samples, path + str(samples[len(cwd):]))
            shutil.copy(samples, path + '/test.png')
            break
    # f = open(path + "/answer.txt", "a")
    # f.write(str(test_image_id))
    results.write(str(test_image_id) + '\n')
    # f.close()
results.close()
