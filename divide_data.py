import os
import random
import cv2
root_dir = 'tests'
training_data_dir = 'training_data'
test_data_dir = 'testing_data'

mx_num = 3
ls = os.listdir(root_dir)
random.shuffle(ls)
for writer in ls:
    if mx_num <= 0:
        break
    lst_images = os.listdir(os.path.join(root_dir, writer))
    if len(lst_images) > 2:
        mx_num -= 1
        random.shuffle(lst_images)
        for i in range(2):
            img = cv2.imread(os.path.join(root_dir, writer, lst_images[i]))
            cv2.imwrite(os.path.join(training_data_dir, lst_images[i]) , img)
        for i in range(2, min(len(lst_images), 6) , 1):
            img = cv2.imread(os.path.join(root_dir, writer, lst_images[i]))
            cv2.imwrite(os.path.join(test_data_dir, lst_images[i]) , img)