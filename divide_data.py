import os
import random
import cv2
root_dir = 'all_writers'
training_data_dir = 'training_data'
test_data_dir = 'testing_data'

mx_num = 657
ls = os.listdir(root_dir)
random.shuffle(ls)
for writer in ls:
    if mx_num <= 0:
        break
    lst_images = os.listdir(os.path.join(root_dir, writer))

    mx_num -= 1
    random.shuffle(lst_images)
    mid = (len(lst_images)+1) // 2
    for i in range(mid):
        img = cv2.imread(os.path.join(root_dir, writer, lst_images[i]))
        cv2.imwrite(os.path.join(training_data_dir, lst_images[i]) , img)
    for i in range(mid, len(lst_images) , 1):
        img = cv2.imread(os.path.join(root_dir, writer, lst_images[i]))
        cv2.imwrite(os.path.join(test_data_dir, lst_images[i]) , img)