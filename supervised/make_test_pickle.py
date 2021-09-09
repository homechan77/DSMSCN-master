import os
import cv2 as cv
import pickle
import numpy as np

def read_data_test():
    path = './data/ACD/Szada/testset'
    test_img_1 = []
    test_img_2 = []
    test_label = []
    file_names = sorted(os.listdir(path))
    for file_name in file_names:
        if file_name[-4:].upper() == '.BMP':
            img = cv.imread(os.path.join(path, file_name))
            '''
            if img.shape[0] > img.shape[1]:
                img = img[0:784, :, :]
            elif img.shape[0] < img.shape[1]:
                img = img[:, 0:784, :]
            '''
            if 'gt.bmp' in file_name.lower():
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                test_label.append(img)
            elif 'im1.bmp' in file_name.lower():
                test_img_1.append(img)
            elif 'im2.bmp' in file_name.lower():
                test_img_2.append(img)
    with open(os.path.join(path, 'test_sample_1.pickle'), 'wb') as file:
        pickle.dump(test_img_1, file)
    with open(os.path.join(path, 'test_sample_2.pickle'), 'wb') as file:
        pickle.dump(test_img_2, file)
    with open(os.path.join(path, 'test_label.pickle'), 'wb') as file:
        pickle.dump(test_label, file)