import os
import cv2 as cv
import pickle
import numpy as np

os.chdir('C:/Users/Ko/Documents/Ko/Deeplearning_Change_detection/DSMSCN-master/supervised')

def calculate_weight():
    train_path = 'data/ACD/Szada/train'
    test_path = 'data/ACD/Szada/test'
    all_count = 0
    true_count = 0
    for file_name in os.listdir(train_path):
        if file_name[-4:].upper() == '.BMP':
            img = cv.imread(os.path.join(train_path, file_name))
            if img.shape[0] > img.shape[1]:
                img = img[0:944, :, :]
            elif img.shape[0] < img.shape[1]:
                img = img[:, 0:944, :]
            if 'gt.bmp' in file_name.lower():
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                count = np.sum((img == 255))
                true_count += count
                all_count += img.shape[0] * img.shape[1]
    for file_name in os.listdir(test_path):
        if file_name[-4:].upper() == '.BMP':
            img = cv.imread(os.path.join(test_path, file_name))
            if img.shape[0] > img.shape[1]:
                img = img[0:944, :, :]
            elif img.shape[0] < img.shape[1]:
                img = img[:, 0:944, :]
            if 'gt.bmp' in file_name.lower():
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                count = np.sum((img == 255))
                true_count += count
                all_count += img.shape[0] * img.shape[1]
    weight = true_count / (all_count - true_count)
    print(weight)


def read_data_train():
    path = './data/ACD/Szada/train'
    train_img_1 = []
    train_img_2 = []
    train_label = []
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
                train_label.append(img)
            elif 'im1.bmp' in file_name.lower():
                train_img_1.append(img)
            elif 'im2.bmp' in file_name.lower():
                train_img_2.append(img)
    with open('data/ACD/Szada/train_sample_1.pickle', 'wb') as file:
        pickle.dump(train_img_1, file)
    with open('data/ACD/Szada/train_sample_2.pickle', 'wb') as file:
        pickle.dump(train_img_2, file)
    with open('data/ACD/Szada/train_label.pickle', 'wb') as file:
        pickle.dump(train_label, file)
    # return train_img_1, train_img_2, train_label

def read_data_test():
    path = './data/ACD/Szada/test'
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
    with open('data/ACD/Szada/test_sample_1.pickle', 'wb') as file:
        pickle.dump(test_img_1, file)
    with open('data/ACD/Szada/test_sample_2.pickle', 'wb') as file:
        pickle.dump(test_img_2, file)
    with open('data/ACD/Szada/test_label.pickle', 'wb') as file:
        pickle.dump(test_label, file)
        


def crop_img():
    path = 'D:/Workspace/Python/RSExperiment/Adata/ACD/Tiszadob/3'
    for env_file_name in os.listdir(path):
        if env_file_name[-4:].upper() == '.BMP':
            img = cv.imread(os.path.join(path, env_file_name))[0:448, 0:784, :]
            cv.imwrite(os.path.join(path, 'crop_' + env_file_name), img)


def rename_img():
    path = 'D:/Workspace/Python/RSExperiment/Mutil-Temp Conf/train_data/Tisza/rotate_270'
    count = 57
    for env_file_name in os.listdir(path):
        if env_file_name[-4:].upper() == '.BMP':
            if 'gt.bmp' in env_file_name.lower():
                os.rename(os.path.join(path, env_file_name), os.path.join(path, str(count) + '_gt.bmp'))
            elif 'im1.bmp' in env_file_name.lower():
                os.rename(os.path.join(path, env_file_name), os.path.join(path, str(count) + '_im1.bmp'))
            elif 'im2.bmp' in env_file_name.lower():
                os.rename(os.path.join(path, env_file_name), os.path.join(path, str(count) + '_im2.bmp'))
                count += 1


calculate_weight()
