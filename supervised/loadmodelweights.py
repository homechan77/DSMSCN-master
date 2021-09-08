import tensorflow as tf
from tensorflow import keras
import argparse
import cv2 as cv
import os
import pickle
import numpy as np
from keras.optimizers import Adam


from seg_model.U_net.FC_Siam_Diff import get_FCSD_model
from net_util import weight_binary_cross_entropy
from acc_util import Recall, Precision, F1_score
from acc_ass import accuracy_assessment
from make_test_pickle import read_data_test

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='./DSMSCN-master/supervised/data', help='data path')
parser.add_argument('--data_set_name', default='ACD/Szada/load_weights_test', help='dataset name')
parser.add_argument('--new_result_path', default='load_weights_test', help='making new result path')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate[default: 3e-4]')
parser.add_argument('--result_save_path', default='./DSMSCN-master/supervised/result', help='result path')
parser.add_argument('--weight_save_path', default='./DSMSCN-master/supervised/model_param', help='weight path')

FLAGS = parser.parse_args()

DATA_PATH = FLAGS.data_path
DATA_SET_NAME = FLAGS.data_set_name
NEW_RESULT_PATH = FLAGS.new_result_path
LEARNING_RATE = FLAGS.learning_rate
RESULT_SAVE_PATH = FLAGS.result_save_path
WEIGHT_SAVE_PATH = FLAGS.weight_save_path
##--------------------------------------------------------------------------##
#best_weights_model
weight_lis = os.path.join(WEIGHT_SAVE_PATH, ) #result 폴더 경로 출력
result_listdir = os.listdir(weight_lis) #result 폴더 내 모든 파일들을 리스트로 리턴 

bmplist = []
for i in result_listdir:
    if i[-7:] == 'bcm.bmp':
        bmplist.append(int(i[:-8]))
best_result = str(max(bmplist))
best_result2 = best_result + '_bcm.bmp'
best_result_path = os.path.join(weight_lis, best_result2)
##--------------------------------------------------------------------------##



# 테스트 데이터셋 불러오기
def get_data():
    te = read_data_test()

    #path정의('data\ACD/Szada')
    dataset_path = os.path.join(DATA_PATH, DATA_SET_NAME)

    #X,Y,label로의 데이터 분류
    test_X, test_Y, test_label = load_test_data(dataset_path)

    #정규화(test)
    test_X = np.array(test_X) / 255.
    test_Y = np.array(test_Y) / 255.
    test_label = np.array(test_label) / 255.

    return test_X, test_Y, test_label

def load_test_data(dataset_path):
    with open(os.path.join(dataset_path, 'test_sample_1.pickle'), 'rb') as file:
        test_X = pickle.load(file)
    with open(os.path.join(dataset_path, 'test_sample_2.pickle'), 'rb') as file:
        test_Y = pickle.load(file)
    with open(os.path.join(dataset_path, 'test_label.pickle'), 'rb') as file:
        test_label = pickle.load(file)

    return test_X, test_Y, test_label

test_X, test_Y, test_label = get_data()


def load_weights_test():

    result_path = os.path.join(RESULT_SAVE_PATH, DATA_SET_NAME, NEW_RESULT_PATH)
    os.makedirs(result_path, exist_ok=True)

    # model 불러오기
    deep_model = get_FCSD_model(input_size=[None, None, 3])

    # 저장한 model_weights 파일 불러오기
    
    deep_model.load_weights('C:/Users/Ko/Documents/Ko/Deeplearning_Change_detection/DSMSCN-master/supervised/model_param/ACD/Szada/166_model.h5')

    # 모델 컴파일 
    opt = Adam(lr=LEARNING_RATE)
    deep_model.compile(optimizer=opt, 
    loss=weight_binary_cross_entropy, metrics=['accuracy', Recall, Precision, F1_score])


    # evaluate
    loss, acc, sen, spe, F1 = deep_model.evaluate(x=[test_X, test_Y], y=test_label, batch_size=1)

    # predict
    binary_change_map = deep_model.predict([test_X, test_Y])
    
    # predict를 통한 결과를 
    binary_change_map = np.reshape(binary_change_map, (binary_change_map.shape[1], binary_change_map.shape[2]))

    idx_1 = binary_change_map > 0.5
    idx_2 = binary_change_map <= 0.5
    binary_change_map[idx_1] = 255
    binary_change_map[idx_2] = 0

    conf_mat, overall_acc, kappa = accuracy_assessment(
        gt_changed=np.reshape(255 * test_label, (test_label.shape[1], test_label.shape[2])),
        gt_unchanged=np.reshape(255. - 255 * test_label, (test_label.shape[1], test_label.shape[2])),
        changed_map=binary_change_map)


    info = 'test loss is %.4f,  test sen is %.4f, test spe is %.4f, test F1 is %.4f, test acc is %.4f, ' \
        'test kappa is %.4f, ' % (loss, sen, spe, F1, overall_acc, kappa)
    
    print(info)
    print('confusion matrix is ', conf_mat)

    print(result_path)
    cv.imwrite(os.path.join(result_path, 'load_weights_test_result.bmp'), binary_change_map)

if __name__ == '__main__':
    load_weights_test()