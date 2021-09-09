import tensorflow as tf
from tensorflow import keras
import argparse
import cv2 as cv
import os
import pickle
import numpy as np
from keras.optimizers import Adam

# 기본 경로 설정
os.chdir('./DSMSCN-master/supervised')

# 모듈 불러오기
from seg_model.U_net.FC_Siam_Diff import get_FCSD_model
from net_util import weight_binary_cross_entropy
from acc_util import Recall, Precision, F1_score
from acc_ass import accuracy_assessment
from make_test_pickle import read_data_test

# 디렉토리 경로 및 learning_rate 설정
parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='./data', help='data path')
parser.add_argument('--result_path', default='./result', help='result path')
parser.add_argument('--model_path', default='./model_param', help='model path')

parser.add_argument('--data_set_name', default='ACD/Szada', help='dataset name')

parser.add_argument('--testset', default='testset', help='testset')
parser.add_argument('--save_result', default='save_result', help='save result')

parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate[default: 3e-4]')

FLAGS = parser.parse_args()

DATA_PATH = FLAGS.data_path
RESULT_PATH = FLAGS.result_path
MODEL_PATH = FLAGS.model_path

DATA_SET_NAME = FLAGS.data_set_name

TESTSET = FLAGS.testset
SAVE_RESULT = FLAGS.save_result

LEARNING_RATE = FLAGS.learning_rate

#best_weights_model_path 탐색 및 설정
weight_list = os.path.join(MODEL_PATH, DATA_SET_NAME)
weight_listdir = os.listdir(weight_list) 

h5list = []
for i in weight_listdir:
    if i[-8:] == 'model.h5':
        h5list.append(int(i[:-9]))
best_weight = str(max(h5list))
best_weight2 = best_weight + '_model.h5'
best_weight_path = os.path.join(weight_list, best_weight2)


# 테스트 데이터셋 불러오기
def get_data():
    #테스트 데이터셋 피클 파일로 만들기
    te = read_data_test()

    #path정의('./data\ACD/Szada\testset')
    dataset_path = os.path.join(DATA_PATH, DATA_SET_NAME, TESTSET)

    #X,Y,label로의 데이터 분류
    test_X, test_Y, test_label = load_test_data(dataset_path)

    #정규화
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

# 불러온 테스트 데이터셋을 최적의 가중치 모델을 가지고 테스트 진행
def load_weights_test():
    # 테스트 결과 저장 경로 지정
    resultsavepath = os.path.join(RESULT_PATH, DATA_SET_NAME, SAVE_RESULT)
    os.makedirs(resultsavepath, exist_ok=True)

    # model 불러오기
    deep_model = get_FCSD_model(input_size=[None, None, 3])

    # 최적의 가중치 모델 파일 불러오기
    deep_model.load_weights(best_weight_path)

    # 모델 컴파일 
    opt = Adam(lr=LEARNING_RATE)
    deep_model.compile(optimizer=opt, 
    loss=weight_binary_cross_entropy, metrics=['accuracy', Recall, Precision, F1_score])

    # evaluate
    loss, acc, sen, spe, F1 = deep_model.evaluate(x=[test_X, test_Y], y=test_label, batch_size=1)

    # predict
    binary_change_map = deep_model.predict([test_X, test_Y])
    
    # predict를 통해 생성된 예측 확률값의 binary_change_map을 2차원 배열로 변경
    binary_change_map = np.reshape(binary_change_map, (binary_change_map.shape[1], binary_change_map.shape[2]))

    # 예측 확률이 50%를 초과하는 것은 255로 그 이하는 0으로 값을 치환
    idx_1 = binary_change_map > 0.5
    idx_2 = binary_change_map <= 0.5
    binary_change_map[idx_1] = 255
    binary_change_map[idx_2] = 0

    # accuracy_assessment 모듈을 활용한 confusion_matrix 생성 및 정확도, kappa계수 산출
    conf_mat, overall_acc, kappa = accuracy_assessment(
        # test_label 데이터의 1값이 255가 되는 get_changed 객체, 0값이 255가 되는 gest_unchanged 객체
        gt_changed=np.reshape(255 * test_label, (test_label.shape[1], test_label.shape[2])),
        gt_unchanged=np.reshape(255. - 255 * test_label, (test_label.shape[1], test_label.shape[2])),
        changed_map=binary_change_map)

    # 성과분석 결과 출력
    info = 'test loss is %.4f,  test sen is %.4f, test spe is %.4f, test F1 is %.4f, test acc is %.4f, ' \
        'test kappa is %.4f, ' % (loss, sen, spe, F1, overall_acc, kappa)
    
    print(info)
    print('confusion matrix is ', conf_mat)

    # 테스트 결과물 저장
    cv.imwrite(os.path.join(resultsavepath, 'load_weights_test_result.bmp'), binary_change_map)
    with open(os.path.join(resultsavepath, 'log_info.txt'), 'w') as f:
        f.write(info)

if __name__ == '__main__':
    load_weights_test()
