import argparse
import cv2 as cv
import os
import pickle

import numpy as np
from keras.optimizers import Adam
from keras.utils import plot_model

#패키지 모듈 불러오기
from acc_util import Recall, Precision, F1_score
from seg_model.U_net.FC_Siam_Diff import get_FCSD_model
from seg_model.U_net.FC_Siam_Conc import get_FCSC_model
from seg_model.U_net.FC_EF import get_FCEF_model
from acc_ass import accuracy_assessment
from net_util import weight_binary_cross_entropy
from scipy import misc
from get_sample import read_data_train, read_data_test

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=200, help='epoch to run[default: 200]')
parser.add_argument('--batch_size', type=int, default=1, help='batch size during training[default: 512]')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate[default: 3e-4]')
parser.add_argument('--result_save_path', default='./result', help='model param path')
parser.add_argument('--model_save_path', default='./model_param', help='model param path')
parser.add_argument('--data_path', default='data', help='data path')
parser.add_argument('--data_set_name', default='ACD/Szada', help='dataset name')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')

# basic params
FLAGS = parser.parse_args()

BATCH_SZ = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
RESULT_SAVE_PATH = FLAGS.result_save_path
MODEL_SAVE_PATH = FLAGS.model_save_path
DATA_PATH = FLAGS.data_path
DATA_SET_NAME = FLAGS.data_set_name
GPU_NUM = FLAGS.gpu_num
BATCH_PER_GPU = BATCH_SZ // GPU_NUM


#모델 학습
def train_model():
    result_path = os.path.join(RESULT_SAVE_PATH, DATA_SET_NAME) #결과 저장 경로 설정
    model_save_path = os.path.join(MODEL_SAVE_PATH, DATA_SET_NAME) #모델 저장 경로 설정
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_X, train_Y, train_label, test_X, test_Y, test_label = get_data() #학습, 테스트 데이터 불러오기
    # test = np.concatenate([test_X, test_Y], axis=-1)  # For FC-EF
    # deep_model = get_FCEF_model(input_size=[None, None, 6])  # For FC-EF

    # deep_model = get_FCSC_model(input_size=[None, None, 3])  # For FC_Siam_Diff and FC_Siam_Conc
    
    #알고리즘 모델(FCSD) 객체 지정 
    deep_model = get_FCSD_model(input_size=[None, None, 3])

    #모델 컴파일
    opt = Adam(lr=LEARNING_RATE)
    deep_model.compile(optimizer=opt,
                       loss=weight_binary_cross_entropy, metrics=['accuracy', Recall, Precision, F1_score])
    
    deep_model.summary()
    #deep_model.summary() 저장
    with open(os.path.join(result_path, 'model_summary.txt'), 'w') as f:
        deep_model.summary(print_fn = lambda x: f.write(x + '\n'))

    #plot_model(deep_model, 'deep_model.png', show_shapes=True)

    #초기값 설정
    best_acc = 0
    best_loss = 1000
    best_kappa = 0
    best_F1 = 0
    save_best = True


    """
    #fit으로 학습
    for i in range(len(train_X)):
        train_1 = np.reshape(train_X[i], (1, train_X[i].shape[0], train_X[i].shape[1], train_X[i].shape[2]))
        train_2 = np.reshape(train_Y[i], (1, train_Y[i].shape[0], train_Y[i].shape[1], train_Y[i].shape[2]))
        label = np.reshape(train_label[i], (1, train_label[i].shape[0], train_label[i].shape[1]))
        deep_model.fit(x=[train_1, train_2], y=label, epochs=MAX_EPOCH, batch_size=BATCH_SZ)
    """

    #"sen": sensitivity(민감도), "spe": specificity(특이도)
    #MAX_EPOCH(200)을 기준으로 반복적으로 학습
    for _epoch in range(MAX_EPOCH):
        train_loss, train_acc, train_sen, train_spe, train_F1 = 0, 0, 0, 0, 0
        for i in range(len(train_X)):
            #train 4d(1,640,784,3)로의 reshape
            train_1 = np.reshape(train_X[i], (1, train_X[i].shape[0], train_X[i].shape[1], train_X[i].shape[2]))
            train_2 = np.reshape(train_Y[i], (1, train_Y[i].shape[0], train_Y[i].shape[1], train_Y[i].shape[2]))
            #train = np.concatenate([train_1, train_2], axis=-1)  # For FC-EF
            label = np.reshape(train_label[i], (1, train_label[i].shape[0], train_label[i].shape[1]))

            
            #'fit'의 경우 epochs, batch_size를 한번에 넘겨주는데 반해, 'train_on_batch'의 경우는 현재 전달받은 데이터를 모두 활용해서 gradient vector를 계산해서 업데이트 합니다.
            temp_loss, temp_train_acc, temp_train_sen, temp_train_spe, temp_train_F1 = deep_model.train_on_batch(
                x=[train_1, train_2], y=label)  # For FC-Siam-Diff and Conc


            # temp_loss, temp_train_acc, temp_train_sen, temp_train_spe, temp_train_F1 = deep_model.train_on_batch(
            #     x=train, y=label) # For FC-EF
            train_loss += temp_loss
            train_acc += temp_train_acc
            train_sen += temp_train_sen
            train_spe += temp_train_spe
            train_F1 += temp_train_F1

        
        #1개의 데이터마다의 학습과 
        train_loss /= len(train_X)
        train_acc /= len(train_X)
        train_sen /= len(train_X)
        train_spe /= len(train_X)
        train_F1 /= len(train_X)

        train_info = 'epoch %d, train loss is %.4f,  train sen is %.4f, train spe is %.4f, train F1 is %.4f, ' \
                     'train acc is %.4f, ' % (_epoch, train_loss, train_sen, train_spe, train_F1, train_acc)        
        print(train_info)

        with open(os.path.join(result_path, 'test_info.txt'), 'w') as f:
            f.write(train_info + '\n') 

    
        loss, acc, sen, spe, F1 = deep_model.evaluate(x=[test_X, test_Y], y=test_label, batch_size=1)
        # loss, acc, sen, spe, F1 = deep_model.evaluate(x=test, y=test_label, batch_size=1) # FC-EF

        binary_change_map = deep_model.predict([test_X, test_Y])
        #  binary_change_map = deep_model.predict(test)
        
        binary_change_map = np.reshape(binary_change_map, (binary_change_map.shape[1], binary_change_map.shape[2]))
        idx_1 = binary_change_map > 0.5
        idx_2 = binary_change_map <= 0.5
        binary_change_map[idx_1] = 255
        binary_change_map[idx_2] = 0

        conf_mat, overall_acc, kappa = accuracy_assessment(
            gt_changed=np.reshape(255 * test_label, (test_label.shape[1], test_label.shape[2])),
            gt_unchanged=np.reshape(255. - 255 * test_label, (test_label.shape[1], test_label.shape[2])),
            changed_map=binary_change_map)

        info = 'epoch %d, test loss is %.4f,  test sen is %.4f, test spe is %.4f, test F1 is %.4f, test acc is %.4f, ' \
               'test kappa is %.4f, ' % (_epoch, loss, sen, spe, F1, overall_acc, kappa)
        print(info)
        print('confusion matrix is ', conf_mat)

        if best_acc < overall_acc:
            best_acc = overall_acc
            save_best = True
        if loss < best_loss:
            best_loss = loss
            save_best = True
        if kappa > best_kappa:
            best_kappa = kappa
            save_best = True
        if F1 > best_F1:
            best_F1 = F1
            save_best = True
        with open(os.path.join(result_path, 'log_all.txt'), 'a+') as f:
            f.write(info + '\n')
        
        if save_best:
            with open(os.path.join(result_path, 'log_best.txt'), 'a+') as f:
                f.write(info + '\n')
            cv.imwrite(os.path.join(result_path, str(_epoch) + '_bcm.bmp'), binary_change_map)
            deep_model.save(os.path.join(model_save_path, str(_epoch) + '_model.h5'))
            save_best = False

    best_info = 'best loss is %.4f, best F1 is %.4f, best acc is %.4f, best kappa is %.4f, ' % (
        best_loss, best_F1, best_acc, best_kappa)
    print('train is done, ' + best_info)
    param_info = 'learning rate ' + str(LEARNING_RATE) + ', Max Epoch is ' + str(MAX_EPOCH)
    print('parameter is: ' + param_info)
    with open(os.path.join(result_path, 'log_best.txt'), 'a+') as f:
        f.write(best_info + '\n')
        f.write(param_info + '\n')


def get_data():
    #데이터 불러오기
    tr = read_data_train()
    te = read_data_test()
    #path정의('data\ACD/Szada')
    path = os.path.join(DATA_PATH, DATA_SET_NAME)
    #X,Y,label로의 데이터 분류
    train_X, train_Y, train_label = load_train_data(path=path)
    test_X, test_Y, test_label = load_test_data(path=path)

    #정규화(test)
    test_X = np.array(test_X) / 255.
    test_Y = np.array(test_Y) / 255.
    test_label = np.array(test_label) / 255.
    #test_label = np.reshape(test_label, (test_label.shape[0], test_label.shape[1], test_label.shape[2]))
    
    #정규화(train)
    for i in range(len(train_X)):
        train_X[i] = train_X[i] / 255.
        train_Y[i] = train_Y[i] / 255.
        train_label[i] = train_label[i] / 255.

    return train_X, train_Y, train_label, test_X, test_Y, test_label

    
def load_train_data(path):
    with open(os.path.join(path, 'train_sample_1.pickle'), 'rb') as file:
        train_X = pickle.load(file)
    with open(os.path.join(path, 'train_sample_2.pickle'), 'rb') as file:
        train_Y = pickle.load(file)
    with open(os.path.join(path, 'train_label.pickle'), 'rb') as file:
        train_label = pickle.load(file)

    return train_X, train_Y, train_label


def load_test_data(path):
    with open(os.path.join(path, 'test_sample_1.pickle'), 'rb') as file:
        test_X = pickle.load(file)
    with open(os.path.join(path, 'test_sample_2.pickle'), 'rb') as file:
        test_Y = pickle.load(file)
    with open(os.path.join(path, 'test_label.pickle'), 'rb') as file:
        test_label = pickle.load(file)

    return test_X, test_Y, test_label


if __name__ == '__main__':
    train_model()