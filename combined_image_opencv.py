import cv2
import os
import datetime
import numpy as np

alpha = 0.5
dim_x2 =(1568, 896)


path = os.path.dirname(os.path.realpath(__file__)) #현재 파일의 디렉토리 출력

##--------------------------------------------------------------------------##
path_test = os.path.join(path, "supervised/data/ACD/Szada/test") #test 폴더 경로 출력
test_listdir = os.listdir(path_test) #['1_gt.bmp', '1_im1.bmp', '1_im2.bmp']

#before
before = test_listdir[1]
before_path = os.path.join(path_test, before)
img1 = cv2.imread(before_path)
img1 = cv2.resize(img1, dim_x2)

#after
after = test_listdir[2]
after_path = os.path.join(path_test, after)
img2 = cv2.imread(after_path)
img2 = cv2.resize(img2, dim_x2)

#ground_truth
gt = test_listdir[0]
gt_path = os.path.join(path_test, gt)
img_gt = cv2.imread(gt_path)
img_gt = cv2.resize(img_gt, dim_x2)


##--------------------------------------------------------------------------##
#best_result
##경로 탐색
path_result = os.path.join(path, "supervised/result/ACD/Szada") #result 폴더 경로 출력
result_listdir = os.listdir(path_result) #result 폴더 내 모든 파일들을 리스트로 리턴 


bmplist = []
for i in result_listdir:
    if i[-7:] == 'bcm.bmp':
        bmplist.append(int(i[:-8]))
best_result = str(max(bmplist))
best_result2 = best_result + '_bcm.bmp'
best_result_path = os.path.join(path_result, best_result2)

##best_result 불러오기
img_result = cv2.imread(best_result_path)
img_result = cv2.resize(img_result, dim_x2)


##--------------------------------------------------------------------------##
#after 영상에 ground_truth 영상을 입히기
dst_gt = cv2.addWeighted(img2, alpha, img_gt, (1-alpha), 0)
#after 영상에 변화 예측 결과 영상을 입히기
dst_result = cv2.addWeighted(img2, alpha, img_result, (1-alpha), 0)


##--------------------------------------------------------------------------##
#편집된 영상을 창으로 띄워주기
cv2.imshow('dst_gt', dst_gt)
cv2.imshow('dst_result', dst_result)
cv2.imshow('img1', img1)


##--------------------------------------------------------------------------##
#편집 영상들을 저장

a = str(datetime.datetime.now())
md = a[:10]
path_opencv = os.path.join(path, "supervised/result/ACD/Szada/opencv") #opencv 폴더 경로 출력
opencv_listdir = os.listdir(path_opencv)
for i in opencv_listdir:
    if i != md:
        oprepath = os.path.join(path_opencv, md)
        os.makedirs(oprepath, exist_ok=True)

cv2.imwrite(str(oprepath)+'/dst_gt.jpg', dst_gt)
cv2.imwrite(str(oprepath)+'/dst_result.jpg', dst_result)
cv2.imwrite(str(oprepath)+'/img1.jpg', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
