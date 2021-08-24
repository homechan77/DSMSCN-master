import cv2
import numpy as np

alpha = 0.5
dim_x2 =(1568, 896)


#before
img1 = cv2.imread('C:/Users/Ko/Documents/Ko/Deeplearning_Change_detection/DSMSCN-master/supervised/data/ACD/Szada/test/1_im1.bmp')
img1 = cv2.resize(img1, dim_x2)

#after
img2 = cv2.imread('C:/Users/Ko/Documents/Ko/Deeplearning_Change_detection/DSMSCN-master/supervised/data/ACD/Szada/test/1_im2.bmp')
img2 = cv2.resize(img2, dim_x2)

#ground_truth
img_gt = cv2.imread('C:/Users/Ko/Documents/Ko/Deeplearning_Change_detection/DSMSCN-master/supervised/data/ACD/Szada/test/1_gt.bmp')
img_gt = cv2.resize(img_gt, dim_x2)

#best_result
img_result = cv2.imread('C:/Users/Ko/Documents/Ko/Deeplearning_Change_detection/DSMSCN-master/supervised/result/ACD/Szada/135_bcm.bmp')
img_result = cv2.resize(img_result, dim_x2)


#after 영상에 ground_truth 영상을 입히기
dst_gt = cv2.addWeighted(img2, alpha, img_gt, (1-alpha), 0)
#after 영상에 변화 예측 결과 영상을 입히기
dst_result = cv2.addWeighted(img2, alpha, img_result, (1-alpha), 0)


#편집된 영상을 창으로 띄워주기
cv2.imshow('dst_gt', dst_gt)
cv2.imshow('dst_result', dst_result)
cv2.imshow('img1', img1)


#편집 영상들을 저장
cv2.imwrite('C:/Users/Ko/Documents/Ko/Deeplearning_Change_detection/DSMSCN-master/supervised/result/ACD/Szada/opencv/dst_gt.jpg', dst_gt)
cv2.imwrite('C:/Users/Ko/Documents/Ko/Deeplearning_Change_detection/DSMSCN-master/supervised/result/ACD/Szada/opencv/dst_result.jpg', dst_result)
cv2.imwrite('C:/Users/Ko/Documents/Ko/Deeplearning_Change_detection/DSMSCN-master/supervised/result/ACD/Szada/opencv/img.jpg', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()