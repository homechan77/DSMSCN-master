import cv2

image1 = cv2.imread('./data/land/test/0_predict.bmp')
image2 = cv2.imread('./data/land/test/1_predict.bmp')
img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

diff = cv2.subtract(image2, image1)
# diff = cv2.absdiff(image2, image1)

# 차 영상을 극대화 하기 위해 threshold
# _, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)


cv2.imshow('result', diff)
cv2.imwrite('./data/land/test/diff_predict.bmp', diff)
cv2.waitKey()
cv2.destroyAllWindows()