import cv2
import os

normal_dir = './data/'
imgs = os.listdir(normal_dir)
dest_dir = './data_gray/'

i = 0
for img in imgs:
	image = cv2.imread(normal_dir + img, 1)
	#print image, image.shape, img
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(dest_dir + img[0] + '_image' + str(i) + '_gray.png', image)
	if i % 100 == 0:
		print i
	i += 1
