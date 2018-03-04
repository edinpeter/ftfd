import cv2
import os

randoms_dir = '/home/peter/Desktop/ftfd/dice_or_no_dice/randoms_small/'
randoms = os.listdir(randoms_dir)
dest_dir = '/home/peter/Desktop/ftfd/dice_classifier/data/'

i = 0
for rand in randoms:
	img = cv2.imread(randoms_dir + rand, cv2.IMREAD_UNCHANGED)
	cv2.imwrite(dest_dir + '7_image' + str(i) + '.png', img)
	i += 1
