import cv2
import numpy as np
def brightness(image):
	value = 100
	img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	h,s,v = cv2.split(img_hsv)
	
	v_dark = np.where(v <= 0 + value, 0, v - value)

	v_bright = np.where(v <= 255 - value, v + value, 255)

	img_dark = cv2.merge((h,s,v_dark))
	img_bright = cv2.merge((h,s,v_bright))
	img_dark = cv2.cvtColor(img_dark, cv2.COLOR_HSV2RGB)
	img_bright = cv2.cvtColor(img_bright, cv2.COLOR_HSV2RGB)

	cv2.imwrite("brighter.png", img_bright)
	cv2.imwrite("darker.png", img_dark)

img = cv2.imread("data/1_image0.png")
print img.shape
brightness(img)