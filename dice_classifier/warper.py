from skimage import color, transform, io, filters, util
import skimage
import os
import random
import numpy as np
from threading import Thread, Lock
import time
import cv2
import warnings

warnings.filterwarnings("ignore")

image_dir = "/home/peter/Desktop/ftfd/dice_classifier/data/"
imgs = filter((lambda s: '7_image' not in s and '_image' in s) , os.listdir(image_dir))
threads = True

SWIRL_STRENGTH_MAX = 2
SWIRL_RADIUS_MIN = 200
SWIRL_RADIUS_MAX = 400
SWIRL_CENTERED = False
BLUR_ITERATIONS_MIN = 0
BLUR_ITERATIONS_MAX = 5
DISTORTION_OPERATION_PROBABILITY = 0.65
MAX_THREADS = 20
MAX_BRIGHTEN_DARKEN = 100
MIN_BRIGHTEN_DARKEN = 0

assert DISTORTION_OPERATION_PROBABILITY >= 0 and DISTORTION_OPERATION_PROBABILITY <= 1 
assert SWIRL_RADIUS_MAX > SWIRL_RADIUS_MIN
assert BLUR_ITERATIONS_MAX > BLUR_ITERATIONS_MIN

def darken(values, val):
	#print values
	return np.where(values <= 0 + val, 0, values - val)


def brighten(values, val):
	return np.where(values <= 255 - val, values + val, 255)


def brightness(image):
	#print image
	#print image.shape
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	op = brighten
	if random.random() > 0.5:
		op = darken
	v = op(v, random.randint(MIN_BRIGHTEN_DARKEN, MAX_BRIGHTEN_DARKEN))
	#print h.shape
	#print s.shape
	#print v.shape
	#print h
	final_hsv = cv2.merge((h, s, v))

	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	#img = img / 255.0
	return img


def color_(image):
	if random.random() > 0.2:
		image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(image_hsv)

		h = np.where(h != 150, random.randint(80, 120), h)
		s = np.where(s < 50, random.randint(120, 175), s)
		v = np.where(v < 50, random.randint(120, 175), v)

		image_hsv = cv2.merge((h,s,v))
		image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
	return image

def blur(image):
	image2 = image
	for i in range(random.randint(BLUR_ITERATIONS_MIN,BLUR_ITERATIONS_MAX)):
		blur_amount = random.randint(3,15)
		image2 = cv2.blur(image2, (blur_amount,blur_amount))
	return image2

def noise(image):
	types= ['gaussian','localvar','poisson','s&p','speckle']
	return util.random_noise(image, mode=types[random.randint(0, len(types) - 1)])

def swirl(image):
	strength = random.randint(0,SWIRL_STRENGTH_MAX)
	radius = random.randint(SWIRL_RADIUS_MIN,SWIRL_RADIUS_MAX)
	rows, cols, channels = image.shape

	image2 = None
	if not SWIRL_CENTERED:
		center = (random.randint(0,rows - 1), random.randint(0, cols - 1))
		image2 = transform.swirl(image, strength=strength, radius=radius, center=center)
	else:
		image2 = transform.swirl(image, strength=strength, radius=radius)

	return image2

def distort(image_name, image_dir, results, in_lock=None, out_lock=None):

	try:
		if in_lock:
			#in_lock.acquire()
			image = cv2.imread(os.path.join(image_dir,image_name), cv2.IMREAD_UNCHANGED)
			#print image
			#print image.shape
			#in_lock.release()
		else:
			image = cv2.imread(os.path.join(image_dir,image_name), cv2.IMREAD_UNCHANGED)

	except:
		#in_lock.release()
		print "Failed to open ", image_name
		results.append(True)
		return

	functions = [blur, brightness, color_]
	#functions = [brightness]
	for function in functions:
		#print "imname", image_name
		image = function(image)
	try:
		cv2.imwrite(os.path.join(image_dir,image_name), image)

		results.append(False)
	except:
		print "Failed to save ", image_name
		results.append(True)



i = 0
fails = 0
threads_list = list()
results = list()
start = time.time()

in_lock = Lock()
out_lock = Lock()

for img in imgs:
	if not threads:
		failure = distort(img, image_dir, results)
		i = i + 1
		if i % 100 == 0:
			print i
	else:
		if len(threads_list) < MAX_THREADS:
			t = Thread(target = distort, args=(img, image_dir, results, in_lock, out_lock,))
			threads_list.append(t)
			t.start()
		else:
			removed = False
			while not removed:
				for t in threads_list:
					if not t.is_alive():
						threads_list.remove(t)
						removed = True
			t = Thread(target = distort, args=(img, image_dir, results, in_lock, out_lock,))
			threads_list.append(t)
			t.start()	
		i = i + 1
		if i % 100 == 0:
			print i

end = time.time()

print "Fails: ",results.count(True)
print "Finished in %4.2f seconds" % (end - start)
