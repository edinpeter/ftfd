from skimage import color, transform, io, filters, util
import skimage
import os
import random
import numpy as np
from threading import Thread, Lock
import time

image_dir = "/home/peter/Desktop/ftfd/dice_classifier/data/"
imgs = filter((lambda s: '_image' in s) , os.listdir(image_dir))
threads = True

SWIRL_STRENGTH_MAX = 10
SWIRL_RADIUS_MIN = 100
SWIRL_RADIUS_MAX = 400
SWIRL_CENTERED = False
BLUR_ITERATIONS_MAX = 10
DISTORTION_OPERATION_PROBABILITY = 0.5
MAX_THREADS = 20

assert DISTORTION_OPERATION_PROBABILITY >= 0 and DISTORTION_OPERATION_PROBABILITY <= 1 
assert SWIRL_RADIUS_MAX > SWIRL_RADIUS_MIN

def color1(image):
	alpha = random.random()
	img = image
	rows, cols, channels = img.shape
	img = color.rgb2grey(img)
	color_mask = np.zeros((rows, cols, 3))
	color_mask[0:rows-1, 0:cols-1] = [random.random(), random.random(), random.random()]  # Red block
	# Construct RGB version of grey-level image
	img_color = np.dstack((img, img, img))
	img_hsv = color.rgb2hsv(img_color)

	color_mask_hsv = color.rgb2hsv(color_mask)

	img_hsv[..., 0] = color_mask_hsv[..., 0]
	img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

	img_masked = color.hsv2rgb(img_hsv)
	return img_masked

def blur(image):
	image2 = image
	for i in range(random.randint(BLUR_ITERATIONS_MAX,BLUR_ITERATIONS_MAX)):
		modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
		mode=modes[random.randint(0, len(modes) - 1)]
		#random modes make it break =(
		image2 = filters.gaussian(image, sigma=1.9, mode='constant', preserve_range=False, multichannel=True)
	return image2

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

def noise(image):
	types= ['gaussian','localvar','poisson','s&p','speckle']
	return util.random_noise(image, mode=types[random.randint(0, len(types) - 1)])


def distort(image_name, image_dir, results, in_lock=None, out_lock=None):

	try:
		if in_lock:
			in_lock.acquire()
			image = io.imread(os.path.join(image_dir,image_name))
			in_lock.release()
		else:
			image = io.imread(os.path.join(image_dir,image_name))

	except:
		in_lock.release()
		print "Failed to open ", image_name
		results.append(True)
		return

	functions = [blur]

	for function in functions:
		if random.random() > DISTORTION_OPERATION_PROBABILITY:
			image = function(image)
	try:
		if out_lock:
			out_lock.acquire()
			io.imsave(os.path.join(image_dir,image_name), skimage.img_as_float(image))
			out_lock.release()
		results.append(False)
	except:
		out_lock.release()
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
