from skimage import color, transform, io, filters
import skimage
import os
import random
import numpy as np

image_dir = "./data/"
imgs = filter((lambda s: '_image' in s) , os.listdir(image_dir))

def color1(image):
	alpha = 0.6
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
	modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
	#mode=modes[random.randint(0, len(modes) - 1)]
	image2 = filters.gaussian(image, sigma=random.random(), mode='reflect', multichannel=True)
	return image2
def swirl(image):
	strength = 2
	radius = random.randint(100,400)
	image2 = transform.swirl(image, strength=strength, radius=radius)
	return image2

i = 0
fails = 0
for img in imgs:
	image = io.imread(os.path.join(image_dir,img))
	functions = [blur, swirl, color1]

	for function in functions:
		if random.random() > 0.5:
			image = function(image)
	#print image
	try:
		io.imsave(os.path.join(image_dir,img), skimage.img_as_float(image))
	except:
		fails = fails + 1
	i = i + 1
	if i % 100 == 0:
		print i

print "Fails: ",fails

