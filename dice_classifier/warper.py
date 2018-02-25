from skimage import color, transform, io, filters
import skimage
import os
import random

image_dir = "./data/"
imgs = filter((lambda s: '_image' in s) , os.listdir(image_dir))

def color1(image):
	pass

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
	functions = [blur, swirl]

	for function in functions:
		if random.random() > 0.5:
			image = function(image)
	#print image
	try:
		io.imsave(os.path.join(image_dir,img), skimage.img_as_float64(image))
	except:
		fails = fails + 1
	i = i + 1
	if i % 100 == 0:
		print i

print "Fails: ",fails

