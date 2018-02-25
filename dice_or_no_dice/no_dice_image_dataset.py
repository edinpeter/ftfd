import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform, color
import matplotlib.pyplot as plt
class NoDiceImageDataset(Dataset):
    """ Dice Dataset """

    def __init__(self, image_dir, transform=None):       
        self.image_dir = image_dir
        self.image_filenames = filter((lambda s: '.thumbnail' in s) , os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, id):
        img_name = os.path.join(self.image_dir,
                                self.image_filenames[id])
        image = io.imread(img_name)

        try:
        	image = color.gray2rgb(image)
        except:
        	pass
        
        try:
        	image = color.rgba2rgb(image)
        except:
        	pass

        image = transform.resize(image, (100,100))

        sample = {'image': image, 'label': 'no-dice', 'filename' : self.image_filenames[id]}

        return sample

if __name__ == "__main__":
	d = NoDiceImageDataset("/home/peter/Desktop/dice/randoms_small/")
	print d[200]
	plt.imshow(d[200]['image'])
	plt.show()