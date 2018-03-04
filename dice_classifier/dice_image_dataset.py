import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform, color
import matplotlib.pyplot as plt

class DiceImageDataset(Dataset):
    """ Dice Dataset """

    def __init__(self, image_dir, classes, class_max=100, transform=None):       
        self.image_dir = image_dir

        self.dice_imgs = list()
        dice_imgs_temp = list()
        for i in range(0, classes):
            self.dice_imgs.append(filter((lambda s: (str(i+1) + '_image') in s) , os.listdir(image_dir)))
            self.dice_imgs[i] = self.dice_imgs[i][0:min(class_max, len(self.dice_imgs[i]))]
            for img in self.dice_imgs[i]:
                dice_imgs_temp.append(img)
        self.dice_imgs = dice_imgs_temp

    def __len__(self):
        return len(self.dice_imgs)

    def __getitem__(self, id):
        img_name = os.path.join(self.image_dir,
                                self.dice_imgs[id])
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

        return image, int(self.dice_imgs[id][0]), self.dice_imgs[id]

    def filename(self, id):
        return self.dice_imgs[id]

if __name__ == "__main__":

    d = DiceImageDataset("/home/peter/Desktop/ftfd/dice_classifier/data/")
    print d[2]
    print 'filename: ', d.filename(2)
    #plt.imshow(d[2][1])
    #plt.show()

