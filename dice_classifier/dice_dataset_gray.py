import os
import torch
import random
import torchvision
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform, color
import matplotlib.pyplot as plt
from dice_image_dataset_g import DiceImageDataset
import cv2

class DiceDataset_G(Dataset):
    """ Dice Dataset """

    def __init__(self, dice_dir, train, class_max=200, classes=6, train_percent=0.75, equal_datasets=False, transform=None):       
        d = list(DiceImageDataset(dice_dir, classes, class_max))
        random.shuffle(d)
        self.dice_imgs = list()
        if train:
            for i in range(0, int(train_percent * len(d))):
                self.dice_imgs.append(d[i])
        else:
            for i in range(int(train_percent * len(d)), len(d)):
                self.dice_imgs.append(d[i])


    def __len__(self):
        return len(self.dice_imgs)

    def __getitem__(self, id):
        image = self.dice_imgs[id][0]
        label = self.dice_imgs[id][1]
        filename = self.dice_imgs[id][2]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=0)

        return torch.from_numpy(image).float(), label - 1, filename

if __name__ == "__main__":
    d = DiceDataset("./data/", True, 20)

    print "Len set: ", len(d)
    p = DataLoader(d, batch_size=4, num_workers=2, shuffle=True)
    dataiter = iter(p)
    images, labels, filenames = dataiter.next()
    print labels