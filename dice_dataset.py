import os
import torch
import random
import torchvision
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform, color
import matplotlib.pyplot as plt
from dice_image_dataset import DiceImageDataset
from no_dice_image_dataset import NoDiceImageDataset


class DiceDataset(Dataset):
    """ Dice Dataset """

    def __init__(self, dice_dir, no_dice_dir, train, no_dice_max=200, train_percent=0.75, equal_datasets=False, transform=None):       
        self.no_dice_dir = no_dice_dir
        self.dice_dir = dice_dir

        self.dice_image_dataset = DiceImageDataset(dice_dir)
        self.no_dice_image_dataset = NoDiceImageDataset(no_dice_dir)
        
        self.dice = list()
        self.no_dice = list()
        if train:
            last_dice_image = int(len(self.dice_image_dataset) * train_percent)
            last_no_dice_image = int(len(self.no_dice_image_dataset) * train_percent)
            last_no_dice_image = min(last_no_dice_image, no_dice_max)

            for i in range(0, last_dice_image):
                self.dice.append(self.dice_image_dataset[i])

            for i in range(0, last_no_dice_image):
                self.no_dice.append(self.no_dice_image_dataset[i])
        else:
            last_dice_image = int(len(self.dice_image_dataset) * (1.0 - train_percent))
            last_no_dice_image = int(len(self.no_dice_image_dataset) * (1.0 - train_percent))
            last_no_dice_image = min(last_no_dice_image, no_dice_max)

            for i in range(0, last_dice_image):
                self.dice.append(self.dice_image_dataset[i])

            for i in range(0, last_no_dice_image):
                self.no_dice.append(self.no_dice_image_dataset[i])
        self.data = self.dice + self.no_dice

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        dice_dataset = bool(random.getrandbits(1))

        image = self.data[id]['image']
        label = 1 if id < len(self.dice) else 0
        filename = self.data[id]['filename']
        image = np.transpose(image, (2,0,1))

        return torch.from_numpy(image).float(), label, filename

    def len_dice(self):
        return len(self.dice)

    def len_no_dice(self):
        return len(self.no_dice)

if __name__ == "__main__":
    d = DiceDataset("/home/peter/Desktop/dice/dice_small", "/home/peter/Desktop/dice/randoms_small/", train=False)

    print "Len set: ", len(d) 
    print "Len dice: ", d.len_dice()
    print "Len nodice: ", d.len_no_dice()
    p = DataLoader(d, batch_size=4, num_workers=2, shuffle=True)

    dataiter = iter(p)
    images, labels, filenames = dataiter.next()
    print images[1]
    #plt.imshow(torchvision.utils.make_grid(images['image']))
    plt.imshow(images[2])

    plt.show()