import torch
from dice_dataset import DiceDataset
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import random

class Net(nn.Module):
    def __init__(self):
    	self.network_width = 10
        self.mystery_1 = 18
        self.mystery_2 = 20
        self.mystery_3 = 20
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, self.network_width, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.network_width, self.mystery_1, 5)
        self.fc1 = nn.Linear(self.mystery_1 * self.mystery_2 * self.mystery_3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.mystery_1 * self.mystery_2 * self.mystery_3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

testset = DiceDataset("./data/", False, 140, train_percent=0.75)

model = torch.load('classifier.pt')
print "Len: ", len(testset)

start = time.time()
for i in range(0, 10):
    r1 = random.randint(0, len(testset) - 1)
    r2 = random.randint(0, len(testset) - 1)
    r3 = random.randint(0, len(testset) - 1)
    r4 = random.randint(0, len(testset) - 1)
    samples = [r1, r2, r3, r4]
    print r1, r2, r3, r4, len(testset) - 1
    t = torch.cat((testset[r1][0].unsqueeze(0), testset[r2][0].unsqueeze(0), testset[r3][0].unsqueeze(0), testset[r4][0].unsqueeze(0)), dim=0)

    soft = nn.Softmax(1)
    optim = Variable(t)
    outputs = model(optim)
    print "Testing..."
    _, predicted = torch.max(outputs.data, 1)

    print outputs.data
    print soft(Variable(outputs.data))
    print outputs.data[1]

    softed = soft(Variable(outputs.data))
    print "\n\n"
    for i in range(0,4):
    	print "Confidence ", softed[i][predicted[i]]
    	print "Prediction: ", predicted[i] + 1
        print "Actual: ", testset[samples[i]][2]
    	print "\n\n"
    #end = time.clock()

end = time.time()
print "Completed in : ", end - start, " seconds"
