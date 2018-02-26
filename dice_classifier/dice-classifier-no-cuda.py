import torch
import torchvision
import torchvision.transforms as transforms
from dice_dataset import DiceDataset
import matplotlib.pyplot as plt

trainset = DiceDataset("./data/", True, 300, train_percent=0.75)

testset = DiceDataset("./data/", False, 300, train_percent=0.75)

print "Train set length: ", len(trainset)
print "Test set length: ", len(testset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=4)

classes = ('1','2','3','4','5','6')

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

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


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels, filenames = data

        # wrap them in Variable
        #print (inputs[0])
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 50 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
images, labels, filenames = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(Variable(images))

_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0

for data in testloader:
    images, labels, filenames = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the %i test images: %d %%' %  (len(testset), (
    100 * correct / total)))

class_correct = list(0. for i in range(6))
class_total = list(0. for i in range(6))
for data in testloader:
    images, labels, filenames = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()

    for i in range(len(labels)):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

print class_correct
print class_total

for i in range(6):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

torch.save(net, 'classifier.pt')
