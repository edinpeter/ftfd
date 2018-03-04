import torch
import torchvision
import torchvision.transforms as transforms
from dice_dataset import DiceDataset
import matplotlib.pyplot as plt

classes = ('1','2','3','4','5','6')

trainset = DiceDataset("./data/", True, classes=len(classes), class_max=100, train_percent=0.75)

testset = DiceDataset("./data/", False, classes=len(classes), class_max=100, train_percent=0.75)

print "Train set length: ", len(trainset)
print "Test set length: ", len(testset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=8)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=8)


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        self.conv1_kernel = 10
        self.conv2_kernel = 5

    	self.network_width = 24 #out channels
        self.conv2_output_channels = 18
        self.mystery_2 = 20
        self.mystery_3 = 20

        self.outputs = len(classes)

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, self.network_width, self.conv1_kernel)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.network_width, self.conv2_output_channels, self.conv2_kernel)
        self.conv3 = nn.Conv2d(self.conv2_output_channels, 20, 5)

        self.fc1 = nn.Linear(20 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 20 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(200):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels, filenames = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        
        print inputs

        optimizer.zero_grad()
        outputs = net(inputs)
        print outputs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 50 == 0:
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
images, labels, filenames = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(Variable(images.cuda()))

_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0

for data in testloader:
    images, labels, filenames = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the %i test images: %d %%' %  (len(testset), (
    100 * correct / total)))

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
for data in testloader:
    images, labels, filenames = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.cuda()).squeeze()

    for i in range(len(labels)):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

print class_correct
print class_total

for i in range(len(classes)):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

torch.save(net, 'classifier_cuda.pt')
