import torch.nn
import torch.nn as nn
import torch.nn.functional as F

# input shape (3,32,32)
# 两次conv5x5，每次conv后接一个maxPooling + 三次fc


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # N = (w - f + 2p)/2 + 1
        self.max_pooling1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.max_pooling2 = torch.nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(in_features=32*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1.forward(x))  # Input(3,32,32). output(16,28,28).
        x = self.max_pooling1(x)           # output(16,14,14)
        x = F.relu(self.conv2.forward(x))  # output(32, 10, 10)
        x = self.max_pooling2(x)           # output(32, 5, 5)

        x = x.view(-1, 32*5*5)  # output(32*5*5)

        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)


input = torch.randn([4, 3, 32, 32])
lenet = LeNet()
lenet.forward(input)


