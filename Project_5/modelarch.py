import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 28 -> 24 (28-5)/1 + 1 = 24
        self.conv1 = nn.Conv2d(1, 6, 5)
        # batch normalization
        self.batch_norm1 = nn.BatchNorm2d(6)

        # 24 -> 12
        self.pool = nn.MaxPool2d(2, 2)
        # 12 -> 8 (12 - 5)/1 + 1 = 8
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batch_norm2 = nn.BatchNorm2d(16)

        # drop out
        self.drop1 = nn.Dropout(p=0.2)

        # 8 -> 4
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)

        # dropout
        self.drop2 = nn.Dropout(p=0.2)
        
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.drop2(x)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
