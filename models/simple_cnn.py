import torch.nn as nn
import torch.functional as F
import torch


class SimpleCNN(nn.Module):
    def __init__(self, batchnorm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.bn2 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 1, 1)
        self.bn3 = nn.BatchNorm2d(1)
        self.fc_out1 = nn.Linear(26*26, 512)
        self.fc_out2 = nn.Linear(512, 3)
        self.batchnorm = batchnorm

    def forward(self, x):
        batch_dim = x.shape[0]

        # [16, 3, 224, 224] -> [16, 6, 220, 220] -> [16, 6, 110, 110]
        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn1(x)

        x = self.pool(torch.relu(x))

        x=self.conv2(x)

        if self.batchnorm:
            x = self.bn2(x)
        # -> [16, 6, 53, 53]
        x = self.pool(torch.relu(x))

        x=self.conv3(x)

        if self.batchnorm:
            x = self.bn3(x)
        # -> [16, 1, 55, 55] -> [16, 26, 26]
        x = self.pool(torch.relu(x))

        # -> [16, 26*26] -> [16, 512]
        x = self.fc_out1(x.reshape(batch_dim, 26*26))

        # -> [16, 3] position prediction
        x = self.fc_out2(x)
        return x


class SimpleCNN5Conv(nn.Module):
    def __init__(self, batchnorm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.bn2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 24, 3)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 12, 3)
        self.bn4 = nn.BatchNorm2d(12)
        self.conv5 = nn.Conv2d(12, 1, 1)
        self.bn5 = nn.BatchNorm2d(1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_out1 = nn.Linear(25*25, 512)
        self.fc_out2 = nn.Linear(512, 3)
        self.batchnorm = batchnorm

    def forward(self, x):
        batch_dim = x.shape[0]
        # [16, 3, 224, 224] -> [16, 6, 220, 220] -> [16, 6, 110, 110]
        x = self.conv1(x)

        if self.batchnorm:
            x = self.bn1(x)

        x = self.pool(torch.relu(x))

        # -> [16, 6, 53, 53]
        x = self.conv2(x)

        if self.batchnorm:
            x = self.bn2(x)

        x = torch.relu(x)

        # -> [16, 1, 55, 55] -> [16, 26, 26]
        x = self.conv3(x)

        if self.batchnorm:
            x = self.bn3(x)

        x = self.pool(torch.relu(x))

        x = self.conv4(x)
        if self.batchnorm:
            x = self.bn4(x)

        x = torch.relu(x)

        x = self.conv5(x)

        if self.batchnorm:
            x = self.bn5(x)

        x = self.pool(torch.relu(x))

        # -> [16, 26*26] -> [16, 512]
        x = self.fc_out1(x.reshape(batch_dim, 25*25))

        # -> [16, 3] position prediction
        x = self.fc_out2(x)
        return x