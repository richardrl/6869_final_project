import torch.nn as nn
import torch.functional as F
import torch


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.conv3 = nn.Conv2d(6, 1, 1)
        self.fc_out1 = nn.Linear(26*26, 512)
        self.fc_out2 = nn.Linear(512, 3)

    def forward(self, x):
        # [batch_size, h, w, c] -> [batch_size, c, h, w]
        x = x.permute(0, 3, 1, 2).float()
        batch_dim = x.shape[0]

        # [16, 3, 224, 224] -> [16, 6, 220, 220] -> [16, 6, 110, 110]
        x = self.pool(torch.relu(self.conv1(x)))

        # -> [16, 6, 53, 53]
        x = self.pool(torch.relu(self.conv2(x)))

        # -> [16, 1, 55, 55] -> [16, 26, 26]
        x = self.pool(torch.relu(self.conv3(x)))

        # -> [16, 26*26] -> [16, 512]
        x = self.fc_out1(x.reshape(batch_dim, 26*26))

        # -> [16, 3] position prediction
        x = self.fc_out2(x)
        return x

