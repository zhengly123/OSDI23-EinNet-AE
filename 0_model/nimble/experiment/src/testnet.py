import torch
import torch.nn as nn
import math


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        # setting of inverted residual blocks

        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1,1,4)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.feat_size = (48//4)*(48//4)*64
        self.fc1 = nn.Linear(self.feat_size, 256)

    def forward(self, x):
        return self.conv2(x)
        x = self.conv(x)
        x = torch.reshape(x, [-1, self.feat_size])
        x = self.fc1(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
