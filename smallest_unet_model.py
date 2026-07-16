import numpy as np
import torch
from torch import nn
from torchsummary import summary

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.conv = nn.ConvTranspose2d(channel, channel // 2, kernel_size=2, stride=2)

    def forward(self, x, feature_map):
        up = self.conv(x)
        return torch.cat((up, feature_map), dim=1)

class smallest_UNet(nn.Module):
    def __init__(self):
        super(smallest_UNet, self).__init__()
        self.c1 = Conv_Block(3, 32)
        self.d1 = DownSample()
        self.c2 = Conv_Block(32, 64)
        self.d2 = DownSample()
        self.c3 = Conv_Block(64, 128)
        self.d3 = DownSample()  # 最后下采样到 64x64，而不是 32x32

        self.c4 = Conv_Block(128, 256)

        self.u1 = UpSample(256)
        self.c5 = Conv_Block(256, 128)
        self.u2 = UpSample(128)
        self.c6 = Conv_Block(128, 64)
        self.u3 = UpSample(64)
        self.c7 = Conv_Block(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)
        self.th = nn.Sigmoid()

    def forward(self, x):
        r1 = self.c1(x)
        r2 = self.c2(self.d1(r1))
        r3 = self.c3(self.d2(r2))
        r4 = self.c4(self.d3(r3))

        o1 = self.c5(self.u1(r4, r3))
        o2 = self.c6(self.u2(o1, r2))
        o3 = self.c7(self.u3(o2, r1))

        return self.th(self.out(o3))

# 初始化模型并迁移到正确的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fire_model = smallest_UNet().to(device)

# 输出模型结构
summary(fire_model, (3, 128, 128))  # 由于减少了一次下采样，输入尺寸可以改小
