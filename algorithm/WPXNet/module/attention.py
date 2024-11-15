import torch
from torch import nn
import torch.nn.functional as F


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x

        return res


class Enhancer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Enhancer, self).__init__()

        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.tanh = nn.Tanh()

        self.refine1 = nn.Conv2d(in_channels, 20, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)

        self.refine3 = nn.Conv2d(20 + 4, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.functional.interpolate

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()

        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze
