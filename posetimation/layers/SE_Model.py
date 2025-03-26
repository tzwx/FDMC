#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   SE_Block.py
@Contact :   fryang@163.com
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/19 20:29   SangsHT      1.0         None
"""

import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


# SE ResNet50

class SEAttention(nn.Module):
    def __init__(self, channel, expand=2):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # out:1*1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * expand, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * expand, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # out:1*1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SeqAttentionNet(nn.Module):
    """
    改造自SENet, 通道权重
    SANet-18: [2, 2, 2, 2]
    SANet-34: [3, 4, 6, 3]
    SANet-101: [3, 4, 23, 3]
    SANet-152: [3, 8, 36, 3]
    """

    def __init__(self, in_planes, block, layers, out_channels=256):
        self.inplanes = in_planes
        super(SeqAttentionNet, self).__init__()
        self.layer1 = self._make_layer(block, 48, layers[0])
        self.layer2 = self._make_layer(block, 96, layers[1])
        self.layer3 = self._make_layer(block, 192, layers[2])
        self.layer4 = self._make_layer(block, 384, layers[3])
        self.final_conv = nn.Conv2d(384, out_channels, 3, 1, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_conv(x)

        return x


if __name__ == '__main__':
    import torch

    x = torch.randn(6, 48, 96, 72)
    x1 = torch.randn(6, 96, 48, 36)
    x2 = torch.randn(6, 192, 24, 18)
    net = SeqAttentionNet(48, SEBasicBlock, [2, 2, 2, 2])
    net_1 = SeqAttentionNet(96, SEBasicBlock, [2, 2, 2, 2])
    net_2 = SeqAttentionNet(192, SEBasicBlock, [2, 2, 2, 2])
    y = net(x)
    y1 = net_1(x1)
    y2 = net_2(x2)
    print(y.shape)
