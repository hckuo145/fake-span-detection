import torch
import torch.nn as nn

from typing import Any
from torch  import Tensor


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel:int, reduction:int=16) -> None:
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x:Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.linear(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes:int, planes:int, stride:int=1, downsample:Any=None, reduction:int=16) -> None:
        super(SEBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        
        self.downsample = downsample

    def forward(self, x:Tensor) -> Tensor:
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        
        res = self.conv2(res)
        res = self.bn2(res)
        res = self.se(res)

        if self.downsample is not None:
            x = self.downsample(x)

        x = x + res
        x = self.relu(x)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, dropout=0.):
        super(TransformerLayer, self).__init__()

        self.attn = nn.MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.norm1 = nn.LayerNorm(input_size, eps=1e-9)

        self.linear = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout, inplace=True),
                nn.Linear(hidden_size, input_size)
                )
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.norm2 = nn.LayerNorm(input_size, eps=1e-9)

    def forward(self, x):
        res, _ = self.attn(x, x, x)
        res = self.dropout1(res)
        x = self.norm1(x + res)
        
        res = self.linear(x)
        res = self.dropout2(res)
        x = self.norm2(x + res)

        return x