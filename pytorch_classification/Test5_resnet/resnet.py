import torch
from torch import nn


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, input_c, out_c, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_c, out_c, 3, stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_c)

        self.conv2 = torch.nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_c)
        self.relu = torch.nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity

        out = self.relu(out)

        return out


# resnet50, resnet101, resnet152的残差模块：
#   conv1x1: in_channel->out_channel
#   conv3x3: out_channel->out_channel, stride=2/1
#   conv1x1: out_channel->out_channel*4
#   downsample: 跳层连接时的下采样。每个layer的第一个残差块都要下采样。
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_c)

        self.conv2 = torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_c)

        self.conv3 = torch.nn.Conv2d(out_c, out_c*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_c*self.expansion)

        self.relu = torch.nn.ReLU(True)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out
