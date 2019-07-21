''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .res_utils import DownsampleA


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.featureSize = 64

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, num_classes, channels=3, num_of_inc_classes=2, is_model_jm=False):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()

        self.featureSize = 64
        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc2 = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feature=False, T=1, labels=False, scale=None, keep=None, embedding_space=False):

        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x1 = self.stage_1(x)
        x2 = self.stage_2(x1)
        x3 = self.stage_3(x2)
        x = self.avgpool(x3)

        embedded = x.view(x.size(0), -1)

        if feature:
            return embedded / torch.norm(embedded, 2, 1).unsqueeze(1)

        x = self.fc(embedded) / T

        if keep is not None:
            x = x[:, keep[0]:keep[1]]

        if labels:
            return F.softmax(x, dim=1)

        if scale is not None:
            temp = F.softmax(x, dim=1)
            temp = temp * scale
            return temp

        if embedding_space:
            # return F.log_softmax(x, dim=1), x
            # return F.log_softmax(x, dim=1), embedded
            norm_embedded = embedded / torch.norm(embedded, 2, 1).unsqueeze(1)

            a1 = x1
            # a1 = a1.contiguous().view(x1.shape[0], -1)
            # a1 = torch.norm(a1, dim=1)
            # a1 = a1.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            a1 = torch.norm(torch.norm(a1, dim=(2, 3)), dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            norm_x1 = x1.div(a1.expand_as(x1))

            a2 = x2
            # a2 = a2.contiguous().view(x2.shape[0], -1)
            # a2 = torch.norm(a2, dim=1)
            # a2 = a2.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            a2 = torch.norm(torch.norm(a2, dim=(2, 3)), dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            norm_x2 = x2.div(a2.expand_as(x2))

            a3 = x3
            # a3 = a3.contiguous().view(x3.shape[0], -1)
            # a3 = torch.norm(a3, dim=1)
            a3 = torch.norm(torch.norm(a3, dim=(2, 3)), dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            norm_x3 = x3.div(a3.expand_as(x3))

            return F.log_softmax(x, dim=1), norm_embedded, norm_x1, norm_x2, norm_x3

        return F.log_softmax(x, dim=1)

    def forwardFeature(self, x):
        pass


def resnet20(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    return model


def resnet10mnist(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 10, num_classes, 1)
    return model


def resnet20mnist(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes, 1)
    return model


def resnet32mnist(num_classes=10, channels=1):
    model = CifarResNet(ResNetBasicblock, 32, num_classes, channels)
    return model


def resnet32(num_classes=10):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 32, num_classes)
    return model


def resnet44(num_classes=10):
    """Constructs a ResNet-44 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 44, num_classes)
    return model


def resnet56(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 56, num_classes)
    return model


def resnet110(num_classes=10):
    """Constructs a ResNet-110 model for CIFAR-10 (by default)
    Args:
      num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 110, num_classes)
    return model
