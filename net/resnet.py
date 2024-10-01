
import math, sys, os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torchvision.models import resnet18, resnet34, resnet50, resnet101
import torch.utils.model_zoo as model_zoo

# from .bam import *
# from .cbam import *

class Resnet18(nn.Module):
    def __init__(self, embedding_size, pretrained=True, is_norm=True, bn_freeze=True):
        super(Resnet18, self).__init__()

        self.model = resnet18(pretrained)
        # self.checkpoint = torch.load('./pre/resnet18-f37072fd.pth')
        # self.model.load_state_dict(self.checkpoint, strict=True)

        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    # def forward(self, x, is_pa=True):
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x

        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)

        if self.is_norm:
            x = self.l2_norm(x)

        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)


class Resnet50(nn.Module):
    def __init__(self, embedding_size, num_classes, pretrained=False, is_norm=True, bn_freeze=True):
        super(Resnet50, self).__init__()

        self.model = resnet50(pretrained)
        self.checkpoint = torch.load('./pre/resnet50-0676ba61.pth')
        self.model.load_state_dict(self.checkpoint, strict=True)

        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.num_ftrs = self.model.fc.in_features
        self.expansion = 4
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
    
        if self.is_norm:
            x = self.l2_norm(x)

        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)
