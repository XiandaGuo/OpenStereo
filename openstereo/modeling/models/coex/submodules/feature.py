#!/usr/bin/env python
from __future__ import print_function
import os
import numpy as np
from typing import Callable, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .Submodule import SubModule
from .util_conv import BasicConv, Conv2x, BasicBlock, conv3x3, conv1x1
from .util_conv import BasicConv2d, BasicTransposeConv2d

import timm

import pdb


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class PSMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(PSMBasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        # self.cfg = cfg['backbone']
        self.type = 'mobilenetv2_100'
        chans = [16, 24, 32, 96, 160]

        if not self.type == 'psm':
            self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
            self.deconv16_8 = Conv2x(chans[3] * 2, chans[2], deconv=True, concat=True)
            self.deconv8_4 = Conv2x(chans[2] * 2, chans[1], deconv=True, concat=True)
            self.conv4 = BasicConv(chans[1] * 2, chans[1] * 2, kernel_size=3, stride=1, padding=1)

            self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        if self.type == 'psm':
            return featL, featR

        if featR is not None:
            y4, y8, y16, y32 = featR

            x16 = self.deconv32_16(x32, x16)
            y16 = self.deconv32_16(y32, y16)

            x8 = self.deconv16_8(x16, x8)
            y8 = self.deconv16_8(y16, y8)

            x4 = self.deconv8_4(x8, x4)
            y4 = self.deconv8_4(y8, y4)

            x4 = self.conv4(x4)
            y4 = self.conv4(y4)

            return [x4, x8, x16, x32], [y4, y8, y16, y32]
        else:
            x16 = self.deconv32_16(x32, x16)
            x8 = self.deconv16_8(x16, x8)
            x4 = self.deconv8_4(x8, x4)
            x4 = self.conv4(x4)
            return [x4, x8, x16, x32]


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        # self.cfg = cfg['backbone']
        self.type = 'mobilenetv2_100'
        chans = [16, 24, 32, 96, 160]
        layers = [1, 2, 3, 5, 6]
        self.pre_trained = True

        if self.type == 'custom_basic':

            self.conv_stem = nn.Sequential(
                BasicConv(3, chans[0], kernel_size=3, stride=2, padding=1),
                BasicConv(chans[0], chans[0], kernel_size=3, stride=1, padding=1),
            )
            self.block1a = nn.Sequential(
                BasicConv(chans[0], chans[1], kernel_size=3, stride=2, padding=1),
                BasicConv(chans[1], chans[1], kernel_size=3, stride=1, padding=1),
            )
            self.block2a = nn.Sequential(
                BasicConv(chans[1], chans[2], kernel_size=3, stride=2, padding=1),
                BasicConv(chans[2], chans[2], kernel_size=3, stride=1, padding=1),
            )
            self.block3a = nn.Sequential(
                BasicConv(chans[2], chans[3], kernel_size=3, stride=2, padding=1),
                BasicConv(chans[3], chans[3], kernel_size=3, stride=1, padding=1),
            )
            self.block4a = nn.Sequential(
                BasicConv(chans[3], chans[4], kernel_size=3, stride=2, padding=1),
                BasicConv(chans[4], chans[4], kernel_size=3, stride=1, padding=1),
            )

            self.deconv4a = Conv2x(chans[4], chans[3], deconv=True, keep_concat=False)
            self.deconv3a = Conv2x(chans[3], chans[2], deconv=True, keep_concat=False)
            self.deconv2a = Conv2x(chans[2], chans[1], deconv=True, keep_concat=False)
            self.deconv1a = Conv2x(chans[1], chans[0], deconv=True, keep_concat=False)

            self.conv1b = Conv2x(chans[0], chans[1], keep_concat=False)
            self.conv2b = Conv2x(chans[1], chans[2], keep_concat=False)
            self.conv3b = Conv2x(chans[2], chans[3], keep_concat=False)
            self.conv4b = Conv2x(chans[3], chans[4], keep_concat=False)

            self.weight_init()

        elif self.type == 'custom_res':

            self.chans = chans
            block_n = [2, 2, 2, 2]

            self.conv_stem = nn.Sequential(
                nn.Conv2d(3, chans[0], 3, 1, 1),
                nn.BatchNorm2d(chans[0]), nn.ReLU6())

            inp = chans[0]
            self.conv = nn.ModuleList()
            for i in range(len(chans) - 1):
                conv: List[nn.ModuleList] = []
                for n in range(block_n[i]):
                    if n == 0:
                        stride = 2
                        downsample = nn.Sequential(
                            conv1x1(inp, chans[i + 1], stride),
                            nn.BatchNorm2d(chans[i + 1]),
                        )
                    else:
                        stride, downsample = 1, None
                    conv.append(BasicBlock(inp, chans[i + 1], stride, downsample=downsample))
                    inp = chans[i + 1]
                self.conv.append(nn.Sequential(*conv))

            self.weight_init()

        elif self.type == 'psm':
            self.inplanes = 32
            self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                           nn.ReLU(inplace=True),
                                           convbn(32, 32, 3, 1, 1, 1),
                                           nn.ReLU(inplace=True),
                                           convbn(32, 32, 3, 1, 1, 1),
                                           nn.ReLU(inplace=True))

            self.layer1 = self._psm_make_layer(PSMBasicBlock, 32, 3, 1, 1, 1)
            self.layer2 = self._psm_make_layer(PSMBasicBlock, 64, 16, 2, 1, 1)
            self.layer3 = self._psm_make_layer(PSMBasicBlock, 128, 3, 1, 1, 1)
            self.layer4 = self._psm_make_layer(PSMBasicBlock, 128, 3, 1, 1, 2)

            self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                         convbn(128, 32, 1, 1, 0, 1),
                                         nn.ReLU(inplace=True))

            self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                         convbn(128, 32, 1, 1, 0, 1),
                                         nn.ReLU(inplace=True))

            self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                         convbn(128, 32, 1, 1, 0, 1),
                                         nn.ReLU(inplace=True))

            self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                         convbn(128, 32, 1, 1, 0, 1),
                                         nn.ReLU(inplace=True))

            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=1, bias=False))

        else:

            pretrained = self.pre_trained
            model = timm.create_model(self.type, pretrained=pretrained, features_only=True)
            if 'resnet' in self.type:
                self.conv1 = model.conv1
                self.bn1 = model.bn1
                self.maxpool = model.maxpool

                self.layer1 = model.layer1
                self.layer2 = model.layer2
                self.layer3 = model.layer3
                self.layer4 = model.layer4
            else:
                self.conv_stem = model.conv_stem
                self.bn1 = model.bn1

                self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
                self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
                self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
                self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
                self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x):
        if self.type == 'custom_basic':
            x2 = self.conv_stem(x)
            x4 = self.block1a(x2)
            x8 = self.block2a(x4)
            x16 = self.block3a(x8)
            x32 = self.block4a(x16)

            x16 = self.deconv4a(x32, x16)
            x8 = self.deconv3a(x16, x8)
            x4 = self.deconv2a(x8, x4)
            x2 = self.deconv1a(x4, x2)

            x4 = self.conv1b(x2, x4)
            x8 = self.conv2b(x4, x8)
            x16 = self.conv3b(x8, x16)
            x32 = self.conv4b(x16, x32)

            return x2, [x4, x8, x16, x32]

        elif self.type == 'custom_res':
            x = self.conv_stem(x)

            outs = []
            x_ = x
            for i in range(len(self.chans) - 1):
                x_ = self.conv[i](x_)
                outs.append(x_)

            return x, outs

        elif self.type == 'psm':
            output2 = self.firstconv(x)
            output2 = self.layer1(output2)
            output_raw = self.layer2(output2)
            output = self.layer3(output_raw)
            output_skip = self.layer4(output)

            output_branch1 = self.branch1(output_skip)
            output_branch1 = F.upsample(
                output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_branch2 = self.branch2(output_skip)
            output_branch2 = F.upsample(
                output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_branch3 = self.branch3(output_skip)
            output_branch3 = F.upsample(
                output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_branch4 = self.branch4(output_skip)
            output_branch4 = F.upsample(
                output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

            output_feature = torch.cat(
                (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
            output_feature = self.lastconv(output_feature)

            xout = [output_feature, output_feature, output_feature, output_feature]
            return output2, xout

        else:
            if 'resnet' in self.type:
                x2 = self.bn1(self.conv1(x))
                x4 = self.layer1(self.maxpool(x2))
                x8 = self.layer2(x4)
                x16 = self.layer3(x8)
                x32 = self.layer4(x16)
            else:
                x = self.bn1(self.conv_stem(x))
                x2 = self.block0(x)
                x4 = self.block1(x2)

                # return x4,x4,x4,x4
                x8 = self.block2(x4)
                x16 = self.block3(x8)
                x32 = self.block4(x16)

            x_out = [x4, x8, x16, x32]

            return x2, x_out

    def _psm_make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)


class unetUp(nn.Module):
    def __init__(self, in_c1, in_c2, out_c):
        super(unetUp, self).__init__()
        self.up_conv1 = BasicTransposeConv2d(in_c1, in_c1 // 2, 2, 2, 0, 1)
        self.reduce_conv2 = BasicConv2d(in_c1 // 2 + in_c2, out_c, 1, 1, 0, 1)
        self.conv = nn.Sequential(
            BasicConv2d(out_c, out_c, 3, 1, 1, 1),
        )

    def forward(self, inputs1, inputs2):  # small scale, large scale
        layer1 = self.up_conv1(inputs1)
        layer2 = self.reduce_conv2(torch.cat([layer1, inputs2], 1))
        output = self.conv(layer2)
        return output


class FeatUp2(SubModule):
    def __init__(self, cfg):
        super(FeatUp2, self).__init__()
        self.cfg = cfg['backbone']
        self.type = self.cfg['type']
        chans = self.cfg['channels'][self.type]

        self.deconv16_8 = unetUp(chans[4], chans[3], chans[3])
        self.deconv8_4 = unetUp(chans[3], chans[2], chans[2])
        self.deconv4_2 = unetUp(chans[2], chans[1], chans[1])
        self.deconv2_1 = unetUp(chans[1], chans[0], chans[0])

        self.conv_16 = nn.Conv2d(chans[4], chans[4], 1, 1, 0, 1, bias=False)
        self.conv_8 = nn.Conv2d(chans[3], chans[3], 1, 1, 0, 1, bias=False)
        self.conv_4 = nn.Conv2d(chans[2], chans[2], 1, 1, 0, 1, bias=False)
        self.conv_2 = nn.Conv2d(chans[1], chans[1], 1, 1, 0, 1, bias=False)
        self.conv_1 = nn.Conv2d(chans[0], chans[0], 1, 1, 0, 1, bias=False)

        self.weight_init()

    def forward(self, featL, featR=None):
        x, x2, x4, x8, x16 = featL

        if featR is not None:
            y, y2, y4, y8, y16 = featR

            x8 = self.deconv16_8(x16, x8)
            y8 = self.deconv16_8(y16, y8)

            x4 = self.deconv8_4(x8, x4)
            y4 = self.deconv8_4(y8, y4)

            x2 = self.deconv4_2(x4, x2)
            y2 = self.deconv4_2(y4, y2)

            x = self.deconv2_1(x2, x)
            y = self.deconv2_1(y2, y)

            x16 = self.conv_16(x16)
            x8 = self.conv_8(x8)
            x4 = self.conv_4(x4)
            x2 = self.conv_2(x2)
            x = self.conv_1(x)

            y16 = self.conv_16(y16)
            y8 = self.conv_8(y8)
            y4 = self.conv_4(y4)
            y2 = self.conv_2(y2)
            y = self.conv_1(y)

            return [x, x2, x4, x8, x16], [y, y2, y4, y8, y16]
        else:
            x8 = self.deconv16_8(x16, x8)
            x4 = self.deconv8_4(x8, x4)
            x2 = self.deconv4_2(x4, x2)
            x = self.deconv2_1(x2, x)

            x16 = self.conv_16(x16)
            x8 = self.conv_8(x8)
            x4 = self.conv_4(x4)
            x2 = self.conv_2(x2)
            x = self.conv_1(x)

            return [x, x2, x4, x8, x16]


class Feature2(SubModule):
    def __init__(self, cfg):
        super(Feature2, self).__init__()
        self.cfg = cfg['backbone']
        self.type = self.cfg['type']
        chans = self.cfg['channels'][self.type]
        layers = self.cfg['layers'][self.type]

        if self.type == 'lignet2':

            self.conv_stem = nn.Sequential(
                BasicConv2d(3, chans[0], 3, 1, 1, 1),
                BasicConv2d(chans[0], chans[0], 3, 1, 1, 1),
            )
            self.block1a = nn.Sequential(
                BasicConv2d(chans[0], chans[1], 3, 2, 1, 1),
                BasicConv2d(chans[1], chans[1], 3, 1, 1, 1),
                BasicConv2d(chans[1], chans[1], 3, 1, 1, 1)
            )
            self.block2a = nn.Sequential(
                BasicConv2d(chans[1], chans[2], 3, 2, 1, 1),
                BasicConv2d(chans[2], chans[2], 3, 1, 1, 1),
                BasicConv2d(chans[2], chans[2], 3, 1, 1, 1)
            )
            self.block3a = nn.Sequential(
                BasicConv2d(chans[2], chans[3], 3, 2, 1, 1),
                BasicConv2d(chans[3], chans[3], 3, 1, 1, 1),
                BasicConv2d(chans[3], chans[3], 3, 1, 1, 1)
            )
            self.block4a = nn.Sequential(
                BasicConv2d(chans[3], chans[4], 3, 2, 1, 1),
                BasicConv2d(chans[4], chans[4], 3, 1, 1, 1),
                BasicConv2d(chans[4], chans[4], 3, 1, 1, 1)
            )

            self.weight_init()

        else:

            pretrained = False if self.cfg['from_scratch'] else True
            model = timm.create_model(self.type, pretrained=pretrained, features_only=True)

            if 'resnet' in self.type:
                self.conv1 = model.conv1
                self.conv1.stride = (1, 1)
                self.bn1 = model.bn1
                self.act1 = model.act1
                self.maxpool = model.maxpool

                self.layer1 = model.layer1
                self.layer2 = model.layer2
                self.layer3 = model.layer3
                self.layer4 = model.layer4
            else:
                self.conv_stem = model.conv_stem
                self.conv_stem.stride = (1, 1)
                self.bn1 = model.bn1
                self.act1 = model.act1

                self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
                self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
                self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
                self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
                self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x):
        if self.type == 'lignet2':
            x = self.conv_stem(x)
            x2 = self.block1a(x)
            x4 = self.block2a(x2)
            x8 = self.block3a(x4)
            x16 = self.block4a(x8)

            return [x, x2, x4, x8, x16]

        else:
            if 'resnet' in self.type:
                x = self.act1(self.bn1(self.conv1(x)))
                x2 = self.layer1(self.maxpool(x))
                x4 = self.layer2(x2)
                x8 = self.layer3(x4)
                x16 = self.layer4(x8)
            else:
                x = self.act1(self.bn1(self.conv_stem(x)))
                x = self.block0(x)
                x2 = self.block1(x)

                x4 = self.block2(x2)
                x8 = self.block3(x4)
                x16 = self.block4(x8)

            return [x, x2, x4, x8, x16]
