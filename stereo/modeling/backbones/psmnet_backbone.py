# @Time    : 2023/9/2 09:55
# @Author  : zhangchenming
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d


class BasicBlock(nn.Module):
    def __init__(self, downsample, in_planes, out_planes, stride, padding):
        super(BasicBlock, self).__init__()
        self.conv1 = BasicConv2d(in_channels=in_planes, out_channels=out_planes,
                                 norm_layer=nn.BatchNorm2d,
                                 act_layer=partial(nn.ReLU, inplace=True),
                                 kernel_size=3, stride=stride, padding=padding)

        self.conv2 = BasicConv2d(in_channels=out_planes, out_channels=out_planes,
                                 norm_layer=nn.BatchNorm2d, act_layer=None,
                                 kernel_size=3, stride=1, padding=padding)

        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class PsmnetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.firstconv = nn.Sequential(
            BasicConv2d(3, out_channels=32,
                        norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=3, stride=2, padding=1),
            BasicConv2d(32, out_channels=32,
                        norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, out_channels=32,
                        norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=3, stride=1, padding=1)
        )

        self.in_planes = 32
        self.layer1 = self._make_layer(32, 3, 1, 1)
        self.layer2 = self._make_layer(64, 16, 2, 1)
        self.layer3 = self._make_layer(128, 3, 1, 1)
        self.layer4 = self._make_layer(128, 3, 1, 1)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            BasicConv2d(in_channels=128, out_channels=32,
                        norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=1, stride=1),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            BasicConv2d(in_channels=128, out_channels=32,
                        norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=1, stride=1),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            BasicConv2d(in_channels=128, out_channels=32,
                        norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=1, stride=1),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            BasicConv2d(in_channels=128, out_channels=32,
                        norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=1, stride=1),
        )

        self.lastconv = nn.Sequential(
            BasicConv2d(in_channels=320, out_channels=128,
                        norm_layer=nn.BatchNorm2d, act_layer=partial(nn.ReLU, inplace=True),
                        kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 32, kernel_size=1, stride=1, bias=False)
        )

    def _make_layer(self, out_planes, blocks, stride, padding):
        downsample = None
        if stride != 1 or self.in_planes != out_planes:
            downsample = BasicConv2d(in_channels=self.in_planes, out_channels=out_planes,
                                     norm_layer=nn.BatchNorm2d, act_layer=None,
                                     kernel_size=1, stride=stride)

        layers = [BasicBlock(downsample, self.in_planes, out_planes, stride, padding)]
        self.in_planes = out_planes
        for i in range(1, blocks):
            layers.append(
                BasicBlock(None, self.in_planes, out_planes, 1, padding)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        output_2_0 = self.firstconv(x)
        output_2_1 = self.layer1(output_2_0)
        output_4_0 = self.layer2(output_2_1)
        output_4_1 = self.layer3(output_4_0)
        output_8 = self.layer4(output_4_1)

        output_branch1 = self.branch1(output_8)
        output_branch1 = F.interpolate(output_branch1, (output_8.size()[2], output_8.size()[3]),
                                       mode='bilinear', align_corners=True)

        output_branch2 = self.branch2(output_8)
        output_branch2 = F.interpolate(output_branch2, (output_8.size()[2], output_8.size()[3]),
                                       mode='bilinear', align_corners=True)

        output_branch3 = self.branch3(output_8)
        output_branch3 = F.interpolate(output_branch3, (output_8.size()[2], output_8.size()[3]),
                                       mode='bilinear', align_corners=True)

        output_branch4 = self.branch4(output_8)
        output_branch4 = F.interpolate(output_branch4, (output_8.size()[2], output_8.size()[3]),
                                       mode='bilinear', align_corners=True)

        output_feature = torch.cat(
            (output_4_0, output_8, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature
