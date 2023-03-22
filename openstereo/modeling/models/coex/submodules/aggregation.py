from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util_conv import BasicConv
from .utils import channelAtt
from .Submodule import SubModule

import pdb


class Aggregation(SubModule):
    def __init__(self,
                 max_disparity=192,
                 matching_head=1,
                 gce=True,
                 disp_strides=2,
                 channels=[16, 32, 48],
                 blocks_num=[2, 2, 2],
                 spixel_branch_channels=[32, 48]
                 ):
        super(Aggregation, self).__init__()
        backbone_type = 'mobilenetv2_100'
        im_chans = [16,24,32,96,160]
        self.D = int(max_disparity//4)

        self.conv_stem = BasicConv(matching_head, 8,
                                   is_3d=True, kernel_size=3, stride=1, padding=1)

        self.gce = gce
        if gce:
            self.channelAttStem = channelAtt(8,
                                             2*im_chans[1]+spixel_branch_channels[1],
                                             self.D,
                                             )
            self.channelAtt = nn.ModuleList()
            self.channelAttDown = nn.ModuleList()

        self.conv_down = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.conv_skip = nn.ModuleList()
        self.conv_agg = nn.ModuleList()

        channels = [8] + (channels)

        s_disp = disp_strides
        block_n = blocks_num
        inp = channels[0]
        for i in range(3):
            conv: List[nn.Module] = []
            for n in range(block_n[i]):
                stride = (s_disp, 2, 2) if n == 0 else 1
                dilation, kernel_size, padding, bn, relu = 1, 3, 1, True, True
                conv.append(
                    BasicConv(
                        inp, channels[i+1], is_3d=True, bn=bn,
                        relu=relu, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation))
                inp = channels[i+1]
            self.conv_down.append(nn.Sequential(*conv))

            if gce:
                cdfeat_mul = 1 if i == 2 else 2
                self.channelAttDown.append(channelAtt(channels[i+1],
                                           cdfeat_mul*im_chans[i+2],
                                           self.D//(2**(i+1)),
                                           ))

            if i == 0:
                out_chan, bn, relu = 1, False, False
            else:
                out_chan, bn, relu = channels[i], True, True
            self.conv_up.append(
                BasicConv(
                    channels[i+1], out_chan, deconv=True, is_3d=True, bn=bn,
                    relu=relu, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(s_disp, 2, 2)))

            self.conv_agg.append(nn.Sequential(
                BasicConv(
                    channels[i], channels[i], is_3d=True, kernel_size=3, padding=1, stride=1),
                BasicConv(
                    channels[i], channels[i], is_3d=True, kernel_size=3, padding=1, stride=1),))

            self.conv_skip.append(BasicConv(2*channels[i], channels[i], is_3d=True, kernel_size=1, padding=0, stride=1))

            if gce:
                self.channelAtt.append(channelAtt(channels[i],
                                                  2*im_chans[i+1],
                                                  self.D//(2**(i)),
                                                  ))

        self.weight_init()

    def forward(self, img, cost):
        b, c, h, w = img[0].shape

        cost = cost.reshape(b, -1, self.D, h, w)
        cost = self.conv_stem(cost)
        if self.gce:
            cost = self.channelAttStem(cost, img[0])
        
        cost_feat = [cost]

        cost_up = cost
        for i in range(3):
            cost_ = self.conv_down[i](cost_up)
            if self.gce:
                cost_ = self.channelAttDown[i](cost_, img[i+1])

            cost_feat.append(cost_)
            cost_up = cost_

        cost_ = cost_feat[-1]
        for i in range(3):

            cost_ = self.conv_up[-i-1](cost_)
            if cost_.shape != cost_feat[-i-2].shape:
                target_d, target_h, target_w = cost_feat[-i-2].shape[-3:]
                cost_ = F.interpolate(
                    cost_,
                    size=(target_d, target_h, target_w),
                    mode='nearest')
            if i == 2:
                break

            cost_ = torch.cat([cost_, cost_feat[-i-2]], 1)
            cost_ = self.conv_skip[-i-1](cost_)
            cost_ = self.conv_agg[-i-1](cost_)

            if self.gce:
                cost_ = self.channelAtt[-i-1](cost_, img[-i-2])
            
        cost = cost_

        return cost
