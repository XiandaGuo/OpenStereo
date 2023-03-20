from __future__ import print_function

import torch
import torch.nn as nn

from .util_conv import BasicConv
from .Submodule import SubModule

import pdb


class CostVolume(nn.Module):
    def __init__(self, maxdisp, glue=False, group=1):
        super(CostVolume, self).__init__()
        self.maxdisp = maxdisp+1
        self.glue = glue
        self.group = group
        self.unfold = nn.Unfold((1, maxdisp+1), 1, 0, 1)
        self.left_pad = nn.ZeroPad2d((maxdisp, 0, 0, 0))

    def forward(self, x, y, v=None):
        b, c, h, w = x.shape

        unfolded_y = self.unfold(self.left_pad(y)).reshape(
            b, self.group, c//self.group, self.maxdisp, h, w)
        x = x.reshape(b, self.group, c//self.group, 1, h, w)
        
        cost = (x*unfolded_y).sum(2)
        cost = torch.flip(cost, [2])

        if self.glue:
            cross = self.unfold(self.left_pad(v)).reshape(
                b, c, self.maxdisp, h, w)
            cross = torch.flip(cross, [2])
            return cost, cross
        else:
            return cost


class AttentionCostVolume(nn.Module):
    def __init__(self, max_disparity, in_chan, hidden_chan, head=1, weighted=False):
        super(AttentionCostVolume, self).__init__()
        self.costVolume = CostVolume(int(max_disparity//4), False, head)
        self.conv = BasicConv(in_chan, hidden_chan, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(hidden_chan, hidden_chan, kernel_size=1, padding=0, stride=1)
        self.head = head
        self.weighted = weighted
        if weighted:
            self.weights = nn.Parameter(
                    torch.randn(hidden_chan).reshape(1, hidden_chan, 1, 1))

    def forward(self, imL, imR):
        b, _, h, w = imL.shape
        x = self.conv(imL)
        y = self.conv(imR)

        x_ = self.desc(x)
        y_ = self.desc(y)

        if self.weighted:
            weights = torch.sigmoid(self.weights)
            x_ = x_ * weights
            y_ = y_ * weights
        cost = self.costVolume(x_/torch.norm(x_, 2, 1, True),
                               y_/torch.norm(y_, 2, 1, True))
        
        return cost


class disparityregression(nn.Module):
    def __init__(self):
        super(disparityregression, self).__init__()

    def forward(self, x, reg):
        out = torch.nn.functional.conv2d(x,reg)
        out = torch.squeeze(out,1)
        return out


class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan, D):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):
        '''
        '''
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att)*cv
        return cv
