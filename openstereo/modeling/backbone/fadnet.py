from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal_
from modeling.common.modules import *

    
class FadnetBackbone(nn.Module):

    def __init__(self, resBlock=True, maxdisp=192, input_channel=3, encoder_ratio=16, decoder_ratio=16):
        super(FadnetBackbone, self).__init__()
        
        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.relu = nn.ReLU(inplace=False)
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC*self.eratio
        self.basicD = self.basicC*self.dratio

        self.disp_width = maxdisp // 8 + 16

        # shrink and extract features
        self.conv1  = conv(self.input_channel, self.basicE, 7, 2)
        if resBlock:
            self.conv2 = ResBlock(self.basicE, self.basicE*2, stride=2)
            self.conv3 = ResBlock(self.basicE*2, self.basicE*4, stride=2)
        else:
            self.conv2 = conv(self.basicE, self.basicE*2, stride=2)
            self.conv3 = conv(self.basicE*2, self.basicE*4, stride=2)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img_left,img_right):

        # split left image and right image
        # imgs = torch.chunk(inputs, 2, dim = 1)
        # img_left = imgs[0]
        # img_right = imgs[1]

        conv1_l = self.conv1(img_left) 
        conv2_l = self.conv2(conv1_l) 
        conv3a_l = self.conv3(conv2_l) 

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3a_r = self.conv3(conv2_r)

        return conv1_l, conv2_l, conv3a_l, conv3a_r

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]