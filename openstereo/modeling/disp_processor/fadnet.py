from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal_
from modeling.common.modules  import *

class DispNetRes(nn.Module):

    def __init__(self, in_planes, resBlock=True, input_channel=3, encoder_ratio=16, decoder_ratio=16):
        super(DispNetRes, self).__init__()
        
        self.input_channel = input_channel
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC*self.eratio
        self.basicD = self.basicC*self.dratio
        self.resBlock = resBlock
        self.res_scale = 7  # number of residuals

        # improved with shrink res-block layers
        self.conv1   = conv(in_planes, self.basicE, 7, 2)
        if resBlock:
            self.conv2   = ResBlock(self.basicE, self.basicE*2, stride=2)
            self.conv3   = ResBlock(self.basicE*2, self.basicE*4, stride=2)
            self.conv3_1 = ResBlock(self.basicE*4, self.basicE*4)
            self.conv4   = ResBlock(self.basicE*4, self.basicE*8, stride=2)
            self.conv4_1 = ResBlock(self.basicE*8, self.basicE*8)
            self.conv5   = ResBlock(self.basicE*8, self.basicE*16, stride=2)
            self.conv5_1 = ResBlock(self.basicE*16, self.basicE*16)
            self.conv6   = ResBlock(self.basicE*16, self.basicE*32, stride=2)
            self.conv6_1 = ResBlock(self.basicE*32, self.basicE*32)
        else:
            self.conv2   = conv(self.basicE, self.basicE*2, stride=2)
            self.conv3   = conv(self.basicE*2, self.basicE*4, stride=2)
            self.conv3_1 = conv(self.basicE*4, self.basicE*4)
            self.conv4   = conv(self.basicE*4, self.basicE*8, stride=2)
            self.conv4_1 = conv(self.basicE*8, self.basicE*8)
            self.conv5   = conv(self.basicE*8, self.basicE*16, stride=2)
            self.conv5_1 = conv(self.basicE*16, self.basicE*16)
            self.conv6   = conv(self.basicE*16, self.basicE*32, stride=2)
            self.conv6_1 = conv(self.basicE*32, self.basicE*32)

        self.pred_res6 = predict_flow(self.basicE*32)

        # iconv with deconv layers
        self.iconv5 = nn.ConvTranspose2d((self.basicD+self.basicE)*16+1, self.basicD*16, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d((self.basicD+self.basicE)*8+1, self.basicD*8, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d((self.basicD+self.basicE)*4+1, self.basicD*4, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d((self.basicD+self.basicE)*2+1, self.basicD*2, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d((self.basicD+self.basicE)*1+1, self.basicD, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(self.basicD+self.input_channel+1, self.basicD, 3, 1, 1)

        # expand and produce disparity
        self.upconv5 = deconv(self.basicE*32, self.basicD*16)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res5 = predict_flow(self.basicD*16)

        self.upconv4 = deconv(self.basicD*16, self.basicD*8)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res4 = predict_flow(self.basicD*8)

        self.upconv3 = deconv(self.basicD*8, self.basicD*4)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res3 = predict_flow(self.basicD*4)

        self.upconv2 = deconv(self.basicD*4, self.basicD*2)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res2 = predict_flow(self.basicD*2)

        self.upconv1 = deconv(self.basicD*2, self.basicD)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res1 = predict_flow(self.basicD)

        self.upconv0 = deconv(self.basicD, self.basicD)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_res0 = predict_flow(self.basicD)

        self.relu = nn.ReLU(inplace=False) 

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, inputs, flows, get_features=False):

        input_features = inputs
        if type(flows) == tuple:
            base_flow = flows
        else:
            base_flow = [F.interpolate(flows, scale_factor=2**(-i)) for i in range(7)]

        conv1 = self.conv1(input_features)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3(conv2)
        conv3b = self.conv3_1(conv3a)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)

        pr6_res = self.pred_res6(conv6b)
        pr6 = pr6_res + base_flow[6]

        upconv5 = self.upconv5(conv6b)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        pr5_res = self.pred_res5(iconv5)
        pr5 = pr5_res + base_flow[5]

        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)

        pr4_res = self.pred_res4(iconv4)
        pr4 = pr4_res + base_flow[4]
        
        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        pr3_res = self.pred_res3(iconv3)
        pr3 = pr3_res + base_flow[3]

        upconv2 = self.upconv2(iconv3)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2), 1)
        iconv2 = self.iconv2(concat2)

        pr2_res = self.pred_res2(iconv2)
        pr2 = pr2_res + base_flow[2]

        upconv1 = self.upconv1(iconv2)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1), 1)
        iconv1 = self.iconv1(concat1)

        pr1_res = self.pred_res1(iconv1)
        pr1 = pr1_res + base_flow[1]

        upconv0 = self.upconv0(iconv1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, input_features[:, :self.input_channel, :, :]), 1)
        iconv0 = self.iconv0(concat0)

        # predict flow residual
        pr0_res = self.pred_res0(iconv0)
        pr0 = pr0_res + base_flow[0]

        # apply ReLU
        pr0 = self.relu(pr0)
        pr1 = self.relu(pr1)
        pr2 = self.relu(pr2)
        pr3 = self.relu(pr3)
        pr4 = self.relu(pr4)
        pr5 = self.relu(pr5)
        pr6 = self.relu(pr6)

        if get_features:
            return pr0, pr1, pr2, pr3, pr4, pr5, pr6, iconv0
        else:
            return pr0, pr1, pr2, pr3, pr4, pr5, pr6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]