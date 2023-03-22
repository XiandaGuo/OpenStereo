from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal_
from modeling.common.modules  import *
import copy

MAX_RANGE=400

class FADAggregator(nn.Module):

    def __init__(self, resBlock=True, maxdisp=192, input_channel=3, encoder_ratio=16, decoder_ratio=16):
        super(FADAggregator, self).__init__()
        
        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.relu = nn.ReLU(inplace=False)
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC*self.eratio
        self.basicD = self.basicC*self.dratio

        self.disp_width = maxdisp // 8 + 16

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        if resBlock:
            self.conv_redir = ResBlock(self.basicE*4, self.basicE, stride=1)
            self.conv3_1 = DyRes(MAX_RANGE//8+16+self.basicE, self.basicE*4)
            self.conv4   = ResBlock(self.basicE*4, self.basicE*8, stride=2)
            self.conv4_1 = ResBlock(self.basicE*8, self.basicE*8)
            self.conv5   = ResBlock(self.basicE*8, self.basicE*16, stride=2)
            self.conv5_1 = ResBlock(self.basicE*16, self.basicE*16)
            self.conv6   = ResBlock(self.basicE*16, self.basicE*32, stride=2)
            self.conv6_1 = ResBlock(self.basicE*32, self.basicE*32)
        else:
            self.conv_redir = conv(self.basicE*4, self.basicE, stride=1)
            self.conv3_1 = conv(self.disp_width+self.basicE, self.basicE*4)
            self.conv4   = conv(self.basicE*4, self.basicE*8, stride=2)
            self.conv4_1 = conv(self.basicE*8, self.basicE*8)
            self.conv5   = conv(self.basicE*8, self.basicE*16, stride=2)
            self.conv5_1 = conv(self.basicE*16, self.basicE*16)
            self.conv6   = conv(self.basicE*16, self.basicE*32, stride=2)
            self.conv6_1 = conv(self.basicE*32, self.basicE*32)

        self.pred_flow6 = predict_flow(self.basicE*32)

        # # iconv with resblock
      

        # iconv with deconv
        self.iconv5 = nn.ConvTranspose2d((self.basicD+self.basicE)*16+1, self.basicD*16, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d((self.basicD+self.basicE)*8+1, self.basicD*8, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d((self.basicD+self.basicE)*4+1, self.basicD*4, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d((self.basicD+self.basicE)*2+1, self.basicD*2, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d((self.basicD+self.basicE)*1+1, self.basicD, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(self.basicD+self.input_channel+1, self.basicD, 3, 1, 1)

        # expand and produce disparity
        self.upconv5 = deconv(self.basicE*32, self.basicD*16)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow5 = predict_flow(self.basicD*16)

        self.upconv4 = deconv(self.basicD*16, self.basicD*8)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow4 = predict_flow(self.basicD*8)

        self.upconv3 = deconv(self.basicD*8, self.basicD*4)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow3 = predict_flow(self.basicD*4)

        self.upconv2 = deconv(self.basicD*4, self.basicD*2)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow2 = predict_flow(self.basicD*2)

        self.upconv1 = deconv(self.basicD*2, self.basicD)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow1 = predict_flow(self.basicD)

        self.upconv0 = deconv(self.basicD, self.basicD)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow0 = predict_flow(self.basicD)
        

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #self.freeze()
        
    def forward(self,img_left ,img_right, conv1_l, conv2_l, conv3a_l, corr_volume, get_features=False):


        # Correlate corr3a_l and corr3a_r
        #out_corr = self.corr(conv3a_l, conv3a_r)
        #out_corr = build_corr(conv3a_l, conv3a_r, max_disp=self.disp_width)
        out_corr = self.corr_activation(corr_volume)
        out_conv3a_redir = self.conv_redir(conv3a_l)
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

        conv3b = self.conv3_1(in_conv3b)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)

        pr6 = self.pred_flow6(conv6b)
        upconv5 = self.upconv5(conv6b)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        pr5 = self.pred_flow5(iconv5)
        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)
        
        pr4 = self.pred_flow4(iconv4)
        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        pr3 = self.pred_flow3(iconv3)
        upconv2 = self.upconv2(iconv3)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2)

        pr2 = self.pred_flow2(iconv2)
        upconv1 = self.upconv1(iconv2)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1)

        pr1 = self.pred_flow1(iconv1)
        upconv0 = self.upconv0(iconv1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0)

        # predict flow
        pr0 = self.pred_flow0(iconv0)
        pr0 = self.relu(pr0)
       

        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)


        # can be chosen outside
        if get_features:
            features = (iconv5, iconv4, iconv3, iconv2, iconv1, iconv0)
            return disps, features
        else:
            return disps
 
    def freeze(self):
        for name, param in self.named_parameters():
            if ('weight' in name) or ('bias' in name):
                param.requires_grad = False

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


