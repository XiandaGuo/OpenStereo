from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

from .submodules.utils import AttentionCostVolume
from .submodules.utils import channelAtt
from .submodules.util_conv import BasicConv

import pdb


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

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


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    def forward(self, x):
        out = torch.sum(x*self.disp.data,1, keepdim=True)
        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class PSMNet(nn.Module):
    def __init__(self, cfg):
        super(PSMNet, self).__init__()
        self.maxdisp = cfg['max_disparity']
        self.topkreg = cfg['regression']['top_k']

        self.feature_extraction = feature_extraction()

        self.corr_volume = cfg['corr_volume']
        if cfg['corr_volume']:
            self.cost_volume = AttentionCostVolume(self.maxdisp, 32, 32, 1)

            cost_in_chan = 32
            self.conv_stem = BasicConv(1, cost_in_chan,
                                   is_3d=True, kernel_size=3, stride=1, padding=1)

            self.gce = cfg['gce']
            if cfg['gce']:
                self.channelAtt = channelAtt(cost_in_chan, 32, int(self.maxdisp//4))

        else:
            cost_in_chan = 64

            self.gce = cfg['gce']
            if cfg['gce']:
                self.channelAtt = channelAtt(32, 32, int(self.maxdisp//4))

        self.dres0 = nn.Sequential(convbn_3d(cost_in_chan, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.ind_init = False


    def forward(self, left, right, u0=None, v0=None, training=False):

        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        #matching
        if self.corr_volume:
            cost = (self.cost_volume(refimg_fea, targetimg_fea))[:, :, :-1]
            cost = self.conv_stem(cost)

            if self.gce:
                cost = self.channelAtt(cost, refimg_fea)
        else:
            cost = Variable(
                torch.FloatTensor(
                    refimg_fea.size()[0],
                    refimg_fea.size()[1]*2,
                    self.maxdisp//4, 
                    refimg_fea.size()[2], 
                    refimg_fea.size()[3]).zero_()).cuda()
            for i in range(self.maxdisp//4):
                if i > 0 :
                 cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
                 cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
                else:
                 cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                 cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
            cost = cost.contiguous()

        cost0 = self.dres0(cost)

        if not self.corr_volume and self.gce:
            cost0 = self.channelAtt(cost0, refimg_fea)

        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            if self.topkreg == self.maxdisp:
                cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')

                cost1 = torch.squeeze(cost1,1)
                pred1 = F.softmax(cost1,dim=1)

                pred1 = disparityregression(self.maxdisp)(pred1)

                cost2 = torch.squeeze(cost2,1)
                pred2 = F.softmax(cost2,dim=1)

                pred2 = disparityregression(self.maxdisp)(pred2)

            else:
                cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
                cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')

                corr1, disp1 = self.topkpool(cost1, self.topkreg)
                corr1 = F.softmax(corr1, 2)
                disp1 = torch.sum(corr1*disp1, 2, keepdim=True)
                pred1 = disp1.squeeze(2)*1

                corr2, disp2 = self.topkpool(cost2, self.topkreg)
                corr2 = F.softmax(corr2, 2)
                disp2 = torch.sum(corr2*disp2, 2, keepdim=True)
                pred2 = disp2.squeeze(2)*1

        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        
        if self.topkreg == self.maxdisp:
            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3,1)
            pred3 = F.softmax(cost3,dim=1)

            pred3 = disparityregression(self.maxdisp)(pred3)
        else:
            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')

            corr3, disp3 = self.topkpool(cost3, self.topkreg)
            corr3 = F.softmax(corr3, 2)
            disp3 = torch.sum(corr3*disp3, 2, keepdim=True)
            pred3 = disp3.squeeze(2)*1

        if self.training:
            return [pred1.squeeze(1), pred2.squeeze(1), pred3.squeeze(1), 0]
        else:
            return pred3

    def topkpool(self, cost, k):
        if not self.ind_init:
            self.ind_ = torch.arange(
                self.maxdisp, device=cost.device).reshape(1, 1, -1, 1, 1)
            self.ind = self.ind_.repeat(
                1, 1, 1, cost.shape[-2], cost.shape[-1])
            self.ind_init = True
        if self.ind.shape[-2:] != cost.shape[-2:]:
            self.ind = self.ind_.repeat(
                1, 1, 1, cost.shape[-2], cost.shape[-1])

        if k == cost.shape[2]:
            cv = cost
            disp = self.ind

        elif k == 1:
            pool_ind_ = cost.argsort(2, True)[:, :, :k]
            b, _, _, h, w = pool_ind_.shape
            pool_ind = pool_ind_.new_zeros((b, 1, 3, h, w))
            pool_ind[:, :, 1:2] = pool_ind_
            pool_ind[:, :, 0:1] = torch.max(
                pool_ind_-1, pool_ind_.new_zeros(pool_ind_.shape))
            pool_ind[:, :, 2:] = torch.min(
                pool_ind_+1, self.maxdisp*pool_ind_.new_ones(pool_ind_.shape))
            cv = torch.gather(cost, 2, pool_ind)

            disp = torch.gather(self.ind.repeat(
                cv.shape[0], cv.shape[1], 1, 1, 1), 2, pool_ind)

        else:
            pool_ind = cost.argsort(2, True)[:, :, :k]
            cv = torch.gather(cost, 2, pool_ind)

            disp = torch.gather(self.ind.repeat(
                cv.shape[0], cv.shape[1], 1, 1, 1), 2, pool_ind)

        return cv, disp