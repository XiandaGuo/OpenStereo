import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from . import loss_functions as lf
from .deformable_refine import DeformableRefineF
from .submodule import convbn_3d, feature_extraction, DisparityRegression


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8

        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        # print('pre2', pre.size())

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        # print('out', out.size())

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class hourglass_gwcnet(nn.Module):
    def __init__(self, inplanes):
        super(hourglass_gwcnet, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 4, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(inplanes * 4, inplanes * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes * 4, inplanes * 2, kernel_size=3, padding=1,
                                                      output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1,
                                                      output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inplanes))

        self.redir1 = convbn_3d(inplanes, inplanes, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class PSMNet(nn.Module):
    def __init__(self, maxdisp, struct_fea_c, fuse_mode, affinity_settings, udc, refine):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.sfc = struct_fea_c
        self.affinity_settings = affinity_settings
        self.udc = udc
        self.refine = refine

        self.feature_extraction = feature_extraction(self.sfc, fuse_mode, affinity_settings)

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass_gwcnet(32)
        self.dres3 = hourglass_gwcnet(32)
        self.dres4 = hourglass_gwcnet(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        if refine == 'csr':
            self.refine_module = DeformableRefineF(feature_c=64, node_n=2, modulation=True, cost=True)
        else:
            self.refine_module = DeformableRefineF(feature_c=64, node_n=2, modulation=True, cost=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)
        # matching
        cost = torch.zeros(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp // 4,
                           refimg_fea.size()[2], refimg_fea.size()[3]).to(refimg_fea.device)

        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.udc:
            win_s = 5
        else:
            win_s = 0

        if self.training:
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)

            cost1 = F.interpolate(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                                  align_corners=True)
            cost2 = F.interpolate(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                                  align_corners=True)

            cost1 = torch.squeeze(cost1, 1)
            distribute1 = F.softmax(cost1, dim=1)
            pred1 = DisparityRegression(self.maxdisp, win_size=win_s)(distribute1)

            cost2 = torch.squeeze(cost2, 1)
            distribute2 = F.softmax(cost2, dim=1)
            pred2 = DisparityRegression(self.maxdisp, win_size=win_s)(distribute2)

        cost3 = self.classif3(out3)
        cost3 = F.interpolate(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                              align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        distribute3 = F.softmax(cost3, dim=1)
        pred3 = DisparityRegression(self.maxdisp, win_size=win_s)(distribute3)

        if self.refine == 'csr':
            costr, offset, m = self.refine_module(left, cost3.squeeze(1))
            distributer = F.softmax(costr, dim=1)
            predr = DisparityRegression(self.maxdisp, win_size=win_s)(distributer)

        else:
            predr = self.refine_module(left, pred3.unsqueeze(1))
            predr = predr.squeeze(1)

        if self.training:
            return pred1, pred2, pred3, distribute1, distribute2, distribute3, distributer, predr
        else:
            if self.refine:
                return predr
            else:
                return pred3
