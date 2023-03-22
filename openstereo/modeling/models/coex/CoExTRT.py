from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import timm

from .submodules.Submodule import SubModule
from .submodules.utils import channelAtt


class CoExTRT(nn.Module):
    def __init__(self, cfg):
        super(CoExTRT, self).__init__()
        self.cfg = cfg
        self.type = self.cfg['backbone']['type']
        chans = self.cfg['backbone']['channels'][self.type]\

        self.D = int(self.cfg['max_disparity']/4)

        # set up the feature extraction first
        self.feature = Feature(self.cfg)
        self.up = FeatUp(self.cfg)

        self.stem_2 = nn.Sequential(
            BasicConv(3, self.cfg['spixel']['branch_channels'][0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.cfg['spixel']['branch_channels'][0], self.cfg['spixel']['branch_channels'][0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.cfg['spixel']['branch_channels'][0]), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(self.cfg['spixel']['branch_channels'][0], self.cfg['spixel']['branch_channels'][1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.cfg['spixel']['branch_channels'][1], self.cfg['spixel']['branch_channels'][1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.cfg['spixel']['branch_channels'][1]), nn.ReLU()
            )

        self.cost_volume = AttentionCostVolume(
            cfg['max_disparity'],
            chans[1]*2+self.cfg['spixel']['branch_channels'][1],
            chans[1]*2,
            1,
            weighted=cfg['matching_weighted'])
        matching_head = cfg['matching_head']

        self.cost_agg = Aggregation(
            cfg['backbone'],
            max_disparity=cfg['max_disparity'],
            matching_head=matching_head,
            gce=cfg['gce'],
            disp_strides=cfg['aggregation']['disp_strides'],
            channels=cfg['aggregation']['channels'],
            blocks_num=cfg['aggregation']['blocks_num'],
            spixel_branch_channels=cfg['spixel']['branch_channels'])
        self.regression = Regression(
            max_disparity=cfg['max_disparity'],
            top_k=cfg['regression']['top_k'])

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(chans[1], 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(chans[1]*2+self.cfg['spixel']['branch_channels'][1], chans[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(chans[1], chans[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chans[1]), nn.ReLU()
            )

    def forward(self, imL):
            
        b, c, h, w = imL.shape

        v2, v4, v8, v16, v32 = self.feature(imL)
        x2, y2 = v2[:b//2], v2[b//2:]

        v4, v8, v16, v32 = self.up(v4, v8, v16, v32)
        x4, y4 = v4[:b//2], v4[b//2:]
        x8, y8 = v8[:b//2], v8[b//2:]
        x16, y16 = v16[:b//2], v16[b//2:]
        x32, y32 = v32[:b//2], v32[b//2:]

        stem_2v = self.stem_2(imL)
        stem_4v = self.stem_4(stem_2v)
        stem_2x, stem_2y = stem_2v[:b//2], stem_2v[b//2:]
        stem_4x, stem_4y = stem_4v[:b//2], stem_4v[b//2:]

        x_0 = torch.cat((x4, stem_4x), 1)
        y_0 = torch.cat((y4, stem_4y), 1)

        # Cost volume processing

        cost = (self.cost_volume(x_0, y_0))[:, :, :-1]
        cost = self.cost_agg(x_0, x8, x16, x32, cost)
        
        # spixel guide comp
        xspx = self.spx_4(x_0)
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        # Regression
        # disp_pred = self.regression(cost, spx_pred)

        return cost, spx_pred


class FeatUp(SubModule):
    def __init__(self, cfg):
        super(FeatUp, self).__init__()
        self.cfg = cfg['backbone']
        self.type = self.cfg['type']
        chans = self.cfg['channels'][self.type]

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, x4, x8, x16, x32):

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(x4)
        return x4, x8, x16, x32


class Feature(SubModule):
    def __init__(self, cfg):
        super(Feature, self).__init__()
        self.cfg = cfg['backbone']
        self.type = self.cfg['type']
        chans = self.cfg['channels'][self.type]
        layers = self.cfg['layers'][self.type]

        pretrained = False if self.cfg['from_scratch'] else True
        model = timm.create_model(self.type, pretrained=pretrained, features_only=True)

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x):
        x = self.bn1(self.conv_stem(x))
        x2 = self.block0(x)
        x4 = self.block1(x2)

        # return x4,x4,x4,x4
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        return x2, x4, x8, x16, x32


class Aggregation(SubModule):
    def __init__(self,
                 backbone_cfg,
                 max_disparity=192,
                 matching_head=1,
                 gce=True,
                 disp_strides=2,
                 channels=[16, 32, 48],
                 blocks_num=[2, 2, 2],
                 spixel_branch_channels=[32, 48]
                 ):
        super(Aggregation, self).__init__()
        backbone_type = backbone_cfg['type']
        im_chans = backbone_cfg['channels'][backbone_type]
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
                self.conv_up.append(
                    BasicConvF(
                        channels[i+1], out_chan, deconv=True, is_3d=True,
                        kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(s_disp, 2, 2)))
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

    def forward(self, x4, x8, x16, x32, cost):
        b, c, h, w = x4.shape

        cost = cost.reshape(b, -1, self.D, h, w)
        cost = self.conv_stem(cost)
        cost = self.channelAttStem(cost, x4)
        
        cost_feat = [cost]

        ####################################
        cost_up = cost

        cost_ = self.conv_down[0](cost_up)
        cost_ = self.channelAttDown[0](cost_, x8)

        cost_feat.append(cost_)
        cost_up = cost_

        cost_ = self.conv_down[1](cost_up)
        cost_ = self.channelAttDown[1](cost_, x16)

        cost_feat.append(cost_)
        cost_up = cost_

        cost_ = self.conv_down[2](cost_up)
        cost_ = self.channelAttDown[2](cost_, x32)

        cost_feat.append(cost_)

        ####################################
        cost_ = cost_feat[-1]

        cost_ = self.conv_up[-1](cost_)

        cost_ = torch.cat([cost_, cost_feat[-2]], 1)
        cost_ = self.conv_skip[-1](cost_)
        cost_ = self.conv_agg[-1](cost_)

        cost_ = self.channelAtt[-1](cost_, x16)

        cost_ = self.conv_up[-2](cost_)

        cost_ = torch.cat([cost_, cost_feat[-3]], 1)
        cost_ = self.conv_skip[-2](cost_)
        cost_ = self.conv_agg[-2](cost_)

        cost_ = self.channelAtt[-2](cost_, x8)

        cost_ = self.conv_up[-3](cost_)
            
        cost = cost_

        return cost


class Regression(SubModule):
    def __init__(self,
                 max_disparity=192,
                 top_k=2):
        super(Regression, self).__init__()
        self.D = int(max_disparity//4)
        self.top_k = top_k
        self.ind_init = False

    def forward(self, cost, spg):
        b, _, h, w = spg.shape

        corr, ind = cost.squeeze().sort(0, True)
        corr = F.softmax(corr[:2], 0).detach().clone()
        disp = ind[:2].detach().clone()
        # disp_f = disp.to(torch.float)

        # disp_ = torch.mul(corr, disp_f)
        return disp, corr, spg
        # disp_4 = disp_[0] + disp_[1]
        # disp_4 = disp_4.reshape(b, 1, disp_4.shape[-2], disp_4.shape[-1])

        # feat = self.unfold(disp_4)
        # feat = torch.repeat_interleave(feat, 4, 2)
        # feat = torch.repeat_interleave(feat, 4, 3)
        # disp_1a = (feat*spg)
        # disp_1 = disp_1a.sum(1)

        # disp_1 = disp_1*4  # + 1.5

        # return disp_1

    def unfold(self, x):
        x = F.pad(x, (1,1,1,1))
        out = torch.cat([
                x[:, :, :-2, :-2],
                x[:, :, :-2, 1:-1],
                x[:, :, :-2, 2:],
                x[:, :, 1:-1, :-2],
                x[:, :, 1:-1, 1:-1],
                x[:, :, 1:-1, 2:],
                x[:, :, 2:, :-2],
                x[:, :, 2:, 1:-1],
                x[:, :, 2:, 2:],
            ], 1)
        return out

    def topkpool(self, cost):
        cv, ind = cost.sort(0, True)
        cv = cv[:2]
        pool_ind = ind[:2].type(torch.float)

        return cv, pool_ind


class CostVolume(nn.Module):
    def __init__(self, maxdisp):
        super(CostVolume, self).__init__()
        self.maxdisp = maxdisp+1

    def forward(self, x, y):

        b, c, h, w = x.shape

        corr_ = (x * y)
        corr = corr_.sum(1)[None]
        costs = [corr]

        for i in range(1, self.maxdisp):
            corr_ = (x[:, :, :, i:] * y[:, :, :, :-i])
            corr = corr_.sum(1)[None]
            costs.append(
                    F.pad(corr, (i, 0, 0, 0))
                )
        cost = torch.cat(costs, 1).unsqueeze(0)

        return cost


class AttentionCostVolume(nn.Module):
    def __init__(self, max_disparity, in_chan, hidden_chan, head=1, weighted=False):
        super(AttentionCostVolume, self).__init__()
        self.costVolume = CostVolume(int(max_disparity//4))
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

        cost = self.costVolume(x_/torch.norm(x_, 2, 1, True),
                               y_/torch.norm(y_, 2, 1, True))
        
        return cost


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = nn.LeakyReLU()
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)#, inplace=True)
        return x


class BasicConvF(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, **kwargs):
        super(BasicConvF, self).__init__()

        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        x = torch.cat((x, rem), 1)
        x = self.conv2(x)
        return x