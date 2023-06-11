import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.common.model_basics.CoEx import BasicConv


class CostVolume(nn.Module):
    def __init__(self, maxdisp, glue=False, group=1):
        super(CostVolume, self).__init__()
        self.maxdisp = maxdisp + 1
        self.glue = glue
        self.group = group
        self.unfold = nn.Unfold((1, maxdisp + 1), 1, 0, 1)
        self.left_pad = nn.ZeroPad2d((maxdisp, 0, 0, 0))

    def forward(self, x, y, v=None):
        b, c, h, w = x.shape

        unfolded_y = self.unfold(self.left_pad(y)).reshape(
            b, self.group, c // self.group, self.maxdisp, h, w)
        x = x.reshape(b, self.group, c // self.group, 1, h, w)

        cost = (x * unfolded_y).sum(2)
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
        self.costVolume = CostVolume(int(max_disparity // 4), False, head)
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
        cost = self.costVolume(x_ / torch.norm(x_, 2, 1, True),
                               y_ / torch.norm(y_, 2, 1, True))

        return cost


class channelAtt(nn.Module):
    def __init__(self, cv_chan, im_chan, D):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan // 2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att) * cv
        return cv

    def weight_init(self):
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


class Aggregation(nn.Module):
    def __init__(
            self,
            max_disparity=192,
            matching_head=1,
            gce=True,
            disp_strides=2,
            channels=[16, 32, 48],
            blocks_num=[2, 2, 2],
            spixel_branch_channels=[32, 48]
    ):
        super(Aggregation, self).__init__()
        im_chans = [16, 24, 32, 96, 160]
        self.D = int(max_disparity // 4)

        self.conv_stem = BasicConv(matching_head, 8,
                                   is_3d=True, kernel_size=3, stride=1, padding=1)

        self.gce = gce
        if gce:
            self.channelAttStem = channelAtt(
                8,
                2 * im_chans[1] + spixel_branch_channels[1],
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
            conv = []
            for n in range(block_n[i]):
                stride = (s_disp, 2, 2) if n == 0 else 1
                dilation, kernel_size, padding, bn, relu = 1, 3, 1, True, True
                conv.append(
                    BasicConv(
                        inp, channels[i + 1], is_3d=True, bn=bn,
                        relu=relu, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation))
                inp = channels[i + 1]
            self.conv_down.append(nn.Sequential(*conv))

            if gce:
                cdfeat_mul = 1 if i == 2 else 2
                self.channelAttDown.append(channelAtt(channels[i + 1],
                                                      cdfeat_mul * im_chans[i + 2],
                                                      self.D // (2 ** (i + 1)),
                                                      ))

            if i == 0:
                out_chan, bn, relu = 1, False, False
            else:
                out_chan, bn, relu = channels[i], True, True
            self.conv_up.append(
                BasicConv(
                    channels[i + 1], out_chan, deconv=True, is_3d=True, bn=bn,
                    relu=relu, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(s_disp, 2, 2)))

            self.conv_agg.append(nn.Sequential(
                BasicConv(
                    channels[i], channels[i], is_3d=True, kernel_size=3, padding=1, stride=1),
                BasicConv(
                    channels[i], channels[i], is_3d=True, kernel_size=3, padding=1, stride=1), ))

            self.conv_skip.append(
                BasicConv(2 * channels[i], channels[i], is_3d=True, kernel_size=1, padding=0, stride=1))

            if gce:
                self.channelAtt.append(channelAtt(channels[i],
                                                  2 * im_chans[i + 1],
                                                  self.D // (2 ** (i)),
                                                  ))
        self.weight_init()

    def weight_init(self):
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
                cost_ = self.channelAttDown[i](cost_, img[i + 1])

            cost_feat.append(cost_)
            cost_up = cost_

        cost_ = cost_feat[-1]
        for i in range(3):

            cost_ = self.conv_up[-i - 1](cost_)
            if cost_.shape != cost_feat[-i - 2].shape:
                target_d, target_h, target_w = cost_feat[-i - 2].shape[-3:]
                cost_ = F.interpolate(
                    cost_,
                    size=(target_d, target_h, target_w),
                    mode='nearest')
            if i == 2:
                break

            cost_ = torch.cat([cost_, cost_feat[-i - 2]], 1)
            cost_ = self.conv_skip[-i - 1](cost_)
            cost_ = self.conv_agg[-i - 1](cost_)

            if self.gce:
                cost_ = self.channelAtt[-i - 1](cost_, img[-i - 2])

        cost = cost_

        return cost


class CoExCostProcessor(nn.Module):
    def __init__(
            self,
            max_disp=192,
            gce=True,
            matching_weighted=False,
            spixel_branch_channels=[32, 48],
            matching_head=1,
            aggregation_disp_strides=2,
            aggregation_channels=[16, 32, 48],
            aggregation_blocks_num=[2, 2, 2],
            chans=[16, 24, 32, 96, 160]
    ):
        super().__init__()
        self.max_disp = max_disp
        self.spixel_branch_channels = spixel_branch_channels
        self.matching_weighted = matching_weighted
        self.matching_head = matching_head
        self.gce = gce
        self.aggregation_disp_strides = aggregation_disp_strides
        self.aggregation_channels = aggregation_channels
        self.aggregation_blocks_num = aggregation_blocks_num
        self.chans = chans

        self.cost_volume = AttentionCostVolume(
            self.max_disp,
            self.chans[1] * 2 + self.spixel_branch_channels[1],
            self.chans[1] * 2,
            1,
            weighted=self.matching_weighted
        )
        self.cost_agg = Aggregation(
            max_disparity=self.max_disp,
            matching_head=self.matching_head,
            gce=self.gce,
            disp_strides=self.aggregation_disp_strides,
            channels=self.aggregation_channels,
            blocks_num=self.aggregation_blocks_num,
            spixel_branch_channels=self.spixel_branch_channels
        )

    def forward(self, inputs):
        x = inputs['ref_feature']
        y = inputs['tgt_feature']
        cost = (self.cost_volume(x[0], y[0]))[:, :, :-1, :, :]
        cost = self.cost_agg(x, cost)
        return {
            "cost_volume": cost
        }
